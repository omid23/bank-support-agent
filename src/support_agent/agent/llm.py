"""Model adapters with tool calling: Anthropic Messages API and OpenAI-compatible chat.

The agent works on a provider-neutral transcript:

* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}``
* ``{"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}``

Each adapter converts that to its wire format. Plain httpx, no SDKs, so the
request shapes are visible and testable.

Token accounting is the same for both: ``input_tokens`` is the uncached part of the prompt, and
``cache_read_tokens`` / ``cache_write_tokens`` the parts served from or written to the prompt cache.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMError(Exception):
    pass


class LLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[dict], tools: Sequence[ToolSpec]) -> LLMResponse: ...


RETRY = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_RETRY_AFTER = 30.0


def _retry_after(resp: httpx.Response | None) -> float | None:
    """The server's requested wait in seconds (``retry-after``), capped; None if absent or unparseable."""
    value = resp.headers.get("retry-after") if resp is not None else None
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER) if value is not None else None
    except ValueError:
        return None


def _post_with_retry(client: httpx.Client, url: str, headers: dict, body: dict, retries: int) -> dict:
    last = ""
    for attempt in range(retries + 1):
        resp = None
        try:
            resp = client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in RETRY:
                break
        if attempt < retries:
            wait = _retry_after(resp)
            time.sleep(wait if wait is not None else min(2 ** attempt, 8))
    raise LLMError(last)


class AnthropicLLM:
    """Anthropic Messages API, with prompt caching on by default.

    The policy, the system prompt and ~60 tool schemas are the same on every call of a conversation, and an
    agent turn makes several calls. Two cache breakpoints keep the model from re-reading them each time:

    * an explicit one on the system block, which caches tools + system (they render in that order), and
    * top-level automatic caching, which moves forward with the conversation so each call reads the
      previous call's prefix and only the new tail is processed.

    ``warm()`` writes the tools + system entry before the first turn (a voice call's greeting is a good moment).
    ``cache=False`` sends plain requests, for gateways that reject ``cache_control``.
    """

    name = "anthropic"
    CACHE = {"type": "ephemeral"}

    def __init__(self, model: str = "claude-sonnet-5-5", *, max_tokens: int = 2048, api_key_env: str = "ANTHROPIC_API_KEY",
                 base_url: str = "https://api.anthropic.com", retries: int = 3, cache: bool = True,
                 client: httpx.Client | None = None):
        self.model, self.max_tokens, self.api_key_env = model, max_tokens, api_key_env
        self.base_url, self.retries, self.cache = base_url.rstrip("/"), retries, cache
        self._client = client or httpx.Client(timeout=120)

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if m.get("is_error"):
                    block["is_error"] = True
                # Consecutive tool results belong in one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def _body(self, system: str, tools: Sequence[ToolSpec], max_tokens: int) -> dict[str, Any]:
        """The request without messages. complete() and warm() share it so their cached prefixes are byte-identical."""
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens}
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        if self.cache and system:
            body["system"] = [{"type": "text", "text": system, "cache_control": self.CACHE}]
        else:
            body["system"] = system
        return body

    def _post(self, body: dict) -> tuple[dict, float]:
        headers = {"x-api-key": os.environ.get(self.api_key_env, ""), "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        started = time.perf_counter()
        data = _post_with_retry(self._client, f"{self.base_url}/v1/messages", headers, body, self.retries)
        return data, (time.perf_counter() - started) * 1000

    def complete(self, system: str, messages: list[dict], tools: Sequence[ToolSpec]) -> LLMResponse:
        body = self._body(system, tools, self.max_tokens)
        body["messages"] = self.to_wire(messages)
        if self.cache:
            body["cache_control"] = self.CACHE  # automatic breakpoint on the conversation tail
        data, ms = self._post(body)
        text, calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        return self._response(data, "".join(text), calls, ms)

    def warm(self, system: str, tools: Sequence[ToolSpec]) -> LLMResponse:
        """Write the tools + system cache entry ahead of the first real call: prefill only, no output tokens.

        The breakpoint is the explicit one on the system block; top-level caching is left off so the
        placeholder message is not part of what gets cached."""
        if not self.cache:
            raise LLMError("prompt caching is off")
        body = self._body(system, tools, 0)
        body["messages"] = [{"role": "user", "content": "warmup"}]
        data, ms = self._post(body)
        return self._response(data, "", [], ms)

    def _response(self, data: dict, text: str, calls: list[ToolCall], ms: float) -> LLMResponse:
        usage = data.get("usage") or {}
        return LLMResponse(text, calls, data.get("stop_reason", ""), int(usage.get("input_tokens") or 0),
                           int(usage.get("output_tokens") or 0), data.get("model", self.model), ms,
                           int(usage.get("cache_read_input_tokens") or 0),
                           int(usage.get("cache_creation_input_tokens") or 0))


class OpenAICompatLLM:
    """OpenAI, Azure OpenAI (via base_url), vLLM, Ollama and other compatible servers."""

    name = "openai"

    def __init__(self, model: str = "gpt-6-luna", *, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 2048, retries: int = 3,
                 client: httpx.Client | None = None):
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.max_tokens, self.retries = max_tokens, retries
        self._client = client or httpx.Client(timeout=120)

    @staticmethod
    def to_wire(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                          "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                         for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def complete(self, system: str, messages: list[dict], tools: Sequence[ToolSpec]) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": self.to_wire(system, messages),
                                "max_completion_tokens": self.max_tokens}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                                "parameters": t.parameters}} for t in tools]
        headers = {"Authorization": f"Bearer {os.environ.get(self.api_key_env, '')}"}
        started = time.perf_counter()
        data = _post_with_retry(self._client, f"{self.base_url}/chat/completions", headers, body, self.retries)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError("response has no choices") from exc
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(tc.get("id", ""), fn.get("name", ""), args if isinstance(args, dict) else {}))
        usage = data.get("usage") or {}
        # OpenAI caches long prefixes on its own; prompt_tokens includes the cached part.
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return LLMResponse(msg.get("content") or "", calls, choice.get("finish_reason", ""), prompt - cached,
                           int(usage.get("completion_tokens") or 0), data.get("model", self.model),
                           (time.perf_counter() - started) * 1000, cache_read_tokens=cached)


class ScriptedLLM:
    """Deterministic stand-in for tests and demos: replays responses or calls a policy function."""

    name = "scripted"
    model = "scripted"

    def __init__(self, script: Sequence[LLMResponse] | Callable[[str, list[dict]], LLMResponse]):
        self._script = script
        self._i = 0
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.calls.append(list(messages))
        if callable(self._script):
            return self._script(system, messages)
        if self._i >= len(self._script):
            raise LLMError("script exhausted")
        resp = self._script[self._i]
        self._i += 1
        return resp


def make_llm(provider: str, model: str | None = None, **kwargs) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model or "claude-sonnet-5-5", **kwargs)
    if provider == "openai":
        return OpenAICompatLLM(model or "gpt-6-luna", **kwargs)
    raise ValueError(f"unknown provider '{provider}' (anthropic | openai)")
