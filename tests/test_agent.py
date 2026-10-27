import json

import httpx
import pytest

from support_agent.agent.backend import ScriptedBackend, base_tool_specs
from support_agent.agent.llm import (
    AnthropicLLM,
    LLMError,
    LLMResponse,
    OpenAICompatLLM,
    ScriptedLLM,
    ToolCall,
    ToolSpec,
    make_llm,
)
from support_agent.agent.runtime import AgentState, BankCalls, LocalSession, Reply, SupportAgent
from support_agent.kb.store import policy_text

from .conftest import REWARDS_DOC, REWARDS_TOOL


def _agent(script, guard, searcher):
    return SupportAgent(ScriptedLLM(script), base_tool_specs(), guard, searcher, policy_text())


def test_search_then_answer(guard, searcher):
    agent = _agent([LLMResponse("", [ToolCall("s", "search_knowledge", {"query": "correct rewards points"})]),
                    LLMResponse("I can help with that.")], guard, searcher)
    state = AgentState()
    agent.on_user(state, "My points are wrong")
    out = agent.next(state)
    assert isinstance(out, Reply) and out.text == "I can help with that."
    tool_msg = next(m for m in state.messages if m["role"] == "tool")
    assert "Internal tools named in this document" in tool_msg["content"]
    assert state.guard.read_docs and state.usage["model_calls"] == 2
    assert "Rho-Bank" in agent.system and "log_verification" in agent.system


def test_read_document_lists_tool_signatures(guard, searcher):
    agent = _agent([LLMResponse("", [ToolCall("r", "read_document", {"doc_id": REWARDS_DOC})]), LLMResponse("ok")],
                   guard, searcher)
    state = AgentState()
    agent.on_user(state, "hi")
    agent.next(state)
    content = next(m for m in state.messages if m["role"] == "tool")["content"]
    assert "Tool signatures:" in content and REWARDS_TOOL in content
    assert REWARDS_DOC in state.guard.read_docs


def test_blocked_call_never_reaches_the_bank(guard, searcher):
    backend = ScriptedBackend(base_tool_specs())
    agent = _agent([LLMResponse("", [ToolCall("c", "get_credit_card_accounts_by_user", {"user_id": "u_bob"})]),
                    LLMResponse("First I need to verify your identity.")], guard, searcher)
    session = LocalSession(agent, backend)
    assert session.send("show my cards") == "First I need to verify your identity."
    assert backend.calls == []
    blocked = [m for m in session.state.messages if m["role"] == "tool" and m["content"].startswith("Blocked")]
    assert blocked and "(identity)" in blocked[0]["content"]
    assert any(e["kind"] == "guard" and not e["allowed"] for e in session.state.events)


def test_verified_flow_executes_bank_calls(guard, searcher):
    backend = ScriptedBackend(base_tool_specs(), {"log_verification": "Verification logged.",
                                                  "get_credit_card_accounts_by_user": '[{"account_id": "cc_1"}]'})
    script = [LLMResponse("", [ToolCall("v", "log_verification", {"user_id": "u_bob", "name": "Bob"})]),
              LLMResponse("", [ToolCall("c", "get_credit_card_accounts_by_user", {"user_id": "u_bob"})]),
              LLMResponse("You have one card.")]
    session = LocalSession(_agent(script, guard, searcher), backend)
    assert session.send("I'm Bob") == "You have one card."
    assert [c.name for c in backend.calls] == ["log_verification", "get_credit_card_accounts_by_user"]
    assert session.trace()["guard"]["verified"] == ["u_bob"]


def test_tau2_style_step_api(guard, searcher):
    agent = _agent([LLMResponse("", [ToolCall("v", "log_verification", {"user_id": "u_bob"})]), LLMResponse("Done.")],
                   guard, searcher)
    state = AgentState()
    agent.on_user(state, "hi")
    out = agent.next(state)
    assert isinstance(out, BankCalls) and out.calls[0].id == "v" and "v" in state.pending
    agent.on_tool_results(state, [("v", "ok", False)])
    assert "u_bob" in state.guard.verified and not state.pending
    assert isinstance(agent.next(state), Reply)


def test_dict_arguments_are_serialised(guard, searcher):
    guard.rules = set()
    agent = _agent([LLMResponse("", [ToolCall("c", "call_discoverable_agent_tool",
                                              {"agent_tool_name": REWARDS_TOOL, "arguments": {"transaction_id": "t"}})])],
                   guard, searcher)
    state = AgentState()
    agent.on_user(state, "x")
    out = agent.next(state)
    assert json.loads(out.calls[0].arguments["arguments"]) == {"transaction_id": "t"}


def test_unknown_tool_and_model_failure(guard, searcher):
    agent = _agent([LLMResponse("", [ToolCall("x", "drop_tables", {})])], guard, searcher)
    state = AgentState()
    agent.on_user(state, "x")
    out = agent.next(state)  # script runs out after the unknown tool -> LLMError -> polite handoff
    assert isinstance(out, Reply) and "transfer" in out.text
    assert any(m["content"].startswith("Error: unknown tool") for m in state.messages if m["role"] == "tool")
    assert state.events[-1]["kind"] == "model_error"


def test_step_budget(guard, searcher):
    loop = lambda s, m: LLMResponse("", [ToolCall("s", "search_knowledge", {"query": "fees"})])  # noqa: E731
    agent = SupportAgent(ScriptedLLM(loop), base_tool_specs(), guard, searcher, policy_text(), max_inner_steps=3)
    state = AgentState()
    agent.on_user(state, "x")
    assert isinstance(agent.next(state), Reply) and state.events[-1]["kind"] == "step_budget"


def test_model_adapters_wire_format(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    seen = {}

    def anthropic(req: httpx.Request) -> httpx.Response:
        seen["a"] = json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "tool_use", "id": "t", "name": "search_knowledge",
                                                      "input": {"query": "q"}}],
                                         "usage": {"input_tokens": 5, "output_tokens": 2}})

    def openai(req: httpx.Request) -> httpx.Response:
        seen["o"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})

    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("t1", "search_knowledge", {"query": "q"})]},
            {"role": "tool", "tool_call_id": "t1", "name": "search_knowledge", "content": "x", "is_error": False}]
    tools = [ToolSpec("search_knowledge", "d", {"type": "object", "properties": {}})]
    r = AnthropicLLM(client=httpx.Client(transport=httpx.MockTransport(anthropic))).complete("sys", msgs, tools)
    assert r.tool_calls[0].arguments == {"query": "q"}
    assert seen["a"]["messages"][2]["content"][0]["type"] == "tool_result"
    r = OpenAICompatLLM(client=httpx.Client(transport=httpx.MockTransport(openai))).complete("sys", msgs, tools)
    assert r.text == "done" and seen["o"]["messages"][0] == {"role": "system", "content": "sys"}
    with pytest.raises(ValueError):
        make_llm("nope")


def test_scripted_llm_exhausts():
    llm = ScriptedLLM([LLMResponse("a")])
    llm.complete("s", [], [])
    with pytest.raises(LLMError):
        llm.complete("s", [], [])


def _capturing_anthropic(bodies: list, usage: dict | None = None, **kwargs) -> AnthropicLLM:
    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                                         "usage": usage or {}})

    return AnthropicLLM(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_anthropic_prompt_caching(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    bodies: list = []
    llm = _capturing_anthropic(bodies, {"input_tokens": 40, "output_tokens": 3, "cache_read_input_tokens": 3900,
                                        "cache_creation_input_tokens": 120})
    tools = base_tool_specs()
    r = llm.complete("policy", [{"role": "user", "content": "hi"}], tools)
    assert (r.input_tokens, r.cache_read_tokens, r.cache_write_tokens) == (40, 3900, 120)
    llm.warm("policy", tools)
    real, warm = bodies
    # tools + system cached by the marker on the system block; the conversation tail by automatic caching
    assert real["system"] == [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}]
    assert real["cache_control"] == {"type": "ephemeral"}
    # the pre-warm writes exactly the prefix the real call reads, and nothing after it
    assert warm["max_tokens"] == 0 and "cache_control" not in warm
    assert json.dumps([warm["tools"], warm["system"]]) == json.dumps([real["tools"], real["system"]])


def test_anthropic_caching_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    bodies: list = []
    llm = _capturing_anthropic(bodies, cache=False)
    llm.complete("policy", [{"role": "user", "content": "hi"}], [])
    assert bodies[0]["system"] == "policy" and "cache_control" not in bodies[0]
    with pytest.raises(LLMError):
        llm.warm("policy", [])


def test_openai_cached_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 5000, "completion_tokens": 9,
                                                   "prompt_tokens_details": {"cached_tokens": 4864}}})

    r = OpenAICompatLLM(client=httpx.Client(transport=httpx.MockTransport(handler))).complete("s", [], [])
    assert (r.input_tokens, r.cache_read_tokens, r.output_tokens) == (136, 4864, 9)


def test_retry_after_is_honoured(monkeypatch):
    from support_agent.agent import llm as llm_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    slept: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", slept.append)
    replies = iter([httpx.Response(429, headers={"retry-after": "2"}, text="slow down"),
                    httpx.Response(529, headers={"retry-after": "9999"}, text="overloaded"),
                    httpx.Response(503, text="unavailable"),
                    httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})])
    llm = AnthropicLLM(client=httpx.Client(transport=httpx.MockTransport(lambda req: next(replies))))
    assert llm.complete("s", [{"role": "user", "content": "x"}], []).text == "ok"
    assert slept == [2.0, llm_mod.MAX_RETRY_AFTER, 4]  # header, capped header, then exponential backoff


def test_agent_warm_and_cache_usage(guard, searcher):
    class WarmLLM(ScriptedLLM):
        def __init__(self, script, fail=False):
            super().__init__(script)
            self.warmed, self.fail = [], fail

        def warm(self, system, tools):
            if self.fail:
                raise LLMError("HTTP 400: bad request")
            self.warmed.append((system, [t.name for t in tools]))
            return LLMResponse("", cache_write_tokens=3800)

    llm = WarmLLM([LLMResponse("Hello.", input_tokens=12, output_tokens=2, cache_read_tokens=3800)])
    agent = SupportAgent(llm, base_tool_specs(), guard, searcher, policy_text())
    session = LocalSession(agent, ScriptedBackend(base_tool_specs()))
    assert session.warm() is True
    assert llm.warmed == [(agent.system, [t.name for t in agent.specs])]
    session.send("hi")
    usage = session.trace()["usage"]
    assert (usage["cache_write_tokens"], usage["cache_read_tokens"], usage["input_tokens"]) == (3800, 3800, 12)
    assert [e["kind"] for e in session.state.events][:2] == ["warm", "model"]

    assert _agent([], guard, searcher).warm(AgentState()) is False  # model without warm(): nothing to do
    failing = SupportAgent(WarmLLM([], fail=True), base_tool_specs(), guard, searcher, policy_text())
    state = AgentState()
    assert failing.warm(state) is False and state.events[-1]["kind"] == "warm_error"


def _vertex(bodies: list, urls: list, headers: list, **kwargs):
    from support_agent.agent.llm import VertexLLM

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        urls.append(str(req.url))
        headers.append(req.headers)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 7, "cache_read_input_tokens": 3900}})

    kwargs.setdefault("token_provider", lambda: "ya29.token")
    return VertexLLM(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_vertex_request(monkeypatch):
    for key in ("CLOUD_ML_REGION", "ANTHROPIC_VERTEX_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    bodies, urls, headers = [], [], []
    llm = _vertex(bodies, urls, headers, project_id="bank-prod")
    tools = base_tool_specs()
    r = llm.complete("policy", [{"role": "user", "content": "hi"}], tools)
    assert r.text == "hi" and r.cache_read_tokens == 3900
    assert urls[0] == ("https://aiplatform.googleapis.com/v1/projects/bank-prod/locations/global/publishers/"
                       "anthropic/models/claude-sonnet-5-5:rawPredict")
    assert headers[0]["authorization"] == "Bearer ya29.token" and "x-api-key" not in headers[0]
    body = bodies[0]
    assert "model" not in body and body["anthropic_version"] == "vertex-2023-10-16"
    # caching and the pre-warm carry over from the Anthropic adapter unchanged
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"} and body["cache_control"] == {"type": "ephemeral"}
    llm.warm("policy", tools)
    assert bodies[1]["max_tokens"] == 0 and json.dumps([bodies[1]["tools"], bodies[1]["system"]]) == \
        json.dumps([body["tools"], body["system"]])


@pytest.mark.parametrize("region, host", [("us-east5", "https://us-east5-aiplatform.googleapis.com/v1"),
                                          ("eu", "https://aiplatform.eu.rep.googleapis.com/v1")])
def test_vertex_regions_from_env(monkeypatch, region, host):
    monkeypatch.setenv("CLOUD_ML_REGION", region)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "bank-prod")
    monkeypatch.delenv("ANTHROPIC_VERTEX_BASE_URL", raising=False)
    urls: list = []
    _vertex([], urls, [], model="claude-opus-5-5").complete("s", [{"role": "user", "content": "x"}], [])
    assert urls[0] == f"{host}/projects/bank-prod/locations/{region}/publishers/anthropic/models/claude-opus-5-5:rawPredict"


def test_vertex_auth_and_project_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_VERTEX_PROJECT_ID"):
        _vertex([], [], [])

    def expired():
        raise RuntimeError("Reauthentication is needed")

    llm = _vertex([], [], [], project_id="p", token_provider=expired)
    with pytest.raises(LLMError, match="Google Cloud auth failed"):  # the runtime turns this into a polite handoff
        llm.complete("s", [{"role": "user", "content": "x"}], [])


def test_vertex_application_default_credentials(monkeypatch):
    import sys
    import types

    from support_agent.agent.llm import CLOUD_PLATFORM_SCOPE

    class Creds:
        token, expired, refreshed = None, False, 0

        def refresh(self, request):
            self.refreshed += 1
            self.token, self.expired = f"tok{self.refreshed}", False

    creds, seen = Creds(), {}

    def default(scopes):
        seen["scopes"] = scopes
        return creds, "adc-project"

    google = types.ModuleType("google")
    auth = types.ModuleType("google.auth")
    auth.default = default
    transport = types.ModuleType("google.auth.transport")
    requests_mod = types.ModuleType("google.auth.transport.requests")
    requests_mod.Request = object
    google.auth = auth
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.auth", auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_mod)
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)

    urls, headers = [], []
    llm = _vertex([], urls, headers, token_provider=None)
    assert seen["scopes"] == [CLOUD_PLATFORM_SCOPE] and llm.project_id == "adc-project"
    msgs = [{"role": "user", "content": "x"}]
    llm.complete("s", msgs, [])
    llm.complete("s", msgs, [])
    creds.expired = True
    llm.complete("s", msgs, [])
    assert [h["authorization"] for h in headers] == ["Bearer tok1", "Bearer tok1", "Bearer tok2"]
    assert "/projects/adc-project/" in urls[0]


def test_default_llm_name(monkeypatch):
    from support_agent.agent.llm import default_llm_name

    for key in ("SUPPORT_AGENT_LLM", "ANTHROPIC_API_KEY", "ANTHROPIC_VERTEX_PROJECT_ID"):
        monkeypatch.delenv(key, raising=False)
    assert default_llm_name() == "anthropic/claude-sonnet-5-5"
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "bank-prod")
    assert default_llm_name() == "vertex/claude-sonnet-5-5"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert default_llm_name() == "anthropic/claude-sonnet-5-5"  # an Anthropic key wins when both are set
    monkeypatch.setenv("SUPPORT_AGENT_LLM", "openai/gpt-5")
    assert default_llm_name() == "openai/gpt-5"
