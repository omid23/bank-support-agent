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
