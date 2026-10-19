"""τ³-bench integration: the support agent as a tau2 ``HalfDuplexAgent``, a backend over a tau2 environment,
and an oracle run that drives reference trajectories through tau2's own orchestrator and grader.

Requires ``pip install 'bank-support-agent[tau2]'`` (tau2-bench needs Python 3.12+).
"""

from __future__ import annotations

import json
from typing import Any

from ..guard.ownership import OwnershipIndex, base_db, merge_initial_state
from ..guard.policy import PolicyGuard
from ..kb.search import Searcher
from ..kb.store import default_kb, policy_text
from .llm import LLM, ToolCall, ToolSpec
from .runtime import AgentState, Reply, SupportAgent

AGENT_NAME = "guarded_rag_agent"
_SEARCHER: Searcher | None = None


def shared_searcher() -> Searcher:
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = Searcher(default_kb())
    return _SEARCHER


def specs_from_tau2(tools: list) -> list[ToolSpec]:
    out = []
    for t in tools:
        fn = t.openai_schema["function"]
        out.append(ToolSpec(fn["name"], fn.get("description", ""), fn.get("parameters") or {"type": "object",
                                                                                              "properties": {}}))
    return out


def _db_dict(tools: list) -> dict | None:
    """The live TransactionalDB behind the harness's tools, as a plain dict (for ownership checks)."""
    for t in tools:
        func = getattr(t, "_func", None) or getattr(t, "func", None)
        owner = getattr(func, "__self__", None)
        db = getattr(owner, "db", None)
        if db is not None and hasattr(db, "model_dump"):
            return db.model_dump()
    return None


def make_tau2_agent_class():
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
    from tau2.data_model.message import ToolCall as Tau2ToolCall

    class GuardedRagAgent(HalfDuplexAgent[AgentState]):
        """Knowledge search + policy guard + any tool-calling model, inside tau2's turn-based harness."""

        def __init__(self, tools: list, domain_policy: str, llm: LLM, task: Any = None, max_inner_steps: int = 12):
            super().__init__(tools=tools, domain_policy=domain_policy)
            db = _db_dict(tools) or merge_initial_state(base_db(), getattr(task, "initial_state", None) and
                                                        task.initial_state.model_dump())
            self.ownership = OwnershipIndex(db)
            self.core = SupportAgent(llm, specs_from_tau2(tools), PolicyGuard(self.ownership), shared_searcher(),
                                     policy_text(), max_inner_steps=max_inner_steps)

        def get_init_state(self, message_history: list | None = None) -> AgentState:
            state = AgentState()
            for m in message_history or []:
                if isinstance(m, AssistantMessage) and m.content:
                    self.core.on_assistant_text(state, m.content)
                elif isinstance(m, UserMessage) and m.content:
                    self.core.on_user(state, m.content)
            return state

        def generate_next_message(self, message, state: AgentState):
            if isinstance(message, UserMessage):
                self.core.on_user(state, message.content or "")
            elif isinstance(message, ToolMessage):
                self.core.on_tool_results(state, [(message.id, message.content or "", bool(message.error))])
            elif isinstance(message, MultiToolMessage):
                self.core.on_tool_results(state, [(m.id, m.content or "", bool(m.error)) for m in message.tool_messages])
            self._refresh_ownership()
            out = self.core.next(state)
            if isinstance(out, Reply):
                return AssistantMessage(role="assistant", content=out.text), state
            calls = [Tau2ToolCall(id=c.id, name=c.name, arguments=c.arguments, requestor="assistant") for c in out.calls]
            return AssistantMessage(role="assistant", content=None, tool_calls=calls), state

        def _refresh_ownership(self) -> None:
            live = _db_dict(self.tools)
            if live is not None:
                self.ownership.db = live
                self.ownership._account_owner = {k: v.get("user_id") for k, v in
                                                 (live.get("accounts", {}).get("data", {}) or {}).items()}

        def set_seed(self, seed: int) -> None:
            pass

    return GuardedRagAgent


def create_guarded_agent(tools, domain_policy, **kwargs):
    """Factory for tau2's registry. ``--agent-llm`` takes provider/model, e.g. anthropic/claude-sonnet-5-5."""
    from .llm import make_llm

    llm_name = kwargs.get("llm") or "anthropic/claude-sonnet-5-5"
    provider, _, model = llm_name.partition("/")
    if provider not in ("anthropic", "openai"):
        provider, model = "openai", llm_name
    cls = make_tau2_agent_class()
    return cls(tools, domain_policy, make_llm(provider, model or None), task=kwargs.get("task"))


def register() -> None:
    from tau2.registry import registry

    try:
        registry.register_agent_factory(create_guarded_agent, AGENT_NAME)
    except (ValueError, KeyError):
        pass  # already registered


# -- oracle: reference trajectories through the real harness ------------------------------------------------

class OraclePolicy:
    """A scripted 'model' that reads the task's required documents, then makes the reference trajectory's agent
    calls in order and hands the turn to the customer whenever the customer acts next. It shows whether the
    agent runtime + guard + harness reproduce the reference end state, without any model."""

    def __init__(self, task_actions: list[dict], required_docs: list[str]):
        self.actions = task_actions
        self.docs = required_docs
        self.i = 0
        self.read = False
        self.handed_over = False
        self.blocked: list[str] = []

    def complete(self, system: str, messages: list[dict], tools: list[ToolSpec]):
        from .llm import LLMResponse

        last = messages[-1] if messages else {}
        if last.get("role") == "tool" and str(last.get("content", "")).startswith("Blocked by policy guard"):
            self.blocked.append(f"{self.actions[self.i - 1]['name']}: {last['content'][:120]}")
        if not self.read:
            self.read = True
            return LLMResponse("", [ToolCall(f"read_{j}", "read_document", {"doc_id": d}) for j, d in enumerate(self.docs)])
        # The customer answered after we handed over: their batch of actions is done.
        if self.handed_over and last.get("role") == "user":
            self.handed_over = False
            while self.i < len(self.actions) and self.actions[self.i]["requestor"] == "user":
                self.i += 1
        if self.i < len(self.actions) and self.actions[self.i]["requestor"] == "assistant":
            a = self.actions[self.i]
            self.i += 1
            return LLMResponse("", [ToolCall(f"oracle_{self.i}", a["name"], a["arguments"])])
        if self.i < len(self.actions):
            self.handed_over = True
            return LLMResponse("Please go ahead with that on your side now.")
        return LLMResponse("Everything is done. Is there anything else I can help with?")


class ScriptedUserFactory:
    """Builds a tau2 HalfDuplexUser that performs the reference trajectory's customer actions in order."""

    @staticmethod
    def build(task_actions: list[dict], opening: str):
        from tau2.data_model.message import ToolCall as Tau2ToolCall
        from tau2.data_model.message import UserMessage
        from tau2.user.user_simulator_base import HalfDuplexUser, UserState

        class ScriptedUser(HalfDuplexUser[UserState]):
            def __init__(self):
                super().__init__(instructions="scripted")
                self.actions = [a for a in task_actions if a["requestor"] == "user"]
                self.agent_left = sum(1 for a in task_actions if a["requestor"] == "assistant")
                self.order = task_actions
                self.pos = 0  # index into task_actions
                self.started = False
                self.waiting_tools = False

            def get_init_state(self, message_history=None) -> UserState:
                return UserState(messages=list(message_history or []), system_messages=[])

            def set_seed(self, seed: int) -> None:
                pass

            def generate_next_message(self, message, state):
                if not self.started:
                    self.started = True
                    return UserMessage(role="user", content=opening), state
                if self.waiting_tools:
                    self.waiting_tools = False
                    return UserMessage(role="user", content="Done, I did that."), state
                # Find the next customer action the agent has handed over to us.
                nxt = [a for a in self.order[self.pos:]]
                while nxt and nxt[0]["requestor"] == "assistant":
                    nxt.pop(0)
                    self.pos += 1
                if nxt and nxt[0]["requestor"] == "user":
                    batch = []
                    while self.pos < len(self.order) and self.order[self.pos]["requestor"] == "user":
                        a = self.order[self.pos]
                        batch.append(Tau2ToolCall(id=f"user_{self.pos}", name=a["name"], arguments=a["arguments"],
                                                  requestor="user"))
                        self.pos += 1
                    self.waiting_tools = True
                    return UserMessage(role="user", content=None, tool_calls=batch), state
                return UserMessage(role="user", content="Thanks, that's all. ###STOP###"), state

        return ScriptedUser()


def run_oracle(task_id: str, guard_on: bool = True) -> dict:
    """Run one task's reference trajectory through tau2's orchestrator with the guarded agent; return the reward."""
    from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import run_simulation

    task = next(t for t in get_tasks() if t.id == task_id)
    raw_actions = [a.model_dump() for a in task.evaluation_criteria.actions]
    env = get_environment(retrieval_variant="no_knowledge", task=task,
                          read_log_allowlist={a["name"] for a in raw_actions} |
                          {str((a.get("arguments") or {}).get("agent_tool_name")) for a in raw_actions})
    from ..eval.tasks import task_by_id

    ours = task_by_id(task_id)
    oracle = OraclePolicy(raw_actions, ours.required_documents)
    agent = make_tau2_agent_class()(env.get_tools(), env.get_policy(), oracle, task=task, max_inner_steps=60)
    if not guard_on:
        agent.core.guard.rules = set()
    user = ScriptedUserFactory.build(raw_actions, ours.opening())
    orch = Orchestrator(domain="banking_knowledge", agent=agent, user=user, environment=env, task=task,
                        max_steps=400, max_errors=50, seed=0)
    # NL assertions need an LLM judge; without one, such tasks are graded on their DB component only.
    nl = "NL_ASSERTION" in [str(getattr(b, "value", b)) for b in task.evaluation_criteria.reward_basis or []]
    result = run_simulation(orch, evaluation_type=EvaluationType.ENV if nl else EvaluationType.ALL,
                            env_kwargs={"retrieval_variant": "no_knowledge"})
    info = result.reward_info
    return {"task": task_id, "reward": info.reward if info else None, "termination": str(result.termination_reason),
            "graded": "db only (NL judge skipped)" if nl else "reward_basis",
            "blocked": oracle.blocked, "db_match": getattr(getattr(info, "db_check", None), "db_match", None),
            "note": json.dumps((info.info or {}) if info and info.info else {})[:200]}
