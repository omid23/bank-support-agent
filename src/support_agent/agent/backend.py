"""Backends that execute bank tool calls for the standalone and voice modes.

``Tau2Backend`` runs the τ³-bench bank (its database and all 60 tools) in-process, optionally seeded with a
task's customers, so the app and the voice bot work against the same simulated bank the benchmark uses.
``ScriptedBackend`` returns canned results for tests and demos without tau2.
"""

from __future__ import annotations

from collections.abc import Callable

from ..guard.ownership import OwnershipIndex, base_db
from .llm import ToolCall, ToolSpec


class Tau2Backend:
    def __init__(self, task_id: str | None = None):
        from tau2.data_model.message import ToolCall as Tau2ToolCall
        from tau2.domains.banking_knowledge.environment import get_environment, get_tasks

        self._ToolCall = Tau2ToolCall
        task = next((t for t in get_tasks() if t.id == task_id), None) if task_id else None
        self.env = get_environment(retrieval_variant="no_knowledge", task=task)
        if task is not None and task.initial_state is not None:
            init = task.initial_state
            self.env.set_state(initialization_data=init.initialization_data,
                               initialization_actions=init.initialization_actions, message_history=[])

    def tool_specs(self) -> list[ToolSpec]:
        from .tau2_plugin import specs_from_tau2

        return specs_from_tau2(self.env.get_tools())

    def execute(self, call: ToolCall) -> tuple[str, bool]:
        msg = self.env.get_response(self._ToolCall(id=call.id, name=call.name, arguments=call.arguments,
                                                   requestor="assistant"))
        return msg.content or "", bool(msg.error)

    def ownership(self) -> OwnershipIndex:
        return OwnershipIndex(self.env.tools.db.model_dump())


class ScriptedBackend:
    """Tool results from a function or a fixed table; records every call."""

    def __init__(self, specs: list[ToolSpec], respond: Callable[[ToolCall], str] | dict[str, str] | None = None):
        self.specs, self.respond = specs, respond or {}
        self.calls: list[ToolCall] = []

    def tool_specs(self) -> list[ToolSpec]:
        return self.specs

    def execute(self, call: ToolCall) -> tuple[str, bool]:
        self.calls.append(call)
        out = self.respond(call) if callable(self.respond) else self.respond.get(call.name, f"{call.name}: ok")
        return out, out.lower().startswith("error")

    def ownership(self) -> OwnershipIndex:
        return OwnershipIndex(base_db())


def base_tool_specs() -> list[ToolSpec]:
    """The bank's always-available agent tools (from the catalog), for running without tau2."""
    from ..kb.store import tool_catalog

    out = []
    for t in tool_catalog()["base_agent_tools"]:
        fn = t["function"]
        out.append(ToolSpec(fn["name"], fn.get("description", ""), fn.get("parameters") or {"type": "object",
                                                                                              "properties": {}}))
    return out
