"""Integration with τ³-bench itself (skipped when tau2-bench is not installed; it needs Python 3.12+)."""

import pytest

pytest.importorskip("tau2")

from support_agent.agent.tau2_plugin import AGENT_NAME, register, run_oracle  # noqa: E402


@pytest.mark.parametrize("task_id", ["task_001", "task_026", "task_102"])
def test_reference_solution_passes_with_guard(task_id):
    r = run_oracle(task_id, guard_on=True)
    assert r["reward"] == 1.0 and not r["blocked"], r


def test_guard_cost_is_visible_in_reward():
    off, on = run_oracle("task_005", guard_on=False), run_oracle("task_005", guard_on=True)
    assert off["reward"] == 1.0 and on["reward"] == 0.0
    assert "ownership" in on["blocked"][0]


def test_registers_with_tau2():
    from tau2.registry import registry

    register()
    register()  # idempotent
    assert registry.get_agent_factory(AGENT_NAME) is not None


def test_tau2_backend_executes_tools():
    from support_agent.agent.backend import Tau2Backend
    from support_agent.agent.llm import ToolCall

    b = Tau2Backend("task_026")
    names = {s.name for s in b.tool_specs()}
    assert {"log_verification", "unlock_discoverable_agent_tool"} <= names
    out, err = b.execute(ToolCall("1", "get_user_information_by_id", {"user_id": "755bcb4d5d"}))
    assert not err and "Amara" in out
    assert b.ownership().owner("transaction_id", "txn_a8f1c2d3e403") == "755bcb4d5d"
