import json

from support_agent.eval.guard_eval import PERTURBATIONS
from support_agent.eval.guard_eval import evaluate as guard_eval
from support_agent.eval.retrieval import evaluate as retrieval_eval
from support_agent.eval.tasks import load_tasks
from support_agent.guard.ownership import ID_ARGS, OwnershipIndex, base_db, merge_initial_state
from support_agent.guard.policy import ConversationState, parse_args

from .conftest import REWARDS_DOC, REWARDS_TOOL


def _call(tool, **inner):
    return {"agent_tool_name": tool, "arguments": json.dumps(inner)}


def _ready(state, verified="u_alice"):
    state.read_docs.add(REWARDS_DOC)
    state.unlocked.add(REWARDS_TOOL)
    if verified:
        state.verified.add(verified)
    return state


def test_ownership(bank_db):
    o = OwnershipIndex(bank_db)
    assert o.owner("account_id", "acc_b") == "u_bob"
    assert o.owner("card_id", "dbc_b") == "u_bob"  # via the card's account
    assert o.owner("transaction_id", "txn_a") == "u_alice"
    assert o.owner("user_id", "u_nobody") is None
    assert o.guardians("u_kid") == {"u_kid", "u_alice"}
    assert o.owners_in({"account_id": "acc_a", "amount": 5}) == {"account_id=acc_a": "u_alice"}
    assert "transaction_id" in ID_ARGS


def test_merge_initial_state_adds_rows():
    t = next(t for t in load_tasks() if t.id == "task_040")
    db = merge_initial_state(base_db(), t.initial_state)
    assert db["transaction_disputes"]["data"]["dsp_ed3ab3dce038"]["user_id"] == "01f21c9970"
    assert "dsp_ed3ab3dce038" not in json.dumps(base_db())  # base untouched
    assert OwnershipIndex(db).owner("transaction_id", "txn_a8f1c2d3e403") == "755bcb4d5d"


def test_known_and_documented_tool(guard):
    s = ConversationState()
    v = guard.check("unlock_discoverable_agent_tool", {"agent_tool_name": "update_transaction_rewards_9999"}, s)
    assert not v.allowed and v.rule == "known_tool"
    v = guard.check("unlock_discoverable_agent_tool", {"agent_tool_name": REWARDS_TOOL}, s)
    assert not v.allowed and v.rule == "documented_tool"
    s.read_docs.add(REWARDS_DOC)
    assert guard.check("unlock_discoverable_agent_tool", {"agent_tool_name": REWARDS_TOOL}, s).allowed


def test_user_tool_given_as_agent_tool_is_explained(guard):
    v = guard.check("unlock_discoverable_agent_tool", {"agent_tool_name": "submit_cash_back_dispute_0589"},
                    ConversationState())
    assert v.rule == "known_tool" and "user tool" in v.message


def test_unlock_before_call(guard):
    s = _ready(ConversationState())
    s.unlocked.clear()
    v = guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, transaction_id="txn_a"), s)
    assert v.rule == "unlock_first"


def test_identity_then_ownership(guard):
    s = _ready(ConversationState(), verified=None)
    v = guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, transaction_id="txn_a"), s)
    assert v.rule == "identity"
    guard.observe("log_verification", {"user_id": "u_alice"}, "Verification logged", s)
    assert guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, transaction_id="txn_a"), s).allowed
    v = guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, transaction_id="txn_b"), s)
    assert not v.allowed and v.rule == "ownership"
    v = guard.check("get_credit_card_accounts_by_user", {"user_id": "u_bob"}, s)
    assert v.rule == "ownership"


def test_failed_verification_does_not_count(guard):
    s = ConversationState()
    guard.observe("log_verification", {"user_id": "u_alice"}, "Error: details do not match", s)
    assert not s.verified


def test_parent_may_act_for_minor_but_not_reverse(guard):
    s = _ready(ConversationState(), verified="u_alice")
    assert guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, account_id="acc_k"), s).allowed
    s2 = _ready(ConversationState(), verified="u_kid")
    assert guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, account_id="acc_a"), s2).rule == "ownership"


def test_unknown_ids_are_advisory(guard):
    s = _ready(ConversationState())
    v = guard.check("call_discoverable_agent_tool", _call(REWARDS_TOOL, account_id="acc_new"), s)
    assert v.allowed and v.advisory


def test_no_duplicate_give(guard, kb):
    tool = "submit_cash_back_dispute_0589"
    s = ConversationState(read_docs=set(kb.docs_mentioning(tool)))
    assert guard.check("give_discoverable_user_tool", {"discoverable_tool_name": tool}, s).allowed
    guard.observe("give_discoverable_user_tool", {"discoverable_tool_name": tool}, "given", s)
    assert guard.check("give_discoverable_user_tool", {"discoverable_tool_name": tool}, s).rule == "no_duplicate_give"


def test_rules_can_be_switched_off(guard):
    guard.rules = set()
    assert guard.check("unlock_discoverable_agent_tool", {"agent_tool_name": "made_up_0000"}, ConversationState()).allowed


def test_transfer_consent_uses_decision_model(guard):
    class No:
        def ask(self, question, context):
            return False

    class Unsure:
        def ask(self, question, context):
            return None

    guard.decision_model = No()
    assert guard.check("transfer_to_human_agents", {}, ConversationState()).rule == "transfer_consent"
    guard.decision_model = Unsure()
    v = guard.check("transfer_to_human_agents", {}, ConversationState())
    assert v.allowed and v.advisory


def test_parse_args():
    assert parse_args('{"a": 1}') == {"a": 1}
    assert parse_args("not json") == {} and parse_args(None) == {} and parse_args({"b": 2}) == {"b": 2}


def test_guard_eval_on_all_reference_trajectories():
    rep = guard_eval(list(load_tasks()))
    assert rep["tasks"] == 97 and rep["assistant_calls_checked"] > 800
    assert {b["task"] for b in rep["false_blocks"]} == {"task_005", "task_051", "task_083"}
    assert set(rep["catches"]) == set(PERTURBATIONS)
    for kind, c in rep["catches"].items():
        assert c["cases"] >= 60 and c["rate"] == 1.0, kind


def test_retrieval_eval_scores_an_oracle(kb):
    tasks = list(load_tasks())[:10]
    by_open = {t.opening(): t.required_documents for t in tasks} | {t.scenario(): t.required_documents for t in tasks}
    rep = retrieval_eval(kb, tasks, {"oracle": lambda q, k: by_open[q][:k],
                                     "nothing": lambda q, k: []})
    o = rep["results"]["oracle | opening"]
    assert o["recall@20"] == 1.0 and o["mrr"] == 1.0
    assert rep["results"]["nothing | opening"]["recall@10"] == 0.0
