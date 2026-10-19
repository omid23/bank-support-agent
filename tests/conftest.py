import pytest

from support_agent.guard.ownership import OwnershipIndex
from support_agent.guard.policy import PolicyGuard
from support_agent.kb.store import default_kb

REWARDS_TOOL = "update_transaction_rewards_3847"
REWARDS_DOC = "doc_credit_cards_credit_cards_(general)_004"


@pytest.fixture(scope="session")
def kb():
    return default_kb()


@pytest.fixture(scope="session")
def searcher(kb):
    from support_agent.agent.tau2_plugin import shared_searcher

    return shared_searcher()


@pytest.fixture
def bank_db():
    return {
        "users": {"data": {"u_alice": {"name": "Alice"}, "u_bob": {"name": "Bob"},
                           "u_kid": {"name": "Kid", "parent_user_id": "u_alice"}}},
        "accounts": {"data": {"acc_a": {"user_id": "u_alice"}, "acc_b": {"user_id": "u_bob"},
                              "acc_k": {"user_id": "u_kid"}}},
        "credit_card_transaction_history": {"data": {"txn_a": {"user_id": "u_alice"}, "txn_b": {"user_id": "u_bob"}}},
        "debit_cards": {"data": {"dbc_b": {"account_id": "acc_b"}}},
    }


@pytest.fixture
def guard(bank_db, kb):
    return PolicyGuard(OwnershipIndex(bank_db), kb)
