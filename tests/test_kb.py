import pytest

from support_agent.eval.tasks import load_tasks, task_by_id
from support_agent.kb.search import Searcher, stem, tokenize
from support_agent.kb.store import TOOL_NAME, tool_catalog

from .conftest import REWARDS_DOC, REWARDS_TOOL


def test_knowledge_base_loads(kb):
    assert len(kb) == 698
    assert sum(1 for d in kb.documents if d.tools) == 44
    assert REWARDS_TOOL in kb.by_id[REWARDS_DOC].tools
    assert kb.docs_mentioning(REWARDS_TOOL) == [REWARDS_DOC]
    assert all(d.sections for d in kb.documents)


def test_tool_names_and_catalog():
    assert TOOL_NAME.findall("call file_credit_card_transaction_dispute_4829 now") == [
        "file_credit_card_transaction_dispute_4829"]
    assert not TOOL_NAME.findall("the year 2024 and account_ids")
    cat = tool_catalog()
    assert len(cat["agent_discoverable"]) == 44 and len(cat["user_discoverable"]) == 4
    assert {"log_verification", "call_discoverable_agent_tool"} <= {t["function"]["name"] for t in cat["base_agent_tools"]}


def test_tokenize_and_stem():
    assert stem("disputes") == stem("disputed") and stem("charges") == stem("charged")
    assert stem("cities") == "city" and stem("2024") == "2024"
    assert "the" not in tokenize("The card was charged twice")
    assert "charg" in "".join(tokenize("charged"))


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes(searcher, mode):
    hits = searcher.search("I was charged twice for the same purchase and want to dispute it", 10, mode)
    assert len(hits) == 10 and len({h.doc_id for h in hits}) == 10
    assert hits[0].score >= hits[-1].score
    assert any("dispute" in h.title.lower() for h in hits)


def test_hybrid_finds_the_rewards_procedure(searcher):
    hits = searcher.search("my rewards points on a transaction are wrong, please correct them", 10)
    assert REWARDS_DOC in [h.doc_id for h in hits]


def test_searcher_without_dense(kb):
    s = Searcher(kb, dense_dims=0)
    assert s.search("overdraft fee", 3)
    assert s.search("", 3) == [] or all(h.doc_id for h in s.search("", 3))


def test_tasks_and_split():
    tasks = load_tasks()
    assert len(tasks) == 97
    assert sum(t.split == "dev" for t in tasks) == 47 and sum(t.split == "test" for t in tasks) == 50
    assert all(t.opening() and t.required_documents for t in tasks)
    t = task_by_id("task_026")
    assert REWARDS_TOOL in t.gold_tools
    assert "###" not in t.scenario()
    with pytest.raises(KeyError):
        task_by_id("task_999")
