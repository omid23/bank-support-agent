import json

import pytest
from starlette.testclient import TestClient

from support_agent.cli import main
from support_agent.web.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_status_and_page(client):
    assert client.get("/api/status").json()["documents"] == 698
    page = client.get("/").text
    assert "<title>Rho-Bank Support Agent</title>" in page


def test_search_endpoint(client):
    r = client.get("/api/search", params={"q": "charged twice dispute", "mode": "bm25", "k": 5}).json()
    assert len(r["hits"]) == 5 and {"id", "title", "tools", "snippet"} <= set(r["hits"][0])
    assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400
    assert client.get("/api/search").json()["hits"] == []


def test_trace_reference_and_broken(client):
    ref = client.get("/api/trace", params={"task": "task_026"}).json()
    agent_steps = [s for s in ref["steps"] if s["who"] == "assistant"]
    assert agent_steps and all(s["verdict"]["allowed"] for s in agent_steps)
    broken = client.get("/api/trace", params={"task": "task_026", "perturb": "skip_verification"}).json()
    hit = next(s for s in broken["steps"] if s.get("verdict") and not s["verdict"]["allowed"])
    assert hit["target"] and hit["verdict"]["rule"] == "identity"
    assert client.get("/api/trace", params={"task": "task_999"}).status_code == 404
    assert client.get("/api/trace", params={"task": "task_026", "perturb": "bad"}).status_code == 400


def test_tasks_and_reports(client, tmp_path, monkeypatch):
    t = client.get("/api/tasks").json()
    assert len(t["tasks"]) == 97 and "invented_tool" in t["perturbations"]
    monkeypatch.setenv("SUPPORT_AGENT_REPORTS", str(tmp_path))
    (tmp_path / "harness.json").write_text(json.dumps({"summary": {"tasks": 97}, "rows": []}))
    assert client.get("/api/reports").json() == {"harness": {"tasks": 97}}


def test_chat_needs_a_model(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert client.post("/api/chat", json={"text": "hi"}).status_code == 503


def test_cli_search_and_guard_report(capsys, tmp_path):
    assert main(["search", "lost debit card", "-k", "3"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    assert main(["eval-guard", "--out", str(tmp_path / "guard")]) == 0
    assert json.loads((tmp_path / "guard.json").read_text())["tasks"] == 97
    assert (tmp_path / "guard.md").read_text().startswith("# Policy guard")
