"""Retrieval quality on the τ³ banking tasks, measured without a model.

Each task lists the documents a correct answer needs (``required_documents``) and its reference
trajectory names the discoverable tools it uses. A search is good if it surfaces those documents,
and in particular the documents that name the tools (an agent cannot use a tool it never read about).

Queries come from the task itself:
* ``opening``  the customer's scripted first line when the scenario has one, else their situation
* ``scenario`` the whole scenario without role-play instructions (an upper bound on one-shot retrieval)

A real agent searches several times as the conversation unfolds, so these are one-shot lower bounds.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from ..kb.store import KnowledgeBase
from .tasks import Task

KS = (5, 10, 20)


def evaluate(kb: KnowledgeBase, tasks: list[Task], methods: dict[str, Callable[[str, int], list[str]]],
             query_kinds: tuple[str, ...] = ("opening", "scenario")) -> dict:
    report: dict = {"tasks": len(tasks), "results": {}}
    for kind in query_kinds:
        for name, search in methods.items():
            rows = []
            t0 = time.perf_counter()
            for task in tasks:
                q = task.opening() if kind == "opening" else task.scenario()
                ranked = search(q, max(KS))
                rows.append(_score(kb, task, ranked))
            ms = (time.perf_counter() - t0) * 1000 / max(len(tasks), 1)
            report["results"][f"{name} | {kind}"] = {"method": name, "query": kind, "ms_per_query": round(ms, 2),
                                                     **_summarize(rows)}
    return report


def _score(kb: KnowledgeBase, task: Task, ranked: list[str]) -> dict:
    req = set(task.required_documents)
    row: dict = {"id": task.id}
    first = next((i for i, d in enumerate(ranked, 1) if d in req), None)
    row["mrr"] = 1 / first if first else 0.0
    for k in KS:
        top = ranked[:k]
        row[f"recall@{k}"] = len(req & set(top)) / len(req) if req else 1.0
        row[f"hit@{k}"] = bool(req & set(top))
        tools = task.gold_tools
        if tools:
            row[f"tools@{k}"] = sum(any(t in kb.by_id[d].content for d in top) for t in tools) / len(tools)
    return row


def _summarize(rows: list[dict]) -> dict:
    out = {"mrr": round(statistics.mean(r["mrr"] for r in rows), 3)}
    for k in KS:
        out[f"hit@{k}"] = round(statistics.mean(r[f"hit@{k}"] for r in rows), 3)
        out[f"recall@{k}"] = round(statistics.mean(r[f"recall@{k}"] for r in rows), 3)
        with_tools = [r[f"tools@{k}"] for r in rows if f"tools@{k}" in r]
        out[f"tools@{k}"] = round(statistics.mean(with_tools), 3) if with_tools else None
    out["tasks_with_tools"] = sum(1 for r in rows if "tools@5" in r)
    return out


def to_markdown(report: dict, title: str) -> str:
    lines = [f"# {title}", "",
             f"{report['tasks']} tasks. `recall@k`: share of a task's required documents in the top k. "
             "`tools@k`: share of the discoverable tools in the reference trajectory that some top-k document names. "
             "`hit@k`: at least one required document in the top k.", "",
             "| method | query | hit@5 | recall@5 | recall@10 | recall@20 | tools@10 | tools@20 | MRR | ms/query |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["results"].values():
        lines.append(f"| {r['method']} | {r['query']} | {r['hit@5']:.1%} | {r['recall@5']:.1%} | {r['recall@10']:.1%} | "
                     f"{r['recall@20']:.1%} | {_pct(r['tools@10'])} | {_pct(r['tools@20'])} | {r['mrr']:.3f} | "
                     f"{r['ms_per_query']:.1f} |")
    return "\n".join(lines) + "\n"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"
