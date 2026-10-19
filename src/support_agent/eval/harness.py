"""Reference trajectories through τ³-bench's own orchestrator and grader, with the guarded agent in the loop.

For every task, a scripted policy plays the reference trajectory through ``SupportAgent`` (knowledge reads,
guard, tool plumbing) inside tau2's ``Orchestrator``; a scripted customer performs the customer-side
actions; tau2's evaluator grades the final database state. Reward 1.0 means the runtime and the guard
reproduce a correct solution end to end. Any task that loses reward with the guard on but not with it
off is a cost of the guard.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor


def _one(args: tuple[str, bool]) -> dict:
    task_id, guard_on = args
    from ..agent.tau2_plugin import run_oracle

    t = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_oracle(task_id, guard_on=guard_on)
    except Exception as exc:  # report, keep going
        r = {"task": task_id, "reward": None, "termination": "exception", "blocked": [], "note": repr(exc)[:300]}
    r["guard"] = guard_on
    r["seconds"] = round(time.perf_counter() - t, 2)
    return r


def run_all(task_ids: list[str], workers: int = 4) -> dict:
    jobs = [(t, g) for t in task_ids for g in (True, False)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one, jobs))
    on = {r["task"]: r for r in rows if r["guard"]}
    off = {r["task"]: r for r in rows if not r["guard"]}
    summary = {
        "tasks": len(task_ids),
        "reward_1_guard_on": sum(1 for r in on.values() if r["reward"] == 1.0),
        "reward_1_guard_off": sum(1 for r in off.values() if r["reward"] == 1.0),
        "lost_to_guard": sorted(t for t in task_ids if off[t]["reward"] == 1.0 and on[t]["reward"] != 1.0),
        "fail_without_guard": sorted(t for t in task_ids if off[t]["reward"] != 1.0),
    }
    return {"summary": summary, "rows": rows}


def to_markdown(rep: dict) -> str:
    s = rep["summary"]
    lines = ["# Reference trajectories through the τ³-bench harness", "",
             "Each task's reference solution is played through the guarded agent runtime inside tau2's orchestrator, "
             "and graded by tau2's evaluator (final database state).", "",
             "| | tasks at reward 1.0 |", "|---|---:|",
             f"| guard off | {s['reward_1_guard_off']} / {s['tasks']} |",
             f"| guard on | {s['reward_1_guard_on']} / {s['tasks']} |", ""]
    rows = {(r["task"], r["guard"]): r for r in rep["rows"]}
    partial = sorted({r["task"] for r in rep["rows"] if r.get("graded", "reward_basis") != "reward_basis"})
    if partial:
        lines += [f"Graded on the database only (their natural-language assertions need an LLM judge): "
                  f"{', '.join(partial)}.", ""]
    if s["lost_to_guard"]:
        lines += ["Tasks that pass without the guard but not with it:", "", "| task | blocked call |", "|---|---|"]
        for t in s["lost_to_guard"]:
            lines.append(f"| {t} | {'; '.join(rows[(t, True)]['blocked'])[:200]} |")
        lines.append("")
    if s["fail_without_guard"]:
        lines += ["Tasks whose reference trajectory does not reach reward 1.0 even without the guard:", "",
                  "| task | termination | note |", "|---|---|---|"]
        for t in s["fail_without_guard"]:
            r = rows[(t, False)]
            lines.append(f"| {t} | {r['termination']} | {str(r.get('note', ''))[:140]} |")
        lines.append("")
    return "\n".join(lines)


def dump(rep: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
