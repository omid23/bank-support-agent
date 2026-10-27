"""support-agent: knowledge search, evals, chat and voice for the guarded Rho-Bank support agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _searcher():
    from .agent.tau2_plugin import shared_searcher

    return shared_searcher()


def cmd_search(a) -> int:
    s = _searcher()
    for h in s.search(a.query, a.k, a.mode):
        d = s.kb.by_id[h.doc_id]
        print(f"{h.score:7.4f}  {h.doc_id}  {h.title}" + (f"   tools: {', '.join(d.tools)}" if d.tools else ""))
    return 0


def cmd_eval_retrieval(a) -> int:
    from .eval.retrieval import evaluate, to_markdown
    from .eval.tasks import load_tasks

    s = _searcher()
    tasks = [t for t in load_tasks() if a.split == "all" or t.split == a.split]
    methods = {}
    try:
        from .kb.search import Tau2BM25

        base = Tau2BM25(s.kb)
        methods["tau2 bm25 (baseline)"] = lambda q, k: [h.doc_id for h in base.search(q, k)]
    except ImportError:
        print("rank-bm25 not installed: skipping the tau2 baseline", file=sys.stderr)
    for mode, label in (("bm25", "bm25 (ours)"), ("dense", "dense (LSA)"), ("hybrid", "hybrid")):
        methods[label] = (lambda m: lambda q, k: [h.doc_id for h in s.search(q, k, m)])(mode)
    rep = evaluate(s.kb, tasks, methods)
    md = to_markdown(rep, f"{a.split} split")
    _write(a.out, md, rep)
    print(md)
    return 0


def cmd_eval_guard(a) -> int:
    from .eval.guard_eval import evaluate, to_markdown
    from .eval.tasks import load_tasks

    rep = evaluate(list(load_tasks()), seed=a.seed)
    md = to_markdown(rep, "Policy guard on τ³-Banking reference trajectories")
    _write(a.out, md, rep)
    print(md)
    return 0


def cmd_eval_harness(a) -> int:
    from .eval.harness import run_all, to_markdown
    from .eval.tasks import load_tasks

    ids = a.tasks or [t.id for t in load_tasks()]
    rep = run_all(ids, workers=a.workers)
    md = to_markdown(rep)
    _write(a.out, md, rep)
    print(md)
    return 0 if rep["summary"]["reward_1_guard_on"] == len(ids) - len(rep["summary"]["lost_to_guard"]) else 1


def cmd_bench_voice(a) -> int:
    from .eval.tasks import load_tasks
    from .voice.bench import run, to_markdown

    utts = [t.opening() for t in load_tasks()][: a.n]
    rows = [run(utts, m, filler_after=a.filler_after, end_grace=a.end_grace) for m in a.model_ms]
    md = to_markdown(rows)
    _write(a.out, md, rows)
    print(md)
    return 0


def cmd_tau2_run(a) -> int:
    """Run the real benchmark with a model: registers the agent and calls tau2's runner."""
    from .agent.tau2_plugin import AGENT_NAME, register

    register()
    from tau2.cli import main as tau2_main

    argv = ["tau2", "run", "--domain", "banking_knowledge", "--agent", AGENT_NAME, "--agent-llm", a.llm,
            "--user-llm", a.user_llm, "--retrieval-config", "no_knowledge", "--num-trials", str(a.trials),
            "--max-concurrency", str(a.concurrency)]
    if a.tasks:
        argv += ["--task-ids", *a.tasks]
    sys.argv = argv
    return tau2_main() or 0


def cmd_chat(a) -> int:
    from .voice.pipeline import make_session

    session = make_session(a.llm, a.task)
    print("Rho-Bank support (blank line to quit)")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        print(f"agent> {session.send(text)}")
        if a.trace:
            for e in session.state.events[-12:]:
                if e["kind"] in ("search", "guard", "bank"):
                    print("   ", json.dumps(e)[:200])
    print(json.dumps(session.trace()["guard"], indent=1))
    print("usage:", json.dumps(session.trace()["usage"]))
    return 0


def cmd_voice(a) -> int:
    missing = [k for k in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"set {', '.join(missing)} (and ANTHROPIC_API_KEY, ANTHROPIC_VERTEX_PROJECT_ID or OPENAI_API_KEY)",
              file=sys.stderr)
        return 2
    from pipecat.runner.run import main as pipecat_main

    from .voice import pipeline

    if a.llm:
        os.environ["SUPPORT_AGENT_LLM"] = a.llm
    if a.task:
        os.environ["SUPPORT_AGENT_TASK"] = a.task
    sys.modules["__main__"].bot = pipeline.bot  # pipecat's runner looks up bot() on __main__
    sys.argv = ["support-agent", "-t", "webrtc", "--port", str(a.port)]
    pipecat_main()
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
    return 0


def _write(out: str | None, md: str, data) -> None:
    if not out:
        return
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".md").write_text(md, encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="support-agent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search the policy knowledge base")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("eval-retrieval", help="retrieval quality on the τ³ tasks")
    s.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval_retrieval)

    s = sub.add_parser("eval-guard", help="guard on reference trajectories and perturbed (unsafe) variants")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval_guard)

    s = sub.add_parser("eval-harness", help="reference trajectories through tau2's orchestrator and grader [tau2]")
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval_harness)

    s = sub.add_parser("bench-voice", help="offline voice-turn latency through the Pipecat processors [voice]")
    s.add_argument("-n", type=int, default=97)
    s.add_argument("--model-ms", type=float, nargs="*", default=[0.0, 400.0])
    s.add_argument("--filler-after", type=float, default=0.8)
    s.add_argument("--end-grace", type=float, default=0.25)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_bench_voice)

    s = sub.add_parser("tau2-run", help="run τ³-Banking with a model (needs API keys) [tau2]")
    s.add_argument("--llm", default="anthropic/claude-sonnet-5-5")
    s.add_argument("--user-llm", default="gpt-4.1")
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--trials", type=int, default=1)
    s.add_argument("--concurrency", type=int, default=4)
    s.set_defaults(fn=cmd_tau2_run)

    for name, fn, hlp in (("chat", cmd_chat, "chat with the agent in the terminal (needs a model key)"),
                          ("voice", cmd_voice, "talk to the agent in the browser over WebRTC [voice]")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--llm", help="provider/model, e.g. anthropic/claude-sonnet-5-5, vertex/claude-sonnet-5-5 "
                                         "or openai/gpt-5")
        s.add_argument("--task", help="seed the bank with a τ³ task's customers, e.g. task_026")
        if name == "chat":
            s.add_argument("--trace", action="store_true")
        else:
            s.add_argument("--port", type=int, default=7860)
        s.set_defaults(fn=fn)

    s = sub.add_parser("serve", help="the web console: knowledge search, guard check, eval reports")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
