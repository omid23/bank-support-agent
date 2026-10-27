"""Web console: search the policy base, watch the guard judge a task's reference trajectory (or a broken copy
of it), read the eval reports, and chat with the agent when a model key is set."""

from __future__ import annotations

import copy
import json
import os
import random
import threading
import time
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from ..eval.guard_eval import PERTURBATIONS, _guard_for, perturb
from ..eval.tasks import load_tasks, task_by_id
from ..guard.policy import ConversationState

PAGE = Path(__file__).with_name("index.html")


def _reports_dir() -> Path:
    return Path(os.environ.get("SUPPORT_AGENT_REPORTS", "reports"))


def create_app() -> Starlette:
    from ..agent.tau2_plugin import shared_searcher

    searcher = shared_searcher()
    sessions: dict[str, object] = {}
    lock = threading.Lock()

    async def index(_: Request):
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    async def search(req: Request):
        q = req.query_params.get("q", "").strip()
        mode = req.query_params.get("mode", "hybrid")
        if mode not in ("hybrid", "bm25", "dense"):
            return JSONResponse({"error": "mode must be hybrid, bm25 or dense"}, 400)
        k = max(1, min(int(req.query_params.get("k", 8)), 20))
        if not q:
            return JSONResponse({"hits": [], "ms": 0})
        t = time.perf_counter()
        hits = searcher.search(q, k, mode)
        ms = round((time.perf_counter() - t) * 1000, 1)
        out = []
        for h in hits:
            d = searcher.kb.by_id[h.doc_id]
            out.append({"id": d.id, "title": d.title, "category": d.category, "score": round(h.score, 4),
                        "section": h.section, "snippet": d.snippet(320), "tools": d.tools})
        return JSONResponse({"hits": out, "ms": ms})

    async def tasks(_: Request):
        return JSONResponse({"tasks": [{"id": t.id, "split": t.split, "opening": t.opening()[:160],
                                        "calls": sum(1 for a in t.actions if a.requestor == "assistant")}
                                       for t in load_tasks()], "perturbations": list(PERTURBATIONS)})

    async def trace(req: Request):
        try:
            task = task_by_id(req.query_params.get("task", ""))
        except (KeyError, StopIteration):
            return JSONResponse({"error": "unknown task"}, 404)
        kind = req.query_params.get("perturb") or ""
        actions, target = task.actions, None
        if kind:
            if kind not in PERTURBATIONS:
                return JSONResponse({"error": f"perturb must be one of {', '.join(PERTURBATIONS)}"}, 400)
            p = perturb(task, kind, random.Random(0), _guard_for(task, searcher.kb))
            if p is None:
                return JSONResponse({"error": f"'{kind}' does not apply to {task.id}"}, 409)
            actions, target = p
        guard = _guard_for(task, searcher.kb)
        state = ConversationState(read_docs=set(task.required_documents))
        steps, stopped = [], False
        for i, a in enumerate(copy.deepcopy(actions)):
            row = {"i": i, "who": a.requestor, "name": a.name, "tool": a.tool, "args": a.arguments,
                   "target": i == target}
            if a.requestor == "assistant" and not stopped:
                t = time.perf_counter()
                v = guard.check(a.name, a.arguments, state)
                row["us"] = round((time.perf_counter() - t) * 1e6, 1)
                row["verdict"] = v.to_dict()
                if v.allowed:
                    guard.observe(a.name, a.arguments, "ok", state)
                else:
                    stopped = True
            elif stopped:
                row["skipped"] = True
            steps.append(row)
        return JSONResponse({"task": task.id, "opening": task.opening(), "required_documents": task.required_documents,
                             "perturbation": kind, "steps": steps, "state": state.to_dict()})

    async def reports(_: Request):
        out = {}
        d = _reports_dir()
        for name in ("harness", "guard", "retrieval_dev", "retrieval_test", "voice_latency"):
            p = d / f"{name}.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                out[name] = data.get("summary", data) if isinstance(data, dict) and name == "harness" else data
                if name.startswith("retrieval"):
                    out[name] = {"tasks": data.get("tasks"), "results": data.get("results")}
                if name == "guard":
                    out[name] = {k: v for k, v in data.items() if k != "false_blocks"} | {
                        "false_blocks": data.get("false_blocks", [])}
        return JSONResponse(out)

    def _has_model() -> bool:
        from ..agent.llm import MODEL_ENV

        return any(os.environ.get(k) for k in MODEL_ENV)

    async def chat(req: Request):
        if not _has_model():
            return JSONResponse({"error": "Set ANTHROPIC_API_KEY, ANTHROPIC_VERTEX_PROJECT_ID or OPENAI_API_KEY to chat with the agent."}, 503)
        body = await req.json()
        text = str(body.get("text", "")).strip()[:2000]
        sid = str(body.get("session") or uuid.uuid4().hex)
        if not text:
            return JSONResponse({"error": "empty message"}, 400)
        from starlette.concurrency import run_in_threadpool

        from ..voice.pipeline import make_session

        with lock:
            session = sessions.get(sid)
            if session is None:
                session = sessions[sid] = make_session(body.get("llm"), body.get("task"))
        n = len(session.state.events)
        reply = await run_in_threadpool(session.send, text)
        return JSONResponse({"session": sid, "reply": reply, "events": session.state.events[n:],
                             "guard": session.state.guard.to_dict()})

    async def status(_: Request):
        return JSONResponse({"documents": len(searcher.kb.documents), "tasks": len(load_tasks()),
                             "chat": _has_model()})

    return Starlette(routes=[
        Route("/", index), Route("/api/status", status), Route("/api/search", search), Route("/api/tasks", tasks),
        Route("/api/trace", trace), Route("/api/reports", reports), Route("/api/chat", chat, methods=["POST"]),
    ])
