"""Offline voice-turn latency: what the pipeline itself adds between the end of the caller's speech and the
first sentence handed to TTS.

Each τ³ task's opening line is spoken into ``AgentProcessor`` as a transcript. The agent runs for real
(knowledge search, guard, tool plumbing); the model is a stand-in that answers after a fixed delay, so the
numbers isolate our overhead plus that delay. STT, TTS and network time are not included: they depend on the
providers and need live keys. ``model_ms`` lets you plug in a measured model latency to see where the filler
kicks in.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from ..agent.llm import LLMResponse, ScriptedLLM, ToolCall


def _policy(model_ms: float):
    def respond(system: str, messages: list[dict]) -> LLMResponse:
        time.sleep(model_ms / 1000)
        last = messages[-1]
        if last["role"] == "user":
            return LLMResponse("", [ToolCall("s1", "search_knowledge", {"query": last["content"], "k": 5})])
        title = str(last.get("content", "")).split("\n", 1)[0]
        return LLMResponse(f"I found {title[:80]}. Before I go further, can you confirm your date of birth?")
    return respond


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * len(s)))], 1)


async def _run(utterances: list[str], model_ms: float, filler_after: float, end_grace: float) -> dict:
    from pipecat.frames.frames import TranscriptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame, run_test

    from ..agent.backend import ScriptedBackend, base_tool_specs
    from ..agent.runtime import LocalSession, SupportAgent
    from ..agent.tau2_plugin import shared_searcher
    from ..guard.policy import PolicyGuard
    from ..kb.store import policy_text
    from .processors import AgentProcessor, LatencyTracker

    backend = ScriptedBackend(base_tool_specs())
    agent = SupportAgent(ScriptedLLM(_policy(model_ms)), backend.tool_specs(), PolicyGuard(backend.ownership()),
                         shared_searcher(), policy_text())
    sessions: list = []

    def respond(text: str) -> str:
        s = LocalSession(agent, backend)  # one fresh conversation per utterance
        sessions.append(s)
        return s.send(text)

    proc = AgentProcessor(respond, filler_after=filler_after, end_grace=end_grace)
    first_out = LatencyTracker("text")
    wait = end_grace + (2 * model_ms) / 1000 + 0.25
    frames = []
    for u in utterances:
        frames += [UserStartedSpeakingFrame(), TranscriptionFrame(u, "caller", ""), UserStoppedSpeakingFrame(),
                   SleepFrame(wait)]
    await run_test(Pipeline([proc, first_out]), frames_to_send=frames)
    agent_ms = [t["agent_ms"] for t in proc.turns if "agent_ms" in t]
    search_ms = [e["ms"] for s in sessions for e in s.state.events if e["kind"] == "search"]
    lat = first_out.samples_ms
    return {
        "turns": len(lat), "model_ms_per_call": model_ms, "model_calls_per_turn": 2,
        "end_grace_ms": end_grace * 1000, "filler_after_ms": filler_after * 1000,
        "first_output_p50_ms": round(statistics.median(lat), 1), "first_output_p95_ms": _pct(lat, 0.95),
        "agent_turn_p50_ms": round(statistics.median(agent_ms), 1), "agent_turn_p95_ms": _pct(agent_ms, 0.95),
        "search_p50_ms": round(statistics.median(search_ms), 2), "search_p95_ms": _pct(search_ms, 0.95),
        "fillers": sum(1 for t in proc.turns if t.get("filler")),
    }


def run(utterances: list[str], model_ms: float = 0.0, filler_after: float = 0.8, end_grace: float = 0.25) -> dict:
    shared_searcher_warm()
    return asyncio.run(_run(utterances, model_ms, filler_after, end_grace))


def shared_searcher_warm() -> None:
    from ..agent.tau2_plugin import shared_searcher

    shared_searcher().search("warm up", 1)


def to_markdown(rows: list[dict]) -> str:
    lines = ["# Voice turn latency (offline)", "",
             "End of caller speech → first sentence handed to TTS, through `AgentProcessor` with the real agent "
             "runtime (knowledge search, guard). The model is a stand-in with a fixed delay per call (2 calls per "
             "turn: search, then answer). STT, TTS and network are not included.", "",
             "| model ms / call | turns | first output p50 | p95 | agent turn p50 | p95 | search p50 | p95 | fillers |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model_ms_per_call']:.0f} | {r['turns']} | {r['first_output_p50_ms']} | "
                     f"{r['first_output_p95_ms']} | {r['agent_turn_p50_ms']} | {r['agent_turn_p95_ms']} | "
                     f"{r['search_p50_ms']} | {r['search_p95_ms']} | {r['fillers']} |")
    lines += ["", f"End-of-turn grace: {rows[0]['end_grace_ms']:.0f} ms; filler after {rows[0]['filler_after_ms']:.0f} ms "
              "(the filler is the first output when the agent is slower than that)."]
    return "\n".join(lines) + "\n"
