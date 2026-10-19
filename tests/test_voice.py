import time

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

from support_agent.voice.processors import AgentProcessor, LatencyTracker, sentences, speakable  # noqa: E402


def tr(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def _texts(frames, kind=TextFrame):
    return [f.text for f in frames if type(f) is kind]


def test_speakable_strips_markdown():
    s = speakable("**Fee:** $5\n\n- first item\n- second item\nSee [the doc](http://x).")
    assert "**" not in s and "- " not in s and "http" not in s
    assert "first item" in s and "the doc" in s


def test_sentences_split_and_cut_long():
    assert sentences("Hi there. How can I help? Sure!") == ["Hi there.", "How can I help?", "Sure!"]
    long = ", ".join(["word " * 8] * 10) + "."
    assert all(len(p) <= 220 for p in sentences(long))


async def test_turn_replies_in_sentences():
    calls = []
    agent = AgentProcessor(lambda t: (calls.append(t), "Sure. Your balance is $40.")[1], end_grace=0.05)
    down, _ = await run_test(agent, frames_to_send=[
        UserStartedSpeakingFrame(), tr("what's my"), tr("balance"), UserStoppedSpeakingFrame(), SleepFrame(0.3)])
    assert calls == ["what's my balance"]
    assert _texts(down) == ["Sure.", "Your balance is $40."]
    kinds = [type(f) for f in down]
    assert kinds.index(LLMFullResponseStartFrame) < kinds.index(TextFrame) < kinds.index(LLMFullResponseEndFrame)


async def test_late_final_transcript_still_answered():
    agent = AgentProcessor(lambda t: f"You said {t}.", end_grace=0.0)
    down, _ = await run_test(agent, frames_to_send=[
        UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), SleepFrame(0.1), tr("hello"), SleepFrame(0.2)])
    assert _texts(down) == ["You said hello."]


async def test_filler_when_agent_is_slow():
    def slow(_):
        time.sleep(0.3)
        return "Done."
    agent = AgentProcessor(slow, filler_after=0.1, end_grace=0.0)
    down, _ = await run_test(agent, frames_to_send=[
        UserStartedSpeakingFrame(), tr("dispute a charge"), UserStoppedSpeakingFrame(), SleepFrame(0.6)])
    fillers = _texts(down, TTSSpeakFrame)
    assert len(fillers) == 1 and _texts(down) == ["Done."]
    assert agent.turns[-1]["filler"] is True


async def test_no_filler_when_fast():
    agent = AgentProcessor(lambda t: "Hi.", filler_after=0.5, end_grace=0.0)
    down, _ = await run_test(agent, frames_to_send=[
        UserStartedSpeakingFrame(), tr("hi"), UserStoppedSpeakingFrame(), SleepFrame(0.2)])
    assert not _texts(down, TTSSpeakFrame)


async def test_barge_in_drops_stale_reply():
    def slow(t):
        time.sleep(0.25)
        return f"Answer to {t}."
    agent = AgentProcessor(slow, filler_after=5, end_grace=0.0)
    down, _ = await run_test(agent, frames_to_send=[
        UserStartedSpeakingFrame(), tr("first question"), UserStoppedSpeakingFrame(), SleepFrame(0.05),
        UserStartedSpeakingFrame(), InterruptionFrame(), tr("actually, second"), UserStoppedSpeakingFrame(),
        SleepFrame(0.7)])
    assert _texts(down) == ["Answer to actually, second."]
    assert any(t.get("spoken") is False for t in agent.turns)


async def test_latency_tracker():
    agent = AgentProcessor(lambda t: "Okay.", end_grace=0.0)
    lat = LatencyTracker()
    from pipecat.pipeline.pipeline import Pipeline

    await run_test(Pipeline([agent, lat]), frames_to_send=[
        UserStartedSpeakingFrame(), tr("one"), UserStoppedSpeakingFrame(), SleepFrame(0.15),
        UserStartedSpeakingFrame(), tr("two"), UserStoppedSpeakingFrame(), SleepFrame(0.15)])
    s = lat.summary()
    assert s["turns"] == 2 and 0 <= s["p50_ms"] < 150
