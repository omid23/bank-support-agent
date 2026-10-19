"""Pipecat processors that put the guarded support agent on a phone line.

``AgentProcessor`` sits where an LLM service normally sits in a cascaded pipeline
(VAD → STT → turn detection → **agent** → TTS):

* collects the final transcripts of a user turn and runs one agent turn when the turn ends
  (a short grace period catches a final transcript that arrives just after the user stops);
* runs the agent off the event loop, so audio keeps flowing while it searches and calls bank tools;
* says a short filler ("One moment while I check that.") if the turn takes longer than ``filler_after``,
  so the caller is not left in silence during tool calls;
* drops a reply that is no longer wanted when the caller barges in;
* turns the reply into speakable sentences and pushes them one at a time, so TTS can start on the first.

``LatencyTracker`` measures, per turn, end of user speech → first agent text and → first audio.
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time
from collections.abc import Callable

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

FILLERS = ("One moment while I check that.", "Let me look into that for you.", "Just a second, I'm checking now.")


def speakable(text: str) -> str:
    """Chat formatting to speech: no markdown, no bullets, no raw ids longer than needed."""
    t = re.sub(r"\*\*|__|`|#+\s*", "", text)
    t = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", t)
    t = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", t)
    t = re.sub(r"\n{2,}", ". ", t)
    t = re.sub(r"\s*\n\s*", ", ", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\.\s*\.", ".", t)
    return t.strip(" ,")


def sentences(text: str, max_len: int = 220) -> list[str]:
    """Split for incremental TTS; long sentences are cut at a comma."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        while len(p) > max_len and "," in p[:max_len]:
            cut = p[:max_len].rfind(",") + 1
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        out.append(p)
    return out


class AgentProcessor(FrameProcessor):
    def __init__(self, respond: Callable[[str], str], *, filler_after: float = 0.8, end_grace: float = 0.25,
                 fillers: tuple[str, ...] = FILLERS, **kwargs):
        super().__init__(**kwargs)
        self.respond = respond  # blocking: user text -> agent reply
        self.filler_after, self.end_grace, self.fillers = filler_after, end_grace, fillers
        self._buffer: list[str] = []
        self._speaking = False
        self._generation = 0
        self._turn_task: asyncio.Task | None = None
        self._pending_end: asyncio.Task | None = None
        self.turns: list[dict] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._speaking = True
            if self._turn_task and not self._turn_task.done():
                self._generation += 1  # barge-in: whatever is in flight will not be spoken
                self.turns.append({"interrupted": True})
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._generation += 1
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._buffer.append(frame.text.strip())
            if not self._speaking and self._pending_end is None and self._buffer:
                await self._start_turn()  # a final transcript that arrived after the turn ended
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            await self.push_frame(frame, direction)
            self._pending_end = self.create_task(self._end_after_grace(), "agent-end-grace")
        elif isinstance(frame, EndFrame):
            for task in (self._pending_end, self._turn_task):  # finish the turn in progress before closing
                if task and not task.done():
                    await task
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _end_after_grace(self) -> None:
        await asyncio.sleep(self.end_grace)
        self._pending_end = None
        if self._buffer and not self._speaking:
            await self._start_turn()

    async def _start_turn(self) -> None:
        text = " ".join(self._buffer)
        self._buffer = []
        self._generation += 1
        self._turn_task = self.create_task(self._run_turn(text, self._generation), "agent-turn")

    async def _run_turn(self, text: str, generation: int) -> None:
        started = time.perf_counter()
        work = asyncio.ensure_future(asyncio.to_thread(self.respond, text))
        filler_said = False
        try:
            await asyncio.wait_for(asyncio.shield(work), timeout=self.filler_after)
        except asyncio.TimeoutError:
            if generation == self._generation:
                filler_said = True
                await self.push_frame(TTSSpeakFrame(self.fillers[len(self.turns) % len(self.fillers)]))
        reply = await work
        record = {"user": text, "reply": reply, "agent_ms": round((time.perf_counter() - started) * 1000, 1),
                  "filler": filler_said, "spoken": generation == self._generation}
        self.turns.append(record)
        if generation != self._generation:
            return  # the caller started talking again; this answer is stale
        await self.push_frame(LLMFullResponseStartFrame())
        for s in sentences(speakable(reply)):
            await self.push_frame(TextFrame(s))
        await self.push_frame(LLMFullResponseEndFrame())


class LatencyTracker(FrameProcessor):
    """Passthrough. Put one after the agent (text latency) and one after TTS (audio latency)."""

    def __init__(self, label: str = "text", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self._stopped_at: float | None = None
        self.samples_ms: list[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._stopped_at = time.perf_counter()
        elif self._stopped_at is not None and isinstance(frame, (TextFrame, TTSSpeakFrame, TTSAudioRawFrame)) \
                and not (self.label == "audio" and not isinstance(frame, TTSAudioRawFrame)):
            self.samples_ms.append((time.perf_counter() - self._stopped_at) * 1000)
            self._stopped_at = None
        await self.push_frame(frame, direction)

    def summary(self) -> dict:
        s = sorted(self.samples_ms)
        if not s:
            return {"turns": 0}
        return {"turns": len(s), "p50_ms": round(statistics.median(s), 1),
                "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1), "max_ms": round(s[-1], 1)}
