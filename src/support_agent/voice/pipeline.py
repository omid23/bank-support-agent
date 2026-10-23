"""The phone/browser voice bot: Pipecat cascade with the guarded support agent in the middle.

    mic → Silero VAD → Deepgram STT → turn detection → AgentProcessor → Cartesia TTS → speaker
                                                         │
                         knowledge search · policy guard · bank tools (τ³ bank)

Run with ``support-agent voice`` (SmallWebRTC; open the printed URL). Needs DEEPGRAM_API_KEY,
CARTESIA_API_KEY and a model key (ANTHROPIC_API_KEY or OPENAI_API_KEY), plus
``pip install 'bank-support-agent[voice,tau2]'``.
"""

from __future__ import annotations

import asyncio
import os

GREETING = "Thanks for calling Rho-Bank. How can I help you today?"
DEFAULT_VOICE = "71a7ad14-091c-4e8e-a314-022ece01c121"  # a Cartesia stock voice; override with CARTESIA_VOICE_ID


def make_session(llm_name: str | None = None, task_id: str | None = None):
    """A LocalSession against the τ³ bank (or canned tools when tau2 is not installed)."""
    from ..agent.backend import ScriptedBackend, base_tool_specs
    from ..agent.llm import make_llm
    from ..agent.runtime import LocalSession, SupportAgent
    from ..agent.tau2_plugin import shared_searcher
    from ..guard.policy import PolicyGuard
    from ..kb.store import policy_text

    try:
        from ..agent.backend import Tau2Backend

        backend = Tau2Backend(task_id)
    except ImportError:
        backend = ScriptedBackend(base_tool_specs())
    provider, _, model = (llm_name or os.environ.get("SUPPORT_AGENT_LLM", "anthropic/claude-sonnet-5-5")).partition("/")
    agent = SupportAgent(make_llm(provider, model or None), backend.tool_specs(), PolicyGuard(backend.ownership()),
                         shared_searcher(), policy_text())
    return LocalSession(agent, backend)


def build_pipeline(transport, respond, *, deepgram_key: str, cartesia_key: str, voice_id: str | None = None):
    """Assemble the processors; returns (pipeline, agent_processor, text_latency, audio_latency)."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.turns.user_turn_processor import UserTurnProcessor

    from .processors import AgentProcessor, LatencyTracker

    agent = AgentProcessor(respond)
    text_latency, audio_latency = LatencyTracker("text"), LatencyTracker("audio")
    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer()),
        DeepgramSTTService(api_key=deepgram_key),
        UserTurnProcessor(),
        agent,
        text_latency,
        CartesiaTTSService(api_key=cartesia_key, settings=CartesiaTTSService.Settings(voice=voice_id or DEFAULT_VOICE)),
        audio_latency,
        transport.output(),
    ])
    return pipeline, agent, text_latency, audio_latency


async def bot(runner_args) -> None:
    """Entry point for Pipecat's development runner."""
    from loguru import logger
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    transport = await create_transport(runner_args, {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    })
    session = make_session(os.environ.get("SUPPORT_AGENT_LLM"), os.environ.get("SUPPORT_AGENT_TASK"))
    pipeline, agent, text_lat, audio_lat = build_pipeline(
        transport, session.send, deepgram_key=os.environ["DEEPGRAM_API_KEY"],
        cartesia_key=os.environ["CARTESIA_API_KEY"], voice_id=os.environ.get("CARTESIA_VOICE_ID"))
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    warming: list[asyncio.Task] = []

    @transport.event_handler("on_client_connected")
    async def _connected(_transport, _client):
        # Write the model's prompt cache while the greeting plays, so the caller's first turn starts warm.
        warming.append(asyncio.ensure_future(asyncio.to_thread(session.warm)))
        await task.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_transport, _client):
        logger.info(f"latency to first text: {text_lat.summary()}; to first audio: {audio_lat.summary()}")
        logger.info(f"guard: {session.trace()['guard']}; usage: {session.trace()['usage']}")
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)
