"""FastAPI app composition for the Rex laptop server.

Loads the Whisper model, Kokoro voice, and LangChain agent once at startup
and exposes them as HTTP/WS endpoints for the Reachy Mini client.

Run with::

    uv run uvicorn server.app:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.tools import ReminderManager, TimerManager
from core.events import EventBus
from stt import Transcriber
from tts import load_voice

from .chat_api import router as chat_router
from .stt_api import router as stt_router
from .tts_api import router as tts_router
from .voice_agent import VoiceAgent

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Whisper transcriber...")
    app.state.transcriber = Transcriber()

    log.info("Loading Kokoro voice...")
    app.state.tts_voice = load_voice()

    log.info("Building voice agent...")
    bus = EventBus()
    # The audio manager is intentionally left out — the laptop never plays
    # sound directly in this deployment. Tools that try to use it will degrade
    # gracefully (timer alarms become text only).
    timer_mgr = TimerManager(event_bus=bus, audio_manager=None)
    reminder_mgr = ReminderManager(event_bus=bus)
    app.state.timer_manager = timer_mgr
    app.state.reminder_manager = reminder_mgr
    app.state.voice_agent = VoiceAgent(timer_mgr, reminder_mgr)

    log.info("rex-server ready.")
    try:
        yield
    finally:
        log.info("Shutting down rex-server...")
        try:
            timer_mgr.cleanup()
        except Exception:
            log.exception("timer cleanup failed")


app = FastAPI(title="rex-server", lifespan=lifespan)
app.include_router(stt_router, tags=["stt"])
app.include_router(chat_router, tags=["chat"])
app.include_router(tts_router, tags=["tts"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
