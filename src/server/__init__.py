"""HTTP + WebSocket front-end for the Rex voice stack.

Exposes the existing Whisper transcriber, LangChain agent, and Kokoro TTS so
that a remote client (the Reachy Mini app) can drive a conversation without
running any heavy models on-device.

The actual FastAPI app lives in ``server.app:app``.
"""

from .app import app

__all__ = ["app"]
