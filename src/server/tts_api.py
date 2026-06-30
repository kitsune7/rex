"""WS /tts — stream Kokoro audio chunks to the client.

Protocol:

    client → server   {"text": "...", "voice": "am_fenrir"}
    server → client   binary frames: float32 PCM mono @ 24 kHz (Kokoro native)
    server → client   {"event": "end"}

The client should treat each binary frame as a numpy ``float32`` buffer and
push it straight into the speaker. Resampling, if any, is the client's job —
on the Reachy we resample to 16 kHz to match the ReSpeaker output rate.

There is no explicit "start" event; the first binary frame implicitly signals
synthesis has begun. The client may close the socket at any time to interrupt
the speech (handy for barge-in via wake word).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from tts.tts import KokoroVoice

router = APIRouter()
log = logging.getLogger(__name__)


@router.websocket("/tts")
async def tts(ws: WebSocket) -> None:
    await ws.accept()
    voice: KokoroVoice = ws.app.state.tts_voice
    try:
        while True:
            payload = await ws.receive_json()
            text = payload.get("text", "").strip()
            if not text:
                await ws.send_json({"event": "error", "message": "empty text"})
                continue

            await _stream_synthesis(ws, voice, text)
            await ws.send_json({"event": "end"})
    except WebSocketDisconnect:
        log.info("TTS client disconnected")
    except Exception:
        log.exception("TTS error")
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"event": "error", "message": "internal error"})
            await ws.close()


async def _stream_synthesis(ws: WebSocket, voice: KokoroVoice, text: str) -> None:
    """Iterate Kokoro's chunk generator and push each one across the wire.

    Kokoro's pipeline is sync. We run each chunk-generating step in a worker
    thread so we don't block the asyncio event loop while Kokoro is busy.
    """
    loop = asyncio.get_running_loop()
    gen = voice.synthesize(text)

    while True:
        chunk = await loop.run_in_executor(None, _next_or_none, gen)
        if chunk is None:
            return
        if isinstance(chunk, np.ndarray):
            pcm = chunk.astype(np.float32, copy=False)
        else:
            # Kokoro may yield torch tensors — coerce.
            pcm = np.asarray(chunk, dtype=np.float32)
        await ws.send_bytes(pcm.tobytes())

        # If the client disconnects mid-stream, bail out gracefully.
        if ws.application_state != WebSocketState.CONNECTED:
            return


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None
