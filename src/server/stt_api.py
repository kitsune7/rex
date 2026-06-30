"""POST /stt — transcribe an uploaded WAV/PCM clip via Whisper."""

from __future__ import annotations

import io
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from stt import Transcriber

router = APIRouter()


class STTResponse(BaseModel):
    text: str
    duration_ms: int


def get_transcriber(request: Request) -> Transcriber:
    transcriber = getattr(request.app.state, "transcriber", None)
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Transcriber not ready")
    return transcriber


@router.post("/stt", response_model=STTResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(..., description="WAV file at any sample rate, mono or stereo"),
) -> STTResponse:
    """Transcribe a single audio clip.

    Accepts any format `soundfile` understands. Whisper expects 16 kHz mono
    float32, so we resample/downmix here if needed.
    """
    transcriber = get_transcriber(request)
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    try:
        samples, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    if samples.ndim == 2:
        # Downmix to mono.
        samples = samples.mean(axis=1).astype(np.int16)

    if sr != 16000:
        samples = _resample_linear(samples, sr, 16000)

    t0 = time.monotonic()
    text = transcriber.transcribe(samples, strip_wake_word=True)
    dt_ms = int((time.monotonic() - t0) * 1000)
    return STTResponse(text=text, duration_ms=dt_ms)


def _resample_linear(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Cheap linear resampler — fine for speech going into Whisper."""
    if from_rate == to_rate:
        return samples
    ratio = to_rate / from_rate
    new_len = int(len(samples) * ratio)
    old_idx = np.arange(len(samples))
    new_idx = np.linspace(0, len(samples) - 1, new_len)
    return np.interp(new_idx, old_idx, samples.astype(np.float32)).astype(np.int16)
