"""
Speech-to-Text transcription using faster-whisper.
"""

import re

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """Transcribes audio using Whisper."""

    def __init__(self):
        print("Loading Whisper model (this may take a moment on first run)...")
        self._model = WhisperModel("small", device="cpu", compute_type="int8")

    def transcribe(self, audio_data: np.ndarray, strip_wake_word: bool = True) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: numpy array of int16 audio samples at 16kHz
            strip_wake_word: if True, remove "hey rex" from start of transcription

        Returns:
            Transcribed text, or empty string if no speech detected.
        """
        if len(audio_data) == 0:
            return ""

        # Convert int16 to float32 for Whisper
        audio_float = audio_data.astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio_float,
            language="en",
            beam_size=5,  # This is already the default but it's kept for clarity
            vad_filter=True,
        )

        text = " ".join(seg.text for seg in segments).strip()

        if strip_wake_word:
            text = self._strip_wake_word(text)

        return text

    def _strip_wake_word(self, text: str) -> str:
        """Remove wake word variations from the start of transcription."""
        # Common variations Whisper might produce
        patterns = [
            r"^hey\s*rex[,.\s]*",
            r"^hay\s*rex[,.\s]*",
            r"^hey\s*racks[,.\s]*",
            r"^hey\s*wrecks[,.\s]*",
        ]
        result = text
        for pattern in patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result.strip()
