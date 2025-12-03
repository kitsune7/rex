"""
Text-to-Speech synthesis using Kokoro.

Routes all audio through AudioManager to avoid race conditions.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from kokoro import KPipeline

if TYPE_CHECKING:
    from audio.manager import AudioManager


class KokoroVoice:
    def __init__(self, lang_code, voice):
        self.voice = voice
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
            self.pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        self.sample_rate = 24000

    def synthesize(self, text):
        """Generator that yields audio chunks for the given text and voice"""
        for _, _, audio in self.pipeline(text, voice=self.voice):
            yield audio


def load_voice(lang_code="a", voice="am_fenrir"):
    return KokoroVoice(lang_code, voice)


def speak_text(
    text: str,
    voice_obj: KokoroVoice,
    audio_manager: AudioManager,
    interrupt_check: Callable[[], bool] | None = None,
) -> bool:
    """
    Speak text using TTS, with optional interruption support.

    Routes audio through AudioManager for coordinated playback.

    Args:
        text: Text to speak
        voice_obj: KokoroVoice instance
        audio_manager: AudioManager instance for audio output
        interrupt_check: Optional callable returning True to interrupt

    Returns:
        True if interrupted, False if completed normally
    """
    for chunk in voice_obj.synthesize(text):
        if interrupt_check is not None and interrupt_check():
            audio_manager.stop_current()
            return True

        was_interrupted = audio_manager.queue_audio_blocking(
            chunk, voice_obj.sample_rate, interrupt_check
        )
        if was_interrupted:
            return True

    return False
