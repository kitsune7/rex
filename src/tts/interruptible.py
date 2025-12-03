"""
Interruptible TTS playback with wake word detection.

Allows users to interrupt Rex's speech by saying "Hey Rex".
When interrupted, captures the user's speech so it can be processed
without requiring them to repeat themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wake_word import WakeWordMonitor

from .tts import KokoroVoice, speak_text

if TYPE_CHECKING:
    from audio.manager import AudioManager


class InterruptibleSpeaker:
    """
    Handles TTS playback with the ability to interrupt via wake word.

    Uses WakeWordMonitor to detect when the user says "Hey Rex" during playback.
    When interrupted, captures the user's speech including any words spoken
    after the wake word.
    """

    def __init__(
        self,
        voice: KokoroVoice,
        audio_manager: AudioManager,
        model_path: str,
        threshold: float = 0.5,
    ):
        """
        Initialize the interruptible speaker.

        Args:
            voice: KokoroVoice instance for TTS synthesis.
            audio_manager: AudioManager instance for audio output.
            model_path: Path to the wake word .onnx model file.
            threshold: Wake word detection threshold (0.0 to 1.0).
        """
        self.voice = voice
        self._audio_manager = audio_manager
        self._monitor = WakeWordMonitor(model_path, threshold=threshold)

    def speak_interruptibly(self, text: str) -> tuple[bool, np.ndarray | None]:
        """
        Speak the given text, allowing interruption via wake word.

        When the user interrupts with the wake word, their speech is captured
        (including words spoken after the wake word) so it can be processed
        immediately without requiring them to repeat themselves.

        Args:
            text: The text to speak.

        Returns:
            Tuple of (was_interrupted, captured_audio):
            - was_interrupted: True if speech was interrupted by wake word
            - captured_audio: numpy array of captured audio if interrupted, else None
        """
        self._monitor.start()

        try:
            was_interrupted = speak_text(
                text, self.voice, self._audio_manager, interrupt_check=self._monitor.was_detected
            )

            if was_interrupted:
                # Play listening tone to indicate we're capturing their speech
                self._audio_manager.play_listening_tone()

                # Wait a moment for the monitor to finish capturing audio
                self._monitor.stop()

                # Play done tone to indicate we've captured their speech
                self._audio_manager.play_done_tone()

                return True, self._monitor.get_captured_audio()

            return False, None
        finally:
            # Always stop monitoring when done
            self._monitor.stop()
