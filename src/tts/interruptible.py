"""
Interruptible TTS playback with wake word detection.

Allows users to interrupt Rex's speech by saying "Hey Rex".
"""

from piper import PiperVoice

from wake_word import WakeWordMonitor

from .tts import speak_text


class InterruptibleSpeaker:
    """
    Handles TTS playback with the ability to interrupt via wake word.

    Uses WakeWordMonitor to detect when the user says "Hey Rex" during playback.
    """

    def __init__(self, voice: PiperVoice, model_path: str, threshold: float = 0.5):
        """
        Initialize the interruptible speaker.

        Args:
            voice: PiperVoice instance for TTS synthesis.
            model_path: Path to the wake word .onnx model file.
            threshold: Wake word detection threshold (0.0 to 1.0).
        """
        self.voice = voice
        self._monitor = WakeWordMonitor(model_path, threshold=threshold)

    def speak_interruptibly(self, text: str) -> bool:
        """
        Speak the given text, allowing interruption via wake word.

        Args:
            text: The text to speak.

        Returns:
            True if speech was interrupted by wake word, False if completed normally.
        """
        self._monitor.start()

        try:
            return speak_text(text, self.voice, interrupt_check=self._monitor.was_detected)
        finally:
            # Always stop monitoring when done
            self._monitor.stop()
