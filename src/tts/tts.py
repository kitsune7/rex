from collections.abc import Callable

import sounddevice as sd
from piper import PiperVoice

models = {
    "danny": "en_US-danny-low.onnx",
    "joe": "en_US-joe-medium.onnx",
    "hfc_male": "en_US-hfc_male-medium.onnx",
}


def load_voice(voice_model="hfc_male"):
    return PiperVoice.load(f"models/tts/{models[voice_model]}")


def speak_text(text, voice, interrupt_check: Callable[[], bool] | None = None) -> bool:
    """
    Streams and plays audio chunks as they arrive.

    Args:
        text: The text to speak.
        voice: The PiperVoice instance to use.
        interrupt_check: Optional callable that returns True if speech should be
                        interrupted. Called between audio chunks.

    Returns:
        True if speech was interrupted, False if it completed normally.
    """
    for chunk in voice.synthesize(text):
        # Check for interruption before playing each chunk
        if interrupt_check is not None and interrupt_check():
            sd.stop()
            return True

        sd.play(chunk.audio_float_array, samplerate=chunk.sample_rate, blocking=True)

    sd.wait()
    return False


def stop_speaking():
    """Immediately stop any currently playing speech."""
    sd.stop()
