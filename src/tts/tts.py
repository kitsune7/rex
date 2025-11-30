import warnings
from collections.abc import Callable

import sounddevice as sd
from kokoro import KPipeline


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


def speak_text(text, voice_obj, interrupt_check: Callable[[], bool] | None = None) -> bool:
    for chunk in voice_obj.synthesize(text):
        if interrupt_check is not None and interrupt_check():
            sd.stop()
            return True
        sd.play(chunk, samplerate=voice_obj.sample_rate, blocking=True)
    sd.wait()
    return False


def stop_speaking():
    """Immediately stop any currently playing speech."""
    sd.stop()
