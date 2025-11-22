import sounddevice as sd
from piper import PiperVoice

models = {
    "danny": "en_US-danny-low.onnx",
    "joe": "en_US-joe-medium.onnx",
    "hfc_male": "en_US-hfc_male-medium.onnx",
}


def load_voice(voice_model="hfc_male"):
    return PiperVoice.load(f"voice_models/{models[voice_model]}")


def speak_text(text, voice):
    """Streams and plays audio chunks as they arrive"""
    for chunk in voice.synthesize(text):
        sd.play(chunk.audio_float_array, samplerate=chunk.sample_rate, blocking=True)
    sd.wait()
