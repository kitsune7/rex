"""
Audio feedback tones for wake word detection.

Plays short musical tones to indicate listening state:
- Ascending C→G: Rex is now listening
- Descending G→C: Rex has finished listening
"""

import threading

import numpy as np
import sounddevice as sd

# Note frequencies (Hz)
C4 = 261.63  # Middle C
G4 = 392.00  # Perfect fifth above C4

# Tone parameters
SAMPLE_RATE = 44100
NOTE_DURATION = 0.1  # seconds per note
GAP_DURATION = 0.05  # seconds between notes


def _generate_tone(frequency: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave tone with smooth attack and release."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Apply envelope to avoid clicks (10ms attack/release)
    envelope_samples = int(0.01 * sample_rate)
    envelope = np.ones_like(tone)
    envelope[:envelope_samples] = np.linspace(0, 1, envelope_samples)
    envelope[-envelope_samples:] = np.linspace(1, 0, envelope_samples)

    return (tone * envelope * 0.3).astype(np.float32)  # 0.3 = volume


def _generate_two_tone_sequence(freq1: float, freq2: float) -> np.ndarray:
    """Generate a two-note sequence with a small gap between notes."""
    tone1 = _generate_tone(freq1, NOTE_DURATION)
    gap = np.zeros(int(SAMPLE_RATE * GAP_DURATION), dtype=np.float32)
    tone2 = _generate_tone(freq2, NOTE_DURATION)

    return np.concatenate([tone1, gap, tone2])


def _play_audio(audio: np.ndarray):
    """Play audio (intended to be run in a thread)."""
    try:
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass  # Silently ignore audio errors


def play_listening_tone():
    """Play ascending C→G tone to indicate Rex is listening. Non-blocking."""
    audio = _generate_two_tone_sequence(C4, G4)
    thread = threading.Thread(target=_play_audio, args=(audio,), daemon=True)
    thread.start()


def play_done_tone():
    """Play descending G→C tone to indicate Rex has finished listening. Non-blocking."""
    audio = _generate_two_tone_sequence(G4, C4)
    thread = threading.Thread(target=_play_audio, args=(audio,), daemon=True)
    thread.start()

