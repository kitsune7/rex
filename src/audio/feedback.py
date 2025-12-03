"""
Audio feedback tones for Rex voice assistant.

Plays short musical tones to indicate listening state:
- Ascending C→G: Rex is now listening
- Descending G→C: Rex has finished listening
- Looping D→A: Rex is thinking (waiting for LLM)
"""

import threading

import numpy as np
import sounddevice as sd

# Note frequencies (Hz)
C4 = 261.63  # Middle C
G4 = 392.00  # Perfect fifth above C4
D4 = 293.66  # D above middle C
A4 = 440.00  # A above middle C

# Tone parameters
SAMPLE_RATE = 44100
NOTE_DURATION = 0.1  # seconds per note
GAP_DURATION = 0.05  # seconds between notes

# Thinking tone parameters (slower tempo, softer)
THINKING_NOTE_DURATION = 0.4
THINKING_GAP_DURATION = 0.05
THINKING_VOLUME = 0.2


def _generate_tone(
    frequency: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 0.3,
) -> np.ndarray:
    """Generate a sine wave tone with smooth attack and release."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Apply raised cosine envelope to avoid clicks (20ms attack/release)
    envelope_samples = int(0.02 * sample_rate)
    envelope = np.ones_like(tone)
    # Fade in: 0.5 * (1 - cos(pi * t)) goes from 0 to 1 smoothly
    envelope[:envelope_samples] = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, envelope_samples)))
    # Fade out: 0.5 * (1 + cos(pi * t)) goes from 1 to 0 smoothly
    envelope[-envelope_samples:] = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, envelope_samples)))

    return (tone * envelope * volume).astype(np.float32)


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


class ThinkingTone:
    """
    Looping D→A tone that plays while waiting for LLM inference.

    Can be used as a context manager:
        with ThinkingTone():
            # tone plays during this block
            result = slow_operation()
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # Don't suppress exceptions

    def _generate_thinking_sequence(self) -> np.ndarray:
        """Generate a D→A tone sequence with slower timing."""
        tone1 = _generate_tone(D4, THINKING_NOTE_DURATION, volume=THINKING_VOLUME)
        gap = np.zeros(int(SAMPLE_RATE * THINKING_GAP_DURATION), dtype=np.float32)
        tone2 = _generate_tone(A4, THINKING_NOTE_DURATION, volume=THINKING_VOLUME)
        trailing_gap = np.zeros(int(SAMPLE_RATE * THINKING_GAP_DURATION), dtype=np.float32)
        return np.concatenate([tone1, gap, tone2, trailing_gap])

    def _play_loop(self):
        """Play the thinking tone on loop until stopped."""
        audio = self._generate_thinking_sequence()

        while not self._stop_event.is_set():
            try:
                sd.play(audio, SAMPLE_RATE)
                sd.wait()  # Proper sync - no overlap
            except Exception:
                pass  # Silently ignore audio errors

    def start(self):
        """Start playing the thinking tone. Non-blocking."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the thinking tone."""
        self._stop_event.set()
        # Don't call sd.stop() - let current sequence finish to avoid pop
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # Allow time for sequence to complete
            self._thread = None
