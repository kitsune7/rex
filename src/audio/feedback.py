"""
Audio feedback tones for Rex voice assistant.

Plays short musical tones to indicate listening state:
- Ascending C→G: Rex is now listening
- Descending G→C: Rex has finished listening
- Looping D→A: Rex is thinking (waiting for LLM)

All tones are played through AudioManager for coordinated audio output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from audio.manager import AudioManager

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
    envelope_duration: float = 0.02,
) -> np.ndarray:
    """Generate a sine wave tone with smooth attack and release.

    Args:
        frequency: Tone frequency in Hz
        duration: Tone duration in seconds
        sample_rate: Audio sample rate
        volume: Volume multiplier (0.0 to 1.0)
        envelope_duration: Duration of fade-in/fade-out in seconds (default 20ms)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Apply raised cosine envelope to avoid clicks
    envelope_samples = int(envelope_duration * sample_rate)
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


def generate_listening_tone() -> np.ndarray:
    """Generate ascending C→G tone to indicate Rex is listening."""
    return _generate_two_tone_sequence(C4, G4)


def generate_done_tone() -> np.ndarray:
    """Generate descending G→C tone to indicate Rex has finished listening."""
    return _generate_two_tone_sequence(G4, C4)


def generate_thinking_sequence() -> np.ndarray:
    """Generate a D→A tone sequence with slower timing for thinking feedback."""
    # Use longer 50ms envelope for smoother fade-in (reduces pop at loop start)
    tone1 = _generate_tone(D4, THINKING_NOTE_DURATION, volume=THINKING_VOLUME, envelope_duration=0.05)
    gap = np.zeros(int(SAMPLE_RATE * THINKING_GAP_DURATION), dtype=np.float32)
    tone2 = _generate_tone(A4, THINKING_NOTE_DURATION, volume=THINKING_VOLUME, envelope_duration=0.05)
    trailing_gap = np.zeros(int(SAMPLE_RATE * THINKING_GAP_DURATION), dtype=np.float32)
    return np.concatenate([tone1, gap, tone2, trailing_gap])


class ThinkingTone:
    """
    Looping D→A tone that plays while waiting for LLM inference.

    Can be used as a context manager:
        with ThinkingTone(audio_manager):
            # tone plays during this block
            result = slow_operation()

    Uses AudioManager's persistent stream for pop-free looping.
    """

    def __init__(self, audio_manager: AudioManager):
        self._audio_manager = audio_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # Don't suppress exceptions

    def start(self):
        """Start playing the thinking tone. Non-blocking."""
        audio = generate_thinking_sequence()
        self._audio_manager.start_loop(audio)

    def stop(self):
        """Stop the thinking tone."""
        self._audio_manager.stop_loop()
