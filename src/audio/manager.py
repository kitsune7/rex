"""
Unified audio manager for Rex voice assistant.

Coordinates all audio I/O operations:
- Input stream management for wake word and speech capture
- Output management for TTS and alarm sounds
- Priority system for competing audio sources
- Muting support during conversations
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from core.events import EventBus


class SoundPriority(IntEnum):
    """
    Priority levels for audio playback.

    Higher priority sounds can interrupt lower priority ones.
    """

    LOW = 0  # Background sounds
    NORMAL = 1  # Feedback tones
    HIGH = 2  # TTS speech
    URGENT = 3  # Alarms


@dataclass
class AudioConfig:
    """Audio configuration constants."""

    # Sample rates
    INPUT_SAMPLE_RATE: int = 16000  # For STT (Whisper expects 16kHz)
    OUTPUT_SAMPLE_RATE: int = 22050  # For TTS (Piper default)
    FEEDBACK_SAMPLE_RATE: int = 44100  # For generated tones

    # Input settings
    INPUT_CHANNELS: int = 1
    INPUT_DTYPE: type = np.int16
    CHUNK_SIZE: int = 1280  # 80ms chunks at 16kHz


class AudioManager:
    """
    Unified manager for all audio operations in Rex.

    This class provides:
    - Coordinated access to audio hardware
    - Priority-based sound playback
    - Muting support for conversations
    - Resource cleanup on shutdown
    """

    def __init__(self, event_bus: "EventBus | None" = None):
        """
        Initialize the audio manager.

        Args:
            event_bus: Optional event bus for emitting audio events
        """
        self._event_bus = event_bus
        self._config = AudioConfig()

        # State
        self._muted = False
        self._current_priority = SoundPriority.LOW
        self._lock = threading.RLock()

        # Sound cache
        self._sound_cache: dict[str, tuple[np.ndarray, int]] = {}

        # Active streams
        self._input_stream: sd.InputStream | None = None

    @property
    def config(self) -> AudioConfig:
        """Get audio configuration."""
        return self._config

    @property
    def is_muted(self) -> bool:
        """Check if audio output is muted."""
        return self._muted

    def mute(self) -> None:
        """Mute audio output. Stops any currently playing sound."""
        with self._lock:
            self._muted = True
            sd.stop()

    def unmute(self) -> None:
        """Unmute audio output."""
        with self._lock:
            self._muted = False

    @contextmanager
    def muted_context(self):
        """Context manager for temporary muting."""
        self.mute()
        try:
            yield
        finally:
            self.unmute()

    def create_input_stream(
        self,
        callback=None,
        blocksize: int | None = None,
    ) -> sd.InputStream:
        """
        Create an audio input stream with standard settings.

        Args:
            callback: Optional callback for stream processing
            blocksize: Block size in samples (defaults to config chunk size)

        Returns:
            Configured InputStream ready to use
        """
        return sd.InputStream(
            samplerate=self._config.INPUT_SAMPLE_RATE,
            channels=self._config.INPUT_CHANNELS,
            dtype=self._config.INPUT_DTYPE,
            blocksize=blocksize or self._config.CHUNK_SIZE,
            callback=callback,
        )

    def play_sound(
        self,
        audio: np.ndarray,
        sample_rate: int,
        priority: SoundPriority = SoundPriority.NORMAL,
        blocking: bool = False,
    ) -> bool:
        """
        Play an audio array with priority handling.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            priority: Priority level for this sound
            blocking: If True, wait for playback to complete

        Returns:
            True if sound was played, False if muted or preempted
        """
        with self._lock:
            if self._muted and priority < SoundPriority.URGENT:
                return False

            # Stop lower priority sounds
            if priority > self._current_priority:
                sd.stop()

            self._current_priority = priority

        try:
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
            return True
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
        finally:
            with self._lock:
                if not blocking:
                    # Reset priority after non-blocking play starts
                    self._current_priority = SoundPriority.LOW

    def play_sound_file(
        self,
        path: str | Path,
        priority: SoundPriority = SoundPriority.NORMAL,
        blocking: bool = False,
        loop: bool = False,
    ) -> bool:
        """
        Play a sound file with optional caching.

        Args:
            path: Path to the audio file
            priority: Priority level for this sound
            blocking: If True, wait for playback to complete
            loop: If True, loop the sound (blocking must be False)

        Returns:
            True if sound was played, False if muted or file not found
        """
        path = Path(path)
        cache_key = str(path)

        # Load from cache or file
        if cache_key in self._sound_cache:
            audio, sample_rate = self._sound_cache[cache_key]
        else:
            if not path.exists():
                return False
            try:
                audio, sample_rate = sf.read(path)
                self._sound_cache[cache_key] = (audio, sample_rate)
            except Exception as e:
                print(f"Error loading sound file {path}: {e}")
                return False

        if loop:
            # Looping requires special handling - caller manages the loop
            return self.play_sound(audio, sample_rate, priority, blocking=False)

        return self.play_sound(audio, sample_rate, priority, blocking)

    def stop(self) -> None:
        """Stop all currently playing audio."""
        sd.stop()
        with self._lock:
            self._current_priority = SoundPriority.LOW

    def wait(self) -> None:
        """Wait for current audio playback to complete."""
        sd.wait()

    def get_sound_duration(self, path: str | Path) -> float:
        """
        Get the duration of a sound file in seconds.

        Args:
            path: Path to the audio file

        Returns:
            Duration in seconds, or 0.0 if file not found
        """
        path = Path(path)
        cache_key = str(path)

        if cache_key in self._sound_cache:
            audio, sample_rate = self._sound_cache[cache_key]
            return len(audio) / sample_rate

        if not path.exists():
            return 0.0

        try:
            audio, sample_rate = sf.read(path)
            self._sound_cache[cache_key] = (audio, sample_rate)
            return len(audio) / sample_rate
        except Exception:
            return 0.0

    def cleanup(self) -> None:
        """Clean up audio resources."""
        self.stop()
        if self._input_stream is not None:
            try:
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        self._sound_cache.clear()
