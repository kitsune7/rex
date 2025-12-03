"""
Unified audio manager for Rex voice assistant.

Coordinates all audio I/O operations:
- Input stream management for wake word and speech capture
- Output management for TTS and alarm sounds
- Priority system for competing audio sources
- Muting support during conversations
"""

import queue
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

        # Persistent output stream for feedback tones (avoids cold-start pops)
        self._output_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._current_output: np.ndarray | None = None
        self._output_position = 0
        self._loop_audio: np.ndarray | None = None
        self._output_lock = threading.Lock()

        self._output_stream = sd.OutputStream(
            samplerate=self._config.FEEDBACK_SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self._output_callback,
            blocksize=1024,
        )
        self._output_stream.start()

    @property
    def config(self) -> AudioConfig:
        """Get audio configuration."""
        return self._config

    @property
    def is_muted(self) -> bool:
        """Check if audio output is muted."""
        return self._muted

    def _output_callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        """
        Fill output buffer from queued audio, loop audio, or silence.

        Called by sounddevice from a separate thread.
        """
        output_pos = 0

        while output_pos < frames:
            # If we have current audio being played, continue it
            if self._current_output is not None:
                remaining = len(self._current_output) - self._output_position
                to_copy = min(remaining, frames - output_pos)
                outdata[output_pos : output_pos + to_copy, 0] = self._current_output[
                    self._output_position : self._output_position + to_copy
                ]
                self._output_position += to_copy
                output_pos += to_copy

                # Check if we finished this audio
                if self._output_position >= len(self._current_output):
                    self._current_output = None
                    self._output_position = 0
                continue

            # Try to get next audio from queue (non-blocking)
            try:
                self._current_output = self._output_queue.get_nowait()
                self._output_position = 0
                continue
            except queue.Empty:
                pass

            # Check for loop audio
            with self._output_lock:
                if self._loop_audio is not None:
                    self._current_output = self._loop_audio.copy()
                    self._output_position = 0
                    continue

            # No audio available - output silence for remaining frames
            outdata[output_pos:, 0] = 0.0
            break

    def queue_audio(self, audio: np.ndarray) -> None:
        """
        Queue audio for playback through the persistent stream.

        Args:
            audio: Audio data as float32 numpy array at FEEDBACK_SAMPLE_RATE
        """
        if not self._muted:
            self._output_queue.put(audio.astype(np.float32))

    def start_loop(self, audio: np.ndarray) -> None:
        """
        Start looping audio through the persistent stream.

        Used for continuous feedback like the thinking tone.

        Args:
            audio: Audio data as float32 numpy array at FEEDBACK_SAMPLE_RATE
        """
        with self._output_lock:
            self._loop_audio = audio.astype(np.float32)

    def stop_loop(self) -> None:
        """Stop any currently looping audio."""
        with self._output_lock:
            self._loop_audio = None

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
    ) -> bool:
        """
        Play a sound file with optional caching.

        Args:
            path: Path to the audio file
            priority: Priority level for this sound
            blocking: If True, wait for playback to complete

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
        self.stop_loop()

        # Stop persistent output stream
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass

        if self._input_stream is not None:
            try:
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        self._sound_cache.clear()
