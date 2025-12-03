"""
Unified audio manager for Rex voice assistant.

Coordinates all audio I/O operations:
- Input stream management for wake word and speech capture
- Output management for TTS, alarms, and feedback sounds
- Single persistent output stream to avoid race conditions
- Muting support during conversations
"""

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from core.events import EventBus


@dataclass
class AudioConfig:
    """Audio configuration constants."""

    # Sample rates
    INPUT_SAMPLE_RATE: int = 16000  # For STT (Whisper expects 16kHz)
    OUTPUT_SAMPLE_RATE: int = 44100  # Unified output sample rate
    TTS_SAMPLE_RATE: int = 24000  # Kokoro TTS native rate

    # Input settings
    INPUT_CHANNELS: int = 1
    INPUT_DTYPE: type = np.int16
    CHUNK_SIZE: int = 1280  # 80ms chunks at 16kHz


class AudioManager:
    """
    Unified manager for all audio operations in Rex.

    All audio output goes through a single persistent stream to avoid
    race conditions from multiple sd.play()/sd.stop() calls. This class provides:
    - Single output stream for all audio (TTS, alarms, feedback tones)
    - Automatic resampling to unified sample rate
    - Muting support during conversations
    - Completion signaling for blocking playback
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
        self._lock = threading.RLock()

        # Sound cache (stores audio already resampled to OUTPUT_SAMPLE_RATE)
        self._sound_cache: dict[str, np.ndarray] = {}

        # Active streams
        self._input_stream: sd.InputStream | None = None

        # Persistent output stream (single stream for all audio output)
        # Using a sentinel value to signal completion for blocking playback
        self._COMPLETION_SENTINEL = object()
        self._output_queue: queue.Queue[np.ndarray | object] = queue.Queue()
        self._current_output: np.ndarray | None = None
        self._output_position = 0
        self._loop_audio: np.ndarray | None = None
        self._output_lock = threading.Lock()
        self._completion_event = threading.Event()
        self._stop_requested = False

        self._output_stream = sd.OutputStream(
            samplerate=self._config.OUTPUT_SAMPLE_RATE,
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

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """
        Resample audio from one sample rate to another using linear interpolation.

        Args:
            audio: Input audio array
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio array
        """
        if from_rate == to_rate:
            return audio

        # Calculate the resampling ratio and new length
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)

        # Use linear interpolation for resampling
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(new_indices, old_indices, audio)

    def _output_callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        """
        Fill output buffer from queued audio, loop audio, or silence.

        Called by sounddevice from a separate thread.
        """
        # Check if stop was requested (clear current audio immediately)
        if self._stop_requested:
            with self._output_lock:
                self._current_output = None
                self._output_position = 0
                # Clear the queue
                while not self._output_queue.empty():
                    try:
                        item = self._output_queue.get_nowait()
                        if item is self._COMPLETION_SENTINEL:
                            self._completion_event.set()
                    except queue.Empty:
                        break
                self._stop_requested = False
            outdata[:, 0] = 0.0
            return

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
                item = self._output_queue.get_nowait()
                # Check for completion sentinel
                if item is self._COMPLETION_SENTINEL:
                    self._completion_event.set()
                    continue
                self._current_output = item
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

    def queue_audio(
        self, audio: np.ndarray, sample_rate: int | None = None
    ) -> None:
        """
        Queue audio for playback through the persistent stream.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio (defaults to OUTPUT_SAMPLE_RATE)
        """
        if self._muted:
            return

        if sample_rate is None:
            sample_rate = self._config.OUTPUT_SAMPLE_RATE

        # Resample if necessary
        if sample_rate != self._config.OUTPUT_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self._config.OUTPUT_SAMPLE_RATE)

        # Normalize to float32 in range [-1, 1]
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            # Assume int16 range, normalize
            audio = audio / 32768.0

        self._output_queue.put(audio)

    def queue_audio_blocking(
        self, audio: np.ndarray, sample_rate: int | None = None, interrupt_check=None
    ) -> bool:
        """
        Queue audio and wait for it to finish playing.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio (defaults to OUTPUT_SAMPLE_RATE)
            interrupt_check: Optional callable that returns True if playback should stop

        Returns:
            True if playback was interrupted, False if completed normally
        """
        if self._muted:
            return False

        # Queue the audio
        self.queue_audio(audio, sample_rate)

        # Queue a completion sentinel
        self._completion_event.clear()
        self._output_queue.put(self._COMPLETION_SENTINEL)

        # Wait for completion, checking for interrupts
        while not self._completion_event.is_set():
            if interrupt_check is not None and interrupt_check():
                self.stop_current()
                return True
            self._completion_event.wait(timeout=0.05)

        return False

    def start_loop(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """
        Start looping audio through the persistent stream.

        Used for continuous feedback like the thinking tone or alarm sounds.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio (defaults to OUTPUT_SAMPLE_RATE)
        """
        if sample_rate is None:
            sample_rate = self._config.OUTPUT_SAMPLE_RATE

        # Resample if necessary
        if sample_rate != self._config.OUTPUT_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self._config.OUTPUT_SAMPLE_RATE)

        # Normalize to float32
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        with self._output_lock:
            self._loop_audio = audio

    def stop_loop(self) -> None:
        """Stop any currently looping audio."""
        with self._output_lock:
            self._loop_audio = None

    def stop_current(self) -> None:
        """Stop current playback and clear the queue (but not loops)."""
        self._stop_requested = True

    def mute(self) -> None:
        """Mute audio output. Stops any currently playing sound."""
        with self._lock:
            self._muted = True
            self.stop_current()
            self.stop_loop()

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

    def play_sound_file(
        self,
        path: str | Path,
        blocking: bool = False,
    ) -> bool:
        """
        Play a sound file with optional caching.

        Args:
            path: Path to the audio file
            blocking: If True, wait for playback to complete

        Returns:
            True if sound was played, False if muted or file not found
        """
        path = Path(path)
        cache_key = str(path)

        # Load from cache or file
        if cache_key in self._sound_cache:
            audio = self._sound_cache[cache_key]
        else:
            if not path.exists():
                return False
            try:
                raw_audio, sample_rate = sf.read(path)
                # Convert to mono if stereo
                if len(raw_audio.shape) > 1:
                    raw_audio = raw_audio.mean(axis=1)
                # Resample and cache
                audio = self._resample(
                    raw_audio.astype(np.float32),
                    sample_rate,
                    self._config.OUTPUT_SAMPLE_RATE,
                )
                self._sound_cache[cache_key] = audio
            except Exception as e:
                print(f"Error loading sound file {path}: {e}")
                return False

        if blocking:
            self.queue_audio_blocking(audio)
        else:
            self.queue_audio(audio)
        return True

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
            audio = self._sound_cache[cache_key]
            return len(audio) / self._config.OUTPUT_SAMPLE_RATE

        if not path.exists():
            return 0.0

        try:
            raw_audio, sample_rate = sf.read(path)
            if len(raw_audio.shape) > 1:
                raw_audio = raw_audio.mean(axis=1)
            audio = self._resample(
                raw_audio.astype(np.float32),
                sample_rate,
                self._config.OUTPUT_SAMPLE_RATE,
            )
            self._sound_cache[cache_key] = audio
            return len(audio) / self._config.OUTPUT_SAMPLE_RATE
        except Exception:
            return 0.0

    # --- Feedback tone convenience methods ---

    def play_listening_tone(self) -> None:
        """Play ascending tone to indicate Rex is listening. Non-blocking."""
        from .feedback import generate_listening_tone

        self.queue_audio(generate_listening_tone())

    def play_done_tone(self) -> None:
        """Play descending tone to indicate Rex has finished listening. Non-blocking."""
        from .feedback import generate_done_tone

        self.queue_audio(generate_done_tone())

    def start_thinking_tone(self) -> None:
        """Start looping thinking tone. Non-blocking."""
        from .feedback import generate_thinking_sequence

        self.start_loop(generate_thinking_sequence())

    def stop_thinking_tone(self) -> None:
        """Stop the thinking tone loop."""
        self.stop_loop()

    def cleanup(self) -> None:
        """Clean up audio resources."""
        self.stop_current()
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
