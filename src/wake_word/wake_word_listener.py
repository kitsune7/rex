"""
Wake Word Listener with Rolling Audio Buffer

Continuously captures audio into a rolling buffer while detecting wake words.
When wake word is detected, continues recording until silence, then returns
the complete audio (buffered + new) for transcription.
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import torch
from openwakeword import Model
from silero_vad import load_silero_vad

from .model_utils import ensure_openwakeword_models

if TYPE_CHECKING:
    from audio.manager import AudioManager


class WakeWordMonitor:
    """
    Background wake word detector for interruption during TTS playback.

    Runs in a separate thread and sets an event when wake word is detected.
    Also captures audio in a rolling buffer so interrupted speech can be
    processed immediately without requiring the user to repeat themselves.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        buffer_duration: float = 3.0,
        silence_duration: float = 1.5,
    ):
        """
        Initialize the wake word monitor.

        Args:
            model_path: Path to the .onnx wake word model file
            threshold: Detection threshold (0.0 to 1.0)
            buffer_duration: Seconds of audio to keep in rolling buffer
            silence_duration: Seconds of silence to stop recording after wake word
        """
        self.threshold = threshold
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms chunks
        self.buffer_duration = buffer_duration
        self.silence_duration = silence_duration

        # Ensure openwakeword models are available
        if not ensure_openwakeword_models():
            raise RuntimeError("Failed to download required models")

        # Load wake word model (separate instance for this monitor)
        self._wake_model = Model(wakeword_models=[model_path], inference_framework="onnx")

        # Load VAD model for end-of-speech detection
        self._vad_model = load_silero_vad()

        # Rolling buffer for audio capture
        buffer_samples = int(self.sample_rate * buffer_duration)
        self._ring_buffer: collections.deque = collections.deque(maxlen=buffer_samples)

        # Threading state
        self._detected_event = threading.Event()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Captured audio after wake word detection
        self._captured_audio: np.ndarray | None = None
        self._audio_lock = threading.Lock()

    def _monitor_loop(self):
        """Background thread that listens for wake word and captures audio."""
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=self.chunk_size,
            )
            with stream:
                self._ready_event.set()  # Signal that we're ready to detect
                self._wake_model.reset()
                while not self._stop_event.is_set():
                    audio_chunk, _ = stream.read(self.chunk_size)
                    audio_chunk = audio_chunk.flatten()

                    # Add to rolling buffer
                    self._ring_buffer.extend(audio_chunk)

                    predictions = self._wake_model.predict(audio_chunk)
                    for score in predictions.values():
                        if score >= self.threshold:
                            self._detected_event.set()
                            # Continue recording until silence
                            self._capture_until_silence(stream)
                            return
        except Exception as e:
            print(f"⚠️ Wake word monitor error: {e}")

    def _capture_until_silence(self, stream: sd.InputStream):
        """Continue recording after wake word until VAD detects silence."""
        # Start with buffered audio
        audio_chunks = [np.array(self._ring_buffer, dtype=np.int16)]

        last_speech_time = time.time()
        speech_detected = False
        vad = VADProcessor(self._vad_model, self.sample_rate)

        while not self._stop_event.is_set():
            try:
                audio_chunk, _ = stream.read(self.chunk_size)
                audio_chunk = audio_chunk.flatten()
                audio_chunks.append(audio_chunk)
                vad.add_audio(audio_chunk)

                for speech_prob in vad.process():
                    if speech_prob > 0.5:
                        speech_detected = True
                        last_speech_time = time.time()

                    # Stop if speech was detected and now silent
                    if speech_detected and (time.time() - last_speech_time) > self.silence_duration:
                        with self._audio_lock:
                            self._captured_audio = np.concatenate(audio_chunks)
                        return
            except Exception:
                break

        # Save whatever we captured
        if audio_chunks:
            with self._audio_lock:
                self._captured_audio = np.concatenate(audio_chunks)

    def start(self):
        """Start monitoring for wake word in the background."""
        # Ensure any previous thread is fully stopped
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=0.5)

        self.reset()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # Longer timeout to allow audio capture to complete
            self._thread = None

    def was_detected(self) -> bool:
        """Check if wake word was detected (non-blocking)."""
        return self._detected_event.is_set()

    def get_captured_audio(self) -> np.ndarray | None:
        """
        Get the audio captured after wake word detection.

        Returns:
            numpy array of audio samples, or None if no audio captured.
        """
        with self._audio_lock:
            return self._captured_audio

    def reset(self):
        """Reset the detection state."""
        self._detected_event.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._ring_buffer.clear()
        with self._audio_lock:
            self._captured_audio = None

    def wait_until_ready(self, timeout: float = 1.0) -> bool:
        """
        Wait for the monitor to be ready to detect wake words.

        Args:
            timeout: Maximum seconds to wait for the monitor to be ready.

        Returns:
            True if the monitor is ready, False if timeout occurred.
        """
        return self._ready_event.wait(timeout=timeout)


class VADProcessor:
    """Processes audio chunks through VAD model to detect speech."""

    def __init__(self, vad_model, sample_rate: int, chunk_size: int = 512):
        self._vad_model = vad_model
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._buffer: list[np.ndarray] = []

    def add_audio(self, chunk: np.ndarray):
        """Add an audio chunk to the buffer."""
        self._buffer.append(chunk)

    def process(self) -> list[float]:
        """Process buffered audio and return speech probabilities for complete chunks."""
        probabilities = []
        total_samples = sum(len(chunk) for chunk in self._buffer)

        while total_samples >= self._chunk_size:
            vad_audio = np.concatenate(self._buffer)
            vad_chunk = vad_audio[: self._chunk_size]
            remaining = vad_audio[self._chunk_size :]
            self._buffer = [remaining] if len(remaining) > 0 else []
            total_samples = len(remaining)

            audio_tensor = torch.from_numpy(vad_chunk.astype(np.float32))
            speech_prob = self._vad_model(audio_tensor, self._sample_rate).item()
            probabilities.append(speech_prob)

        return probabilities

    def reset(self):
        """Clear the audio buffer."""
        self._buffer.clear()


class WakeWordListener:
    def __init__(
        self,
        model_path: str,
        audio_manager: AudioManager,
        threshold: float = 0.5,
        buffer_duration: float = 3.0,
        silence_duration: float = 1.5,
    ):
        """
        Initialize the wake word listener with rolling buffer.

        Args:
            model_path: Path to the .onnx wake word model file
            audio_manager: AudioManager instance for audio output
            threshold: Detection threshold (0.0 to 1.0)
            buffer_duration: Seconds of audio to keep in rolling buffer
            silence_duration: Seconds of silence to stop recording after wake word
        """
        self._audio_manager = audio_manager
        self.threshold = threshold
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms chunks for wake word detection
        self.buffer_duration = buffer_duration
        self.silence_duration = silence_duration

        # Rolling buffer: stores last N seconds of audio samples
        buffer_samples = int(self.sample_rate * buffer_duration)
        self._ring_buffer = collections.deque(maxlen=buffer_samples)

        # Ensure openwakeword models are available
        if not ensure_openwakeword_models():
            raise RuntimeError("Failed to download required openwakeword models")

        # Load wake word model
        print(f"Loading wake word model from: {model_path}")
        self._wake_model = Model(wakeword_models=[model_path], inference_framework="onnx")

        # Load VAD model for end-of-speech detection
        print("Loading VAD model...")
        self._vad_model = load_silero_vad()

        # State (using Event for thread-safe interruption)
        self._interrupted = threading.Event()
        self._stream = None

    def _create_audio_stream(self) -> sd.InputStream:
        """Create and return an audio input stream with standard settings."""
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=self.chunk_size,
        )

    def wait_for_wake_word_and_speech(self, on_wake_word: Callable[[], None] | None = None) -> np.ndarray | None:
        """
        Wait for wake word, then capture speech until silence.

        Args:
            on_wake_word: Optional callback invoked when wake word is detected,
                          before recording starts. Use this to mute other audio sources.

        Returns:
            numpy array of audio samples (including buffered pre-wake-word audio),
            or None if interrupted.
        """
        self._interrupted.clear()
        self._ring_buffer.clear()
        self._wake_model.reset()

        self._stream = self._create_audio_stream()

        try:
            with self._stream:
                # Phase 1: Listen for wake word while filling buffer
                if not self._wait_for_wake_word():
                    return None

                print("🔔 Wake word detected! Transcribing audio...")

                # Play ascending tone to indicate listening
                self._audio_manager.play_listening_tone()

                # Invoke callback (e.g., to mute timer sounds)
                if on_wake_word:
                    on_wake_word()

                # Phase 2: Continue recording until silence
                audio = self._record_until_silence()

                # Play descending tone to indicate done listening
                self._audio_manager.play_done_tone()

                return audio

        except Exception as e:
            print(f"❌ Audio error: {e}")
            return None

    def _wait_for_wake_word(self) -> bool:
        """
        Listen for wake word while maintaining rolling buffer.

        Returns:
            True if wake word detected, False if interrupted.
        """
        while not self._interrupted.is_set():
            audio_chunk, _ = self._stream.read(self.chunk_size)
            audio_chunk = audio_chunk.flatten()

            # Add to rolling buffer
            self._ring_buffer.extend(audio_chunk)

            # Check for wake word
            predictions = self._wake_model.predict(audio_chunk)
            for score in predictions.values():
                if score >= self.threshold:
                    return True

        return False

    def _record_until_silence(self, include_buffer: bool = True) -> np.ndarray:
        """
        Continue recording after wake word until VAD detects silence.

        Args:
            include_buffer: If True, include rolling buffer contents in output.

        Returns:
            Complete audio: optionally rolling buffer contents + new audio.
        """
        # Start with buffered audio if requested
        if include_buffer:
            audio_chunks = [np.array(self._ring_buffer, dtype=np.int16)]
        else:
            audio_chunks = []

        last_speech_time = time.time()
        speech_detected = False
        vad = VADProcessor(self._vad_model, self.sample_rate)

        while not self._interrupted.is_set():
            audio_chunk, _ = self._stream.read(self.chunk_size)
            audio_chunk = audio_chunk.flatten()
            audio_chunks.append(audio_chunk)
            vad.add_audio(audio_chunk)

            for speech_prob in vad.process():
                if speech_prob > 0.5:
                    speech_detected = True
                    last_speech_time = time.time()

                # Stop if speech was detected and now silent
                if speech_detected and (time.time() - last_speech_time) > self.silence_duration:
                    return np.concatenate(audio_chunks)

        return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.int16)

    def listen_for_speech(
        self, timeout: float = 5.0, play_tones: bool = True
    ) -> np.ndarray | None:
        """
        Listen for speech without requiring wake word detection.

        Waits for speech to begin (up to timeout), then records until silence.
        Used for follow-up questions where the user can respond without
        saying the wake word again.

        Args:
            timeout: Maximum seconds to wait for speech to begin.
            play_tones: If True, play listening/done tones. Set to False if
                        the caller has already played a ready tone.

        Returns:
            numpy array of audio samples if speech detected,
            or None if timeout occurred without speech.
        """
        self._interrupted.clear()
        self._ring_buffer.clear()

        vad = VADProcessor(self._vad_model, self.sample_rate)
        start_time = time.time()

        self._stream = self._create_audio_stream()

        try:
            with self._stream:
                # Wait for speech to start (with timeout)
                while not self._interrupted.is_set():
                    # Check timeout
                    if (time.time() - start_time) > timeout:
                        return None

                    audio_chunk, _ = self._stream.read(self.chunk_size)
                    audio_chunk = audio_chunk.flatten()

                    # Keep audio in ring buffer in case speech started
                    self._ring_buffer.extend(audio_chunk)
                    vad.add_audio(audio_chunk)

                    for speech_prob in vad.process():
                        if speech_prob > 0.5:
                            # Trim ring buffer to only recent audio
                            # to avoid including long periods of silence from wait phase
                            recent_samples = int(0.5 * self.sample_rate)  # 0.5 seconds
                            buffer_array = np.array(self._ring_buffer, dtype=np.int16)
                            self._ring_buffer.clear()
                            self._ring_buffer.extend(
                                buffer_array[-recent_samples:] if len(buffer_array) > recent_samples else buffer_array
                            )
                            audio = self._record_until_silence(include_buffer=True)

                            # Play descending tone to indicate done listening
                            if play_tones:
                                self._audio_manager.play_done_tone()

                            return audio

                return None

        except Exception as e:
            print(f"❌ Audio error: {e}")
            return None

    def stop(self):
        """Stop listening and clean up."""
        self._interrupted.set()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass

    def is_interrupted(self) -> bool:
        """Check if the listener has been interrupted (thread-safe)."""
        return self._interrupted.is_set()

    def interrupt(self) -> None:
        """Interrupt the listener (thread-safe). Same as stop() but doesn't close stream."""
        self._interrupted.set()
