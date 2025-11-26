"""
Wake Word Listener with Rolling Audio Buffer

Continuously captures audio into a rolling buffer while detecting wake words.
When wake word is detected, continues recording until silence, then returns
the complete audio (buffered + new) for transcription.
"""

import collections
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from openwakeword import Model
from silero_vad import load_silero_vad

from .model_utils import ensure_openwakeword_models


class WakeWordListener:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        buffer_duration: float = 3.0,
        silence_duration: float = 1.5,
    ):
        """
        Initialize the wake word listener with rolling buffer.

        Args:
            model_path: Path to the .onnx wake word model file
            threshold: Detection threshold (0.0 to 1.0)
            buffer_duration: Seconds of audio to keep in rolling buffer
            silence_duration: Seconds of silence to stop recording after wake word
        """
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
            print("❌ Failed to download required models")
            sys.exit(1)

        # Load wake word model
        print(f"Loading wake word model from: {model_path}")
        self._wake_model = Model(wakeword_models=[model_path], inference_framework="onnx")

        # Load VAD model for end-of-speech detection
        print("Loading VAD model...")
        self._vad_model = load_silero_vad()

        # State
        self._interrupted = False
        self._stream = None
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self._interrupted = True

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
        self._interrupted = False
        self._ring_buffer.clear()
        self._wake_model.reset()

        # Open audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=self.chunk_size,
        )

        try:
            with self._stream:
                # Phase 1: Listen for wake word while filling buffer
                if not self._wait_for_wake_word():
                    return None

                print("🔔 Wake word detected! Transcribing audio...")

                # Invoke callback (e.g., to mute timer sounds)
                if on_wake_word:
                    on_wake_word()

                # Phase 2: Continue recording until silence
                return self._record_until_silence()

        except Exception as e:
            print(f"❌ Audio error: {e}")
            return None

    def _wait_for_wake_word(self) -> bool:
        """
        Listen for wake word while maintaining rolling buffer.

        Returns:
            True if wake word detected, False if interrupted.
        """
        while not self._interrupted:
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
        vad_chunk_size = 512  # Silero VAD requires 512 samples at 16kHz
        vad_buffer = []

        while not self._interrupted:
            audio_chunk, _ = self._stream.read(self.chunk_size)
            audio_chunk = audio_chunk.flatten()
            audio_chunks.append(audio_chunk)
            vad_buffer.append(audio_chunk)

            # Process VAD when we have enough samples
            total_samples = sum(len(c) for c in vad_buffer)
            while total_samples >= vad_chunk_size:
                vad_audio = np.concatenate(vad_buffer)
                vad_chunk = vad_audio[:vad_chunk_size]
                remaining = vad_audio[vad_chunk_size:]
                vad_buffer = [remaining] if len(remaining) > 0 else []
                total_samples = len(remaining)

                # Run VAD
                audio_tensor = torch.from_numpy(vad_chunk.astype(np.float32))
                speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:
                    speech_detected = True
                    last_speech_time = time.time()

                # Stop if speech was detected and now silent
                if speech_detected and (time.time() - last_speech_time) > self.silence_duration:
                    return np.concatenate(audio_chunks)

        return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.int16)

    def listen_for_speech(self, timeout: float = 5.0) -> np.ndarray | None:
        """
        Listen for speech without requiring wake word detection.

        Waits for speech to begin (up to timeout), then records until silence.
        Used for follow-up questions where the user can respond without
        saying the wake word again.

        Args:
            timeout: Maximum seconds to wait for speech to begin.

        Returns:
            numpy array of audio samples if speech detected,
            or None if timeout occurred without speech.
        """
        self._interrupted = False
        self._ring_buffer.clear()

        vad_chunk_size = 512  # Silero VAD requires 512 samples at 16kHz
        vad_buffer = []
        start_time = time.time()

        # Open audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=self.chunk_size,
        )

        try:
            with self._stream:
                # Phase 1: Wait for speech to start (with timeout)
                while not self._interrupted:
                    # Check timeout
                    if (time.time() - start_time) > timeout:
                        return None

                    audio_chunk, _ = self._stream.read(self.chunk_size)
                    audio_chunk = audio_chunk.flatten()

                    # Keep audio in ring buffer in case speech started
                    self._ring_buffer.extend(audio_chunk)
                    vad_buffer.append(audio_chunk)

                    # Process VAD when we have enough samples
                    total_samples = sum(len(c) for c in vad_buffer)
                    while total_samples >= vad_chunk_size:
                        vad_audio = np.concatenate(vad_buffer)
                        vad_chunk = vad_audio[:vad_chunk_size]
                        remaining = vad_audio[vad_chunk_size:]
                        vad_buffer = [remaining] if len(remaining) > 0 else []
                        total_samples = len(remaining)

                        # Run VAD
                        audio_tensor = torch.from_numpy(vad_chunk.astype(np.float32))
                        speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()

                        if speech_prob > 0.5:
                            # Speech detected! Trim ring buffer to only recent audio
                            # to avoid including long periods of silence from wait phase
                            recent_samples = int(0.5 * self.sample_rate)  # 0.5 seconds
                            buffer_array = np.array(self._ring_buffer, dtype=np.int16)
                            self._ring_buffer.clear()
                            self._ring_buffer.extend(
                                buffer_array[-recent_samples:]
                                if len(buffer_array) > recent_samples
                                else buffer_array
                            )
                            return self._record_until_silence(include_buffer=True)

                return None

        except Exception as e:
            print(f"❌ Audio error: {e}")
            return None

    def stop(self):
        """Stop listening and clean up."""
        self._interrupted = True
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass


def run_wake_word_listener(args):
    """CLI entry point for standalone wake word testing."""
    base_model_path = Path("models/wake_word_models")
    model_path = base_model_path / args.model / f"{args.model}.onnx"

    if not model_path.exists():
        print(f"❌ Error: Model file not found at: {model_path}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)

    listener = WakeWordListener(model_path=str(model_path), threshold=args.threshold)

    print(f"\n🎤 Listening for wake word...")
    print("   Press Ctrl+C to stop\n")

    try:
        while True:
            audio = listener.wait_for_wake_word_and_speech()
            if audio is None:
                break
            print(f"🔔 Captured {len(audio) / 16000:.1f}s of audio")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print("\nGoodbye! 👋")
