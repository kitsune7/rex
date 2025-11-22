import queue
import time

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


class SpeechRecorder:
    """Records speech with Voice Activity Detection and transcribes using Whisper."""

    def __init__(self, sample_rate=16000, silence_duration=1.5):
        """
        Initialize the speech recorder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            silence_duration: Duration of silence in seconds to stop recording (default: 1.5)
        """
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.audio_queue = queue.Queue()
        self.recording = False

        # Load Silero VAD model
        self.vad_model = load_silero_vad()

        # Load faster-whisper model (small for balanced speed/accuracy)
        print("Loading Whisper model (this may take a moment on first run)...")
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    def audio_callback(self, in_data, frames, time_info, status):
        """Callback function for audio stream - puts audio data into queue."""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(in_data.copy())

    def record_until_silence(self):
        """
        Record audio until VAD detects silence for the specified duration.

        Returns:
            numpy.ndarray: The recorded audio data
        """
        # Clear any existing audio in the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        self.recording = True
        audio_chunks = []
        last_speech_time = time.time()
        speech_detected = False

        # Silero VAD requires exactly 512 samples for 16kHz (32ms chunks)
        vad_chunk_size = 512
        vad_buffer = []

        stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )

        with stream:
            while self.recording:
                try:
                    # Get audio data from queue
                    data = self.audio_queue.get(timeout=0.1)
                    audio_chunks.append(data)
                    vad_buffer.append(data.flatten())

                    # Accumulate enough samples for VAD
                    total_samples = sum(len(chunk) for chunk in vad_buffer)

                    # Process VAD when we have enough data
                    while total_samples >= vad_chunk_size:
                        # Concatenate buffer
                        vad_audio = np.concatenate(vad_buffer)

                        # Extract exactly vad_chunk_size samples
                        vad_chunk = vad_audio[:vad_chunk_size]
                        remaining = vad_audio[vad_chunk_size:]

                        # Update buffer with remaining data
                        vad_buffer = [remaining] if len(remaining) > 0 else []
                        total_samples = len(remaining)

                        # Run VAD on the chunk
                        audio_tensor = torch.from_numpy(vad_chunk).float()
                        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                        # Speech detected if probability > 0.5
                        if speech_prob > 0.5:
                            speech_detected = True
                            last_speech_time = time.time()

                        # Stop recording if we've detected speech and then silence
                        if speech_detected and (time.time() - last_speech_time) > self.silence_duration:
                            self.recording = False
                            break

                except queue.Empty:
                    # Check for timeout if speech was detected
                    if speech_detected and (time.time() - last_speech_time) > self.silence_duration:
                        self.recording = False
                        break
                    continue

        # Concatenate all audio chunks
        if audio_chunks:
            audio_np = np.concatenate(audio_chunks, axis=0).flatten()
            return audio_np
        return np.array([])

    def transcribe(self, audio_data):
        """
        Transcribe audio data to text using faster-whisper.

        Args:
            audio_data: numpy array of audio samples

        Returns:
            str: Transcribed text
        """
        if len(audio_data) == 0:
            return ""

        # Faster-whisper expects float32 audio
        audio_float32 = audio_data.astype(np.float32)

        # Transcribe
        segments, info = self.whisper_model.transcribe(
            audio_float32,
            language="en",
            beam_size=5,
            vad_filter=True,  # Use built-in VAD filtering for better accuracy
        )

        # Combine all segments into a single transcription
        transcription = " ".join([segment.text for segment in segments]).strip()

        return transcription

    def listen_and_transcribe(self):
        """
        Complete workflow: record speech with VAD and transcribe.

        Returns:
            str: Transcribed text
        """
        print("Listening for speech...")
        audio_data = self.record_until_silence()

        if len(audio_data) == 0:
            return ""

        print("Transcribing...")
        transcription = self.transcribe(audio_data)
        return transcription
