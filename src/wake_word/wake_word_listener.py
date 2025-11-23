"""
Custom Wake Word Detector using openWakeWord
Continuously listens for a custom wake word and prints detections.
"""

import os
import sys
import time
import numpy as np
import pyaudio
from openwakeword import Model
import argparse
from collections import deque
import threading
from pathlib import Path

from .model_utils import ensure_openwakeword_models


class WakeWordListener:
    def __init__(self, model_path, threshold=0.5, chunk_size=1280):
        """
        Initialize the wake word listener.

        Args:
            model_path: Path to the .onnx model file
            threshold: Detection threshold (0.0 to 1.0)
            chunk_size: Audio chunk size (default 1280 for 80ms at 16kHz)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.chunk_size = chunk_size

        # Audio parameters (openWakeWord expects 16kHz, mono, int16)
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # Ensure openwakeword resource models are available
        if not ensure_openwakeword_models():
            print("❌ Failed to download required models")
            sys.exit(1)

        # Initialize model
        print(f"Loading model from: {model_path}")
        self.model = Model(wakeword_models=[model_path], inference_framework="onnx")

        # Get model name for display
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Detection state
        self.is_running = False
        self.detection_count = 0

    def start_listening(self):
        """Start the audio stream and begin listening."""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None,  # We'll use blocking mode
            )

            self.is_running = True
            print(f"\n🎤 Listening for wake word: '{self.model_name}'")
            print(f"   Threshold: {self.threshold}")
            print(f"   Press Ctrl+C to stop\n")
            print("-" * 50)

            # Main listening loop
            while self.is_running:
                # Read audio chunk
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Run inference
                prediction = self.model.predict(audio_array)

                # Check detection for our model
                for mdl_name, score in prediction.items():
                    if score >= self.threshold:
                        self.handle_detection(mdl_name, score)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping listener...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.stop_listening()

    def handle_detection(self, model_name, score):
        """Handle wake word detection."""
        self.detection_count += 1
        timestamp = time.strftime("%H:%M:%S")

        # Print detection with formatting
        print(f"🔔 [{timestamp}] WAKE WORD DETECTED! #{self.detection_count}")
        print(f"   Model: {model_name}")
        print(f"   Confidence: {score:.3f}")
        print("-" * 50)

        # Optional: Add a small cooldown to prevent multiple rapid detections
        time.sleep(0.5)

        # Here you can add any additional logic for what happens after detection
        # For example:
        # - Play a sound
        # - Trigger another action
        # - Start recording for speech recognition

    def stop_listening(self):
        """Clean up audio resources."""
        self.is_running = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()

        print(f"\n📊 Total detections: {self.detection_count}")
        print("Goodbye! 👋")


def run_wake_word_listener(args):
    base_model_path = Path("models/wake_word_models")
    model_path = base_model_path / args.model / f"{args.model}.onnx"

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at: {model_path}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)

    listener = WakeWordListener(model_path=str(model_path), threshold=args.threshold, chunk_size=args.chunk_size)
    listener.start_listening()
