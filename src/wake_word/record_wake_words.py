import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import sys
import time
from datetime import datetime
import queue
import threading


class WakeWordRecorder:
    def __init__(self, output_dir="recordings", sample_rate=16000):
        self.output_dir = output_dir
        self.sample_rate = sample_rate

        self.recording = False
        self.audio_queue = queue.Queue()

        # Each wake word needs its own directory with positive and negative sub-directories
        self.wake_words = ["hey_rex", "rex", "captain_rex"]
        for word in self.wake_words:
            os.makedirs(os.path.join(output_dir, word, "positive"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, word, "negative"), exist_ok=True)

    def audio_callback(self, in_data, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(in_data.copy())

    def record_sample(self, duration=2.0, wake_word="rex", sample_type="positive"):
        """Record a single sample"""
        print(f"\nRecording '{wake_word}' ({sample_type}) in 3 seconds...")
        time.sleep(1)
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(0.5)
        print("SPEAK NOW!")

        while not self.audio_queue.empty():
            self.audio_queue.get()

        self.recording = True
        audio_data = []

        start_time = time.time()
        while (time.time() - start_time) < duration:
            try:
                data = self.audio_queue.get(timeout=0.1)
                audio_data.append(data)
            except queue.Empty:
                continue

        self.recording = False
        print("Recording complete!")

        self.save_audio(audio_data, wake_word)

    def save_audio(self, audio_data, wake_word):
        if audio_data:
            audio_np = np.concatenate(audio_data, axis=0)

            # i.e. 2025-09-07_05-03-08pm (Sep. 7, 2025 at 5:03:08pm)
            timestamp = datetime.now().strftime("%Y-%m%d_%I-%M-%S%p")

            filename = f"{wake_word}"

    def batch_record(self, wake_word, num_samples=20, sample_type="positive"):
        pass


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
