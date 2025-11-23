import argparse
import sys

from .wake_word_listener import run_wake_word_listener


def main():
    parser = argparse.ArgumentParser(description="Wake Word Tester - Test wake word detection")

    parser.add_argument(
        "--model", default="hey_rex", help="The specific wake word to test in models/wake_word_models (i.e. hey_rex)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--chunk-size", type=int, default=1280, help="Audio chunk size in samples (default: 1280)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown period between detections in seconds (default: 2.0)")

    args = parser.parse_args()

    run_wake_word_listener(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
