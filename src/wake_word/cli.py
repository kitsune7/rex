import argparse
import sys
from pathlib import Path

from .wake_word_listener import WakeWordListener


def main():
    parser = argparse.ArgumentParser(description="Wake Word Tester - Test wake word detection")

    parser.add_argument(
        "--model", default="hey_rex", help="The specific wake word to test in models/wake_word_models (i.e. hey_rex)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0, default: 0.5)")

    args = parser.parse_args()

    base_model_path = Path("models/wake_word_models")
    model_path = base_model_path / args.model / f"{args.model}.onnx"

    if not model_path.exists():
        print(f"❌ Error: Model file not found at: {model_path}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print("❌ Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)

    listener = WakeWordListener(model_path=str(model_path), threshold=args.threshold)

    print("\n🎤 Listening for wake word...")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
