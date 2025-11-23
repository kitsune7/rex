import argparse
import sys

from .wake_word_recorder import WakeWordRecorder
from .wake_word_listener import run_wake_word_listener


def main():
    parser = argparse.ArgumentParser(description="Wake Word Recorder - Train wake word detection")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    test_pretrained_parser = subparsers.add_parser(
        "test-pretrained", help="Manually test the pretained wake word effectiveness"
    )
    test_pretrained_parser.add_argument("model_path", help="Path to the .onnx model file")
    test_pretrained_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0, default: 0.5)"
    )
    test_pretrained_parser.add_argument(
        "--chunk-size", type=int, default=1280, help="Audio chunk size in samples (default: 1280)"
    )

    # Record positive samples
    positive_parser = subparsers.add_parser("record-positive", help="Record positive samples (with the wake word)")
    positive_parser.add_argument("wake_word", choices=["hey_rex", "rex", "captain_rex"], help="The wake word to record")
    positive_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to record (default: 20)",
    )
    positive_parser.add_argument(
        "-o",
        "--output-dir",
        default="recordings",
        help="Output directory for recordings (default: recordings)",
    )

    # Record negative samples
    negative_parser = subparsers.add_parser("record-negative", help="Record negative samples (without the wake word)")
    negative_parser.add_argument(
        "wake_word",
        choices=["hey_rex", "rex", "captain_rex"],
        help="The wake word category for negative samples",
    )
    negative_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to record (default: 20)",
    )
    negative_parser.add_argument(
        "-o",
        "--output-dir",
        default="recordings",
        help="Output directory for recordings (default: recordings)",
    )

    # Test wake word detection
    test_parser = subparsers.add_parser("test", help="Test wake word detection effectiveness")
    test_parser.add_argument("wake_word", choices=["hey_rex", "rex", "captain_rex"], help="The wake word to test")
    test_parser.add_argument(
        "-o",
        "--output-dir",
        default="recordings",
        help="Directory containing training recordings (default: recordings)",
    )

    args = parser.parse_args()

    if args.command == "record-positive":
        return cmd_record_positive(args)
    elif args.command == "record-negative":
        return cmd_record_negative(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "test-pretrained":
        return run_wake_word_listener(args)
    else:
        parser.print_help()
        return 0


def cmd_record_positive(args):
    recorder = WakeWordRecorder(output_dir=args.output_dir, sample_rate=16000)
    recorder.batch_record(wake_word=args.wake_word, num_samples=args.num_samples, sample_type="positive")
    return 0


def cmd_record_negative(args):
    recorder = WakeWordRecorder(output_dir=args.output_dir, sample_rate=16000)
    recorder.batch_record(wake_word=args.wake_word, num_samples=args.num_samples, sample_type="negative")
    return 0


def cmd_test(args):
    print(f'Testing wake word detection for "{args.wake_word}"...')
    print("\nThis feature is coming soon!")
    print("It will:")
    print("- Load your trained recordings")
    print("- Test detection accuracy on positive samples")
    print("- Test false positive rate on negative samples")
    print("- Provide a summary of effectiveness metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
