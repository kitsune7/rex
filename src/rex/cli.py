import argparse
import sys
import numpy as np
import sounddevice as sd
from pathlib import Path

from agent import run_voice_agent, run_text_agent
from tts import load_voice, speak_text
from stt import get_transcription
from wake_word.wake_word_listener import WakeWordListener


def play_beep():
    """Play a simple beep to indicate wake word detection."""
    duration = 0.15  # seconds
    frequency = 800  # Hz
    sample_rate = 16000

    t = np.linspace(0, duration, int(sample_rate * duration))
    beep = 0.3 * np.sin(2 * np.pi * frequency * t)

    sd.play(beep, sample_rate)
    sd.wait()


def main():
    parser = argparse.ArgumentParser(description="Rex AI Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("start", help="Start Rex AI Assistant")

    query_parser = subparsers.add_parser("query", help="Query Rex directly")
    query_parser.add_argument("text", help="Text to pass to Rex")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start()
    elif args.command == "query":
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


def cmd_start():
    """Main voice assistant loop with wake word detection."""
    print("🚀 Starting Rex Voice Assistant...")

    model_path = Path("models/wake_word_models/hey_rex/hey_rex.onnx")
    if not model_path.exists():
        print(f"❌ Error: Wake word model not found at {model_path}")
        print("Please ensure the wake word model is trained and saved.")
        return 1

    wake_word_listener = WakeWordListener(model_path=str(model_path), threshold=0.5, cooldown_period=2.0)

    print("Loading voice model...")
    voice = load_voice()

    print("\n✅ Rex is ready!")
    print("🎤 Listening for 'hey rex'...")
    print("   Press Ctrl+C to exit\n")

    try:
        while True:
            # Wait for wake word
            detected = wake_word_listener.wait_for_detection()
            if not detected:  # False means it was interrupted
                break
            play_beep()

            # Record and transcribe speech (with retry logic)
            error_count = 0
            max_retries = 2
            transcription = None

            while error_count < max_retries:
                try:
                    transcription = get_transcription(debug=False)

                    # Empty transcription - go back to wake word
                    if not transcription or transcription == "No speech detected or transcription failed.":
                        transcription = None
                        break

                    # Success - reset error count and continue
                    error_count = 0
                    break

                except Exception as e:
                    error_count += 1
                    print(f"❌ Transcription error: {e}")

                    if error_count < max_retries:
                        # Speak error and retry
                        speak_text("Sorry, I didn't catch that.", voice)
                    else:
                        # Max retries reached - go back to wake word
                        transcription = None
                        break

            # If no valid transcription, go back to wake word listening
            if not transcription:
                print("🎤 Listening for 'hey rex'...")
                continue

            print(f"\n💬 You said: {transcription}\n")

            # Check for stop command
            if transcription.strip().lower() == "stop":
                print("🎤 Listening for 'hey rex'...")
                continue

            # Run agent with transcription
            try:
                response = run_voice_agent(transcription)
                print(f"\n🤖 Rex: {response}\n")

                # Speak agent's response via TTS
                speak_text(response, voice)

            except Exception as e:
                print(f"❌ Agent error: {e}")
                error_message = "Sorry, I encountered an error processing your request."
                speak_text(error_message, voice)

            # Go back to wake word listening
            print("🎤 Listening for 'hey rex'...")

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")

    finally:
        wake_word_listener.stop_listening()

    return 0


def cmd_query(args):
    response = run_text_agent(args.text)
    print(response)


if __name__ == "__main__":
    sys.exit(main())
