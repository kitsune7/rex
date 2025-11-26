import sys

from pathlib import Path

from agent import run_voice_agent
from tts import load_voice, speak_text
from stt import Transcriber
from wake_word import WakeWordListener


def main():
    """Main voice assistant loop with wake word detection and rolling buffer."""
    print("🚀 Starting Rex Voice Assistant...")

    model_path = Path("models/wake_word_models/hey_rex/hey_rex.onnx")
    if not model_path.exists():
        print(f"❌ Error: Wake word model not found at {model_path}")
        return 1

    # Initialize components
    listener = WakeWordListener(model_path=str(model_path), threshold=0.5)
    voice = load_voice()
    transcriber = Transcriber()

    print("\n✅ Rex is ready!")
    print("🎤 Listening for 'hey rex'...")
    print("   Press Ctrl+C to exit\n")

    try:
        while True:
            audio = listener.wait_for_wake_word_and_speech()

            if audio is None or listener._interrupted:
                break

            transcription = transcriber.transcribe(audio)

            if not transcription:
                print("🎤 Listening for 'hey rex'...")
                continue

            print(f"\n💬 You said: {transcription}\n")

            if transcription.strip().lower() == "stop":
                print("🎤 Listening for 'hey rex'...")
                continue

            try:
                response = run_voice_agent(transcription)
                print(f"\n🤖 Rex: {response}\n")
                speak_text(response, voice)
            except Exception as e:
                print(f"❌ Agent error: {e}")
                speak_text("Sorry, I encountered an error processing your request.", voice)

            print("🎤 Listening for 'hey rex'...")

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
