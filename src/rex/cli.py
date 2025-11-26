import sys

from pathlib import Path

from agent import run_voice_agent
from agent.tools import get_timer_manager
from tts import load_voice, speak_text
from stt import Transcriber
from wake_word import WakeWordListener

# Timeout in seconds for waiting for follow-up responses
FOLLOW_UP_TIMEOUT = 5.0


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

    timer_manager = get_timer_manager()

    # Conversation state
    history = None

    try:
        while True:
            if history is None:
                # Normal wake word flow - wait for "hey rex"
                audio = listener.wait_for_wake_word_and_speech(on_wake_word=timer_manager.mute)
            else:
                # Follow-up flow - listen for speech without wake word
                print("🎤 Listening for response...")
                audio = listener.listen_for_speech(timeout=FOLLOW_UP_TIMEOUT)

                if audio is None:
                    # Timeout - user didn't respond, end conversation
                    print("⏱️ No response received, ending conversation.")
                    history = None
                    timer_manager.unmute()
                    print("🎤 Listening for 'hey rex'...")
                    continue

            if audio is None or listener._interrupted:
                break

            # Transcribe - strip wake word only for initial messages
            transcription = transcriber.transcribe(audio, strip_wake_word=(history is None))

            # Unmute timer after transcription is complete
            timer_manager.unmute()

            if not transcription:
                if history is None:
                    print("🎤 Listening for 'hey rex'...")
                continue

            print(f"\n💬 You said: {transcription}\n")

            # Handle timer stop command (works in any conversation state)
            if transcription.strip().lower() in ("stop", "stop the timer"):
                if timer_manager.stop_any_ringing():
                    print("🔕 Timer alarm stopped.")
                history = None  # End conversation after stop command
                print("🎤 Listening for 'hey rex'...")
                continue

            try:
                response, history = run_voice_agent(transcription, history)
                print(f"\n🤖 Rex: {response}\n")
                speak_text(response, voice)

                # Check if Rex asked a follow-up question
                if not response.strip().endswith("?"):
                    # Rex didn't ask a question - conversation is done
                    history = None
                    print("🎤 Listening for 'hey rex'...")
                # If Rex asked a question, keep history and loop will listen for follow-up

            except Exception as e:
                print(f"❌ Agent error: {e}")
                speak_text("Sorry, I encountered an error processing your request.", voice)
                history = None  # Reset conversation on error
                print("🎤 Listening for 'hey rex'...")

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
