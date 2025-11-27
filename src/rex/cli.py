import sys
from pathlib import Path

from langchain_core.messages import AIMessage

from agent import run_voice_agent
from agent.tools import get_timer_manager
from tts import load_voice, speak_text, InterruptibleSpeaker
from stt import Transcriber
from wake_word import WakeWordListener
from wake_word.audio_feedback import ThinkingTone

# Timeout in seconds for waiting for follow-up responses
FOLLOW_UP_TIMEOUT = 5.0

# Phrases that end the conversation after interruption
STOP_PHRASES = ("stop", "nevermind", "never mind", "cancel", "forget it")


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
    speaker = InterruptibleSpeaker(voice, model_path=str(model_path))
    timer_manager = get_timer_manager()

    print("\n✅ Rex is ready!")
    print("🎤 Listening for 'hey rex'...")
    print("   Press Ctrl+C to exit\n")

    # Conversation state: None = waiting for wake word, list = in conversation
    history = None

    try:
        while True:
            # --- Get audio ---
            if history is None:
                # Wait for wake word, then capture speech
                audio = listener.wait_for_wake_word_and_speech(on_wake_word=timer_manager.mute)
            else:
                # In conversation - listen for follow-up (with timeout)
                print("🎤 Listening for response...")
                audio = listener.listen_for_speech(timeout=FOLLOW_UP_TIMEOUT)
                if audio is None:
                    print("⏱️ No response received, ending conversation.")
                    history = _end_conversation(history, timer_manager)
                    continue

            # Check for shutdown
            if audio is None or listener._interrupted:
                break

            # --- Transcribe ---
            transcription = transcriber.transcribe(audio, strip_wake_word=(history is None))
            timer_manager.unmute()

            if not transcription:
                if history is None:
                    print("🎤 Listening for 'hey rex'...")
                continue

            print(f"\n💬 You said: {transcription}\n")

            # --- Handle special commands ---
            normalized = transcription.strip().lower()

            # Timer stop command (works anytime)
            if normalized in ("stop", "stop the timer"):
                if timer_manager.stop_any_ringing():
                    print("🔕 Timer alarm stopped.")
                history = _end_conversation(history, timer_manager)
                continue

            # Stop phrases end conversation (only during follow-up)
            if history is not None and normalized in STOP_PHRASES:
                history = _end_conversation(history, timer_manager)
                continue

            # --- Process with agent ---
            try:
                print("🤔 Thinking...")
                with ThinkingTone():
                    response, history = run_voice_agent(transcription, history)

                print(f"\n🤖 Rex: {response}\n")
                was_interrupted = speaker.speak_interruptibly(response)

                if was_interrupted:
                    print("🛑 Interrupted!")
                    # Mark the response as interrupted in history
                    if history and isinstance(history[-1], AIMessage):
                        history[-1] = AIMessage(content=history[-1].content + " [interrupted]")
                    # Loop continues - will listen for follow-up
                    continue

                # If Rex didn't ask a question, end conversation
                if not response.strip().endswith("?"):
                    history = _end_conversation(history, timer_manager)

            except Exception as e:
                print(f"❌ Agent error: {e}")
                speak_text("Sorry, I encountered an error processing your request.", voice)
                history = _end_conversation(history, timer_manager)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        listener.stop()

    return 0


def _end_conversation(history, timer_manager):
    """End conversation and return to wake word listening mode."""
    if history is not None:
        timer_manager.unmute()
    print("🎤 Listening for 'Hey Rex'...")
    return None


if __name__ == "__main__":
    sys.exit(main())
