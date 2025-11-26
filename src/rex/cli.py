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


def _end_conversation(history, timer_manager):
    """End conversation and return to wake word listening mode."""
    if history is not None:
        timer_manager.unmute()
    print("🎤 Listening for 'Hey Rex'...")
    return None


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
                    history = _end_conversation(history, timer_manager)
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
                history = _end_conversation(history, timer_manager)
                continue

            try:
                # Play thinking tone while waiting for LLM
                print("🤔 Thinking...")
                thinking_tone = ThinkingTone()
                thinking_tone.start()
                try:
                    response, history = run_voice_agent(transcription, history)
                finally:
                    thinking_tone.stop()

                print(f"\n🤖 Rex: {response}\n")
                was_interrupted = speaker.speak_interruptibly(response)

                if was_interrupted:
                    # User interrupted with "Hey Rex" - modify history and continue
                    print("🛑 Interrupted!")
                    if history and isinstance(history[-1], AIMessage):
                        # Mark the response as interrupted in history
                        history[-1] = AIMessage(content=history[-1].content + " [interrupted]")

                    # Listen for the user's follow-up command
                    print("🎤 Listening...")
                    audio = listener.listen_for_speech(timeout=FOLLOW_UP_TIMEOUT)

                    if audio is None:
                        # User didn't say anything after interrupting
                        print("⏱️ No follow-up received, ending conversation.")
                        history = _end_conversation(history, timer_manager)
                        continue

                    # Transcribe the follow-up (no wake word to strip)
                    follow_up = transcriber.transcribe(audio, strip_wake_word=False)

                    if not follow_up:
                        history = _end_conversation(history, timer_manager)
                        continue

                    print(f"\n💬 You said: {follow_up}\n")

                    # Check if user wants to stop entirely
                    if follow_up.strip().lower() in STOP_PHRASES:
                        history = _end_conversation(history, timer_manager)
                        continue

                    # Process the follow-up as a new message in the conversation
                    print("🤔 Thinking...")
                    thinking_tone = ThinkingTone()
                    thinking_tone.start()
                    try:
                        response, history = run_voice_agent(follow_up, history)
                    finally:
                        thinking_tone.stop()

                    print(f"\n🤖 Rex: {response}\n")
                    # Speak the response (could be interrupted again)
                    was_interrupted = speaker.speak_interruptibly(response)

                    if was_interrupted:
                        # Interrupted again - just end conversation for simplicity
                        if history and isinstance(history[-1], AIMessage):
                            history[-1] = AIMessage(content=history[-1].content + " [interrupted]")
                        history = _end_conversation(history, timer_manager)
                    elif not response.strip().endswith("?"):
                        history = _end_conversation(history, timer_manager)

                elif not response.strip().endswith("?"):
                    # Rex didn't ask a question - conversation is done
                    history = _end_conversation(history, timer_manager)
                # If Rex asked a question, keep history and loop will listen for follow-up

            except Exception as e:
                print(f"❌ Agent error: {e}")
                speak_text("Sorry, I encountered an error processing your request.", voice)
                history = _end_conversation(history, timer_manager)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
