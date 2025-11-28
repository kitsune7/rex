import re
import sys
import threading
from pathlib import Path

from langchain_core.messages import AIMessage

from agent import run_voice_agent, confirm_tool_call, PendingConfirmation
from agent.tools import get_timer_manager
from tts import load_voice, speak_text, InterruptibleSpeaker
from stt import Transcriber
from wake_word import WakeWordListener
from wake_word.audio_feedback import ThinkingTone
from .audio_setup import suppress_portaudio_warnings
from .reminder_scheduler import ReminderScheduler, ReminderDelivery

suppress_portaudio_warnings()

# Timeout in seconds for waiting for follow-up responses
FOLLOW_UP_TIMEOUT = 5.0

# Timeout for confirmation responses
CONFIRMATION_TIMEOUT = 10.0

# Phrases that end the conversation after interruption
STOP_PHRASES = ("stop", "nevermind", "never mind", "cancel", "forget it")

# Confirmation phrases
CONFIRM_PHRASES = ("yes", "yeah", "yep", "sure", "okay", "ok", "confirm", "do it", "go ahead", "proceed")
REJECT_PHRASES = ("no", "nope", "cancel", "nevermind", "never mind", "don't", "stop")


def _handle_special_command(normalized: str, history: list | None, timer_manager) -> tuple[bool, list | None]:
    """
    Handle special voice commands (stop, nevermind, etc).

    Returns:
        (was_handled, updated_history) - if was_handled is True, skip agent processing
    """
    # Timer stop command (works anytime)
    if normalized in ("stop", "stop the timer"):
        if timer_manager.stop_any_ringing():
            print("🔕 Timer alarm stopped.")
        return True, None

    # Stop phrases end conversation (only during follow-up)
    if history is not None and normalized in STOP_PHRASES:
        return True, None

    return False, history


def _is_confirmation(text: str) -> bool:
    """Check if text is a confirmation."""
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in CONFIRM_PHRASES)


def _is_rejection(text: str) -> bool:
    """Check if text is a rejection."""
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in REJECT_PHRASES)


def _parse_snooze_duration(text: str) -> int | None:
    """
    Parse a snooze duration from text like "remind me in 30 minutes".

    Returns:
        Number of minutes, or None if not a snooze request.
    """
    # Patterns for snooze requests
    patterns = [
        r"(?:remind|tell|ask)\s+me\s+(?:again\s+)?in\s+(\d+)\s*(?:minute|min)",
        r"(?:snooze|delay|postpone)(?:\s+(?:it|for))?\s+(\d+)\s*(?:minute|min)",
        r"(\d+)\s*(?:minute|min)(?:\s+(?:later|from now))?",
    ]

    normalized = text.strip().lower()

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))

    return None


def _is_pure_rejection(text: str) -> bool:
    """
    Check if text is a pure rejection (just "no", "cancel", etc.) vs a modification request.

    A pure rejection exactly matches a REJECT_PHRASE. Anything else with additional
    instructions (e.g., "no, make it 11pm") is a modification request.
    """
    normalized = text.strip().lower()
    return normalized in REJECT_PHRASES


def _handle_confirmation(
    pending: PendingConfirmation,
    listener: WakeWordListener,
    transcriber: Transcriber,
    speaker: InterruptibleSpeaker,
    voice,
) -> tuple[str, list, str | None]:
    """
    Handle human-in-the-loop confirmation for a tool call.

    Args:
        pending: The pending confirmation details
        listener: Wake word listener for audio capture
        transcriber: Speech transcriber
        speaker: Interruptible speaker
        voice: TTS voice

    Returns:
        Tuple of (response_text, updated_history, modification_text) where:
        - response_text is the agent's response after confirmation/rejection
        - modification_text is None unless user requested a modification
    """
    # Speak the confirmation prompt
    print(f"\n🤖 Rex: {pending.confirmation_prompt}\n")
    speaker.speak_interruptibly(pending.confirmation_prompt)

    # Listen for response
    print("🎤 Listening for confirmation...")
    audio = listener.listen_for_speech(timeout=CONFIRMATION_TIMEOUT)

    if audio is None:
        # No response - treat as rejection
        print("⏱️ No confirmation received, cancelling.")
        response, history = confirm_tool_call(pending, confirmed=False)
        return response, history, None

    transcription = transcriber.transcribe(audio)
    if not transcription:
        print("⏱️ Could not understand response, cancelling.")
        response, history = confirm_tool_call(pending, confirmed=False)
        return response, history, None

    print(f"\n💬 You said: {transcription}\n")

    if _is_confirmation(transcription):
        print("✅ Confirmed!")
        response, history = confirm_tool_call(pending, confirmed=True)
        return response, history, None
    elif _is_pure_rejection(transcription):
        print("❌ Cancelled.")
        response, history = confirm_tool_call(pending, confirmed=False)
        return response, history, None
    else:
        # User provided modification instructions
        print("🔄 Modification requested.")
        response, history = confirm_tool_call(pending, confirmed=False, user_response=transcription)
        return response, history, transcription


def _handle_reminder_delivery(
    delivery: ReminderDelivery,
    scheduler: ReminderScheduler,
    listener: WakeWordListener,
    transcriber: Transcriber,
    speaker: InterruptibleSpeaker,
    voice,
    timer_manager,
) -> bool:
    """
    Handle proactive reminder delivery.

    Args:
        delivery: The reminder delivery details
        scheduler: Reminder scheduler
        listener: Wake word listener
        transcriber: Speech transcriber
        speaker: Interruptible speaker
        voice: TTS voice
        timer_manager: Timer manager (for muting)

    Returns:
        True if reminder was handled (cleared or snoozed), False to retry later
    """
    reminder = delivery.reminder

    # Mute any timer alarms during reminder delivery
    timer_manager.mute()

    # Play ding sound
    print("🔔 Reminder!")
    scheduler.play_ding()

    # Speak the reminder
    reminder_text = f"You have a reminder: {reminder.message}. Would you like to clear this reminder?"
    print(f"\n🤖 Rex: {reminder_text}\n")
    speaker.speak_interruptibly(reminder_text)

    # Listen for response
    print("🎤 Listening for response...")
    audio = listener.listen_for_speech(timeout=FOLLOW_UP_TIMEOUT)

    timer_manager.unmute()

    if audio is None:
        # No response - schedule retry
        print("⏱️ No response, will retry later.")
        scheduler.schedule_retry(reminder.id)
        return False

    transcription = transcriber.transcribe(audio)
    if not transcription:
        print("⏱️ Could not understand response, will retry later.")
        scheduler.schedule_retry(reminder.id)
        return False

    print(f"\n💬 You said: {transcription}\n")
    normalized = transcription.strip().lower()

    # Check for snooze request
    snooze_minutes = _parse_snooze_duration(transcription)
    if snooze_minutes:
        scheduler.snooze_reminder(reminder.id, snooze_minutes)
        response = f"Okay, I'll remind you again in {snooze_minutes} minutes."
        print(f"\n🤖 Rex: {response}\n")
        speak_text(response, voice)
        return True

    # Check for clear/acknowledge
    if _is_confirmation(transcription) or "clear" in normalized or "done" in normalized or "got it" in normalized:
        scheduler.mark_delivered(reminder.id)
        response = "Reminder cleared."
        print(f"\n🤖 Rex: {response}\n")
        speak_text(response, voice)
        return True

    # Check for explicit rejection/snooze request without time
    if _is_rejection(transcription) or "later" in normalized or "not now" in normalized:
        scheduler.schedule_retry(reminder.id)
        from .settings import get_settings

        retry_mins = get_settings().reminders.retry_minutes
        response = f"Okay, I'll remind you again in {retry_mins} minutes."
        print(f"\n🤖 Rex: {response}\n")
        speak_text(response, voice)
        return True

    # Unclear response - retry later
    print("⏱️ Unclear response, will retry later.")
    scheduler.schedule_retry(reminder.id)
    return False


def _process_turn(
    transcription: str,
    history: list | None,
    speaker: InterruptibleSpeaker,
    voice,
    listener: WakeWordListener,
    transcriber: Transcriber,
    thread_id: str | None = None,
) -> tuple[list | None, bool, str | None]:
    """
    Process one conversation turn: run agent, speak response, handle interruption.

    Returns:
        (updated_history, should_continue_conversation, thread_id)
    """
    try:
        print("🤔 Thinking...")
        with ThinkingTone():
            result, history, thread_id = run_voice_agent(transcription, history, thread_id)

        # Handle confirmation loop (may iterate if user requests modifications)
        while isinstance(result, PendingConfirmation):
            confirmation_result, history, modification = _handle_confirmation(
                result, listener, transcriber, speaker, voice
            )

            if modification:
                # User requested a modification - re-run the agent with their request
                # This will process the modification and potentially propose new parameters
                print("🤔 Processing modification...")
                with ThinkingTone():
                    result, history, thread_id = run_voice_agent(modification, history, thread_id)
                # Loop back to check if we got a new PendingConfirmation
                continue
            else:
                # Confirmed or rejected (no modification) - speak response and end
                response = confirmation_result
                print(f"\n🤖 Rex: {response}\n")
                speaker.speak_interruptibly(response)
                return history, False, thread_id

        response = result
        print(f"\n🤖 Rex: {response}\n")
        was_interrupted = speaker.speak_interruptibly(response)

        if was_interrupted:
            print("🛑 Interrupted!")
            if history and isinstance(history[-1], AIMessage):
                history[-1] = AIMessage(content=history[-1].content + " [interrupted]")
            return history, True, thread_id  # Continue conversation

        # End conversation if Rex didn't ask a question
        should_continue = response.strip().endswith("?")
        return history, should_continue, thread_id

    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback

        traceback.print_exc()
        speak_text("Sorry, I encountered an error processing your request.", voice)
        return None, False, None


def _end_conversation(history, timer_manager):
    """End conversation and return to wake word listening mode."""
    if history is not None:
        timer_manager.unmute()
    print("🎤 Listening for 'Hey Rex'...")
    return None, None  # Return (history, thread_id)


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

    # Initialize reminder scheduler
    reminder_interrupt = threading.Event()

    def on_reminder_due(delivery: ReminderDelivery):
        """Callback when a reminder is due - signal the main loop."""
        reminder_interrupt.set()
        # Just set the interrupted flag - don't close the stream from another thread
        # as that's not thread-safe and can cause hangs
        listener._interrupted = True

    scheduler = ReminderScheduler(on_reminder_due=on_reminder_due)
    scheduler.start()

    print("\n✅ Rex is ready!")
    print("🎤 Listening for 'hey rex'...")
    print("   Press Ctrl+C to exit\n")

    # Conversation state: None = waiting for wake word, list = in conversation
    history = None
    thread_id = None

    try:
        while True:
            # Check for pending reminder delivery (proactive)
            if reminder_interrupt.is_set():
                reminder_interrupt.clear()
                # Reset listener state so it can be used for reminder delivery
                listener._interrupted = False
                delivery = scheduler.get_pending_delivery()
                if delivery:
                    # End any current conversation
                    if history is not None:
                        print("📢 Interrupting for reminder...")
                        history, thread_id = _end_conversation(history, timer_manager)

                    _handle_reminder_delivery(
                        delivery,
                        scheduler,
                        listener,
                        transcriber,
                        speaker,
                        voice,
                        timer_manager,
                    )
                    print("🎤 Listening for 'Hey Rex'...")
                    continue

            # --- Get audio ---
            if history is None:
                # Wait for wake word, then capture speech
                audio = listener.wait_for_wake_word_and_speech(
                    on_wake_word=timer_manager.mute,
                )

                # Check if we were interrupted for a reminder
                if reminder_interrupt.is_set() and audio is None:
                    # Reset listener state for next iteration
                    listener._interrupted = False
                    continue

                if audio is None:
                    # Listener was stopped (shutdown or other interrupt)
                    if listener._interrupted:
                        break
                    continue
            else:
                # In conversation - listen for follow-up (with timeout)
                print("🎤 Listening for response...")
                audio = listener.listen_for_speech(timeout=FOLLOW_UP_TIMEOUT)
                if audio is None:
                    print("⏱️ No response received, ending conversation.")
                    history, thread_id = _end_conversation(history, timer_manager)
                    continue

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
            was_handled, history = _handle_special_command(normalized, history, timer_manager)
            if was_handled:
                history, thread_id = _end_conversation(history, timer_manager)
                continue

            # --- Process with agent ---
            history, should_continue, thread_id = _process_turn(
                transcription, history, speaker, voice, listener, transcriber, thread_id
            )
            if not should_continue:
                history, thread_id = _end_conversation(history, timer_manager)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        scheduler.stop()
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
