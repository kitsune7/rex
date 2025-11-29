"""
Delivering reminder state handler.

Handles proactive reminder delivery when a reminder becomes due.
"""

import re
from typing import TYPE_CHECKING

from core.state_machine import ConversationState, StateHandler, StateResult
from tts import speak_text

if TYPE_CHECKING:
    from core.context import AppContext
    from rex.reminder_scheduler import ReminderDelivery, ReminderScheduler
    from stt import Transcriber
    from tts import InterruptibleSpeaker
    from wake_word import WakeWordListener

# Timeout for reminder responses
REMINDER_RESPONSE_TIMEOUT = 5.0

# Confirmation phrases for clearing reminders
CONFIRM_PHRASES = ("yes", "yeah", "yep", "sure", "okay", "ok", "clear", "done", "got it")
REJECT_PHRASES = ("no", "nope", "cancel", "nevermind", "never mind", "don't", "stop")


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


class DeliveringReminderHandler(StateHandler):
    """
    Handler for the DELIVERING_REMINDER state.

    Proactively delivers a due reminder to the user and handles
    their response (acknowledge, snooze, or ignore).
    """

    def __init__(
        self,
        listener: "WakeWordListener",
        transcriber: "Transcriber",
        speaker: "InterruptibleSpeaker",
        voice,
        scheduler: "ReminderScheduler",
    ):
        self._listener = listener
        self._transcriber = transcriber
        self._speaker = speaker
        self._voice = voice
        self._scheduler = scheduler
        self._delivery: "ReminderDelivery | None" = None

    @property
    def state(self) -> ConversationState:
        return ConversationState.DELIVERING_REMINDER

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Store the reminder delivery info."""
        if data:
            self._delivery = data.get("delivery")
        else:
            self._delivery = None

        # Mute timer alarms during reminder delivery
        if ctx.timer_manager:
            ctx.timer_manager.mute()

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Deliver the reminder and handle user response.

        Returns:
            WAITING_FOR_WAKE_WORD after handling (or scheduling retry)
        """
        if self._delivery is None:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        reminder = self._delivery.reminder

        # Play ding sound
        print("🔔 Reminder!")
        self._scheduler.play_ding()

        # Speak the reminder
        reminder_text = f"You have a reminder: {reminder.message}. Would you like to clear this reminder?"
        print(f"\n🤖 Rex: {reminder_text}\n")
        self._speaker.speak_interruptibly(reminder_text)

        # Listen for response
        print("🎤 Listening for response...")
        audio = self._listener.listen_for_speech(timeout=REMINDER_RESPONSE_TIMEOUT)

        if audio is None:
            # No response - schedule retry
            print("⏱️ No response, will retry later.")
            self._scheduler.schedule_retry(reminder.id)
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        transcription = self._transcriber.transcribe(audio)
        if not transcription:
            print("⏱️ Could not understand response, will retry later.")
            self._scheduler.schedule_retry(reminder.id)
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        print(f"\n💬 You said: {transcription}\n")
        normalized = transcription.strip().lower()

        # Check for snooze request
        snooze_minutes = _parse_snooze_duration(transcription)
        if snooze_minutes:
            self._scheduler.snooze_reminder(reminder.id, snooze_minutes)
            response = f"Okay, I'll remind you again in {snooze_minutes} minutes."
            print(f"\n🤖 Rex: {response}\n")
            speak_text(response, self._voice)
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Check for clear/acknowledge
        if _is_confirmation(transcription) or "clear" in normalized or "done" in normalized or "got it" in normalized:
            self._scheduler.mark_delivered(reminder.id)
            response = "Reminder cleared."
            print(f"\n🤖 Rex: {response}\n")
            speak_text(response, self._voice)
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Check for explicit rejection/snooze without time
        if _is_rejection(transcription) or "later" in normalized or "not now" in normalized:
            self._scheduler.schedule_retry(reminder.id)
            retry_mins = ctx.settings.reminders.retry_minutes if ctx.settings else 10
            response = f"Okay, I'll remind you again in {retry_mins} minutes."
            print(f"\n🤖 Rex: {response}\n")
            speak_text(response, self._voice)
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Unclear response - retry later
        print("⏱️ Unclear response, will retry later.")
        self._scheduler.schedule_retry(reminder.id)
        return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

    def exit(self, ctx: "AppContext") -> None:
        """Unmute timer and clear delivery info."""
        if ctx.timer_manager:
            ctx.timer_manager.unmute()
        self._delivery = None
