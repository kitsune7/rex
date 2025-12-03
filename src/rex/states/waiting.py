"""
Waiting for wake word state handler.

This is the idle state where Rex listens for the wake word to start a conversation.
"""

from typing import TYPE_CHECKING

from core.state_machine import ConversationState, StateHandler, StateResult

if TYPE_CHECKING:
    from core.context import AppContext
    from rex.reminder_scheduler import ReminderScheduler
    from wake_word import WakeWordListener


class WaitingForWakeWordHandler(StateHandler):
    """
    Handler for the WAITING_FOR_WAKE_WORD state.

    Waits for the user to say the wake word or for a reminder to become due.
    """

    def __init__(
        self,
        listener: "WakeWordListener",
        scheduler: "ReminderScheduler",
    ):
        self._listener = listener
        self._scheduler = scheduler

    @property
    def state(self) -> ConversationState:
        return ConversationState.WAITING_FOR_WAKE_WORD

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Reset conversation state and print listening message."""
        ctx.reset_conversation()

        # Unmute timer sounds when returning to idle
        if ctx.timer_manager:
            ctx.timer_manager.unmute()

        print(f"🎤 Listening for '{ctx.settings.wake_word.display_name}'...")

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Wait for wake word detection or reminder interrupt.

        Returns:
            LISTENING if wake word detected
            DELIVERING_REMINDER if a reminder is due
            SHUTTING_DOWN if interrupted
        """
        # Check for pending reminder first
        if self._scheduler.has_pending_delivery():
            delivery = self._scheduler.get_pending_delivery()
            if delivery:
                return StateResult(
                    next_state=ConversationState.DELIVERING_REMINDER,
                    data={"delivery": delivery},
                )

        # Wait for wake word with callback to mute timer
        def on_wake_word():
            if ctx.timer_manager:
                ctx.timer_manager.mute()

        audio = self._listener.wait_for_wake_word_and_speech(on_wake_word=on_wake_word)

        # Check if we were interrupted for a reminder (using thread-safe method)
        if self._listener.is_interrupted():
            if self._scheduler.has_pending_delivery():
                # Clear interruption state by calling stop() which also resets the event
                # We don't actually want to stop, just clear the flag, so we do it manually
                # by accessing the clear method indirectly through a new listen call
                delivery = self._scheduler.get_pending_delivery()
                if delivery:
                    return StateResult(
                        next_state=ConversationState.DELIVERING_REMINDER,
                        data={"delivery": delivery},
                    )
            # Otherwise it's a shutdown
            return StateResult(next_state=ConversationState.SHUTTING_DOWN)

        if audio is None:
            # No audio captured, stay in waiting state
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Wake word detected with audio captured
        return StateResult(
            next_state=ConversationState.LISTENING,
            data={
                "audio": audio,
                "is_wake_word_trigger": True,
            },
        )

    def exit(self, ctx: "AppContext") -> None:
        """No cleanup needed."""
        pass
