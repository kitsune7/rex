"""
Awaiting confirmation state handler.

Handles human-in-the-loop confirmation for tool calls that
require explicit user approval (e.g., creating reminders).
"""

from typing import TYPE_CHECKING

from agent import PendingConfirmation, confirm_tool_call
from core.state_machine import ConversationState, StateHandler, StateResult

from .phrases import is_confirmation

if TYPE_CHECKING:
    from core.context import AppContext
    from stt import Transcriber
    from tts import InterruptibleSpeaker
    from wake_word import WakeWordListener

# Timeout for confirmation responses
CONFIRMATION_TIMEOUT = 10.0


class AwaitingConfirmationHandler(StateHandler):
    """
    Handler for the AWAITING_CONFIRMATION state.

    Prompts the user to confirm or reject a pending tool call,
    then processes the response.
    """

    def __init__(
        self,
        listener: "WakeWordListener",
        transcriber: "Transcriber",
        speaker: "InterruptibleSpeaker",
    ):
        self._listener = listener
        self._transcriber = transcriber
        self._speaker = speaker
        self._pending: PendingConfirmation | None = None

    @property
    def state(self) -> ConversationState:
        return ConversationState.AWAITING_CONFIRMATION

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Store the pending confirmation."""
        if data:
            self._pending = data.get("pending")
        else:
            self._pending = None

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Handle the confirmation flow.

        Returns:
            SPEAKING with the result of confirmation/rejection
        """
        if self._pending is None:
            return StateResult(
                next_state=ConversationState.SPEAKING,
                data={"response": "Something went wrong with the confirmation."},
            )

        # Speak the confirmation prompt
        print(f"\n🤖 Rex: {self._pending.confirmation_prompt}\n")
        self._speaker.speak_interruptibly(self._pending.confirmation_prompt)

        # Listen for response
        print("🎤 Listening for confirmation...")
        audio = self._listener.listen_for_speech(timeout=CONFIRMATION_TIMEOUT)

        if audio is None:
            # No response - treat as rejection
            print("⏱️ No confirmation received, cancelling.")
            response, history = confirm_tool_call(self._pending, confirmed=False)
            ctx.conversation_history = history
            return StateResult(
                next_state=ConversationState.SPEAKING,
                data={"response": response, "force_end_conversation": True},
            )

        transcription = self._transcriber.transcribe(audio)
        if not transcription:
            print("⏱️ Could not understand response, cancelling.")
            response, history = confirm_tool_call(self._pending, confirmed=False)
            ctx.conversation_history = history
            return StateResult(
                next_state=ConversationState.SPEAKING,
                data={"response": response, "force_end_conversation": True},
            )

        print(f"\n💬 You said: {transcription}\n")

        if is_confirmation(transcription):
            print("✅ Confirmed!")
            response, history = confirm_tool_call(self._pending, confirmed=True)
        else:
            print("❌ Cancelled.")
            response, history = confirm_tool_call(self._pending, confirmed=False)

        ctx.conversation_history = history

        return StateResult(
            next_state=ConversationState.SPEAKING,
            data={"response": response, "force_end_conversation": True},
        )

    def exit(self, ctx: "AppContext") -> None:
        """Clear pending confirmation."""
        self._pending = None
