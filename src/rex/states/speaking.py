"""
Speaking state handler.

Handles TTS playback of the agent's response with interruption support.
"""

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from core.state_machine import ConversationState, StateHandler, StateResult

if TYPE_CHECKING:
    from core.context import AppContext
    from tts import InterruptibleSpeaker


class SpeakingHandler(StateHandler):
    """
    Handler for the SPEAKING state.

    Speaks the agent's response and handles interruption.
    Determines whether to continue the conversation based on
    whether the response ends with a question.
    """

    def __init__(self, speaker: "InterruptibleSpeaker"):
        self._speaker = speaker
        self._response = ""
        self._force_end_conversation = False

    @property
    def state(self) -> ConversationState:
        return ConversationState.SPEAKING

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Store the response to speak."""
        if data:
            self._response = data.get("response", "")
            self._force_end_conversation = data.get("force_end_conversation", False)
        else:
            self._response = ""
            self._force_end_conversation = False

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Speak the response and determine next state.

        Returns:
            LISTENING if interrupted or response ends with question
            WAITING_FOR_WAKE_WORD if conversation should end
        """
        if not self._response:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        print(f"\n🤖 Rex: {self._response}\n")

        # Speak with interruption support - returns (was_interrupted, captured_audio)
        was_interrupted, captured_audio = self._speaker.speak_interruptibly(self._response)

        if was_interrupted:
            print("🛑 Interrupted!")

            # Mark the last message as interrupted
            if ctx.conversation_history and isinstance(ctx.conversation_history[-1], AIMessage):
                ctx.conversation_history[-1] = AIMessage(
                    content=ctx.conversation_history[-1].content + " [interrupted]"
                )

            # Continue conversation with captured audio (user wants to say something)
            # Pass the captured audio so we don't need to listen again
            return StateResult(
                next_state=ConversationState.LISTENING,
                data={
                    "audio": captured_audio,
                    "is_wake_word_trigger": True,  # Strip wake word from transcription
                },
            )

        # Check if we should continue the conversation
        if self._force_end_conversation:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Continue if Rex asked a question
        should_continue = self._response.strip().endswith("?")

        if should_continue:
            return StateResult(
                next_state=ConversationState.LISTENING,
                data={"audio": None, "is_wake_word_trigger": False},
            )
        else:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

    def exit(self, ctx: "AppContext") -> None:
        """Clear response."""
        self._response = ""
        self._force_end_conversation = False
