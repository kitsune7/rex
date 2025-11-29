"""
Processing state handler.

Sends the user's query to the agent and handles the response.
"""

from typing import TYPE_CHECKING

from agent import PendingConfirmation, run_voice_agent
from audio.feedback import ThinkingTone
from core.state_machine import ConversationState, StateHandler, StateResult

if TYPE_CHECKING:
    from core.context import AppContext


class ProcessingHandler(StateHandler):
    """
    Handler for the PROCESSING state.

    Invokes the LLM agent with the user's query and determines
    the next state based on the response.
    """

    def __init__(self):
        self._transcription = ""

    @property
    def state(self) -> ConversationState:
        return ConversationState.PROCESSING

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Store the transcription to process."""
        if data:
            self._transcription = data.get("transcription", "")
        else:
            self._transcription = ""

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Process the query with the agent.

        Returns:
            SPEAKING with response text
            AWAITING_CONFIRMATION if tool needs confirmation
            WAITING_FOR_WAKE_WORD on error
        """
        if not self._transcription:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        try:
            print("🤔 Thinking...")

            # Play thinking tone while waiting for LLM
            with ThinkingTone():
                result, history, thread_id = run_voice_agent(
                    self._transcription,
                    ctx.conversation_history,
                    ctx.thread_id,
                )

            # Update conversation state
            ctx.conversation_history = history
            ctx.thread_id = thread_id

            # Check if we need confirmation
            if isinstance(result, PendingConfirmation):
                return StateResult(
                    next_state=ConversationState.AWAITING_CONFIRMATION,
                    data={"pending": result},
                )

            # Normal response
            response = result
            return StateResult(
                next_state=ConversationState.SPEAKING,
                data={"response": response},
            )

        except Exception as e:
            print(f"❌ Agent error: {e}")
            import traceback

            traceback.print_exc()

            return StateResult(
                next_state=ConversationState.SPEAKING,
                data={
                    "response": "Sorry, I encountered an error processing your request.",
                    "force_end_conversation": True,
                },
            )

    def exit(self, ctx: "AppContext") -> None:
        """Clear transcription."""
        self._transcription = ""
