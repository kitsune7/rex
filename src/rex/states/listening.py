"""
Listening state handler.

Handles speech capture and transcription, either after wake word
detection or during follow-up conversation.
"""

from typing import TYPE_CHECKING

from core.state_machine import ConversationState, StateHandler, StateResult

if TYPE_CHECKING:
    from core.context import AppContext
    from stt import Transcriber
    from wake_word import WakeWordListener

# Phrases that end the conversation
STOP_PHRASES = ("stop", "nevermind", "never mind", "cancel", "forget it")


class ListeningHandler(StateHandler):
    """
    Handler for the LISTENING state.

    Processes captured audio or waits for follow-up speech,
    then transcribes and handles special commands.
    """

    def __init__(
        self,
        listener: "WakeWordListener",
        transcriber: "Transcriber",
    ):
        self._listener = listener
        self._transcriber = transcriber
        self._audio = None
        self._is_wake_word_trigger = False

    @property
    def state(self) -> ConversationState:
        return ConversationState.LISTENING

    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """Store audio data if provided."""
        if data:
            self._audio = data.get("audio")
            self._is_wake_word_trigger = data.get("is_wake_word_trigger", False)
        else:
            self._audio = None
            self._is_wake_word_trigger = False

    def process(self, ctx: "AppContext") -> StateResult:
        """
        Process audio or capture follow-up speech.

        Returns:
            PROCESSING if transcription successful
            WAITING_FOR_WAKE_WORD if timeout or stop command
        """
        # If no audio provided (follow-up mode), listen for speech
        if self._audio is None:
            # Play ready tone BEFORE waiting for speech so user knows Rex is listening
            ctx.audio_manager.play_listening_tone()
            print("🎤 Listening for response...")
            self._audio = self._listener.listen_for_speech(
                timeout=ctx.settings.listening_timeout, play_tones=True
            )

            if self._audio is None:
                print("⏱️ No response received, ending conversation.")
                # Play done tone to indicate we're no longer listening
                ctx.audio_manager.play_done_tone()
                return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Transcribe the audio
        strip_wake_word = self._is_wake_word_trigger
        transcription = self._transcriber.transcribe(self._audio, strip_wake_word=strip_wake_word)

        if not transcription:
            # Empty transcription - go back to appropriate state
            if ctx.is_in_conversation():
                # Stay listening for follow-up
                return StateResult(
                    next_state=ConversationState.LISTENING,
                    data={"audio": None, "is_wake_word_trigger": False},
                )
            else:
                return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        print(f"\n💬 You said: {transcription}\n")

        # Handle special commands
        normalized = transcription.strip().lower()

        # Timer stop command (works anytime)
        if normalized in ("stop", "stop the timer"):
            if ctx.timer_manager and ctx.timer_manager.stop_any_ringing():
                print("🔕 Timer alarm stopped.")
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Stop phrases end conversation (only during follow-up)
        if ctx.is_in_conversation() and normalized in STOP_PHRASES:
            return StateResult(next_state=ConversationState.WAITING_FOR_WAKE_WORD)

        # Pass transcription to processing state
        return StateResult(
            next_state=ConversationState.PROCESSING,
            data={"transcription": transcription},
        )

    def exit(self, ctx: "AppContext") -> None:
        """Clear audio data."""
        self._audio = None
        self._is_wake_word_trigger = False
