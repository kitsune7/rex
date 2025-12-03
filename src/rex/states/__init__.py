"""
Conversation state handlers for Rex voice assistant.

Each state in the conversation flow has a handler that:
1. Performs the work for that state
2. Returns the next state to transition to
"""

from .confirming import AwaitingConfirmationHandler
from .listening import ListeningHandler
from .processing import ProcessingHandler
from .reminder import DeliveringReminderHandler
from .speaking import SpeakingHandler
from .waiting import WaitingForWakeWordHandler

__all__ = [
    "WaitingForWakeWordHandler",
    "ListeningHandler",
    "ProcessingHandler",
    "SpeakingHandler",
    "AwaitingConfirmationHandler",
    "DeliveringReminderHandler",
]


def create_all_handlers(
    listener,
    transcriber,
    speaker,
    voice,
    scheduler,
):
    """
    Create all state handlers with shared dependencies.

    Args:
        listener: WakeWordListener instance
        transcriber: Transcriber instance
        speaker: InterruptibleSpeaker instance
        voice: Loaded TTS voice
        scheduler: ReminderScheduler instance

    Returns:
        List of all state handlers
    """
    return [
        WaitingForWakeWordHandler(listener=listener, scheduler=scheduler),
        ListeningHandler(listener=listener, transcriber=transcriber),
        ProcessingHandler(),
        SpeakingHandler(speaker=speaker),
        AwaitingConfirmationHandler(listener=listener, transcriber=transcriber, speaker=speaker),
        DeliveringReminderHandler(
            listener=listener,
            transcriber=transcriber,
            speaker=speaker,
            voice=voice,
            scheduler=scheduler,
        ),
    ]
