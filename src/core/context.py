"""
Application context for dependency injection.

The AppContext holds all shared dependencies and is passed
to components that need them. This replaces global singletons
and makes dependencies explicit.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .events import EventBus

if TYPE_CHECKING:
    from agent.tools.reminder import ReminderManager
    from agent.tools.timer import TimerManager
    from audio.manager import AudioManager
    from rex.settings import Settings


@dataclass
class AppContext:
    """
    Container for all application dependencies.

    This is the central place where all shared components are held.
    Components receive the context (or specific dependencies from it)
    rather than accessing global singletons.

    Attributes:
        event_bus: Central event bus for component communication
        timer_manager: Manages active timers
        reminder_manager: Manages reminder persistence
        settings: Application configuration
        audio_manager: Unified audio I/O management
    """

    event_bus: EventBus = field(default_factory=EventBus)
    timer_manager: "TimerManager | None" = None
    reminder_manager: "ReminderManager | None" = None
    settings: "Settings | None" = None
    audio_manager: "AudioManager | None" = None

    # Conversation state - shared across components
    thread_id: str | None = None
    conversation_history: list | None = None

    def reset_conversation(self) -> None:
        """Reset conversation state to start fresh."""
        self.thread_id = None
        self.conversation_history = None

    def is_in_conversation(self) -> bool:
        """Check if we're currently in an active conversation."""
        return self.conversation_history is not None


def create_app_context(
    timer_manager: "TimerManager | None" = None,
    reminder_manager: "ReminderManager | None" = None,
    settings: "Settings | None" = None,
    audio_manager: "AudioManager | None" = None,
) -> AppContext:
    """
    Factory function to create a fully initialized AppContext.

    This is the preferred way to create an AppContext as it ensures
    all dependencies are properly wired together.

    Args:
        timer_manager: Optional TimerManager instance (created if not provided)
        reminder_manager: Optional ReminderManager instance (created if not provided)
        settings: Optional Settings instance (loaded if not provided)
        audio_manager: Optional AudioManager instance (created if not provided)

    Returns:
        Fully initialized AppContext
    """
    from agent.tools import ReminderManager, TimerManager
    from audio.manager import AudioManager
    from rex.settings import load_settings

    event_bus = EventBus()

    audio_mgr = audio_manager or AudioManager(event_bus=event_bus)
    timer_mgr = timer_manager or TimerManager(event_bus=event_bus, audio_manager=audio_mgr)
    reminder_mgr = reminder_manager or ReminderManager(event_bus=event_bus)

    return AppContext(
        event_bus=event_bus,
        timer_manager=timer_mgr,
        reminder_manager=reminder_mgr,
        settings=settings or load_settings(),
        audio_manager=audio_mgr,
    )
