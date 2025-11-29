"""
Core infrastructure for Rex voice assistant.

Provides foundational abstractions:
- Event bus for decoupled communication
- Application context for dependency injection
- State machine for conversation flow management
"""

from .context import AppContext, create_app_context
from .events import Event, EventBus
from .state_machine import ConversationState, StateHandler, StateMachine

__all__ = [
    "Event",
    "EventBus",
    "AppContext",
    "ConversationState",
    "StateHandler",
    "StateMachine",
    "create_app_context",
]
