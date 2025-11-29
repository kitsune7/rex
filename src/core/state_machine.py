"""
State machine for Rex conversation flow.

Defines the possible states of a conversation and provides
the infrastructure for state transitions. Each state has
a handler that processes events and returns the next state.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import AppContext


class ConversationState(Enum):
    """
    Possible states in the Rex conversation flow.

    State diagram:

    WAITING_FOR_WAKE_WORD
        │
        ├─[wake word detected]─→ LISTENING
        │
        └─[reminder due]─→ DELIVERING_REMINDER

    LISTENING
        │
        ├─[speech captured]─→ PROCESSING
        │
        ├─[timeout]─→ WAITING_FOR_WAKE_WORD
        │
        └─[stop command]─→ WAITING_FOR_WAKE_WORD

    PROCESSING
        │
        ├─[response ready]─→ SPEAKING
        │
        ├─[confirmation needed]─→ AWAITING_CONFIRMATION
        │
        └─[error]─→ SPEAKING (error message)

    SPEAKING
        │
        ├─[speech complete + question]─→ LISTENING (follow-up)
        │
        ├─[speech complete + no question]─→ WAITING_FOR_WAKE_WORD
        │
        └─[interrupted]─→ LISTENING

    AWAITING_CONFIRMATION
        │
        ├─[confirmed/rejected]─→ SPEAKING
        │
        └─[timeout]─→ SPEAKING (cancellation message)

    DELIVERING_REMINDER
        │
        ├─[acknowledged/snoozed]─→ WAITING_FOR_WAKE_WORD
        │
        └─[no response]─→ WAITING_FOR_WAKE_WORD (schedules retry)
    """

    WAITING_FOR_WAKE_WORD = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    AWAITING_CONFIRMATION = auto()
    DELIVERING_REMINDER = auto()
    SHUTTING_DOWN = auto()


@dataclass
class StateResult:
    """
    Result of processing a state.

    Attributes:
        next_state: The state to transition to
        data: Optional data to pass to the next state
    """

    next_state: ConversationState
    data: dict | None = None


class StateHandler(ABC):
    """
    Abstract base class for state handlers.

    Each conversation state has a handler that:
    1. Performs the work for that state
    2. Returns the next state to transition to

    Handlers receive the AppContext for access to shared resources.
    """

    @property
    @abstractmethod
    def state(self) -> ConversationState:
        """The state this handler handles."""
        pass

    @abstractmethod
    def enter(self, ctx: "AppContext", data: dict | None = None) -> None:
        """
        Called when entering this state.

        Use this for setup, playing sounds, printing messages, etc.

        Args:
            ctx: Application context with shared resources
            data: Optional data passed from the previous state
        """
        pass

    @abstractmethod
    def process(self, ctx: "AppContext") -> StateResult:
        """
        Process this state and determine the next state.

        This is where the main work happens. The handler should:
        1. Do whatever work is needed for this state
        2. Determine what state to transition to
        3. Return a StateResult with the next state

        Args:
            ctx: Application context with shared resources

        Returns:
            StateResult indicating the next state
        """
        pass

    def exit(self, ctx: "AppContext") -> None:
        """
        Called when leaving this state.

        Use this for cleanup. Default implementation does nothing.

        Args:
            ctx: Application context with shared resources
        """
        pass


class StateMachine:
    """
    Manages state transitions for the conversation flow.

    The state machine:
    1. Maintains the current state
    2. Dispatches to the appropriate handler
    3. Manages state transitions
    4. Continues until reaching SHUTTING_DOWN state
    """

    def __init__(self, ctx: "AppContext", handlers: list[StateHandler]):
        """
        Initialize the state machine.

        Args:
            ctx: Application context with shared resources
            handlers: List of state handlers (one per state)
        """
        self.ctx = ctx
        self._handlers: dict[ConversationState, StateHandler] = {}
        self._current_state = ConversationState.WAITING_FOR_WAKE_WORD
        self._running = False
        self._transition_data: dict | None = None

        # Register handlers
        for handler in handlers:
            self._handlers[handler.state] = handler

        # Validate all states have handlers (except SHUTTING_DOWN)
        for state in ConversationState:
            if state != ConversationState.SHUTTING_DOWN and state not in self._handlers:
                raise ValueError(f"Missing handler for state: {state}")

    @property
    def current_state(self) -> ConversationState:
        """Get the current state."""
        return self._current_state

    def run(self) -> None:
        """
        Run the state machine until shutdown.

        This is the main loop that:
        1. Enters the current state
        2. Processes the state
        3. Exits the current state
        4. Transitions to the next state
        5. Repeats until SHUTTING_DOWN
        """
        self._running = True

        while self._running and self._current_state != ConversationState.SHUTTING_DOWN:
            handler = self._handlers.get(self._current_state)
            if handler is None:
                print(f"No handler for state {self._current_state}, shutting down")
                break

            try:
                handler.enter(self.ctx, self._transition_data)
                self._transition_data = None

                result = handler.process(self.ctx)

                handler.exit(self.ctx)

                self._current_state = result.next_state
                self._transition_data = result.data

            except KeyboardInterrupt:
                self._current_state = ConversationState.SHUTTING_DOWN
            except Exception as e:
                print(f"Error in state {self._current_state}: {e}")
                import traceback

                traceback.print_exc()
                # Try to recover by going back to waiting state
                self._current_state = ConversationState.WAITING_FOR_WAKE_WORD
                self._transition_data = None

        self._running = False

    def stop(self) -> None:
        """Request the state machine to stop."""
        self._running = False
        self._current_state = ConversationState.SHUTTING_DOWN
