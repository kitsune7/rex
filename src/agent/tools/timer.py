"""
Timer tools for Rex voice assistant.

Provides functionality to set, check, and stop named timers with alarm sounds.
"""

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import sounddevice as sd
import soundfile as sf
from langchain_core.tools import tool

if TYPE_CHECKING:
    from core.events import EventBus


class TimerState(Enum):
    PENDING = "pending"
    RINGING = "ringing"


@dataclass
class Timer:
    name: str
    duration_seconds: float
    start_time: float
    state: TimerState
    thread: threading.Timer | None = None


class TimerManager:
    """
    Manages multiple named timers with alarm sound playback.

    This is no longer a singleton - instances should be created via
    the AppContext or create_timer_manager() factory.
    """

    def __init__(
        self,
        event_bus: "EventBus | None" = None,
        sound_path: str | Path = "sounds/fun-timer.mp3",
    ):
        """
        Initialize the timer manager.

        Args:
            event_bus: Optional event bus for emitting timer events
            sound_path: Path to the alarm sound file
        """
        self._event_bus = event_bus
        self._timers: dict[str, Timer] = {}
        self._timers_lock = threading.Lock()
        self._sound_thread: threading.Thread | None = None
        self._stop_sound = threading.Event()
        self._current_ringing: str | None = None
        self._muted = False

        # Load sound file
        sound_path = Path(sound_path)
        if sound_path.exists():
            self._sound_data, self._sample_rate = sf.read(sound_path)
        else:
            self._sound_data = None
            self._sample_rate = None

    def _emit_event(self, event) -> None:
        """Emit an event if event bus is configured."""
        if self._event_bus is not None:
            self._event_bus.emit(event)

    def _play_sound_loop(self):
        """Play the alarm sound on loop until stopped."""
        if self._sound_data is None:
            return

        while not self._stop_sound.is_set():
            # Skip playback if muted
            if self._muted:
                time.sleep(0.1)
                continue

            sd.play(self._sound_data, self._sample_rate)
            # Wait for playback to finish or stop signal
            duration = len(self._sound_data) / self._sample_rate
            # Check stop signal more frequently than sound duration
            check_interval = 0.1
            elapsed = 0.0
            while elapsed < duration and not self._stop_sound.is_set() and not self._muted:
                time.sleep(check_interval)
                elapsed += check_interval

            if self._stop_sound.is_set():
                sd.stop()
                break

            if self._muted:
                sd.stop()

    def _timer_callback(self, name: str):
        """Called when a timer fires."""
        with self._timers_lock:
            if name not in self._timers:
                return
            timer = self._timers[name]
            if timer.state != TimerState.PENDING:
                return
            timer.state = TimerState.RINGING
            self._current_ringing = name

        # Emit event
        from core.events import TimerFired

        self._emit_event(TimerFired(timer_name=name))

        # Start sound playback
        self._stop_sound.clear()
        self._sound_thread = threading.Thread(target=self._play_sound_loop, daemon=True)
        self._sound_thread.start()

    def set_timer(self, name: str, duration_seconds: float) -> str:
        """
        Set a new timer.

        Args:
            name: Name for the timer
            duration_seconds: Duration in seconds

        Returns:
            Confirmation message
        """
        with self._timers_lock:
            # Cancel existing timer with same name
            if name in self._timers:
                old_timer = self._timers[name]
                if old_timer.thread:
                    old_timer.thread.cancel()
                if old_timer.state == TimerState.RINGING:
                    self._stop_alarm()

            # Create new timer
            timer_thread = threading.Timer(duration_seconds, self._timer_callback, args=[name])
            timer_thread.daemon = True

            timer = Timer(
                name=name,
                duration_seconds=duration_seconds,
                start_time=time.time(),
                state=TimerState.PENDING,
                thread=timer_thread,
            )
            self._timers[name] = timer
            timer_thread.start()

        return f"Timer '{name}' set for {self._format_duration(duration_seconds)}"

    def check_timers(self) -> str:
        """Get status of all timers."""
        with self._timers_lock:
            if not self._timers:
                return "No active timers."

            status_lines = []
            for name, timer in self._timers.items():
                if timer.state == TimerState.RINGING:
                    status_lines.append(f"'{name}' is ringing!")
                else:
                    elapsed = time.time() - timer.start_time
                    remaining = max(0, timer.duration_seconds - elapsed)
                    status_lines.append(f"'{name}': {self._format_duration(remaining)} remaining")

            return "\n".join(status_lines)

    def stop_timer(self, name: str | None = None) -> str:
        """
        Stop a timer or the currently ringing alarm.

        Args:
            name: Timer name to stop. If None, stops the currently ringing alarm.

        Returns:
            Confirmation message
        """
        with self._timers_lock:
            # If no name given, stop the ringing alarm
            if name is None:
                if self._current_ringing:
                    name = self._current_ringing
                else:
                    return "No timer is currently ringing."

            if name not in self._timers:
                return f"No timer named '{name}' found."

            timer = self._timers[name]

            if timer.state == TimerState.RINGING:
                self._stop_alarm()
                timer_name = name
                del self._timers[name]
                self._current_ringing = None

                # Emit event
                from core.events import TimerStopped

                self._emit_event(TimerStopped(timer_name=timer_name))

                return f"Stopped alarm for timer '{name}'."
            else:  # PENDING
                if timer.thread:
                    timer.thread.cancel()
                del self._timers[name]
                return f"Cancelled timer '{name}'."

    def stop_any_ringing(self) -> bool:
        """
        Stop any currently ringing alarm. Used by CLI for "stop" command.

        Returns:
            True if an alarm was stopped, False otherwise.
        """
        with self._timers_lock:
            if self._current_ringing and self._current_ringing in self._timers:
                timer = self._timers[self._current_ringing]
                if timer.state == TimerState.RINGING:
                    self._stop_alarm()
                    timer_name = self._current_ringing
                    del self._timers[self._current_ringing]
                    self._current_ringing = None

                    # Emit event
                    from core.events import TimerStopped

                    self._emit_event(TimerStopped(timer_name=timer_name))

                    return True
        return False

    def _stop_alarm(self):
        """Stop the alarm sound playback."""
        self._stop_sound.set()
        sd.stop()
        if self._sound_thread and self._sound_thread.is_alive():
            self._sound_thread.join(timeout=1.0)
        self._sound_thread = None

    def mute(self):
        """Temporarily mute the alarm sound. Call unmute() to resume."""
        self._muted = True
        sd.stop()

    def unmute(self):
        """Resume alarm sound if a timer is still ringing."""
        self._muted = False
        with self._timers_lock:
            # If there's still a ringing timer and we're not already playing, restart sound
            if self._current_ringing and self._current_ringing in self._timers:
                timer = self._timers[self._current_ringing]
                if timer.state == TimerState.RINGING:
                    # Only restart if sound thread isn't running
                    if self._sound_thread is None or not self._sound_thread.is_alive():
                        self._stop_sound.clear()
                        self._sound_thread = threading.Thread(target=self._play_sound_loop, daemon=True)
                        self._sound_thread.start()

    def cleanup(self):
        """Clean up all timers and stop sounds."""
        with self._timers_lock:
            for timer in self._timers.values():
                if timer.thread:
                    timer.thread.cancel()
            self._timers.clear()
        self._stop_alarm()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            parts = [f"{minutes} minute{'s' if minutes != 1 else ''}"]
            if secs > 0:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            return " ".join(parts)
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            return " ".join(parts)


def parse_duration(duration_str: str) -> float | None:
    """
    Parse a duration string into seconds.

    Supports formats like:
    - "5 minutes"
    - "30 seconds"
    - "1 hour 30 minutes"
    - "90 seconds"
    - "2 hours"
    - "1 minute 30 seconds"

    Returns:
        Duration in seconds, or None if parsing fails.
    """
    duration_str = duration_str.lower().strip()

    total_seconds = 0.0
    found_any = False

    # Pattern for hours
    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)", duration_str)
    if hours_match:
        total_seconds += float(hours_match.group(1)) * 3600
        found_any = True

    # Pattern for minutes
    minutes_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)(?!s)", duration_str)
    if minutes_match:
        total_seconds += float(minutes_match.group(1)) * 60
        found_any = True

    # Pattern for seconds
    seconds_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)", duration_str)
    if seconds_match:
        total_seconds += float(seconds_match.group(1))
        found_any = True

    # If nothing matched, try to parse as just a number (assume minutes)
    if not found_any:
        try:
            total_seconds = float(duration_str) * 60
            found_any = True
        except ValueError:
            pass

    return total_seconds if found_any and total_seconds > 0 else None


# Module-level instance for tool access
# This gets set by the AppContext during initialization
_timer_manager: TimerManager | None = None


def set_timer_manager(manager: TimerManager) -> None:
    """Set the module-level timer manager instance for tool access."""
    global _timer_manager
    _timer_manager = manager


def get_timer_manager() -> TimerManager:
    """
    Get the timer manager instance.

    Returns the configured instance, or creates a default one if not set.
    """
    global _timer_manager
    if _timer_manager is None:
        _timer_manager = TimerManager()
    return _timer_manager


@tool
def set_timer(duration: str, name: str = "timer") -> str:
    """
    Set a timer for a specified duration.

    Args:
        duration: How long to set the timer for (e.g., "5 minutes", "30 seconds", "1 hour 30 minutes")
        name: Optional name for the timer (e.g., "pizza timer", "tea timer"). Defaults to "timer".

    Returns:
        Confirmation message with timer details.

    Examples:
        - set_timer("5 minutes") - sets a 5 minute timer
        - set_timer("30 seconds", "egg timer") - sets a 30 second timer named "egg timer"
        - set_timer("1 hour 30 minutes", "meeting") - sets a 90 minute timer named "meeting"
    """
    seconds = parse_duration(duration)
    if seconds is None:
        return (
            f"Could not understand duration '{duration}'. "
            "Try formats like '5 minutes', '30 seconds', or '1 hour 30 minutes'."
        )

    return get_timer_manager().set_timer(name, seconds)


@tool
def check_timers() -> str:
    """
    Check the status of all active timers.

    Returns:
        Status of all timers including remaining time or if they're currently ringing.
    """
    return get_timer_manager().check_timers()


@tool
def stop_timer(name: str | None = None) -> str:
    """
    Stop a ringing timer alarm or cancel a pending timer.

    Args:
        name: The name of the timer to stop. If not provided, stops the currently ringing alarm.

    Returns:
        Confirmation message.

    Examples:
        - stop_timer() - stops the currently ringing alarm
        - stop_timer("pizza timer") - stops or cancels the pizza timer
    """
    return get_timer_manager().stop_timer(name)
