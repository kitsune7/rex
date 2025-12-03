"""
Timer tools for Rex voice assistant.

Provides functionality to set, check, and stop named timers with alarm sounds.
Routes audio through AudioManager to avoid race conditions.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from langchain_core.tools import tool

if TYPE_CHECKING:
    from audio.manager import AudioManager
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

    Routes audio through AudioManager to avoid race conditions with other
    audio sources (TTS, feedback tones, etc.).
    """

    def __init__(
        self,
        event_bus: "EventBus | None" = None,
        audio_manager: AudioManager | None = None,
        sound_path: str | Path = "sounds/fun-timer.mp3",
    ):
        """
        Initialize the timer manager.

        Args:
            event_bus: Optional event bus for emitting timer events
            audio_manager: AudioManager instance for audio output
            sound_path: Path to the alarm sound file
        """
        self._event_bus = event_bus
        self._audio_manager = audio_manager
        self._timers: dict[str, Timer] = {}
        self._timers_lock = threading.Lock()
        self._current_ringing: str | None = None
        self._muted = False

        # Load and prepare sound file
        sound_path = Path(sound_path)
        if sound_path.exists():
            raw_audio, self._sample_rate = sf.read(sound_path)
            # Convert to mono if stereo
            if len(raw_audio.shape) > 1:
                raw_audio = raw_audio.mean(axis=1)
            self._sound_data = raw_audio.astype(np.float32)
        else:
            self._sound_data = None
            self._sample_rate = None

    def _emit_event(self, event) -> None:
        """Emit an event if event bus is configured."""
        if self._event_bus is not None:
            self._event_bus.emit(event)

    def _start_alarm_sound(self):
        """Start the alarm sound loop through AudioManager."""
        if self._sound_data is None or self._audio_manager is None:
            return
        if not self._muted:
            self._audio_manager.start_loop(self._sound_data, self._sample_rate)

    def _stop_alarm_sound(self):
        """Stop the alarm sound loop."""
        if self._audio_manager is not None:
            self._audio_manager.stop_loop()

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

        # Start sound playback through AudioManager
        self._start_alarm_sound()

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
                self._stop_alarm_sound()
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
                    self._stop_alarm_sound()
                    timer_name = self._current_ringing
                    del self._timers[self._current_ringing]
                    self._current_ringing = None

                    # Emit event
                    from core.events import TimerStopped

                    self._emit_event(TimerStopped(timer_name=timer_name))

                    return True
        return False

    def mute(self):
        """Temporarily mute the alarm sound. Call unmute() to resume."""
        self._muted = True
        self._stop_alarm_sound()

    def unmute(self):
        """Resume alarm sound if a timer is still ringing."""
        self._muted = False
        with self._timers_lock:
            # If there's still a ringing timer, restart sound
            if self._current_ringing and self._current_ringing in self._timers:
                timer = self._timers[self._current_ringing]
                if timer.state == TimerState.RINGING:
                    self._start_alarm_sound()

    def cleanup(self):
        """Clean up all timers and stop sounds."""
        with self._timers_lock:
            for timer in self._timers.values():
                if timer.thread:
                    timer.thread.cancel()
            self._timers.clear()
        self._stop_alarm_sound()

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


def create_timer_tools(manager: TimerManager) -> tuple:
    """
    Create timer tools with the given manager injected.

    Args:
        manager: TimerManager instance to use for all operations

    Returns:
        Tuple of (set_timer, check_timers, stop_timer) tools
    """

    @tool
    def set_timer(duration: str, name: str = "timer") -> str:
        """Set a timer.

        Args:
            duration: e.g. "5 minutes", "30 seconds", "1 hour 30 minutes"
            name: Optional timer name (default: "timer")
        """
        seconds = parse_duration(duration)
        if seconds is None:
            return (
                f"Could not understand duration '{duration}'. "
                "Try formats like '5 minutes', '30 seconds', or '1 hour 30 minutes'."
            )

        return manager.set_timer(name, seconds)

    @tool
    def check_timers() -> str:
        """
        Check the status of all active timers.

        Returns:
            Status of all timers including remaining time or if they're currently ringing.
        """
        return manager.check_timers()

    @tool
    def stop_timer(name: str | None = None) -> str:
        """Stop a ringing alarm or cancel a timer. If name is omitted, stops the current alarm."""
        return manager.stop_timer(name)

    return set_timer, check_timers, stop_timer
