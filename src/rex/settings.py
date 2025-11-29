"""
Settings management for Rex voice assistant.

Loads configuration from settings.toml at the repository root.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReminderSettings:
    """Settings related to reminders."""

    retry_minutes: int = 10


@dataclass
class WakeWordSettings:
    """Settings related to wake word."""

    path_label: str = "hey_rex"  # String used for folder and file name in models/wake_word_models
    display_name: str = "Hey Rex"  # What's displayed to the user when talking about the wake word


@dataclass
class Settings:
    """Application settings loaded from settings.toml."""

    reminders: ReminderSettings
    wake_word: WakeWordSettings
    listening_timeout: float = 6.0  # Seconds to wait for a follow-up response


def load_settings(settings_path: str | Path = "settings.toml") -> Settings:
    """
    Load settings from settings.toml.

    Args:
        settings_path: Path to the settings file

    Returns:
        Settings object with all configuration values.
    """
    settings_path = Path(settings_path)

    if settings_path.exists():
        with open(settings_path, "rb") as f:
            data = tomllib.load(f)
    else:
        data = {}

    # Parse reminder settings
    reminder_data = data.get("reminders", {})
    reminder_settings = ReminderSettings(
        retry_minutes=reminder_data.get("retry_minutes", 10),
    )

    wake_word_data = data.get("wake_word", {})
    wake_word_settings = WakeWordSettings(
        path_label=wake_word_data.get("path_label", "hey_rex"),
        display_name=wake_word_data.get("display_name", "Hey Rex"),
    )

    listening_timeout_data = data.get("listening_timeout", 6.0)

    return Settings(
        reminders=reminder_settings,
        wake_word=wake_word_settings,
        listening_timeout=listening_timeout_data,
    )


# Module-level cached settings (for backwards compatibility)
# New code should pass Settings via AppContext
_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the current settings, loading if necessary.

    For new code, prefer passing Settings via AppContext.
    This function exists for backwards compatibility.
    """
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = load_settings()
    return _cached_settings


def set_settings(settings: Settings) -> None:
    """
    Set the module-level cached settings.

    Used by AppContext initialization to ensure consistent settings.
    """
    global _cached_settings
    _cached_settings = settings
