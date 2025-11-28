"""
Settings management for Rex voice assistant.

Loads configuration from settings.toml at the repository root.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_settings = None


@dataclass
class ReminderSettings:
    """Settings related to reminders."""

    retry_minutes: int = 10


@dataclass
class Settings:
    """Application settings loaded from settings.toml."""

    reminders: ReminderSettings


def load_settings() -> Settings:
    """
    Load settings from settings.toml.

    Returns:
        Settings object with all configuration values.
    """
    global _settings

    if _settings is not None:
        return _settings

    settings_path = Path("settings.toml")

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

    _settings = Settings(reminders=reminder_settings)
    return _settings


def get_settings() -> Settings:
    """Get the current settings, loading if necessary."""
    return load_settings()
