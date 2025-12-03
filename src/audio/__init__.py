"""
Audio management for Rex voice assistant.

Provides unified audio I/O management, feedback tones,
and priority handling for competing audio sources.
"""

from .feedback import (
    ThinkingTone,
    init_audio_feedback,
    play_done_tone,
    play_listening_tone,
)
from .manager import AudioManager, SoundPriority

__all__ = [
    "AudioManager",
    "SoundPriority",
    "init_audio_feedback",
    "play_listening_tone",
    "play_done_tone",
    "ThinkingTone",
]
