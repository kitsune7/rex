"""
Audio management for Rex voice assistant.

Provides unified audio I/O management and feedback tones.
All audio output goes through AudioManager to avoid race conditions.
"""

from .feedback import (
    ThinkingTone,
    generate_done_tone,
    generate_listening_tone,
    generate_thinking_sequence,
)
from .manager import AudioManager

__all__ = [
    "AudioManager",
    "ThinkingTone",
    "generate_listening_tone",
    "generate_done_tone",
    "generate_thinking_sequence",
]
