"""
Shared phrase matching utilities for conversation states.

Provides common phrase sets and matching functions used across
multiple state handlers (confirmation, reminder delivery, etc.).
"""

CONFIRM_PHRASES = (
    "yes", "yeah", "yep", "sure", "okay", "ok",
    "confirm", "do it", "go ahead", "proceed",
    "clear", "done", "got it",
)

REJECT_PHRASES = ("no", "nope", "cancel", "nevermind", "never mind", "don't", "stop")


def is_confirmation(text: str) -> bool:
    """Check if text contains a confirmation phrase."""
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in CONFIRM_PHRASES)


def is_rejection(text: str) -> bool:
    """Check if text contains a rejection phrase."""
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in REJECT_PHRASES)

