"""Emotion tagging for Rex responses.

The voice agent is instructed to prefix every response with one of a small set
of tags like ``[emotion:happy]``. This module defines the allowed tags and a
parser that splits the leading tag off the spoken text.
"""

from __future__ import annotations

import re

# Keep this tight — the Reachy side maps each tag to a motion preset, so
# every new emotion needs a corresponding pose. Add tags here AND in
# rex_reachy/motion.py at the same time.
ALLOWED_EMOTIONS = (
    "neutral",
    "happy",
    "thinking",
    "confused",
    "alert",
    "sad",
    "excited",
)

DEFAULT_EMOTION = "neutral"

_EMOTION_PATTERN = re.compile(
    r"^\s*\[emotion:(?P<tag>[a-z_]+)\]\s*",
    re.IGNORECASE,
)


def parse_emotion(text: str) -> tuple[str, str]:
    """Pop a leading ``[emotion:...]`` tag off the text.

    Returns ``(emotion, clean_text)``. If the tag is missing or unrecognized,
    ``DEFAULT_EMOTION`` is used and the input is returned unchanged.
    """
    match = _EMOTION_PATTERN.match(text)
    if not match:
        return DEFAULT_EMOTION, text.strip()

    tag = match.group("tag").lower()
    if tag not in ALLOWED_EMOTIONS:
        tag = DEFAULT_EMOTION

    clean = text[match.end():].strip()
    return tag, clean


EMOTION_SYSTEM_PROMPT_SUFFIX = (
    "\n\n"
    "At the very start of every reply, output exactly one emotion tag from this "
    "list, then continue with your normal spoken response. Tags: "
    f"{', '.join(f'[emotion:{e}]' for e in ALLOWED_EMOTIONS)}. "
    "Pick the tag that best fits the tone of your reply. Example: "
    "'[emotion:happy] Sure thing — the timer is set for five minutes.' "
    "Never mention the tag itself in your spoken words; it is parsed out before "
    "speech synthesis."
)
