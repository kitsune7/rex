from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current time. Use this when the user asks what time it is."""
    now = datetime.now()
    return now.strftime("%-I:%M %p")
