from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current time in 12-hour format with AM/PM."""
    now = datetime.now()
    return now.strftime("%I:%M %p")
