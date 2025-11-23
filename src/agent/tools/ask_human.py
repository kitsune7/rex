from langchain_core.tools import tool
from stt import get_transcription


@tool
def get_human_text_response() -> str:
    """Ask the human user for input when the agent needs help or clarification.
    This tool asks for text directly from the command-line.

    Args:
        question: The question to ask the user

    Returns:
        The user's response
    """
    return input("Your response: ")


@tool
def ask_human_for_voice_response() -> str:
    """Ask the human user for input when the agent needs help or clarification.
    This tool asks the human and transcribes their vocal answer into text.

    Args:
        question: The question to ask the user

    Returns:
        The user's response
    """

    return get_transcription(debug=True)
