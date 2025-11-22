import argparse
import sys

from rex.manual_query import query
from rex.speech_input import SpeechRecorder
from tts import load_voice, speak_text


def main():
    parser = argparse.ArgumentParser(description="Rex AI Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("start", help="Start Rex AI Assistant")

    query_parser = subparsers.add_parser("query", help="Query Rex directly")
    query_parser.add_argument("text", help="The text to give to Rex for a single request")
    default_system_prompt = """
You are a helpful AI assistant that responds to the name Rex. Rex is the name of the whole system the user is
interacting with, and you are that system's brain.

If the grammar of the user's message doesn't make sense, a word or two may have been transcribed incorrectly from STT.

Important Rules:
- Avoid speaking about your internals and the specific role you play as the model unless the user asks specifically
- Because your response is voiced with TTS, avoid characters that can't be voiced such as emojis
    """
    query_parser.add_argument(
        "--system-prompt",
        "-s",
        help="System prompt to set context for the model",
        default=default_system_prompt,
    )

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start()
    elif args.command == "query":
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


def cmd_start():
    # Default system prompt for voice interactions
    default_system_prompt = """
You are a helpful AI assistant that responds to the name Rex. Rex is the name of the whole system the user is
interacting with, and you are that system's brain.

If the grammar of the user's message doesn't make sense, a word or two may have been transcribed incorrectly from STT.

Important Rules:
- Avoid speaking about your internals and the specific role you play as the model unless the user asks specifically
- Because your response is voiced with TTS, avoid characters that can't be voiced such as emojis
    """
    recorder = SpeechRecorder(sample_rate=16000, silence_duration=1.5)

    print("Listening for user speech...")
    audio_data = recorder.record_until_silence()

    print("End of speech detected. Transcribing now...")
    transcription = recorder.transcribe(audio_data)
    print(f"\nTranscription of user speech:\n{transcription}")

    # Pass transcription to LLM for processing and get voice response
    if transcription.strip():
        print("\nProcessing query...")
        query(transcription, system_prompt=default_system_prompt)
    else:
        print("No speech detected or transcription failed.")


def cmd_query(args):
    user_query = args.text
    print(f'User query: "{user_query}"')
    query(user_query, system_prompt=args.system_prompt)


if __name__ == "__main__":
    sys.exit(main())
