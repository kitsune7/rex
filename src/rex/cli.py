import argparse
import sys
from tts import load_voice, speak_text


def main():
    parser = argparse.ArgumentParser(description="Rex AI Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("start", help="Start Rex AI Assistant")

    query_parser = subparsers.add_parser("query", help="Query Rex directly")
    query_parser.add_argument("text", help="The text to give to Rex for a single request")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start()
    elif args.command == "query":
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


def cmd_start():
    voice = load_voice()
    text = """
    Oh, I don't know if I'm ready for that quite yet. The wake_word still needs work and then
    faster whisper needs to be added to transcribe what you say. Get all of that in place and THEN
    we'll talk.
    """
    speak_text(text, voice)


def cmd_query(args):
    user_query = args.text
    print(f'User query: "{user_query}"')
    voice = load_voice()
    text = "Almost there! Just let me get my LLM sorted out..."
    speak_text(text, voice)


if __name__ == "__main__":
    sys.exit(main())
