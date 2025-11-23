import argparse
import sys

from agent import run_voice_agent, run_text_agent
from tts import load_voice, speak_text
from stt import get_transcription


def main():
    parser = argparse.ArgumentParser(description="Rex AI Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("start", help="Start Rex AI Assistant")

    query_parser = subparsers.add_parser("query", help="Query Rex directly")
    query_parser.add_argument("text", help="Text to pass to Rex")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start()
    elif args.command == "query":
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


def cmd_start():
    response = run_voice_agent(get_transcription(debug=True))
    print(response)


def cmd_query(args):
    response = run_text_agent(args.text)
    print(response)


if __name__ == "__main__":
    sys.exit(main())
