import sys
from tts import load_voice, speak_text


def main():
    speak_text("Testing, testing...", load_voice())


if __name__ == "__main__":
    sys.exit(main())
