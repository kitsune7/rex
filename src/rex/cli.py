"""
Main CLI entry point for Rex voice assistant.

Uses the state machine architecture for conversation flow management.
"""

import sys
import threading
from pathlib import Path

from agent import initialize_agent
from core import StateMachine, create_app_context
from rex.reminder_scheduler import ReminderDelivery, ReminderScheduler
from rex.states import create_all_handlers
from stt import Transcriber
from tts import InterruptibleSpeaker, load_voice
from wake_word import WakeWordListener


def main():
    """Main voice assistant entry point using state machine architecture."""
    print("🚀 Starting Rex Voice Assistant...")

    ctx = create_app_context()

    # Initialize the agent with managers from context
    initialize_agent(ctx.timer_manager, ctx.reminder_manager)

    model_path = Path(
        f"models/wake_word_models/{ctx.settings.wake_word.path_label}/{ctx.settings.wake_word.path_label}.onnx",
    )
    if not model_path.exists():
        print(f"❌ Error: Wake word model not found at {model_path}")
        return 1

    print("Loading models...")
    listener = WakeWordListener(
        model_path=str(model_path),
        audio_manager=ctx.audio_manager,
        threshold=0.5,
    )
    voice = load_voice()
    transcriber = Transcriber()
    speaker = InterruptibleSpeaker(
        voice=voice,
        audio_manager=ctx.audio_manager,
        model_path=str(model_path),
    )

    # Initialize reminder scheduler with interrupt callback
    reminder_interrupt = threading.Event()

    def on_reminder_due(delivery: ReminderDelivery):
        """Callback when a reminder is due - signal the main loop."""
        reminder_interrupt.set()
        listener.interrupt()  # Thread-safe interrupt

    scheduler = ReminderScheduler(
        reminder_manager=ctx.reminder_manager,
        reminder_settings=ctx.settings.reminders,
        on_reminder_due=on_reminder_due,
        event_bus=ctx.event_bus,
        audio_manager=ctx.audio_manager,
    )
    scheduler.start()

    handlers = create_all_handlers(
        listener=listener,
        transcriber=transcriber,
        speaker=speaker,
        voice=voice,
        scheduler=scheduler,
    )

    state_machine = StateMachine(ctx, handlers)

    print("\n✅ Rex is ready!")
    print("   Press Ctrl+C to exit\n")

    try:
        state_machine.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Rex...")
    finally:
        scheduler.stop()
        listener.stop()
        if ctx.timer_manager:
            ctx.timer_manager.cleanup()
        if ctx.audio_manager:
            ctx.audio_manager.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
