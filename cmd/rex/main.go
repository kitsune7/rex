// Command rex is the Go entry point for the Rex voice assistant.
//
// Stages 1–2 of the Go port: this binary wires up config, audio, the event
// bus, the SQLite reminder store, and the timer manager, then waits for
// SIGINT/SIGTERM before tearing everything down. Later stages will plug in
// wake-word detection, STT, TTS, the agent, and the state machine.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"rex/internal/app"
	"rex/internal/core"
)

func main() {
	os.Exit(run())
}

func run() int {
	settingsPath := flag.String("settings", "settings.toml", "path to settings.toml")
	flag.Parse()

	a, err := app.New(*settingsPath)
	if err != nil {
		log.Printf("rex: startup failed: %v", err)
		return 1
	}
	defer a.Cleanup()

	logLoadedSettings(a)
	subscribeEventLogs(a)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	log.Print("rex: ready — press Ctrl-C to exit")

	<-ctx.Done()

	log.Print("rex: signal received, shutting down")
	return 0
}

func logLoadedSettings(a *app.App) {
	s := a.Context.Settings
	log.Printf("rex: settings loaded — listening_timeout=%.1fs retry_minutes=%d wake_word=%q reminders_db=%q",
		s.ListeningTimeout, s.Reminders.RetryMinutes, s.WakeWord.DisplayName, s.Reminders.DBPath)
	if a.WakeWordListener() != nil {
		log.Print("rex: wake-word detection ready")
	} else {
		log.Print("rex: wake-word detection disabled (see earlier log for reason)")
	}
	if a.Transcriber() != nil {
		log.Print("rex: speech-to-text ready")
	} else {
		log.Print("rex: speech-to-text disabled (see earlier log for reason)")
	}
}

// subscribeEventLogs attaches lightweight observability for the events that
// Stage 2 now produces. Stage 7 will replace these with real state handlers.
func subscribeEventLogs(a *app.App) {
	bus := a.Context.EventBus
	bus.Subscribe("ReminderScheduleChanged", func(core.Event) {
		log.Print("rex: reminder schedule changed")
	})
	bus.Subscribe("TimerFired", func(e core.Event) {
		if t, ok := e.(core.TimerFired); ok {
			log.Printf("rex: timer fired — %q", t.Name)
		}
	})
	bus.Subscribe("TimerStopped", func(e core.Event) {
		if t, ok := e.(core.TimerStopped); ok {
			log.Printf("rex: timer stopped — %q", t.Name)
		}
	})
}
