// Package main is the entry point for the Rex voice assistant.
//
// It wires all components together in dependency order: settings, audio,
// models, managers, agent, state handlers, and state machine. Graceful
// shutdown is handled via signal.NotifyContext on SIGINT/SIGTERM.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	// TODO: Uncomment when internal packages are implemented.
	// "fmt"
	// "rex/internal/agent"
	// "rex/internal/audio"
	// "rex/internal/config"
	// "rex/internal/core"
	// "rex/internal/scheduler"
	// "rex/internal/states"
	// "rex/internal/stt"
	// "rex/internal/tools"
	// "rex/internal/tts"
	// "rex/internal/wakeword"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal error", "err", err)
		os.Exit(1)
	}
}

// run contains the full initialisation and event loop. Returning an error
// from here causes the process to exit with code 1.
func run() error {
	// 1. Structured logging to stderr.
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})))

	// 2. Graceful shutdown context — cancelled on SIGINT or SIGTERM.
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	slog.Info("starting Rex voice assistant")

	// ---------------------------------------------------------------
	// 3. Load settings from settings.toml.
	// ---------------------------------------------------------------
	// TODO: Uncomment when config package exists.
	// settings, err := config.LoadSettings("settings.toml")
	// if err != nil {
	// 	return fmt.Errorf("load settings: %w", err)
	// }

	// ---------------------------------------------------------------
	// 4. Initialise PortAudio (process-wide).
	// ---------------------------------------------------------------
	// TODO: Uncomment when audio package exists.
	// if err := audio.InitPortAudio(); err != nil {
	// 	return fmt.Errorf("init portaudio: %w", err)
	// }
	// defer audio.TerminatePortAudio()

	// ---------------------------------------------------------------
	// 5. Create the central event bus.
	// ---------------------------------------------------------------
	// TODO: Uncomment when core package exists.
	// eventBus := core.NewEventBus()

	// ---------------------------------------------------------------
	// 6. Create the audio manager.
	// ---------------------------------------------------------------
	// TODO: Uncomment when audio package exists.
	// audioManager, err := audio.NewManager(eventBus)
	// if err != nil {
	// 	return fmt.Errorf("create audio manager: %w", err)
	// }
	// defer audioManager.Cleanup()

	// ---------------------------------------------------------------
	// 7. Create the timer manager.
	// ---------------------------------------------------------------
	// TODO: Uncomment when tools package exists.
	// timerManager := tools.NewTimerManager(audioManager, eventBus, "sounds/fun-timer.mp3")
	// defer timerManager.Cleanup()

	// ---------------------------------------------------------------
	// 8. Create the reminder manager (SQLite-backed).
	// ---------------------------------------------------------------
	// TODO: Uncomment when tools package exists.
	// reminderManager, err := tools.NewReminderManager("data/reminders.db", eventBus)
	// if err != nil {
	// 	return fmt.Errorf("create reminder manager: %w", err)
	// }

	// ---------------------------------------------------------------
	// 9. Load the STT model (Whisper via whisper.cpp).
	// ---------------------------------------------------------------
	// TODO: Uncomment when stt package exists.
	// transcriber, err := stt.NewTranscriber("models/whisper/ggml-small.bin")
	// if err != nil {
	// 	return fmt.Errorf("load whisper model: %w", err)
	// }

	// ---------------------------------------------------------------
	// 10. Load the TTS voice (Piper via ONNX Runtime).
	// ---------------------------------------------------------------
	// TODO: Uncomment when tts package exists.
	// voice, err := tts.NewVoice("models/piper/en_US-lessac-medium.onnx")
	// if err != nil {
	// 	return fmt.Errorf("load tts voice: %w", err)
	// }

	// ---------------------------------------------------------------
	// 11. Ensure wake word ONNX models are present.
	// ---------------------------------------------------------------
	// TODO: Uncomment when wakeword package exists.
	// wakeModelDir := "models/wake_word_models/hey_rex"
	// if err := wakeword.EnsureModels(wakeModelDir); err != nil {
	// 	return fmt.Errorf("ensure wake word models: %w", err)
	// }

	// ---------------------------------------------------------------
	// 12. Create the wake word listener (blocks until wake word heard).
	// ---------------------------------------------------------------
	// TODO: Uncomment when wakeword package exists.
	// listener, err := wakeword.NewListener(audioManager, wakeModelDir, 0.5)
	// if err != nil {
	// 	return fmt.Errorf("create wake word listener: %w", err)
	// }
	// defer listener.Stop()

	// ---------------------------------------------------------------
	// 13. Create the wake word monitor (used during TTS for interruption).
	// ---------------------------------------------------------------
	// TODO: Uncomment when wakeword package exists.
	// monitor, err := wakeword.NewMonitor(audioManager, wakeModelDir, 0.5)
	// if err != nil {
	// 	return fmt.Errorf("create wake word monitor: %w", err)
	// }

	// ---------------------------------------------------------------
	// 14. Create the interruptible speaker (TTS + wake word monitor).
	// ---------------------------------------------------------------
	// TODO: Uncomment when tts package exists.
	// speaker := tts.NewInterruptibleSpeaker(voice, audioManager, monitor)

	// ---------------------------------------------------------------
	// 15. Register agent tools.
	// ---------------------------------------------------------------
	// TODO: Uncomment when agent and tools packages exist.
	// agentTools := []agent.Tool{
	// 	tools.NewTimerSetTool(timerManager),
	// 	tools.NewTimerCheckTool(timerManager),
	// 	tools.NewTimerStopTool(timerManager),
	// 	tools.NewReminderCreateTool(reminderManager),
	// 	tools.NewReminderListTool(reminderManager),
	// 	tools.NewReminderUpdateTool(reminderManager),
	// 	tools.NewReminderDeleteTool(reminderManager),
	// 	tools.NewMathTool(),
	// 	tools.NewTimeTool(),
	// }

	// ---------------------------------------------------------------
	// 16. Create the LLM agent (talks to a local LLM server).
	// ---------------------------------------------------------------
	// TODO: Uncomment when agent package exists.
	// llmAgent, err := agent.NewAgent("http://localhost:1234/v1", "not-needed", agentTools)
	// if err != nil {
	// 	return fmt.Errorf("create agent: %w", err)
	// }

	// ---------------------------------------------------------------
	// 17. Start the reminder scheduler in the background.
	// ---------------------------------------------------------------
	// TODO: Uncomment when scheduler package exists.
	// reminderScheduler := scheduler.NewScheduler(
	// 	reminderManager,
	// 	audioManager,
	// 	eventBus,
	// 	settings.Reminders.RetryMinutes,
	// )
	// go reminderScheduler.Start(ctx)

	// ---------------------------------------------------------------
	// 18. Build state handler dependencies.
	// ---------------------------------------------------------------
	// TODO: Uncomment when states package exists.
	// deps := &states.Deps{
	// 	Listener:    listener,
	// 	Transcriber: transcriber,
	// 	Speaker:     speaker,
	// 	Voice:       voice,
	// 	Agent:       llmAgent,
	// 	Scheduler:   reminderScheduler,
	// 	EventBus:    eventBus,
	// }

	// ---------------------------------------------------------------
	// 19. Create all state handlers.
	// ---------------------------------------------------------------
	// TODO: Uncomment when states package exists.
	// handlers := states.CreateAllHandlers(deps)

	// ---------------------------------------------------------------
	// 20. Create and run the state machine.
	// ---------------------------------------------------------------
	// TODO: Uncomment when core package exists.
	// sm := core.NewStateMachine(handlers, core.StateWaitingForWakeWord)

	slog.Info("Rex voice assistant ready", "wake_word", "Hey Rex")

	// TODO: Replace placeholder select with sm.Run(ctx) once the state
	// machine package is available.
	// sm.Run(ctx)
	<-ctx.Done()

	slog.Info("Rex shutting down")
	return nil
}
