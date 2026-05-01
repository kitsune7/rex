// Package app wires Rex's top-level dependencies together and exposes a single
// Cleanup entry point. It is the glue between config, audio, reminders,
// timers, and core.
package app

import (
	"fmt"
	"log"
	"path/filepath"

	"rex/internal/audio"
	"rex/internal/config"
	"rex/internal/core"
	"rex/internal/reminders"
	"rex/internal/stt"
	"rex/internal/timers"
	"rex/internal/wakeword"
)

// App owns the long-lived managers and the AppContext that state handlers use.
// Callers construct an App via New, run whatever they need with App.Context,
// and defer App.Cleanup.
type App struct {
	Context *core.AppContext

	audioMgr      *audio.Manager
	reminderStore *reminders.Store
	timerMgr      *timers.Manager
	wakeListener  *wakeword.Listener
	wakeMonitor   *wakeword.Monitor
	transcriber   *stt.Transcriber
}

// WakeWordListener returns the wake-word listener, or nil if wake-word
// support failed to initialise (e.g. missing model files). Stage 7 handlers
// will drive it from their state machine.
func (a *App) WakeWordListener() *wakeword.Listener { return a.wakeListener }

// WakeWordMonitor returns the background wake-word monitor used for
// barge-in during TTS playback, or nil if wake-word support is disabled.
func (a *App) WakeWordMonitor() *wakeword.Monitor { return a.wakeMonitor }

// Transcriber returns the speech-to-text transcriber, or nil if no
// whisper model was configured. Stage 7 listening handlers will drive
// it once available.
func (a *App) Transcriber() *stt.Transcriber { return a.transcriber }

// New loads settings from settingsPath and initialises the managers that
// Stages 1–2 require: audio output, event bus, reminder store, and the
// timer manager. Later stages will extend this to wake word, STT, TTS, and
// the agent.
func New(settingsPath string) (*App, error) {
	settings, err := config.LoadSettings(settingsPath)
	if err != nil {
		return nil, fmt.Errorf("loading settings: %w", err)
	}

	audioMgr, err := audio.NewManager()
	if err != nil {
		return nil, fmt.Errorf("creating audio manager: %w", err)
	}

	bus := core.NewEventBus()

	reminderStore, err := reminders.Open(settings.Reminders.DBPath, bus)
	if err != nil {
		audioMgr.Cleanup()
		return nil, fmt.Errorf("opening reminder store: %w", err)
	}

	timerMgr := timers.New(timers.Options{
		Bus:       bus,
		Player:    audioMgr,
		SoundPath: settings.Timers.SoundPath,
	})

	ctx := BuildContext(settings, audioMgr, timerMgr, reminderStore)
	ctx.EventBus = bus

	app := &App{
		Context:       ctx,
		audioMgr:      audioMgr,
		reminderStore: reminderStore,
		timerMgr:      timerMgr,
	}

	// Wire wake-word detection. Absent model files are non-fatal so the
	// binary still boots and exercises the rest of the wiring — callers
	// can tell wake-word failed because WakeWordListener() returns nil.
	if listener, monitor, err := buildWakeWord(settings, audioMgr, settingsPath); err != nil {
		log.Printf("rex: wake-word disabled: %v", err)
	} else {
		app.wakeListener = listener
		app.wakeMonitor = monitor
	}

	// Wire STT. Same degradation story: if the model isn't configured
	// or whisper-server can't start, we log and keep going so the
	// other stages remain exercisable.
	if settings.STT.ModelPath == "" {
		log.Print("rex: stt disabled (stt.model_path not set)")
	} else {
		transcriber, err := stt.NewTranscriber(stt.Options{
			ModelPath: settings.STT.ModelPath,
			Binary:    settings.STT.Binary,
			Language:  settings.STT.Language,
		})
		if err != nil {
			log.Printf("rex: stt disabled: %v", err)
		} else {
			app.transcriber = transcriber
		}
	}

	return app, nil
}

// buildWakeWord loads the wake-word listener and the background monitor.
// settingsPath is used to locate the model directory relative to the
// project root (the directory containing settings.toml).
func buildWakeWord(settings *config.Settings, audioMgr *audio.Manager, settingsPath string) (*wakeword.Listener, *wakeword.Monitor, error) {
	root := projectRoot(settingsPath)
	paths := wakeword.DefaultModelPaths(root, settings.WakeWord.PathLabel)
	if err := paths.Verify(); err != nil {
		return nil, nil, err
	}

	listener, err := wakeword.NewListener(wakeword.ListenerOptions{
		Models:    paths,
		Tones:     audioMgr,
		Threshold: settings.WakeWord.Threshold,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("building listener: %w", err)
	}

	monitor, err := wakeword.NewMonitor(wakeword.MonitorOptions{
		Models:    paths,
		Threshold: settings.WakeWord.Threshold,
	})
	if err != nil {
		listener.Close()
		return nil, nil, fmt.Errorf("building monitor: %w", err)
	}

	return listener, monitor, nil
}

// projectRoot returns the directory containing settingsPath, falling back
// to "." when settingsPath is not absolute.
func projectRoot(settingsPath string) string {
	if abs, err := filepath.Abs(settingsPath); err == nil {
		return filepath.Dir(abs)
	}
	return filepath.Dir(settingsPath)
}

// BuildContext assembles an AppContext from already-constructed dependencies.
// It has no I/O, so tests can exercise it with fakes. The returned context
// owns a fresh EventBus; callers that want a shared bus should overwrite
// ctx.EventBus after construction.
func BuildContext(settings *config.Settings, audio core.AudioPlayer, timers core.TimerController, reminders core.ReminderStore) *core.AppContext {
	return &core.AppContext{
		Audio:     audio,
		Timers:    timers,
		Reminders: reminders,
		Settings:  settings,
		EventBus:  core.NewEventBus(),
	}
}

// Cleanup releases all resources owned by the App. Safe to call once.
func (a *App) Cleanup() {
	if a == nil {
		return
	}
	if a.transcriber != nil {
		if err := a.transcriber.Close(); err != nil {
			log.Printf("rex: closing transcriber: %v", err)
		}
		a.transcriber = nil
	}
	if a.wakeMonitor != nil {
		a.wakeMonitor.Close()
		a.wakeMonitor = nil
	}
	if a.wakeListener != nil {
		a.wakeListener.Close()
		a.wakeListener = nil
	}
	if a.timerMgr != nil {
		a.timerMgr.Cleanup()
		a.timerMgr = nil
	}
	if a.reminderStore != nil {
		if err := a.reminderStore.Close(); err != nil {
			log.Printf("rex: closing reminder store: %v", err)
		}
		a.reminderStore = nil
	}
	if a.audioMgr != nil {
		a.audioMgr.Cleanup()
		a.audioMgr = nil
	}
	log.Print("rex: shutdown complete")
}
