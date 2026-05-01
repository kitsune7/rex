// Package config loads application settings from settings.toml and environment
// variables.
package config

import (
	"errors"
	"fmt"
	"os"

	"github.com/joho/godotenv"
	"github.com/pelletier/go-toml/v2"
)

// Settings holds all application configuration loaded from settings.toml.
type Settings struct {
	ListeningTimeout float64          `toml:"listening_timeout"`
	Reminders        ReminderSettings `toml:"reminders"`
	Timers           TimerSettings    `toml:"timers"`
	WakeWord         WakeWordSettings `toml:"wake_word"`
	STT              STTSettings      `toml:"stt"`
}

// ReminderSettings controls reminder behaviour.
type ReminderSettings struct {
	RetryMinutes int    `toml:"retry_minutes"`
	DBPath       string `toml:"db_path"`
}

// TimerSettings controls timer behaviour.
type TimerSettings struct {
	SoundPath string `toml:"sound_path"`
}

// WakeWordSettings controls wake-word detection.
type WakeWordSettings struct {
	PathLabel   string  `toml:"path_label"`
	DisplayName string  `toml:"display_name"`
	Threshold   float32 `toml:"threshold"`
}

// STTSettings controls speech-to-text transcription. ModelPath points
// at a ggml whisper model; when empty the transcriber is disabled and
// later stages will log that fact. Binary overrides the whisper-server
// executable (defaults to "whisper-server" on $PATH).
type STTSettings struct {
	ModelPath string `toml:"model_path"`
	Binary    string `toml:"binary"`
	Language  string `toml:"language"`
}

// LoadSettings reads settings.toml at the given path and returns a Settings
// value with defaults applied for any missing fields. It also loads a .env
// file from the working directory if one exists.
func LoadSettings(path string) (*Settings, error) {
	// Best-effort .env loading; ignore errors if the file is absent.
	_ = godotenv.Load()

	s := &Settings{
		ListeningTimeout: 6.0,
		Reminders: ReminderSettings{
			RetryMinutes: 10,
			DBPath:       "data/reminders.db",
		},
		Timers: TimerSettings{
			SoundPath: "sounds/fun-timer.mp3",
		},
		WakeWord: WakeWordSettings{
			PathLabel:   "hey_rex",
			DisplayName: "Hey Rex",
			Threshold:   0.5,
		},
		STT: STTSettings{
			Language: "en",
		},
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// No config file; return defaults.
			return s, nil
		}
		return nil, fmt.Errorf("reading settings file: %w", err)
	}

	if err := toml.Unmarshal(data, s); err != nil {
		return nil, fmt.Errorf("parsing settings file: %w", err)
	}

	return s, nil
}
