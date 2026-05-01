package app

import (
	"os"
	"path/filepath"
	"testing"

	"rex/internal/config"
)

func TestBuildContextUsesProvidedDependencies(t *testing.T) {
	settings := &config.Settings{
		ListeningTimeout: 4.2,
		Reminders:        config.ReminderSettings{RetryMinutes: 7},
		WakeWord:         config.WakeWordSettings{PathLabel: "hey_rex", DisplayName: "Hey Rex"},
	}

	ctx := BuildContext(settings, nil, nil, nil)

	if ctx.Settings != settings {
		t.Fatalf("settings not wired through: got %#v", ctx.Settings)
	}
	if ctx.EventBus == nil {
		t.Fatal("EventBus should be initialised")
	}
	if ctx.History != nil || ctx.IsInConversation() {
		t.Fatalf("expected fresh conversation state, got History=%v", ctx.History)
	}
}

func TestNewLoadsSettingsFromTempFile(t *testing.T) {
	if testing.Short() {
		t.Skip("requires PortAudio; skipped in -short mode")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "settings.toml")
	dbPath := filepath.Join(dir, "reminders.db")
	body := `
listening_timeout = 2.5

[reminders]
retry_minutes = 15
db_path = "` + dbPath + `"

[wake_word]
path_label = "hey_rex"
display_name = "Hey Rex"
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("writing settings: %v", err)
	}

	a, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(a.Cleanup)

	s := a.Context.Settings
	if s.ListeningTimeout != 2.5 {
		t.Errorf("ListeningTimeout = %v, want 2.5", s.ListeningTimeout)
	}
	if s.Reminders.RetryMinutes != 15 {
		t.Errorf("RetryMinutes = %d, want 15", s.Reminders.RetryMinutes)
	}
	if s.WakeWord.DisplayName != "Hey Rex" {
		t.Errorf("DisplayName = %q, want %q", s.WakeWord.DisplayName, "Hey Rex")
	}
	if a.Context.Audio == nil {
		t.Error("Audio player should be wired")
	}
	if a.Context.EventBus == nil {
		t.Error("EventBus should be wired")
	}
	if a.Context.Timers == nil {
		t.Error("Timers should be wired")
	}
	if a.Context.Reminders == nil {
		t.Error("Reminders should be wired")
	}
}
