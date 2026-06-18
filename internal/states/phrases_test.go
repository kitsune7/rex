package states

import (
	"testing"
	"time"
)

func TestIsConfirmation(t *testing.T) {
	positives := []string{
		"yes", "Yeah", " yep ", "Sure thing", "okay",
		"OK", "go ahead", "Do it", "please", "correct",
		"right", "affirmative", "confirm", "absolutely", "definitely",
		"Yes, go ahead", "  YEAH  ",
	}
	for _, s := range positives {
		if !IsConfirmation(s) {
			t.Errorf("IsConfirmation(%q) = false, want true", s)
		}
	}

	negatives := []string{
		"no", "nah", "nope", "banana", "",
	}
	for _, s := range negatives {
		if IsConfirmation(s) {
			t.Errorf("IsConfirmation(%q) = true, want false", s)
		}
	}
}

func TestIsRejection(t *testing.T) {
	positives := []string{
		"no", "Nah", " nope ", "don't do it", "cancel",
		"stop", "never mind", "forget it", "negative",
	}
	for _, s := range positives {
		if !IsRejection(s) {
			t.Errorf("IsRejection(%q) = false, want true", s)
		}
	}

	negatives := []string{
		"yes", "sure", "banana", "",
	}
	for _, s := range negatives {
		if IsRejection(s) {
			t.Errorf("IsRejection(%q) = true, want false", s)
		}
	}
}

func TestIsStopPhrase(t *testing.T) {
	positives := []string{
		"stop", "nevermind", "never mind", "forget it",
		"that's all", "nothing", "goodbye", "bye", "cancel",
		"  Stop  ", "GOODBYE",
	}
	for _, s := range positives {
		if !IsStopPhrase(s) {
			t.Errorf("IsStopPhrase(%q) = false, want true", s)
		}
	}

	negatives := []string{
		"stop the timer", "forget it all", "bye bye", "yes", "",
	}
	for _, s := range negatives {
		if IsStopPhrase(s) {
			t.Errorf("IsStopPhrase(%q) = true, want false", s)
		}
	}
}

func TestParseSnoozeDuration(t *testing.T) {
	tests := []struct {
		input string
		want  time.Duration
		ok    bool
	}{
		{"snooze for 5 minutes", 5 * time.Minute, true},
		{"remind me in 10 minutes", 10 * time.Minute, true},
		{"remind me again in 30 min", 30 * time.Minute, true},
		{"delay it 15 minutes", 15 * time.Minute, true},
		{"postpone 20 min", 20 * time.Minute, true},
		{"tell me in 3 minutes", 3 * time.Minute, true},
		{"5 minutes later", 5 * time.Minute, true},
		{"10 min from now", 10 * time.Minute, true},
		{"yes", 0, false},
		{"no thanks", 0, false},
		{"", 0, false},
	}
	for _, tt := range tests {
		got, ok := ParseSnoozeDuration(tt.input)
		if ok != tt.ok || got != tt.want {
			t.Errorf("ParseSnoozeDuration(%q) = (%v, %v), want (%v, %v)",
				tt.input, got, ok, tt.want, tt.ok)
		}
	}
}
