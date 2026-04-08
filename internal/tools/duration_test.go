package tools

import (
	"testing"
	"time"
)

func TestParseDuration(t *testing.T) {
	tests := []struct {
		input    string
		expected time.Duration
	}{
		// Minutes
		{"5 minutes", 5 * time.Minute},
		{"1 minute", 1 * time.Minute},
		{"10 mins", 10 * time.Minute},
		{"2 min", 2 * time.Minute},
		{"3m", 3 * time.Minute},

		// Seconds
		{"30 seconds", 30 * time.Second},
		{"1 second", 1 * time.Second},
		{"45 secs", 45 * time.Second},
		{"15 sec", 15 * time.Second},
		{"10s", 10 * time.Second},

		// Hours
		{"1 hour", 1 * time.Hour},
		{"2 hours", 2 * time.Hour},
		{"1 hr", 1 * time.Hour},
		{"3 hrs", 3 * time.Hour},
		{"1h", 1 * time.Hour},

		// Combined
		{"1 hour 30 minutes", 90 * time.Minute},
		{"1 minute 30 seconds", 90 * time.Second},
		{"2 hours 15 minutes 30 seconds", 2*time.Hour + 15*time.Minute + 30*time.Second},
		{"1h 30m", 90 * time.Minute},
		{"1m 30s", 90 * time.Second},

		// Decimal
		{"1.5 hours", 90 * time.Minute},
		{"2.5 minutes", 150 * time.Second},

		// Bare number (minutes)
		{"5", 5 * time.Minute},
		{"10", 10 * time.Minute},

		// Case insensitive
		{"5 MINUTES", 5 * time.Minute},
		{"30 Seconds", 30 * time.Second},
		{"1 HOUR", 1 * time.Hour},

		// Whitespace
		{"  5 minutes  ", 5 * time.Minute},

		// "and" connector
		{"2 hours and 15 minutes", 2*time.Hour + 15*time.Minute},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got, err := ParseDuration(tc.input)
			if err != nil {
				t.Fatalf("ParseDuration(%q) returned error: %v", tc.input, err)
			}
			if got != tc.expected {
				t.Errorf("ParseDuration(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestParseDurationErrors(t *testing.T) {
	errorCases := []string{
		"",
		"invalid",
		"no numbers here",
		"0 minutes",
		"0",
	}

	for _, input := range errorCases {
		t.Run(input, func(t *testing.T) {
			_, err := ParseDuration(input)
			if err == nil {
				t.Errorf("ParseDuration(%q) expected error, got nil", input)
			}
		})
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		input    time.Duration
		expected string
	}{
		{1 * time.Second, "1 second"},
		{30 * time.Second, "30 seconds"},
		{59 * time.Second, "59 seconds"},
		{60 * time.Second, "1 minute"},
		{120 * time.Second, "2 minutes"},
		{300 * time.Second, "5 minutes"},
		{90 * time.Second, "1 minute 30 seconds"},
		{125 * time.Second, "2 minutes 5 seconds"},
		{61 * time.Second, "1 minute 1 second"},
		{3600 * time.Second, "1 hour"},
		{7200 * time.Second, "2 hours"},
		{3660 * time.Second, "1 hour 1 minute"},
		{5400 * time.Second, "1 hour 30 minutes"},
		{7320 * time.Second, "2 hours 2 minutes"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			got := FormatDuration(tc.input)
			if got != tc.expected {
				t.Errorf("FormatDuration(%v) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}
