package tools

import (
	"testing"
	"time"
)

func TestParseDuration(t *testing.T) {
	tests := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		// Minutes
		{"5 minutes", 5 * time.Minute, true},
		{"1 minute", time.Minute, true},
		{"10 mins", 10 * time.Minute, true},
		{"2 min", 2 * time.Minute, true},
		{"3m", 3 * time.Minute, true},

		// Seconds
		{"30 seconds", 30 * time.Second, true},
		{"1 second", time.Second, true},
		{"45 secs", 45 * time.Second, true},
		{"15 sec", 15 * time.Second, true},
		{"10s", 10 * time.Second, true},

		// Hours
		{"1 hour", time.Hour, true},
		{"2 hours", 2 * time.Hour, true},
		{"1 hr", time.Hour, true},
		{"3 hrs", 3 * time.Hour, true},
		{"1h", time.Hour, true},

		// Combinations
		{"1 hour 30 minutes", 90 * time.Minute, true},
		{"1 minute 30 seconds", 90 * time.Second, true},
		{"2 hours 15 minutes 30 seconds", 2*time.Hour + 15*time.Minute + 30*time.Second, true},
		{"1h 30m", 90 * time.Minute, true},
		{"1m 30s", 90 * time.Second, true},

		// Decimal
		{"1.5 hours", 90 * time.Minute, true},
		{"2.5 minutes", 150 * time.Second, true},
		{"30.5 seconds", 30*time.Second + 500*time.Millisecond, true},

		// Bare number assumes minutes
		{"5", 5 * time.Minute, true},
		{"10", 10 * time.Minute, true},

		// Case insensitive
		{"5 MINUTES", 5 * time.Minute, true},
		{"30 Seconds", 30 * time.Second, true},
		{"1 HOUR", time.Hour, true},

		// Whitespace
		{"  5 minutes  ", 5 * time.Minute, true},
		{"1  hour", time.Hour, true},

		// Invalid
		{"", 0, false},
		{"invalid", 0, false},
		{"no numbers here", 0, false},

		// Zero duration
		{"0 minutes", 0, false},
		{"0", 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			got, ok := ParseDuration(tc.in)
			if ok != tc.ok {
				t.Fatalf("ParseDuration(%q) ok=%v, want %v", tc.in, ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("ParseDuration(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		in   time.Duration
		want string
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
		// Truncation to int seconds
		{30700 * time.Millisecond, "30 seconds"},
		{90900 * time.Millisecond, "1 minute 30 seconds"},
	}
	for _, tc := range tests {
		t.Run(tc.in.String(), func(t *testing.T) {
			if got := FormatDuration(tc.in); got != tc.want {
				t.Errorf("FormatDuration(%v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
