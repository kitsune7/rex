package tools

import (
	"testing"
	"time"
)

func TestFormatCurrentTime(t *testing.T) {
	tests := []struct {
		name string
		when time.Time
		want string
	}{
		{"morning", time.Date(2026, 4, 1, 9, 5, 0, 0, time.Local), "9:05 AM"},
		{"noon", time.Date(2026, 4, 1, 12, 0, 0, 0, time.Local), "12:00 PM"},
		{"afternoon", time.Date(2026, 4, 1, 15, 42, 0, 0, time.Local), "3:42 PM"},
		{"midnight", time.Date(2026, 4, 1, 0, 0, 0, 0, time.Local), "12:00 AM"},
		{"one past midnight", time.Date(2026, 4, 1, 0, 1, 0, 0, time.Local), "12:01 AM"},
		{"11 pm", time.Date(2026, 4, 1, 23, 7, 0, 0, time.Local), "11:07 PM"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := FormatCurrentTime(tc.when); got != tc.want {
				t.Errorf("FormatCurrentTime(%v) = %q, want %q", tc.when, got, tc.want)
			}
		})
	}
}
