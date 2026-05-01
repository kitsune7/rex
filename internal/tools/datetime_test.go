package tools

import (
	"testing"
	"time"
)

// fixedNow is the anchor time used by date-relative tests: 2026-04-30 10:00
// local time, which is a Thursday. Wednesday = April 29, Tuesday = April 28.
var fixedNow = time.Date(2026, 4, 30, 10, 0, 0, 0, time.Local)

func TestParseDatetimeExplicit(t *testing.T) {
	got, ok := parseDatetimeAt("2025-12-25 14:30", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Month() != 12 || got.Day() != 25 || got.Hour() != 14 || got.Minute() != 30 {
		t.Errorf("got %v", got)
	}
}

func TestParseDatetimeTimeOnly(t *testing.T) {
	got, ok := parseDatetimeAt("3pm", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 15 {
		t.Errorf("hour = %d, want 15", got.Hour())
	}
}

func TestParseDatetimeTimeWithMinutes(t *testing.T) {
	got, ok := parseDatetimeAt("3:30 pm", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 15 || got.Minute() != 30 {
		t.Errorf("got %d:%02d, want 15:30", got.Hour(), got.Minute())
	}
}

func TestParseDatetimeTomorrow(t *testing.T) {
	got, ok := parseDatetimeAt("tomorrow at 9am", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	tomorrow := fixedNow.AddDate(0, 0, 1)
	if got.Day() != tomorrow.Day() || got.Hour() != 9 {
		t.Errorf("got %v, want day=%d hour=9", got, tomorrow.Day())
	}
}

func TestParseDatetimeMonthName(t *testing.T) {
	got, ok := parseDatetimeAt("December 25th at 10am", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Month() != 12 || got.Day() != 25 || got.Hour() != 10 {
		t.Errorf("got %v", got)
	}
}

func TestParseDatetimeNoon(t *testing.T) {
	got, ok := parseDatetimeAt("noon", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 12 {
		t.Errorf("hour = %d, want 12", got.Hour())
	}
}

func TestParseDatetimeMidnight(t *testing.T) {
	got, ok := parseDatetimeAt("midnight", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 0 {
		t.Errorf("hour = %d, want 0", got.Hour())
	}
}

func TestParseDatetimeNextWeekday(t *testing.T) {
	// fixedNow = Thursday 2026-04-30 → next Tuesday = 2026-05-05.
	got, ok := parseDatetimeAt("next Tuesday at noon", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Year() != 2026 || got.Month() != 5 || got.Day() != 5 || got.Hour() != 12 {
		t.Errorf("got %v, want 2026-05-05 12:00", got)
	}
}

func TestParseDatetimeEmptyReturnsFalse(t *testing.T) {
	if _, ok := parseDatetimeAt("", fixedNow); ok {
		t.Error("expected failure on empty string")
	}
	if _, ok := parseDatetimeAt("   ", fixedNow); ok {
		t.Error("expected failure on whitespace")
	}
}

func TestParseDatetimeAmbiguousPrefersFutureAM(t *testing.T) {
	// At 08:15, "8:45" should disambiguate to 08:45 AM (same day).
	now := time.Date(2026, 4, 30, 8, 15, 0, 0, time.Local)
	got, ok := parseDatetimeAt("8:45", now)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 8 || got.Minute() != 45 {
		t.Errorf("got %d:%02d, want 08:45", got.Hour(), got.Minute())
	}
}

func TestParseDatetimeAmbiguousPrefersFuturePM(t *testing.T) {
	// At 23:30, "3:05" should disambiguate to the next 3:05 AM (tomorrow).
	now := time.Date(2026, 4, 30, 23, 30, 0, 0, time.Local)
	got, ok := parseDatetimeAt("3:05", now)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Hour() != 3 || got.Minute() != 5 {
		t.Errorf("got %d:%02d, want 3:05", got.Hour(), got.Minute())
	}
	// Must be strictly after "now".
	if !got.After(now) {
		t.Errorf("got %v is not after now=%v", got, now)
	}
}

func TestParseDatetimeISOAtTime(t *testing.T) {
	got, ok := parseDatetimeAt("2026-05-15 at 10am", fixedNow)
	if !ok {
		t.Fatal("expected success")
	}
	if got.Year() != 2026 || got.Month() != 5 || got.Day() != 15 || got.Hour() != 10 {
		t.Errorf("got %v", got)
	}
}
