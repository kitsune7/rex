// Package tools provides the pure-logic helpers used by the LLM agent:
// a safe math expression evaluator, a clock tool, and natural-language
// parsers for durations and datetimes.
package tools

import (
	"fmt"
	"time"
)

// GetCurrentTime returns the current local time formatted as "3:04 PM".
func GetCurrentTime() string {
	return FormatCurrentTime(time.Now())
}

// FormatCurrentTime is the time-injectable form of GetCurrentTime, used by tests.
func FormatCurrentTime(t time.Time) string {
	return fmt.Sprintf("%d:%02d %s",
		hour12(t.Hour()), t.Minute(), amPm(t.Hour()))
}

func hour12(h int) int {
	h %= 12
	if h == 0 {
		return 12
	}
	return h
}

func amPm(h int) string {
	if h < 12 {
		return "AM"
	}
	return "PM"
}
