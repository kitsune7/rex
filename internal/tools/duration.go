package tools

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	hoursRe   = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)`)
	minutesRe = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)(?:[^s]|$)`)
	secondsRe = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)`)
)

// ParseDuration parses a natural language duration string into a time.Duration.
//
// Supported formats:
//   - "5 minutes", "1 hour", "90 seconds"
//   - "1 hour 30 minutes", "2 hours and 15 minutes"
//   - "1h 30m", "5m", "10s"
//   - Bare numbers are treated as minutes: "5" -> 5 minutes
func ParseDuration(s string) (time.Duration, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return 0, fmt.Errorf("empty duration string")
	}

	// Remove "and" connectors
	s = strings.ReplaceAll(s, " and ", " ")

	var totalSeconds float64
	found := false

	if m := hoursRe.FindStringSubmatch(s); m != nil {
		v, _ := strconv.ParseFloat(m[1], 64)
		totalSeconds += v * 3600
		found = true
	}

	if m := minutesRe.FindStringSubmatch(s); m != nil {
		v, _ := strconv.ParseFloat(m[1], 64)
		totalSeconds += v * 60
		found = true
	}

	if m := secondsRe.FindStringSubmatch(s); m != nil {
		v, _ := strconv.ParseFloat(m[1], 64)
		totalSeconds += v
		found = true
	}

	// If nothing matched, try bare number (assume minutes)
	if !found {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, fmt.Errorf("could not parse duration: %q", s)
		}
		totalSeconds = v * 60
		found = true
	}

	if totalSeconds <= 0 {
		return 0, fmt.Errorf("duration must be positive")
	}

	return time.Duration(totalSeconds * float64(time.Second)), nil
}

// FormatDuration formats a duration in human-readable form.
func FormatDuration(d time.Duration) string {
	totalSecs := int(d.Seconds())
	if totalSecs < 60 {
		return pluralize(totalSecs, "second")
	}
	if totalSecs < 3600 {
		mins := totalSecs / 60
		secs := totalSecs % 60
		parts := []string{pluralize(mins, "minute")}
		if secs > 0 {
			parts = append(parts, pluralize(secs, "second"))
		}
		return strings.Join(parts, " ")
	}

	hours := totalSecs / 3600
	mins := (totalSecs % 3600) / 60
	parts := []string{pluralize(hours, "hour")}
	if mins > 0 {
		parts = append(parts, pluralize(mins, "minute"))
	}
	return strings.Join(parts, " ")
}

func pluralize(n int, unit string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, unit)
	}
	return fmt.Sprintf("%d %ss", n, unit)
}
