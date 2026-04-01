package tools

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	// Patterns for relative durations: "in 2 hours", "in 30 minutes"
	relativeRe = regexp.MustCompile(`(?i)^in\s+(.+)$`)

	// Pattern for "tomorrow at TIME"
	tomorrowRe = regexp.MustCompile(`(?i)^tomorrow(?:\s+at\s+(.+))?$`)

	// Pattern for "today at TIME"
	todayRe = regexp.MustCompile(`(?i)^today(?:\s+at\s+(.+))?$`)

	// Pattern for "next WEEKDAY at TIME"
	nextWeekdayRe = regexp.MustCompile(`(?i)^next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+at\s+(.+))?$`)

	// Pattern for time: "3pm", "3:30 PM", "15:00", "3:30pm"
	timeRe = regexp.MustCompile(`(?i)^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.?|p\.m\.?)?$`)

	// Pattern for absolute date: "March 15 at 2:30 PM", "December 25th at 9am"
	dateWithTimeRe = regexp.MustCompile(`(?i)^(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:at\s+)?(.+)$`)

	// Pattern for ISO-ish date: "2025-12-25 14:30"
	isoDateRe = regexp.MustCompile(`^(\d{4})-(\d{1,2})-(\d{1,2})\s+(.+)$`)

	// Pattern for "DATE at TIME" where DATE is like "2025-12-25"
	isoDateAtTimeRe = regexp.MustCompile(`^(\d{4}-\d{1,2}-\d{1,2})\s+at\s+(.+)$`)

	// Explicit AM/PM check
	ampmCheckRe = regexp.MustCompile(`(?i)(am|pm|a\.m\.?|p\.m\.?)`)
)

// weekdayMap maps day names to time.Weekday values.
var weekdayMap = map[string]time.Weekday{
	"sunday":    time.Sunday,
	"monday":    time.Monday,
	"tuesday":   time.Tuesday,
	"wednesday": time.Wednesday,
	"thursday":  time.Thursday,
	"friday":    time.Friday,
	"saturday":  time.Saturday,
}

// monthMap maps month names to time.Month values.
var monthMap = map[string]time.Month{
	"january":   time.January,
	"february":  time.February,
	"march":     time.March,
	"april":     time.April,
	"may":       time.May,
	"june":      time.June,
	"july":      time.July,
	"august":    time.August,
	"september": time.September,
	"october":   time.October,
	"november":  time.November,
	"december":  time.December,
}

// ParseDatetime parses a natural language date/time string into a time.Time.
//
// Supported formats:
//   - "tomorrow at 3pm"
//   - "next Tuesday at noon"
//   - "in 2 hours"
//   - "March 15 at 2:30 PM"
//   - "noon", "midnight"
//   - "3pm", "3:30 PM"
//   - "2025-12-25 14:30"
//   - AM/PM disambiguation: if time is ambiguous and in the past, picks the soonest future interpretation
func ParseDatetime(s string) (time.Time, error) {
	return parseDatetimeRelativeTo(s, time.Now())
}

// parseDatetimeRelativeTo is the internal implementation that accepts a "now" parameter for testing.
func parseDatetimeRelativeTo(s string, now time.Time) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, fmt.Errorf("empty datetime string")
	}

	original := strings.ToLower(s)

	// Handle "noon" and "midnight" as standalone
	if original == "noon" {
		t := todayAt(now, 12, 0)
		return disambiguateTime(t, now, true), nil
	}
	if original == "midnight" {
		t := todayAt(now, 0, 0)
		// Midnight is always tomorrow if it's past midnight today
		if !t.After(now) {
			t = t.Add(24 * time.Hour)
		}
		return t, nil
	}

	// Handle "in X hours/minutes" (relative)
	if m := relativeRe.FindStringSubmatch(original); m != nil {
		dur, err := ParseDuration(m[1])
		if err != nil {
			return time.Time{}, fmt.Errorf("could not parse relative duration: %w", err)
		}
		return now.Add(dur), nil
	}

	// Handle "tomorrow at TIME"
	if m := tomorrowRe.FindStringSubmatch(original); m != nil {
		tomorrow := now.AddDate(0, 0, 1)
		if m[1] != "" {
			h, min, err := parseTimeString(m[1])
			if err != nil {
				return time.Time{}, err
			}
			return time.Date(tomorrow.Year(), tomorrow.Month(), tomorrow.Day(), h, min, 0, 0, now.Location()), nil
		}
		return time.Date(tomorrow.Year(), tomorrow.Month(), tomorrow.Day(), 9, 0, 0, 0, now.Location()), nil
	}

	// Handle "today at TIME"
	if m := todayRe.FindStringSubmatch(original); m != nil {
		if m[1] != "" {
			h, min, err := parseTimeString(m[1])
			if err != nil {
				return time.Time{}, err
			}
			t := time.Date(now.Year(), now.Month(), now.Day(), h, min, 0, 0, now.Location())
			return disambiguateTime(t, now, hasExplicitAMPM(m[1])), nil
		}
		return time.Time{}, fmt.Errorf("today requires a time")
	}

	// Handle "next WEEKDAY at TIME"
	if m := nextWeekdayRe.FindStringSubmatch(original); m != nil {
		dayName := strings.ToLower(m[1])
		targetDay, ok := weekdayMap[dayName]
		if !ok {
			return time.Time{}, fmt.Errorf("unknown weekday: %s", dayName)
		}

		// Find next occurrence of that weekday
		daysUntil := (int(targetDay) - int(now.Weekday()) + 7) % 7
		if daysUntil == 0 {
			daysUntil = 7 // Always next week
		}
		target := now.AddDate(0, 0, daysUntil)

		h, min := 9, 0 // default to 9am
		if m[2] != "" {
			var err error
			h, min, err = parseTimeString(m[2])
			if err != nil {
				return time.Time{}, err
			}
		}
		return time.Date(target.Year(), target.Month(), target.Day(), h, min, 0, 0, now.Location()), nil
	}

	// Handle ISO date with "at": "2025-12-25 at 10am"
	if m := isoDateAtTimeRe.FindStringSubmatch(s); m != nil {
		datePart, err := time.Parse("2006-1-2", m[1])
		if err != nil {
			return time.Time{}, fmt.Errorf("could not parse date: %w", err)
		}
		h, min, err := parseTimeString(m[2])
		if err != nil {
			return time.Time{}, err
		}
		return time.Date(datePart.Year(), datePart.Month(), datePart.Day(), h, min, 0, 0, now.Location()), nil
	}

	// Handle ISO date: "2025-12-25 14:30"
	if m := isoDateRe.FindStringSubmatch(s); m != nil {
		year, _ := strconv.Atoi(m[1])
		month, _ := strconv.Atoi(m[2])
		day, _ := strconv.Atoi(m[3])
		h, min, err := parseTimeString(m[4])
		if err != nil {
			return time.Time{}, err
		}
		return time.Date(year, time.Month(month), day, h, min, 0, 0, now.Location()), nil
	}

	// Handle "Month Day at Time": "December 25th at 10am", "March 15 at 2:30 PM"
	if m := dateWithTimeRe.FindStringSubmatch(original); m != nil {
		monthName := strings.ToLower(m[1])
		month, ok := monthMap[monthName]
		if ok {
			day, err := strconv.Atoi(m[2])
			if err != nil {
				return time.Time{}, fmt.Errorf("could not parse day: %w", err)
			}
			h, min, err := parseTimeString(m[3])
			if err != nil {
				return time.Time{}, err
			}
			// Use current year, or next year if the date has passed
			year := now.Year()
			t := time.Date(year, month, day, h, min, 0, 0, now.Location())
			if t.Before(now) {
				t = time.Date(year+1, month, day, h, min, 0, 0, now.Location())
			}
			return t, nil
		}
	}

	// Handle time-only: "3pm", "3:30 PM"
	h, min, err := parseTimeString(original)
	if err == nil {
		t := todayAt(now, h, min)
		return disambiguateTime(t, now, hasExplicitAMPM(original)), nil
	}

	return time.Time{}, fmt.Errorf("could not parse datetime: %q", s)
}

// parseTimeString parses a time string like "3pm", "3:30 PM", "15:00", "noon", "midnight".
func parseTimeString(s string) (hour, minute int, err error) {
	s = strings.TrimSpace(strings.ToLower(s))

	// Handle special words
	if s == "noon" {
		return 12, 0, nil
	}
	if s == "midnight" {
		return 0, 0, nil
	}

	m := timeRe.FindStringSubmatch(s)
	if m == nil {
		return 0, 0, fmt.Errorf("could not parse time: %q", s)
	}

	hour, _ = strconv.Atoi(m[1])
	if m[2] != "" {
		minute, _ = strconv.Atoi(m[2])
	}

	ampm := strings.TrimRight(strings.ToLower(m[3]), ".")
	switch ampm {
	case "pm", "p.m":
		if hour != 12 {
			hour += 12
		}
	case "am", "a.m":
		if hour == 12 {
			hour = 0
		}
	}

	if hour > 23 || minute > 59 {
		return 0, 0, fmt.Errorf("invalid time: %q", s)
	}
	return hour, minute, nil
}

// hasExplicitAMPM returns true if the string contains an explicit AM/PM marker.
func hasExplicitAMPM(s string) bool {
	return ampmCheckRe.MatchString(s)
}

// todayAt returns today's date at the given hour and minute.
func todayAt(now time.Time, hour, minute int) time.Time {
	return time.Date(now.Year(), now.Month(), now.Day(), hour, minute, 0, 0, now.Location())
}

// disambiguateTime resolves ambiguous times by picking the soonest future time.
// If the time has explicit AM/PM, it is returned as-is (but moved to tomorrow if in the past).
// If ambiguous, it tries both AM and PM versions and picks the soonest future one.
func disambiguateTime(t time.Time, now time.Time, explicitAMPM bool) time.Time {
	if explicitAMPM {
		if t.After(now) {
			return t
		}
		return t.Add(24 * time.Hour)
	}

	// Ambiguous: try AM and PM versions
	hour := t.Hour()
	amHour := hour % 12
	pmHour := amHour + 12

	am := time.Date(t.Year(), t.Month(), t.Day(), amHour, t.Minute(), 0, 0, t.Location())
	pm := time.Date(t.Year(), t.Month(), t.Day(), pmHour, t.Minute(), 0, 0, t.Location())

	candidates := []time.Time{am, pm, am.Add(24 * time.Hour), pm.Add(24 * time.Hour)}

	var best time.Time
	for _, c := range candidates {
		if c.After(now) {
			if best.IsZero() || c.Before(best) {
				best = c
			}
		}
	}

	if !best.IsZero() {
		return best
	}
	return t
}
