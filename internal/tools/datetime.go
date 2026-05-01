package tools

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ParseDatetime parses a natural-language datetime string. Returns the parsed
// time and true on success, or a zero time and false on failure.
//
// Supported forms (a subset of what Python's dateutil accepts, but covering
// the Python test corpus for Rex):
//   - "YYYY-MM-DD HH:MM"
//   - "YYYY-MM-DD at 10am"
//   - "tomorrow at 3pm", "today at 9am"
//   - "next Tuesday at noon"
//   - "December 25th at 10am", "Dec 25 at 10am"
//   - bare times: "3pm", "3:30 pm", "noon", "midnight"
//
// When no AM/PM marker is given on a bare time, the parser picks the soonest
// future am/pm interpretation, matching the Python behaviour.
func ParseDatetime(s string) (time.Time, bool) {
	return parseDatetimeAt(s, time.Now())
}

func parseDatetimeAt(s string, now time.Time) (time.Time, bool) {
	if strings.TrimSpace(s) == "" {
		return time.Time{}, false
	}
	original := strings.ToLower(strings.TrimSpace(s))
	modified := original

	// Replace "tomorrow" / "today" with a concrete ISO date.
	if strings.Contains(modified, "tomorrow") {
		modified = strings.ReplaceAll(modified, "tomorrow", now.AddDate(0, 0, 1).Format("2006-01-02"))
	}
	if strings.Contains(modified, "today") {
		modified = strings.ReplaceAll(modified, "today", now.Format("2006-01-02"))
	}
	// "noon" and "midnight" are unambiguous — substitute them AND treat the
	// result as if an AM/PM marker were specified so disambiguation skips them.
	unambiguousTimeOfDay := false
	for _, phrase := range []string{"at noon", " noon"} {
		if strings.Contains(modified, phrase) {
			modified = strings.ReplaceAll(modified, phrase, strings.Replace(phrase, "noon", "12:00 pm", 1))
			unambiguousTimeOfDay = true
		}
	}
	if modified == "noon" {
		modified = "12:00 pm"
		unambiguousTimeOfDay = true
	}
	for _, phrase := range []string{"at midnight", " midnight"} {
		if strings.Contains(modified, phrase) {
			modified = strings.ReplaceAll(modified, phrase, strings.Replace(phrase, "midnight", "00:00 am", 1))
			unambiguousTimeOfDay = true
		}
	}
	if modified == "midnight" {
		modified = "00:00 am"
		unambiguousTimeOfDay = true
	}

	hasAMPM := unambiguousTimeOfDay ||
		strings.Contains(modified, "am") || strings.Contains(modified, "pm") ||
		strings.Contains(modified, "a.m") || strings.Contains(modified, "p.m")

	parsed, ok := tryParse(modified, now)
	if !ok {
		return time.Time{}, false
	}

	if !hasAMPM {
		parsed = soonestAMPM(parsed, now)
	}
	return parsed, true
}

// tryParse walks a list of recognisers and returns the first successful match.
func tryParse(s string, now time.Time) (time.Time, bool) {
	if t, ok := parseExplicit(s, now); ok {
		return t, true
	}
	if t, ok := parseDatePlusTime(s, now); ok {
		return t, true
	}
	if t, ok := parseWeekday(s, now); ok {
		return t, true
	}
	if t, ok := parseMonthNameDate(s, now); ok {
		return t, true
	}
	if t, ok := parseBareTime(s, now); ok {
		return t, true
	}
	return time.Time{}, false
}

var (
	reISO          = regexp.MustCompile(`^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ t](\d{1,2}):(\d{2}))?(?:\s*(am|pm))?$`)
	reISOAtTime    = regexp.MustCompile(`^(\d{4})-(\d{1,2})-(\d{1,2})\s+at\s+(.+)$`)
	reWeekday      = regexp.MustCompile(`^(?:next\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)(?:\s+at\s+(.+))?$`)
	reMonthName    = regexp.MustCompile(`^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+at\s+(.+))?$`)
	reTime         = regexp.MustCompile(`^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.?|p\.m\.?)?$`)
	reAtTimeSuffix = regexp.MustCompile(`\s+at\s+(.+)$`)
)

var monthNames = map[string]time.Month{
	"january": time.January, "jan": time.January,
	"february": time.February, "feb": time.February,
	"march": time.March, "mar": time.March,
	"april": time.April, "apr": time.April,
	"may":  time.May,
	"june": time.June, "jun": time.June,
	"july": time.July, "jul": time.July,
	"august": time.August, "aug": time.August,
	"september": time.September, "sep": time.September, "sept": time.September,
	"october": time.October, "oct": time.October,
	"november": time.November, "nov": time.November,
	"december": time.December, "dec": time.December,
}

var weekdays = map[string]time.Weekday{
	"sunday": time.Sunday, "monday": time.Monday, "tuesday": time.Tuesday,
	"wednesday": time.Wednesday, "thursday": time.Thursday,
	"friday": time.Friday, "saturday": time.Saturday,
}

// parseExplicit handles strings like "2025-12-25 14:30" or "2025-12-25".
func parseExplicit(s string, now time.Time) (time.Time, bool) {
	m := reISO.FindStringSubmatch(s)
	if m == nil {
		return time.Time{}, false
	}
	year, _ := strconv.Atoi(m[1])
	month, _ := strconv.Atoi(m[2])
	day, _ := strconv.Atoi(m[3])
	hour := 0
	minute := 0
	if m[4] != "" {
		hour, _ = strconv.Atoi(m[4])
		minute, _ = strconv.Atoi(m[5])
	}
	if m[6] == "pm" && hour < 12 {
		hour += 12
	} else if m[6] == "am" && hour == 12 {
		hour = 0
	}
	return time.Date(year, time.Month(month), day, hour, minute, 0, 0, now.Location()), true
}

// parseDatePlusTime handles "YYYY-MM-DD at HH(:MM)?(am|pm)?".
func parseDatePlusTime(s string, now time.Time) (time.Time, bool) {
	m := reISOAtTime.FindStringSubmatch(s)
	if m == nil {
		return time.Time{}, false
	}
	year, _ := strconv.Atoi(m[1])
	month, _ := strconv.Atoi(m[2])
	day, _ := strconv.Atoi(m[3])
	hour, minute, ok := parseClockPart(m[4])
	if !ok {
		return time.Time{}, false
	}
	return time.Date(year, time.Month(month), day, hour, minute, 0, 0, now.Location()), true
}

// parseWeekday handles "next Tuesday at 3pm" (and bare weekday).
func parseWeekday(s string, now time.Time) (time.Time, bool) {
	m := reWeekday.FindStringSubmatch(s)
	if m == nil {
		return time.Time{}, false
	}
	wd := weekdays[m[1]]
	// Move forward to the next occurrence of that weekday strictly after today.
	days := (int(wd) - int(now.Weekday()) + 7) % 7
	if days == 0 {
		days = 7
	}
	target := now.AddDate(0, 0, days)
	hour, minute := 0, 0
	if m[2] != "" {
		h, mn, ok := parseClockPart(m[2])
		if !ok {
			return time.Time{}, false
		}
		hour, minute = h, mn
	}
	return time.Date(target.Year(), target.Month(), target.Day(), hour, minute, 0, 0, now.Location()), true
}

// parseMonthNameDate handles "December 25th at 10am".
func parseMonthNameDate(s string, now time.Time) (time.Time, bool) {
	m := reMonthName.FindStringSubmatch(s)
	if m == nil {
		return time.Time{}, false
	}
	month := monthNames[m[1]]
	day, _ := strconv.Atoi(m[2])
	hour, minute := 0, 0
	if m[3] != "" {
		h, mn, ok := parseClockPart(m[3])
		if !ok {
			return time.Time{}, false
		}
		hour, minute = h, mn
	}
	year := now.Year()
	result := time.Date(year, month, day, hour, minute, 0, 0, now.Location())
	if result.Before(now) {
		// Roll to next year if the date has already passed this year.
		result = time.Date(year+1, month, day, hour, minute, 0, 0, now.Location())
	}
	return result, true
}

// parseBareTime handles "3pm", "3:30 pm", "15:00", etc.
func parseBareTime(s string, now time.Time) (time.Time, bool) {
	h, mn, ok := parseClockPart(s)
	if !ok {
		return time.Time{}, false
	}
	return time.Date(now.Year(), now.Month(), now.Day(), h, mn, 0, 0, now.Location()), true
}

// parseClockPart parses a "3pm" / "3:30 pm" / "15:00" style clock expression
// into (hour, minute, ok). Hour is returned in 24-hour form only if an AM/PM
// marker or an hour >= 13 was given; otherwise it is returned as-is (0-12)
// and the caller is responsible for AM/PM disambiguation.
func parseClockPart(s string) (int, int, bool) {
	s = strings.TrimSpace(s)
	m := reTime.FindStringSubmatch(s)
	if m == nil {
		return 0, 0, false
	}
	hour, _ := strconv.Atoi(m[1])
	minute := 0
	if m[2] != "" {
		minute, _ = strconv.Atoi(m[2])
	}
	marker := strings.TrimRight(strings.ReplaceAll(m[3], ".", ""), ".")
	switch marker {
	case "pm":
		if hour < 12 {
			hour += 12
		}
	case "am":
		if hour == 12 {
			hour = 0
		}
	}
	if hour > 23 || minute > 59 {
		return 0, 0, false
	}
	return hour, minute, true
}

// soonestAMPM mirrors the Python logic: for ambiguous times (no AM/PM given),
// consider both interpretations for today and tomorrow, and pick the earliest
// that is strictly in the future.
func soonestAMPM(parsed, now time.Time) time.Time {
	hour := parsed.Hour()
	amHour := hour % 12
	pmHour := amHour + 12

	am := time.Date(parsed.Year(), parsed.Month(), parsed.Day(), amHour, parsed.Minute(), 0, 0, parsed.Location())
	pm := time.Date(parsed.Year(), parsed.Month(), parsed.Day(), pmHour, parsed.Minute(), 0, 0, parsed.Location())

	candidates := []time.Time{am, pm}
	if sameDate(parsed, now) {
		candidates = append(candidates,
			am.AddDate(0, 0, 1),
			pm.AddDate(0, 0, 1),
		)
	}

	var best time.Time
	for _, c := range candidates {
		if c.After(now) {
			if best.IsZero() || c.Before(best) {
				best = c
			}
		}
	}
	if best.IsZero() {
		return parsed
	}
	return best
}

func sameDate(a, b time.Time) bool {
	ay, am, ad := a.Date()
	by, bm, bd := b.Date()
	return ay == by && am == bm && ad == bd
}
