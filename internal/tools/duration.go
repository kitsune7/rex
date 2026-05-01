package tools

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	// Python: r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)"
	durHoursRE = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)`)
	// Python: r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)(?!s)"
	// Go's regexp package lacks lookahead, so we allow an optional trailing
	// character and reject matches where it's 's' ourselves.
	durMinutesRE = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)(\w?)`)
	// Python: r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)"
	durSecondsRE = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)`)
)

// ParseDuration parses a natural-language duration such as "5 minutes",
// "1 hour 30 minutes", "90 seconds". A bare number is interpreted as minutes.
// Returns (0, false) if nothing could be parsed or the total is zero.
func ParseDuration(s string) (time.Duration, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return 0, false
	}

	var (
		total    float64
		foundAny bool
	)

	if m := durHoursRE.FindStringSubmatch(s); m != nil {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			total += v * 3600
			foundAny = true
		}
	}

	// Emulate (?!s) by rejecting matches where the 'm' is followed by 's'.
	if m := durMinutesRE.FindStringSubmatch(s); m != nil {
		suffix := m[2]
		// "m" with trailing "s" would be "ms" which should NOT match per the
		// Python lookahead; but "minutes"/"mins" end with 's' themselves — for
		// those the subpattern consumed the trailing 's' already (m[3] is "").
		if !(suffix == "m" && m[3] == "s") {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				total += v * 60
				foundAny = true
			}
		}
	}

	if m := durSecondsRE.FindStringSubmatch(s); m != nil {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			total += v
			foundAny = true
		}
	}

	if !foundAny {
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			total = v * 60
			foundAny = true
		}
	}

	if !foundAny || total <= 0 {
		return 0, false
	}
	return time.Duration(total * float64(time.Second)), true
}

// FormatDuration produces a human-readable rendering such as "1 hour 30 minutes".
// Seconds are truncated to whole integers (matching the Python implementation).
func FormatDuration(d time.Duration) string {
	total := max(int(d.Seconds()), 0)

	switch {
	case total < 60:
		return pluralize(total, "second")
	case total < 3600:
		minutes := total / 60
		secs := total % 60
		parts := []string{pluralize(minutes, "minute")}
		if secs > 0 {
			parts = append(parts, pluralize(secs, "second"))
		}
		return strings.Join(parts, " ")
	default:
		hours := total / 3600
		minutes := (total % 3600) / 60
		parts := []string{pluralize(hours, "hour")}
		if minutes > 0 {
			parts = append(parts, pluralize(minutes, "minute"))
		}
		return strings.Join(parts, " ")
	}
}

func pluralize(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return strconv.Itoa(n) + " " + unit + "s"
}
