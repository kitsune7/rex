package states

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

// confirmationPhrases are words/phrases that signal user approval.
var confirmationPhrases = []string{
	"yes", "yeah", "yep", "sure", "okay", "ok",
	"go ahead", "do it", "please", "correct", "right",
	"affirmative", "confirm", "absolutely", "definitely",
	"clear", "done", "got it", "proceed",
}

// rejectionPhrases are words/phrases that signal user disapproval.
var rejectionPhrases = []string{
	"no", "nah", "nope", "don't", "cancel",
	"stop", "never mind", "forget it", "negative",
}

// stopPhrases signal the user wants to end the conversation entirely.
var stopPhrases = []string{
	"stop", "nevermind", "never mind", "forget it",
	"that's all", "nothing", "goodbye", "bye", "cancel",
}

// IsConfirmation reports whether text contains a confirmation phrase.
func IsConfirmation(text string) bool {
	normalized := trimLower(text)
	for _, p := range confirmationPhrases {
		if strings.Contains(normalized, p) {
			return true
		}
	}
	return false
}

// IsRejection reports whether text contains a rejection phrase.
func IsRejection(text string) bool {
	normalized := trimLower(text)
	for _, p := range rejectionPhrases {
		if strings.Contains(normalized, p) {
			return true
		}
	}
	return false
}

// IsStopPhrase reports whether text exactly matches a stop phrase.
func IsStopPhrase(text string) bool {
	normalized := trimLower(text)
	for _, p := range stopPhrases {
		if normalized == p {
			return true
		}
	}
	return false
}

// snooze regex patterns used by ParseSnoozeDuration.
var snoozePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?:remind|tell|ask)\s+me\s+(?:again\s+)?in\s+(\d+)\s*(?:minute|min)`),
	regexp.MustCompile(`(?:snooze|delay|postpone)(?:\s+(?:it|for))?\s+(\d+)\s*(?:minute|min)`),
	regexp.MustCompile(`(\d+)\s*(?:minute|min)(?:\s+(?:later|from now))?`),
}

// ParseSnoozeDuration attempts to extract a snooze duration from text such as
// "snooze for 5 minutes" or "remind me in 10 minutes".
// It returns the duration and true on success, or zero and false if no
// snooze request was detected.
func ParseSnoozeDuration(text string) (time.Duration, bool) {
	normalized := trimLower(text)
	for _, re := range snoozePatterns {
		m := re.FindStringSubmatch(normalized)
		if m != nil {
			minutes, err := strconv.Atoi(m[1])
			if err == nil && minutes > 0 {
				return time.Duration(minutes) * time.Minute, true
			}
		}
	}
	return 0, false
}
