// Package stt provides speech-to-text transcription using whisper.cpp.
package stt

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// wakeWordPatterns matches common Whisper transcription variations of the
// "hey rex" wake phrase at the start of a string. Patterns are compiled once
// at package init time.
var wakeWordPatterns = []*regexp.Regexp{
	// "hey/hay rex/racks/wrecks/rax" with optional leading whitespace/commas
	regexp.MustCompile(`(?i)^[\s,]*(?:hey|hay)\s*(?:rex|racks?|wrecks?|rax)[\s,!.]*`),
	// Just the name without "hey/hay"
	regexp.MustCompile(`(?i)^[\s,]*(?:rex|racks?|wrecks?)[\s,!.]*`),
}

// stripWakeWord removes wake-word variations from the beginning of text and
// capitalises the first letter of whatever remains.
func stripWakeWord(text string) string {
	result := text
	for _, pat := range wakeWordPatterns {
		if loc := pat.FindStringIndex(result); loc != nil && loc[0] == 0 {
			result = result[loc[1]:]
			break // only strip the first matching pattern
		}
	}

	result = strings.TrimSpace(result)

	return capitalizeFirst(result)
}

// capitalizeFirst returns s with its first Unicode letter upper-cased.
// Non-letter leading runes (digits, punctuation) are left unchanged.
func capitalizeFirst(s string) string {
	if s == "" {
		return s
	}
	r, size := utf8.DecodeRuneInString(s)
	if unicode.IsLetter(r) {
		return string(unicode.ToUpper(r)) + s[size:]
	}
	return s
}
