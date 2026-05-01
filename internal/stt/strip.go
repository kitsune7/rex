package stt

import (
	"regexp"
	"strings"
	"unicode"
)

// wakeWordPatterns mirror src/stt/stt.py: the common variations Whisper
// produces when transcribing the "hey rex" wake word.
var wakeWordPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^hey\s*rex[,.\s]*`),
	regexp.MustCompile(`(?i)^hay\s*rex[,.\s]*`),
	regexp.MustCompile(`(?i)^hey\s*racks[,.\s]*`),
	regexp.MustCompile(`(?i)^hey\s*wrecks[,.\s]*`),
}

// StripWakeWord removes a leading wake-word variant from text and
// capitalises the first letter of whatever remains. It returns the empty
// string if the input was empty or contained only the wake word.
func StripWakeWord(text string) string {
	result := text
	for _, re := range wakeWordPatterns {
		result = re.ReplaceAllString(result, "")
	}
	result = strings.TrimSpace(result)

	if result == "" {
		return ""
	}

	runes := []rune(result)
	if unicode.IsLetter(runes[0]) {
		runes[0] = unicode.ToUpper(runes[0])
	}
	return string(runes)
}
