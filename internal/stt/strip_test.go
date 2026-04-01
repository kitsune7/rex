package stt

import "testing"

func TestStripWakeWord(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		// Basic "hey rex" variants
		{"hey rex lowercase", "hey rex what time is it", "What time is it"},
		{"Hey Rex title case", "Hey Rex what time is it", "What time is it"},
		{"HEY REX upper case", "HEY REX what time is it", "What time is it"},

		// Punctuation after wake word
		{"comma after wake word", "hey rex, what time is it", "What time is it"},
		{"period after wake word", "hey rex. what time is it", "What time is it"},

		// No space between hey and rex
		{"heyrex no space", "heyrex what time is it", "What time is it"},

		// Misheard variants
		{"hay rex", "hay rex what time is it", "What time is it"},
		{"hey racks", "hey racks what time is it", "What time is it"},
		{"hey wrecks", "hey wrecks what time is it", "What time is it"},

		// No wake word — text preserved (first letter capitalised)
		{"no wake word lowercase", "what time is it", "What time is it"},
		{"no wake word hello", "hello there", "Hello there"},

		// Wake word NOT at start — should be unchanged
		{"wake word mid-sentence", "I said hey rex earlier", "I said hey rex earlier"},

		// Edge cases
		{"empty string", "", ""},
		{"only wake word", "hey rex", ""},
		{"only wake word trailing space", "hey rex ", ""},

		// First-letter capitalisation after strip
		{"capitalise after strip", "hey rex hello", "Hello"},
		{"digit after strip", "hey rex 5 minutes", "5 minutes"},

		// Preserve rest of text exactly
		{"preserve mixed case", "hey rex Set a Timer for 5 Minutes Please", "Set a Timer for 5 Minutes Please"},

		// "Rex" alone (second pattern)
		{"Rex alone at start", "Rex tell me a joke", "Tell me a joke"},

		// Comma before wake word with leading whitespace
		{"hey rex set a timer", "hey rex, set a timer", "Set a timer"},

		// Hay Rex variant from spec
		{"Hay Rex weather", "Hay Rex what's the weather", "What's the weather"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripWakeWord(tt.input)
			if got != tt.want {
				t.Errorf("stripWakeWord(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
