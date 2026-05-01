package stt

import "testing"

// These cases mirror src/tests/test_stt.py.
func TestStripWakeWord(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"hey rex lower", "hey rex what time is it", "What time is it"},
		{"Hey Rex mixed", "Hey Rex what time is it", "What time is it"},
		{"HEY REX upper", "HEY REX what time is it", "What time is it"},
		{"comma", "hey rex, what time is it", "What time is it"},
		{"period", "hey rex. what time is it", "What time is it"},
		{"heyrex no space", "heyrex what time is it", "What time is it"},
		{"hay rex variant", "hay rex what time is it", "What time is it"},
		{"hey racks variant", "hey racks what time is it", "What time is it"},
		{"hey wrecks variant", "hey wrecks what time is it", "What time is it"},
		{"no wake word", "what time is it", "What time is it"},
		{"no wake word hello", "hello there", "Hello there"},
		{"middle match not stripped", "I said hey rex earlier", "I said hey rex earlier"},
		{"empty", "", ""},
		{"only wake word", "hey rex", ""},
		{"only wake word trailing space", "hey rex ", ""},
		{"capitalise first letter", "hey rex hello", "Hello"},
		{"leading digit preserved", "hey rex 5 minutes", "5 minutes"},
		{"preserves rest exactly", "hey rex Set a Timer for 5 Minutes Please", "Set a Timer for 5 Minutes Please"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := StripWakeWord(tc.in)
			if got != tc.want {
				t.Fatalf("StripWakeWord(%q) = %q; want %q", tc.in, got, tc.want)
			}
		})
	}
}
