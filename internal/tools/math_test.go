package tools

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestCalculateBasic(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"2 + 3", "5"},
		{"100 + 200", "300"},
		{"10 - 3", "7"},
		{"100 - 150", "-50"},
		{"6 * 7", "42"},
		{"0 * 100", "0"},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			got := Calculate(tc.expr)
			if got != tc.expected {
				t.Errorf("Calculate(%q) = %q, want %q", tc.expr, got, tc.expected)
			}
		})
	}
}

func TestCalculateDivision(t *testing.T) {
	got := Calculate("10 / 2")
	if got != "5" {
		t.Errorf("Calculate('10 / 2') = %q, want '5'", got)
	}

	got = Calculate("7 / 2")
	if got != "3.5" {
		t.Errorf("Calculate('7 / 2') = %q, want '3.5'", got)
	}
}

func TestCalculateExponentiation(t *testing.T) {
	got := Calculate("2 ** 3")
	if got != "8" {
		t.Errorf("Calculate('2 ** 3') = %q, want '8'", got)
	}

	got = Calculate("10 ** 2")
	if got != "100" {
		t.Errorf("Calculate('10 ** 2') = %q, want '100'", got)
	}
}

func TestCalculateOrderOfOperations(t *testing.T) {
	got := Calculate("2 + 3 * 4")
	if got != "14" {
		t.Errorf("Calculate('2 + 3 * 4') = %q, want '14'", got)
	}

	got = Calculate("(2 + 3) * 4")
	if got != "20" {
		t.Errorf("Calculate('(2 + 3) * 4') = %q, want '20'", got)
	}
}

func TestCalculateFunctions(t *testing.T) {
	tests := []struct {
		expr     string
		expected float64
	}{
		{"sqrt(16)", 4.0},
		{"sqrt(2)", math.Sqrt(2)},
		{"sin(0)", 0.0},
		{"cos(0)", 1.0},
		{"tan(0)", 0.0},
		{"log(1)", 0.0},
		{"log10(100)", 2.0},
		{"exp(0)", 1.0},
		{"exp(1)", math.E},
		{"pow(2, 3)", 8.0},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			got := Calculate(tc.expr)
			f, err := strconv.ParseFloat(got, 64)
			if err != nil {
				t.Fatalf("Calculate(%q) = %q, not a number", tc.expr, got)
			}
			if math.Abs(f-tc.expected) > 1e-9 {
				t.Errorf("Calculate(%q) = %v, want %v", tc.expr, f, tc.expected)
			}
		})
	}
}

func TestCalculateConstants(t *testing.T) {
	piResult := Calculate("pi")
	f, err := strconv.ParseFloat(piResult, 64)
	if err != nil || math.Abs(f-math.Pi) > 1e-9 {
		t.Errorf("Calculate('pi') = %q, want pi", piResult)
	}

	eResult := Calculate("e")
	f, err = strconv.ParseFloat(eResult, 64)
	if err != nil || math.Abs(f-math.E) > 1e-9 {
		t.Errorf("Calculate('e') = %q, want e", eResult)
	}
}

func TestCalculateAbs(t *testing.T) {
	got := Calculate("abs(-5)")
	if got != "5" {
		t.Errorf("Calculate('abs(-5)') = %q, want '5'", got)
	}
}

func TestCalculateRound(t *testing.T) {
	got := Calculate("round(3.7)")
	if got != "4" {
		t.Errorf("Calculate('round(3.7)') = %q, want '4'", got)
	}

	got = Calculate("round(3.14159, 2)")
	if got != "3.14" {
		t.Errorf("Calculate('round(3.14159, 2)') = %q, want '3.14'", got)
	}
}

func TestCalculateCombined(t *testing.T) {
	got := Calculate("2 * pi")
	f, err := strconv.ParseFloat(got, 64)
	if err != nil || math.Abs(f-2*math.Pi) > 1e-4 {
		t.Errorf("Calculate('2 * pi') = %q, want ~%v", got, 2*math.Pi)
	}
}

func TestCalculateErrors(t *testing.T) {
	errorCases := []string{
		"invalid",
		"undefined_func(5)",
		"1 / 0",
	}

	for _, expr := range errorCases {
		t.Run(expr, func(t *testing.T) {
			got := Calculate(expr)
			if !strings.Contains(got, "Error") {
				t.Errorf("Calculate(%q) = %q, expected error", expr, got)
			}
		})
	}
}

func TestCalculateSecurity(t *testing.T) {
	securityCases := []string{
		"__import__('os').system('echo hacked')",
		"eval('1+1')",
		"exec('x=1')",
		"open('/etc/passwd')",
		"__builtins__",
	}

	for _, expr := range securityCases {
		t.Run(expr, func(t *testing.T) {
			got := Calculate(expr)
			if !strings.Contains(got, "Error") {
				t.Errorf("Calculate(%q) = %q, expected error for security", expr, got)
			}
		})
	}
}
