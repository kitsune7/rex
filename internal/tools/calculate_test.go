package tools

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestCalculateArithmetic(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"2 + 2", "4"},
		{"100 + 200", "300"},
		{"10 - 3", "7"},
		{"100 - 150", "-50"},
		{"6 * 7", "42"},
		{"0 * 100", "0"},
		{"10 / 2", "5.0"},
		{"7 / 2", "3.5"},
		{"2 ** 3", "8"},
		{"10 ** 2", "100"},
		{"2 + 3 * 4", "14"},
		{"(2 + 3) * 4", "20"},
		{"10 / 2 + 3", "8.0"},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			if got := Calculate(tc.in); got != tc.want {
				t.Errorf("Calculate(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestCalculateFunctions(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"sqrt(16)", "4.0"},
		{"sqrt(2)", strconv.FormatFloat(math.Sqrt(2), 'g', -1, 64)},
		{"sin(0)", "0.0"},
		{"cos(0)", "1.0"},
		{"tan(0)", "0.0"},
		{"log(1)", "0.0"},
		{"log10(100)", "2.0"},
		{"log(e)", "1.0"},
		{"exp(0)", "1.0"},
		{"exp(1)", strconv.FormatFloat(math.E, 'g', -1, 64)},
		{"pow(2, 3)", "8.0"},
		{"abs(-5)", "5"},
		{"abs(5)", "5"},
		{"round(3.7)", "4"},
		{"round(3.14159, 2)", "3.14"},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			if got := Calculate(tc.in); got != tc.want {
				t.Errorf("Calculate(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestCalculateConstants(t *testing.T) {
	if got := Calculate("pi"); got != strconv.FormatFloat(math.Pi, 'g', -1, 64) {
		t.Errorf("Calculate(pi) = %q", got)
	}
	if got := Calculate("e"); got != strconv.FormatFloat(math.E, 'g', -1, 64) {
		t.Errorf("Calculate(e) = %q", got)
	}

	result, err := strconv.ParseFloat(Calculate("2 * pi"), 64)
	if err != nil {
		t.Fatalf("parse float: %v", err)
	}
	if math.Abs(result-2*math.Pi) > 0.0001 {
		t.Errorf("2*pi = %v", result)
	}
}

func TestCalculateErrors(t *testing.T) {
	tests := []string{
		"invalid",
		"undefined_func(5)",
		"1 / 0",
		// Security: none of these should be accessible.
		"__import__('os').system('echo hacked')",
		"eval('1+1')",
		"exec('x=1')",
		"open('/etc/passwd')",
	}
	for _, expr := range tests {
		t.Run(expr, func(t *testing.T) {
			got := Calculate(expr)
			if !strings.HasPrefix(got, "Error calculating") {
				t.Errorf("Calculate(%q) = %q; want error prefix", expr, got)
			}
		})
	}
}
