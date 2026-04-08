package tools

import (
	"fmt"
	"math"
	"strings"

	"github.com/Knetic/govaluate"
)

// MathTool implements the Tool interface for safe math evaluation.
type MathTool struct{}

func (t *MathTool) Name() string        { return "calculate" }
func (t *MathTool) Description() string  { return "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e." }
func (t *MathTool) RequiresConfirmation() bool { return false }

func (t *MathTool) Parameters() map[string]any {
	return map[string]any{
		"expression": map[string]any{
			"type":        "string",
			"description": "Math expression to evaluate",
			"required":    true,
		},
	}
}

func (t *MathTool) Execute(args map[string]any) (string, error) {
	expr, _ := args["expression"].(string)
	return Calculate(expr), nil
}

// mathFunctions provides safe math functions for expression evaluation.
var mathFunctions = map[string]govaluate.ExpressionFunction{
	"sqrt": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("sqrt: invalid argument")
		}
		return math.Sqrt(v), nil
	},
	"sin": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("sin: invalid argument")
		}
		return math.Sin(v), nil
	},
	"cos": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("cos: invalid argument")
		}
		return math.Cos(v), nil
	},
	"tan": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("tan: invalid argument")
		}
		return math.Tan(v), nil
	},
	"log": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("log: invalid argument")
		}
		return math.Log(v), nil
	},
	"log10": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("log10: invalid argument")
		}
		return math.Log10(v), nil
	},
	"exp": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("exp: invalid argument")
		}
		return math.Exp(v), nil
	},
	"abs": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("abs: invalid argument")
		}
		return math.Abs(v), nil
	},
	"round": func(args ...any) (any, error) {
		v, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("round: invalid argument")
		}
		if len(args) > 1 {
			decimals, ok := toFloat64(args[1])
			if !ok {
				return nil, fmt.Errorf("round: invalid decimals argument")
			}
			pow := math.Pow(10, decimals)
			return math.Round(v*pow) / pow, nil
		}
		return math.Round(v), nil
	},
	"pow": func(args ...any) (any, error) {
		if len(args) < 2 {
			return nil, fmt.Errorf("pow: requires 2 arguments")
		}
		base, ok := toFloat64(args[0])
		if !ok {
			return nil, fmt.Errorf("pow: invalid base argument")
		}
		exp, ok := toFloat64(args[1])
		if !ok {
			return nil, fmt.Errorf("pow: invalid exponent argument")
		}
		return math.Pow(base, exp), nil
	},
}

// Calculate evaluates a math expression safely and returns the result as a string.
func Calculate(expression string) string {
	// Security: reject dangerous strings
	lower := strings.ToLower(expression)
	for _, banned := range []string{"import", "__", "exec", "eval", "open", "builtins"} {
		if strings.Contains(lower, banned) {
			return fmt.Sprintf("Error calculating '%s': forbidden operation", expression)
		}
	}

	// Replace ** with govaluate's exponent operator
	expr := strings.ReplaceAll(expression, "**", "^")

	// Inject constants as parameters
	params := map[string]any{
		"pi": math.Pi,
		"e":  math.E,
	}

	parsed, err := govaluate.NewEvaluableExpressionWithFunctions(expr, mathFunctions)
	if err != nil {
		return fmt.Sprintf("Error calculating '%s': %s", expression, err.Error())
	}

	result, err := parsed.Evaluate(params)
	if err != nil {
		return fmt.Sprintf("Error calculating '%s': %s", expression, err.Error())
	}

	return formatNumber(result)
}

// formatNumber formats a numeric result, removing unnecessary trailing zeros.
func formatNumber(v any) string {
	f, ok := toFloat64(v)
	if !ok {
		return fmt.Sprintf("%v", v)
	}

	// If the result is a whole number, format without decimals
	if f == math.Trunc(f) && math.Abs(f) < 1e15 {
		return fmt.Sprintf("%g", f)
	}
	return fmt.Sprintf("%g", f)
}

func toFloat64(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	default:
		return 0, false
	}
}
