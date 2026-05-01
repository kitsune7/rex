package tools

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

// Calculate evaluates a math expression and returns its string form, or an
// error message matching the Python "Error calculating '<expr>': <msg>" shape.
// Supported: +, -, *, /, **, unary ±, parentheses, numeric literals, and the
// identifiers sqrt, sin, cos, tan, log, log10, exp, pow, abs, round, pi, e.
func Calculate(expression string) string {
	result, err := evalExpression(expression)
	if err != nil {
		return fmt.Sprintf("Error calculating '%s': %s", expression, err.Error())
	}
	return formatResult(result)
}

// formatResult renders a number. Integers come out without a decimal point to
// match Python's str(int) ("4"); floats always include a decimal point or an
// exponent to match str(float) ("5.0", "3.5", "1.4142135623730951").
func formatResult(v value) string {
	if v.isInt {
		return strconv.FormatInt(v.i, 10)
	}
	s := strconv.FormatFloat(v.f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// value carries either an int64 (for exact integer literals/results) or a
// float64. Python distinguishes int and float when stringifying results
// (e.g. "2 + 2" -> "4", "10 / 2" -> "5.0"), so we preserve the distinction.
type value struct {
	isInt bool
	i     int64
	f     float64
}

func intVal(i int64) value     { return value{isInt: true, i: i} }
func floatVal(f float64) value { return value{isInt: false, f: f} }

func (v value) float() float64 {
	if v.isInt {
		return float64(v.i)
	}
	return v.f
}

// tokenKind distinguishes token classes the parser needs to look at.
type tokenKind int

const (
	tkEOF tokenKind = iota
	tkNumber
	tkIdent
	tkPlus
	tkMinus
	tkStar
	tkSlash
	tkPow
	tkLParen
	tkRParen
	tkComma
)

type token struct {
	kind   tokenKind
	text   string
	numInt int64
	isInt  bool
	numF   float64
}

type lexer struct {
	src string
	pos int
}

func (l *lexer) next() (token, error) {
	for l.pos < len(l.src) && unicode.IsSpace(rune(l.src[l.pos])) {
		l.pos++
	}
	if l.pos >= len(l.src) {
		return token{kind: tkEOF}, nil
	}

	c := l.src[l.pos]
	switch c {
	case '+':
		l.pos++
		return token{kind: tkPlus, text: "+"}, nil
	case '-':
		l.pos++
		return token{kind: tkMinus, text: "-"}, nil
	case '*':
		if l.pos+1 < len(l.src) && l.src[l.pos+1] == '*' {
			l.pos += 2
			return token{kind: tkPow, text: "**"}, nil
		}
		l.pos++
		return token{kind: tkStar, text: "*"}, nil
	case '/':
		l.pos++
		return token{kind: tkSlash, text: "/"}, nil
	case '(':
		l.pos++
		return token{kind: tkLParen, text: "("}, nil
	case ')':
		l.pos++
		return token{kind: tkRParen, text: ")"}, nil
	case ',':
		l.pos++
		return token{kind: tkComma, text: ","}, nil
	}

	if c >= '0' && c <= '9' || c == '.' {
		return l.readNumber()
	}
	if isIdentStart(c) {
		return l.readIdent(), nil
	}
	return token{}, fmt.Errorf("unexpected character %q", string(c))
}

func (l *lexer) readNumber() (token, error) {
	start := l.pos
	hasDot := false
	for l.pos < len(l.src) {
		c := l.src[l.pos]
		if c >= '0' && c <= '9' {
			l.pos++
			continue
		}
		if c == '.' && !hasDot {
			hasDot = true
			l.pos++
			continue
		}
		break
	}
	text := l.src[start:l.pos]
	if hasDot {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return token{}, fmt.Errorf("invalid number %q", text)
		}
		return token{kind: tkNumber, text: text, numF: f}, nil
	}
	i, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		// Fallback: very large integer literal — treat as float.
		f, ferr := strconv.ParseFloat(text, 64)
		if ferr != nil {
			return token{}, fmt.Errorf("invalid number %q", text)
		}
		return token{kind: tkNumber, text: text, numF: f}, nil
	}
	return token{kind: tkNumber, text: text, isInt: true, numInt: i}, nil
}

func (l *lexer) readIdent() token {
	start := l.pos
	for l.pos < len(l.src) && isIdentPart(l.src[l.pos]) {
		l.pos++
	}
	return token{kind: tkIdent, text: l.src[start:l.pos]}
}

func isIdentStart(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
}

func isIdentPart(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}

type parser struct {
	lex  *lexer
	peek token
}

func newParser(src string) (*parser, error) {
	p := &parser{lex: &lexer{src: src}}
	if err := p.advance(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *parser) advance() error {
	tok, err := p.lex.next()
	if err != nil {
		return err
	}
	p.peek = tok
	return nil
}

func evalExpression(src string) (value, error) {
	if strings.TrimSpace(src) == "" {
		return value{}, errors.New("empty expression")
	}
	p, err := newParser(src)
	if err != nil {
		return value{}, err
	}
	v, err := p.parseExpr()
	if err != nil {
		return value{}, err
	}
	if p.peek.kind != tkEOF {
		return value{}, fmt.Errorf("unexpected token %q", p.peek.text)
	}
	return v, nil
}

// Grammar:
//   expr    = term { ("+" | "-") term }
//   term    = factor { ("*" | "/") factor }
//   factor  = unary [ "**" factor ]        ; right-associative
//   unary   = ("+" | "-") unary | primary
//   primary = number
//           | ident [ "(" [ expr { "," expr } ] ")" ]
//           | "(" expr ")"

func (p *parser) parseExpr() (value, error) {
	left, err := p.parseTerm()
	if err != nil {
		return value{}, err
	}
	for p.peek.kind == tkPlus || p.peek.kind == tkMinus {
		op := p.peek.kind
		if err := p.advance(); err != nil {
			return value{}, err
		}
		right, err := p.parseTerm()
		if err != nil {
			return value{}, err
		}
		if op == tkPlus {
			left = addValues(left, right)
		} else {
			left = subValues(left, right)
		}
	}
	return left, nil
}

func (p *parser) parseTerm() (value, error) {
	left, err := p.parseFactor()
	if err != nil {
		return value{}, err
	}
	for p.peek.kind == tkStar || p.peek.kind == tkSlash {
		op := p.peek.kind
		if err := p.advance(); err != nil {
			return value{}, err
		}
		right, err := p.parseFactor()
		if err != nil {
			return value{}, err
		}
		if op == tkStar {
			left = mulValues(left, right)
		} else {
			r, err := divValues(left, right)
			if err != nil {
				return value{}, err
			}
			left = r
		}
	}
	return left, nil
}

func (p *parser) parseFactor() (value, error) {
	base, err := p.parseUnary()
	if err != nil {
		return value{}, err
	}
	if p.peek.kind == tkPow {
		if err := p.advance(); err != nil {
			return value{}, err
		}
		// right-associative: recurse into parseFactor, not parseUnary, so
		// "2 ** 3 ** 2" parses as 2 ** (3 ** 2).
		exp, err := p.parseFactor()
		if err != nil {
			return value{}, err
		}
		base = powValues(base, exp)
	}
	return base, nil
}

func (p *parser) parseUnary() (value, error) {
	if p.peek.kind == tkPlus {
		if err := p.advance(); err != nil {
			return value{}, err
		}
		return p.parseUnary()
	}
	if p.peek.kind == tkMinus {
		if err := p.advance(); err != nil {
			return value{}, err
		}
		v, err := p.parseUnary()
		if err != nil {
			return value{}, err
		}
		return negate(v), nil
	}
	return p.parsePrimary()
}

func (p *parser) parsePrimary() (value, error) {
	switch p.peek.kind {
	case tkNumber:
		tok := p.peek
		if err := p.advance(); err != nil {
			return value{}, err
		}
		if tok.isInt {
			return intVal(tok.numInt), nil
		}
		return floatVal(tok.numF), nil
	case tkIdent:
		name := p.peek.text
		if err := p.advance(); err != nil {
			return value{}, err
		}
		if p.peek.kind != tkLParen {
			return lookupConstant(name)
		}
		// Function call.
		if err := p.advance(); err != nil {
			return value{}, err
		}
		var args []value
		if p.peek.kind != tkRParen {
			arg, err := p.parseExpr()
			if err != nil {
				return value{}, err
			}
			args = append(args, arg)
			for p.peek.kind == tkComma {
				if err := p.advance(); err != nil {
					return value{}, err
				}
				arg, err := p.parseExpr()
				if err != nil {
					return value{}, err
				}
				args = append(args, arg)
			}
		}
		if p.peek.kind != tkRParen {
			return value{}, fmt.Errorf("expected ')' after %s arguments", name)
		}
		if err := p.advance(); err != nil {
			return value{}, err
		}
		return callFunction(name, args)
	case tkLParen:
		if err := p.advance(); err != nil {
			return value{}, err
		}
		v, err := p.parseExpr()
		if err != nil {
			return value{}, err
		}
		if p.peek.kind != tkRParen {
			return value{}, errors.New("missing closing paren")
		}
		if err := p.advance(); err != nil {
			return value{}, err
		}
		return v, nil
	default:
		return value{}, fmt.Errorf("unexpected token %q", p.peek.text)
	}
}

func lookupConstant(name string) (value, error) {
	switch name {
	case "pi":
		return floatVal(math.Pi), nil
	case "e":
		return floatVal(math.E), nil
	}
	return value{}, fmt.Errorf("name %q is not defined", name)
}

func callFunction(name string, args []value) (value, error) {
	requireArgs := func(n int) error {
		if len(args) != n {
			return fmt.Errorf("%s expected %d argument(s), got %d", name, n, len(args))
		}
		return nil
	}

	switch name {
	case "sqrt":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Sqrt(args[0].float())), nil
	case "sin":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Sin(args[0].float())), nil
	case "cos":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Cos(args[0].float())), nil
	case "tan":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Tan(args[0].float())), nil
	case "log":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Log(args[0].float())), nil
	case "log10":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Log10(args[0].float())), nil
	case "exp":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return floatVal(math.Exp(args[0].float())), nil
	case "pow":
		if err := requireArgs(2); err != nil {
			return value{}, err
		}
		// Python's math.pow always returns float, even for integer inputs.
		return floatVal(math.Pow(args[0].float(), args[1].float())), nil
	case "abs":
		if err := requireArgs(1); err != nil {
			return value{}, err
		}
		return absValue(args[0]), nil
	case "round":
		switch len(args) {
		case 1:
			return intVal(int64(pyRound(args[0].float(), 0))), nil
		case 2:
			if !args[1].isInt {
				return value{}, errors.New("round: second argument must be an integer")
			}
			return floatVal(pyRound(args[0].float(), int(args[1].i))), nil
		default:
			return value{}, fmt.Errorf("round expected 1 or 2 arguments, got %d", len(args))
		}
	}
	return value{}, fmt.Errorf("name %q is not defined", name)
}

// pyRound implements Python's banker's rounding (round half to even).
// For d >= 0 this also rounds to d decimal places.
func pyRound(x float64, digits int) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	scale := math.Pow(10, float64(digits))
	y := x * scale
	// math.RoundToEven handles the standard cases.
	y = math.RoundToEven(y)
	return y / scale
}

func addValues(a, b value) value {
	if a.isInt && b.isInt {
		return intVal(a.i + b.i)
	}
	return floatVal(a.float() + b.float())
}

func subValues(a, b value) value {
	if a.isInt && b.isInt {
		return intVal(a.i - b.i)
	}
	return floatVal(a.float() - b.float())
}

func mulValues(a, b value) value {
	if a.isInt && b.isInt {
		return intVal(a.i * b.i)
	}
	return floatVal(a.float() * b.float())
}

func divValues(a, b value) (value, error) {
	// Python's "/" always returns float and raises on division by zero.
	if b.float() == 0 {
		return value{}, errors.New("division by zero")
	}
	return floatVal(a.float() / b.float()), nil
}

func powValues(a, b value) value {
	// Python's "**": int ** non-negative int stays int.
	if a.isInt && b.isInt && b.i >= 0 {
		result := int64(1)
		base := a.i
		exp := b.i
		for exp > 0 {
			if exp&1 == 1 {
				result *= base
			}
			base *= base
			exp >>= 1
		}
		return intVal(result)
	}
	return floatVal(math.Pow(a.float(), b.float()))
}

func negate(v value) value {
	if v.isInt {
		return intVal(-v.i)
	}
	return floatVal(-v.f)
}

func absValue(v value) value {
	if v.isInt {
		if v.i < 0 {
			return intVal(-v.i)
		}
		return intVal(v.i)
	}
	return floatVal(math.Abs(v.f))
}
