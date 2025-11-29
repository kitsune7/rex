"""Tests for the calculate tool."""

import math

from agent.tools.math import calculate


def calc(expression: str) -> str:
    """Helper to invoke the calculate tool."""
    return calculate.invoke({"expression": expression})


class TestCalculate:
    """Tests for calculate function."""

    def test_basic_addition(self):
        assert calc("2 + 2") == "4"
        assert calc("100 + 200") == "300"

    def test_basic_subtraction(self):
        assert calc("10 - 3") == "7"
        assert calc("100 - 150") == "-50"

    def test_basic_multiplication(self):
        assert calc("6 * 7") == "42"
        assert calc("0 * 100") == "0"

    def test_basic_division(self):
        assert calc("10 / 2") == "5.0"
        assert calc("7 / 2") == "3.5"

    def test_exponentiation(self):
        assert calc("2 ** 3") == "8"
        assert calc("10 ** 2") == "100"

    def test_complex_expressions(self):
        assert calc("2 + 3 * 4") == "14"
        assert calc("(2 + 3) * 4") == "20"
        assert calc("10 / 2 + 3") == "8.0"

    def test_sqrt(self):
        assert calc("sqrt(16)") == "4.0"
        assert calc("sqrt(2)") == str(math.sqrt(2))

    def test_trigonometric_functions(self):
        assert calc("sin(0)") == "0.0"
        assert calc("cos(0)") == "1.0"
        # tan(0) should be 0
        assert calc("tan(0)") == "0.0"

    def test_logarithms(self):
        assert calc("log(1)") == "0.0"
        assert calc("log10(100)") == "2.0"
        assert calc("log(e)") == "1.0"

    def test_exp(self):
        assert calc("exp(0)") == "1.0"
        assert calc("exp(1)") == str(math.e)

    def test_pow(self):
        assert calc("pow(2, 3)") == "8.0"

    def test_constants(self):
        assert calc("pi") == str(math.pi)
        assert calc("e") == str(math.e)

    def test_abs(self):
        assert calc("abs(-5)") == "5"
        assert calc("abs(5)") == "5"

    def test_round(self):
        assert calc("round(3.7)") == "4"
        assert calc("round(3.14159, 2)") == "3.14"

    def test_combined_functions_and_constants(self):
        result = float(calc("2 * pi"))
        assert abs(result - 2 * math.pi) < 0.0001

    def test_error_on_invalid_expression(self):
        result = calc("invalid")
        assert "Error" in result

    def test_error_on_undefined_function(self):
        result = calc("undefined_func(5)")
        assert "Error" in result

    def test_division_by_zero(self):
        result = calc("1 / 0")
        assert "Error" in result

    def test_security_no_builtins(self):
        # Should not be able to access dangerous builtins
        result = calc("__import__('os').system('echo hacked')")
        assert "Error" in result

    def test_security_no_eval(self):
        # eval should not be accessible
        result = calc("eval('1+1')")
        assert "Error" in result

    def test_security_no_exec(self):
        # exec should not be accessible
        result = calc("exec('x=1')")
        assert "Error" in result

    def test_security_no_open(self):
        # open should not be accessible
        result = calc("open('/etc/passwd')")
        assert "Error" in result
