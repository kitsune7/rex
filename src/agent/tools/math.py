from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    This tool can handle basic arithmetic operations (+, -, *, /),
    exponentiation (**), and common math functions like sqrt, sin, cos, etc.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "sqrt(16)", "3 ** 2")

    Returns:
        The result of the calculation as a string

    Examples:
        - "2 + 2" returns "4"
        - "10 * 5" returns "50"
        - "sqrt(16)" returns "4.0"
    """
    import math

    try:
        # Create a safe evaluation environment with math functions
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pow": math.pow,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
        }

        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"
