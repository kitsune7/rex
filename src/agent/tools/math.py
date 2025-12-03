from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e."""
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
