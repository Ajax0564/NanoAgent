from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib

matplotlib.use(
    "Agg"
)  # Must be set before importing pyplot for headless/server environments
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)


class PlottingTool:
    """
    A tool for generating and saving 2D plots of mathematical expressions using SymPy and Matplotlib.
    """

    def get_schema(self):
        """Return the native tool schema for mathematical expression plotting.

        Inputs:
            None.

        Returns:
            A tool definition containing an expression, optional x-axis bounds,
            and output filename schema.
        """
        return {
            "name": "plot_expression",
            "description": "Generates a 2D line plot of a mathematical expression and saves it as an image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression in terms of x (e.g., 'sin(x) * exp(-x/5)', 'x**3 - 3*x')",
                    },
                    "x_min": {
                        "type": "number",
                        "description": "Minimum value for the x-axis (default: -10)",
                    },
                    "x_max": {
                        "type": "number",
                        "description": "Maximum value for the x-axis (default: 10)",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename for the saved plot (default: 'math_plot.png')",
                    },
                },
                "required": ["expression"],
            },
        }

    def execute(
        self,
        expression: str,
        x_min: float = -10.0,
        x_max: float = 10.0,
        filename: str = "math_plot.png",
    ) -> str:
        """Evaluate and save a sampled 2D plot of an expression in ``x``.

        Args:
            expression: SymPy-compatible expression to plot.
            x_min: Inclusive start of the sampled x range.
            x_max: Inclusive end of the sampled x range.
            filename: Destination image path; defaults to ``math_plot.png``.

        Returns:
            A success message with the output path, or a plotting-error message.
            The function closes its Matplotlib figure in either case.
        """
        try:
            # Setup transformations
            transformations = standard_transformations + (
                implicit_multiplication_application,
            )

            # Parse expression
            x = sp.Symbol("x")
            expr = parse_expr(
                expression, transformations=transformations, evaluate=True
            )

            # Convert SymPy expression to a fast numerical function using numpy
            f = sp.lambdify(x, expr, "numpy")

            # Generate data points
            X = np.linspace(x_min, x_max, 400)

            # Handle potential domain errors (e.g., division by zero, log of negative)
            with np.errstate(all="ignore"):
                Y = f(X)

            # Create the plot
            plt.figure(figsize=(8, 5))
            plt.plot(X, Y, label=f"$f(x) = {expr}$", color="#2b6cb0", linewidth=2)
            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
            plt.title(f"Plot of $f(x) = {expr}$")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()

            # Save figure
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

            return f"Success: Plot saved successfully to '{filename}'."

        except Exception as e:
            plt.close()
            return f"Plotting error: {e}. Please check the expression format."


class CalculatorTool:
    """
    General-purpose mathematical calculation tool.

    Supports:
    - Arithmetic
    - Algebra
    - Equation solving
    - Differentiation
    - Integration
    - Limits
    - Series
    - Matrix operations
    - Statistics
    """

    def __init__(self):
        """Prepare SymPy parsing transformations for implicit multiplication.

        Inputs:
            None.

        Returns:
            None. Stores the parser transformation tuple used by ``parse``.
        """
        self.transformations = standard_transformations + (
            implicit_multiplication_application,
        )

    def get_schema(self):
        """Return the native schema for calculator operations and their inputs.

        Inputs:
            None.

        Returns:
            A tool definition listing the supported operation names and optional
            expression, calculus, matrix, and statistics arguments.
        """
        return {
            "name": "calculator",
            "description": (
                "General-purpose mathematical tool for arithmetic, "
                "algebra, calculus, equations, matrices and statistics. "
                "Select the appropriate operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "calculate",
                            "simplify",
                            "expand",
                            "factor",
                            "solve",
                            "differentiate",
                            "integrate",
                            "limit",
                            "series",
                            "determinant",
                            "matrix_inverse",
                            "matrix_rank",
                            "eigenvalues",
                            "mean",
                            "median",
                            "variance",
                        ],
                        "description": "Mathematical operation to perform",
                    },
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression",
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable, default x",
                    },
                    "point": {
                        "type": "number",
                        "description": "Point for limit calculation",
                    },
                    "lower": {
                        "type": "number",
                        "description": "Lower integration bound",
                    },
                    "upper": {
                        "type": "number",
                        "description": "Upper integration bound",
                    },
                    "order": {
                        "type": "integer",
                        "description": "Series expansion order",
                    },
                    "matrix": {"type": "array", "description": "Input matrix"},
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Statistical dataset",
                    },
                },
                "required": ["operation"],
            },
        }

    def parse(self, expression):
        """Parse a SymPy expression with standard and implicit-multiplication rules.

        Args:
            expression: Source string representing a mathematical expression.

        Returns:
            A SymPy expression tree.

        Raises:
            SyntaxError, TypeError, ValueError: Depending on the parser failure.
        """
        return parse_expr(
            expression, transformations=self.transformations, evaluate=True
        )

    def execute(
        self,
        operation: str,
        expression: str = None,
        variable: str = "x",
        point: float = 0,
        lower: float = None,
        upper: float = None,
        order: int = 5,
        matrix: list = None,
        data=None,
    ):
        """Perform the selected symbolic, numeric, matrix, or statistics operation.

        Args:
            operation: Supported calculator operation such as ``solve``,
                ``differentiate``, ``determinant``, or ``mean``.
            expression: Expression used by algebra/calculus operations.
            variable: Symbol used for equation solving and calculus.
            point: Evaluation point for limits and series.
            lower: Optional lower bound accepted by the tool schema.
            upper: Optional upper bound accepted by the tool schema.
            order: Series expansion order, constrained to 1 through 30.
            matrix: Matrix data for determinant, inverse, rank, or eigenvalues.
            data: Numeric values for mean, median, or variance.

        Returns:
            ``Result: ...`` with the operation result, or ``Calculation error: ...``
            when the operation, input, or SymPy computation fails.

        Notes:
            Although ``lower`` and ``upper`` are accepted, the current operation
            dispatcher does not use them.
        """
        try:
            x = sp.Symbol(variable)

            # BASIC CALCULATOR

            if operation == "calculate":
                result = self.parse(expression)

                if result.free_symbols:
                    result = sp.simplify(result)
                else:
                    result = result.evalf(15)

            # ALGEBRA

            elif operation == "simplify":
                result = sp.simplify(self.parse(expression))

            elif operation == "expand":
                result = sp.expand(self.parse(expression))

            elif operation == "factor":
                result = sp.factor(self.parse(expression))

            elif operation == "solve":
                result = sp.solve(self.parse(expression), x)

            # CALCULUS

            elif operation == "differentiate":
                result = sp.diff(self.parse(expression), x)

            elif operation == "integrate":
                result = sp.integrate(self.parse(expression), x)

            elif operation == "limit":
                result = sp.limit(self.parse(expression), x, point)

            elif operation == "series":
                if not 1 <= order <= 30:
                    raise ValueError("Invalid series order")

                result = sp.series(self.parse(expression), x, point, order)

            # LINEAR ALGEBRA

            elif operation in [
                "determinant",
                "matrix_inverse",
                "matrix_rank",
                "eigenvalues",
            ]:
                if matrix is None:
                    raise ValueError("Matrix is required")

                M = sp.Matrix(matrix)

                if operation == "determinant":
                    result = M.det()

                elif operation == "matrix_inverse":
                    result = M.inv()

                elif operation == "matrix_rank":
                    result = M.rank()

                elif operation == "eigenvalues":
                    result = M.eigenvals()

            # STATISTICS

            elif operation in ["mean", "median", "variance"]:
                if not data:
                    raise ValueError("Dataset is required")

                if operation == "mean":
                    result = sp.N(sp.Add(*data) / len(data))

                elif operation == "median":
                    result = sp.N(sp.median(data))

                elif operation == "variance":
                    if len(data) < 2:
                        raise ValueError("At least two values required")

                    result = sp.N(sp.variance(data))

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return f"Result: {result}"

        except Exception as e:
            return f"Calculation error: {e}"


def get_date_and_time() -> str:
    """Return the current wall-clock time in the Asia/Kolkata timezone.

    Inputs:
        None.

    Returns:
        A formatted weekday, date, time, and timezone string suitable for
        time-sensitive answers.
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return (
        "Current date and time: "
        f"{now.strftime('%A, %B %d, %Y at %H:%M %Z')} (Asia/Kolkata)"
    )
