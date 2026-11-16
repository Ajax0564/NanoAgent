from pathlib import Path

from ..tools import NativeTools


def create_native_code_tools() -> NativeTools:
    """Create the native tool registry used by the command-line agent."""
    from .file_tools import execute_python, list_files, read_file
    from .math_tools import CalculatorTool, PlottingTool, get_date_and_time
    from .websearch_tools import WebSearchTool

    workspace = Path.cwd().resolve()

    def resolve_workspace_path(path: str) -> str:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate

        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as error:
            raise ValueError(
                "File tools can only access the current workspace."
            ) from error
        return str(resolved)

    def list_workspace_files(path: str = ".") -> str:
        """List files and directories within the current workspace."""
        return list_files(resolve_workspace_path(path))

    def read_workspace_file(
        file_path: str, start_line: int = 1, end_line: int = -1
    ) -> str:
        """Read a file within the current workspace."""
        return read_file(resolve_workspace_path(file_path), start_line, end_line)

    tools = NativeTools(requires_approval=["execute_python"])
    tools.add_tool(
        "list_files",
        list_workspace_files,
        "List files and directories in the current workspace.",
    )
    tools.add_tool(
        "read_file",
        read_workspace_file,
        "Read a text, CSV, or supported spreadsheet file in the current workspace.",
    )
    tools.add_tool(
        "execute_python",
        execute_python,
        "Run Python code after user approval.",
    )

    calculator = CalculatorTool()
    tools.add_tool(
        "calculator", calculator.execute, calculator.get_schema()["description"]
    )

    plotter = PlottingTool()
    tools.add_tool(
        "plot_expression", plotter.execute, plotter.get_schema()["description"]
    )

    web_search = WebSearchTool()
    tools.add_tool(
        "web_search", web_search.execute, web_search.get_schema()["description"]
    )
    tools.add_tool(
        "get_date_and_time",
        get_date_and_time,
        "Return the current date and time.",
    )
    return tools
