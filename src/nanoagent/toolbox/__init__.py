from pathlib import Path
from typing import Literal

from ..profiles import AGENT_PROFILES
from ..tools import NativeTools


def _resolve_workspace_path(path: str, workspace: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate

    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("File tools can only access the current workspace.") from error
    return str(resolved)


def create_native_code_tools(agent_name: str = "code_agent") -> NativeTools:
    """Create tools only from the selected agent's tool module."""
    profile = AGENT_PROFILES.get(agent_name)
    if profile is None:
        raise ValueError(f"Unknown agent profile: {agent_name}")

    workspace = Path.cwd().resolve()
    tools = NativeTools()

    if profile.tool_module == "code_tools":
        from . import code_tools

        tools.requires_approval = [
            "write_file",
            "edit_file",
            "run_python",
            "run_python_file",
            "run_tests",
        ]
        tools.add_tool(
            "list_files",
            lambda directory=".": code_tools.list_files(
                _resolve_workspace_path(directory, workspace)
            ),
            "List Python files and folders in the current workspace. Non-Python "
            "files are omitted.",
        )
        tools.add_tool(
            "read_file",
            lambda file_path: code_tools.read_file(
                _resolve_workspace_path(file_path, workspace)
            ),
            "Read a Python source file (.py) in the current workspace. Non-Python "
            "files such as .jsonl logs are not supported; file_path must be a file.",
        )

        def read_workspace_lines(
            file_path: str, start_line: int = 1, end_line: int = 50
        ) -> str:
            return code_tools.read_lines(
                _resolve_workspace_path(file_path, workspace),
                start_line,
                end_line,
                workspace=str(workspace),
            )

        tools.add_tool(
            "read_lines",
            read_workspace_lines,
            "Read selected numbered lines from a Python source file (.py). "
            "Non-Python files such as .jsonl logs are not supported.",
        )
        tools.add_tool(
            "find_files",
            lambda pattern: code_tools.find_files(pattern, workspace=str(workspace)),
            "Find workspace Python files (.py) matching a glob pattern. Non-Python "
            "files such as .jsonl logs are never returned.If the named file isn’t found after a direct read or one find_files lookup, stop.Tell the user the file wasn’t found and ask for its correct name or path.",
        )

        def search_workspace_text(
            query: str,
            file_glob: str = "**/*",
            use_regex: bool = False,
            case_sensitive: bool = False,
            context_lines: int = 2,
            max_results: int = 50,
        ) -> str:
            return code_tools.search_text(
                query,
                file_glob,
                use_regex,
                case_sensitive,
                context_lines,
                max_results,
                workspace=str(workspace),
            )

        tools.add_tool(
            "search_text",
            search_workspace_text,
            "Search text across workspace files using a literal query or regular "
            "expression. Supports file_glob (for example '*.py' or 'src/**/*.toml'), "
            "case_sensitive, context_lines, and max_results. Binary and generated "
            "files are skipped.",
        )
        tools.add_tool(
            "inspect_file",
            lambda file_path: code_tools.inspect_file(
                _resolve_workspace_path(file_path, workspace),
                workspace=str(workspace),
            ),
            "Inspect a Python file's top-level variables, functions, classes, "
            "methods, signatures, decorators, docstrings, and line ranges.",
        )

        def search_workspace_symbols(
            file_path: str,
            query: str = "",
            symbol_type: Literal["all", "variable", "function", "class"] = "all",
        ) -> str:
            return code_tools.search_symbols(
                _resolve_workspace_path(file_path, workspace),
                query,
                symbol_type,
                workspace=str(workspace),
            )

        tools.add_tool(
            "search_symbols",
            search_workspace_symbols,
            "Search a Python file for variable, function, or class names.",
        )
        tools.add_tool(
            "write_file",
            lambda file_path, content: code_tools.write_file(
                _resolve_workspace_path(file_path, workspace), content
            ),
            "Create or replace a Python source file (.py) after approval.",
        )
        tools.add_tool(
            "edit_file",
            lambda file_path, old_text, new_text: code_tools.edit_file(
                _resolve_workspace_path(file_path, workspace),
                old_text,
                new_text,
                workspace=str(workspace),
            ),
            "Replace one exact text occurrence in a Python source file (.py) after "
            "approval.",
        )

        def run_workspace_python(code: str) -> str:
            return code_tools.run_python(code, workspace=str(workspace))

        tools.add_tool(
            "run_python",
            run_workspace_python,
            "Run a Python snippet after approval.",
        )

        def run_workspace_python_file(file_path: str) -> str:
            return code_tools.run_python_file(
                _resolve_workspace_path(file_path, workspace),
                workspace=str(workspace),
            )

        tools.add_tool(
            "run_python_file",
            run_workspace_python_file,
            "Run a Python file in the workspace after approval.",
        )
        tools.add_tool(
            "run_tests",
            lambda test_path="tests": code_tools.run_tests(
                test_path, workspace=str(workspace)
            ),
            "Run unittest discovery in a workspace test directory after approval.",
        )
    elif profile.tool_module == "file_tools":
        from .file_tools import list_files, read_file, unzip_file

        def list_workspace_files(path: str = ".") -> str:
            """List files and directories within the current workspace."""
            return list_files(_resolve_workspace_path(path, workspace))

        def read_workspace_file(
            file_path: str, start_line: int = 1, end_line: int = -1
        ) -> str:
            """Read a supported file within the current workspace."""
            return read_file(
                _resolve_workspace_path(file_path, workspace),
                start_line,
                end_line,
            )

        def unzip_workspace_file(zip_path: str, extract_to: str | None = None) -> str:
            """Extract a workspace archive after approval."""
            resolved_zip = _resolve_workspace_path(zip_path, workspace)
            resolved_destination = (
                _resolve_workspace_path(extract_to, workspace)
                if extract_to is not None
                else None
            )
            return unzip_file(resolved_zip, resolved_destination)

        tools.requires_approval = ["unzip_file"]
        tools.add_tool(
            "list_files",
            list_workspace_files,
            "List files and directories in the current workspace.",
        )
        tools.add_tool(
            "read_file",
            read_workspace_file,
            "Read a supported file in the current workspace.",
        )
        tools.add_tool(
            "unzip_file",
            unzip_workspace_file,
            "Extract a workspace archive after user approval.",
        )
    elif profile.tool_module == "math_tools":
        from .math_tools import CalculatorTool, PlottingTool, get_date_and_time

        calculator = CalculatorTool()
        tools.add_tool(
            "calculator", calculator.execute, calculator.get_schema()["description"]
        )

        plotter = PlottingTool()
        tools.add_tool(
            "plot_expression",
            plotter.execute,
            plotter.get_schema()["description"],
        )
        tools.add_tool(
            "get_date_and_time",
            get_date_and_time,
            "Return the current date and time.",
        )
    else:
        raise ValueError(
            f"Unsupported tool module for agent profile '{agent_name}': "
            f"{profile.tool_module}"
        )

    return tools
