from pathlib import Path
from typing import Literal

from ..profiles import AGENT_PROFILES
from ..tools import NativeTools


def _resolve_workspace_path(path: str, workspace: Path) -> str:
    """Resolve a candidate path and reject targets outside the active workspace.

    Args:
        path: Relative or absolute user-supplied path.
        workspace: Resolved root directory defining the permitted boundary.

    Returns:
        The absolute normalized target path as a string.

    Raises:
        ValueError: If the resolved target is outside ``workspace``.
    """
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
    """Create a native tool registry for one configured agent profile.

    Args:
        agent_name: Profile key selecting which tool module to expose.

    Returns:
        A ``NativeTools`` registry bound to the current working directory, with
        that profile's tools and approval requirements.

    Raises:
        ValueError: If the profile is unknown or its tool module is unsupported.
    """
    profile = AGENT_PROFILES.get(agent_name)
    if profile is None:
        raise ValueError(f"Unknown agent profile: {agent_name}")

    workspace = Path.cwd().resolve()
    tools = NativeTools()

    if profile.tool_module == "code_tools":
        from . import code_tools

        tools.requires_approval = ["run_python", "run_python_file", "run_tests"]

        def list_workspace_python_files(directory="."):
            """List Python files and subdirectories within the workspace.

            Args:
                directory: Workspace-relative or absolute directory path.

            Returns:
                A sorted Python-file and subdirectory listing.
            """
            return code_tools.list_files(
                _resolve_workspace_path(directory, workspace)
            )

        def read_workspace_python_file(file_path):
            """Read one Python source file from within the workspace.

            Args:
                file_path: Workspace-relative or absolute ``.py`` path.

            Returns:
                Complete UTF-8 source text.
            """
            return code_tools.read_file(
                _resolve_workspace_path(file_path, workspace)
            )

        def find_workspace_python_files(pattern):
            """Find matching Python files under the workspace root.

            Args:
                pattern: Glob matched against workspace-relative Python paths.

            Returns:
                A sorted list of matching paths or a no-files message.
            """
            return code_tools.find_files(pattern, workspace=str(workspace))

        def inspect_workspace_file(file_path):
            """Build a structural outline for a workspace Python source file.

            Args:
                file_path: Workspace-relative or absolute ``.py`` path.

            Returns:
                An outline of top-level symbols, class members, and raise sites.
            """
            return code_tools.inspect_file(
                _resolve_workspace_path(file_path, workspace),
                workspace=str(workspace),
            )

        def write_workspace_file(file_path, content):
            """Create or replace a Python source file within the workspace.

            Args:
                file_path: Workspace-relative or absolute ``.py`` destination.
                content: Complete source content to write.

            Returns:
                A status message identifying the written path.
            """
            return code_tools.write_file(
                _resolve_workspace_path(file_path, workspace),
                content,
                workspace=str(workspace),
            )

        def edit_workspace_file(file_path, old_text, new_text):
            """Replace one unique text occurrence in a workspace Python file.

            Args:
                file_path: Workspace-relative or absolute ``.py`` target.
                old_text: Non-empty exact text expected once in the file.
                new_text: Replacement text.

            Returns:
                A status message identifying the updated path.
            """
            return code_tools.edit_file(
                _resolve_workspace_path(file_path, workspace),
                old_text,
                new_text,
                workspace=str(workspace),
            )

        def run_workspace_tests(test_path="tests"):
            """Run unittest discovery in a workspace directory.

            Args:
                test_path: Workspace-relative or absolute test directory.

            Returns:
                Test output or a formatted test-failure result.
            """
            return code_tools.run_tests(test_path, workspace=str(workspace))

        tools.add_tool(
            "list_files",
            list_workspace_python_files,
            "List Python files and folders in the current workspace. Non-Python "
            "files are omitted.",
        )
        tools.add_tool(
            "read_file",
            read_workspace_python_file,
            "Read a Python source file (.py) in the current workspace. Non-Python "
            "files such as .jsonl logs are not supported; file_path must be a file.",
        )

        def read_workspace_lines(
            file_path: str, start_line: int = 1, end_line: int = 50
        ) -> str:
            """Read an inclusive numbered range from a workspace Python file.

            Args:
                file_path: Workspace-relative or absolute Python file path.
                start_line: First 1-based line to include.
                end_line: Last 1-based line to include.

            Returns:
                Selected lines with their original line numbers.
            """
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
            find_workspace_python_files,
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
            """Search permitted workspace text files for literal text or a regex.

            Args:
                query: Non-empty literal or regular-expression pattern.
                file_glob: Glob used to select files.
                use_regex: Whether to interpret ``query`` as a regular expression.
                case_sensitive: Whether matching preserves letter case.
                context_lines: Number of surrounding lines included per match.
                max_results: Maximum matching lines returned.

            Returns:
                Formatted match blocks, or a no-results message.
            """
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
            inspect_workspace_file,
            "Inspect a Python file's top-level variables, functions, classes, "
            "methods, signatures, decorators, docstrings, and line ranges.",
        )

        def search_workspace_symbols(
            file_path: str,
            query: str = "",
            symbol_type: Literal["all", "variable", "function", "class"] = "all",
        ) -> str:
            """Search a Python file's AST for matching symbols.

            Args:
                file_path: Workspace Python file to inspect.
                query: Case-insensitive substring filter; empty matches all names.
                symbol_type: Restrict matches to variables, functions, classes,
                    or all supported kinds.

            Returns:
                Numbered symbol locations, or a no-symbols message.
            """
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
            write_workspace_file,
            "Create or replace a Python source file (.py) in the current workspace.",
        )
        tools.add_tool(
            "edit_file",
            edit_workspace_file,
            "Replace one exact text occurrence in a Python source file (.py) in the "
            "current workspace.",
        )

        def run_workspace_python(code: str) -> str:
            """Execute a Python snippet with the workspace as its working directory.

            Args:
                code: Python source to execute after the registry's approval check.

            Returns:
                Captured stdout, stderr, or a formatted execution failure.
            """
            return code_tools.run_python(code, workspace=str(workspace))

        tools.add_tool(
            "run_python",
            run_workspace_python,
            "Run a Python snippet after approval.",
        )

        def run_workspace_python_file(file_path: str) -> str:
            """Execute a Python file located within the workspace.

            Args:
                file_path: Workspace-relative or absolute path to a ``.py`` file.

            Returns:
                Captured execution output or a formatted execution failure.
            """
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
            run_workspace_tests,
            "Run unittest discovery in a workspace test directory after approval.",
        )
    elif profile.tool_module == "file_tools":
        from .file_tools import list_files, read_file, unzip_file

        def list_workspace_files(path: str = ".") -> str:
            """List visible files and directories at a workspace path.

            Args:
                path: Workspace-relative or absolute directory path.

            Returns:
                A formatted listing, or a message describing a missing/non-directory
                path.
            """
            return list_files(_resolve_workspace_path(path, workspace))

        def read_workspace_file(
            file_path: str, start_line: int = 1, end_line: int = -1
        ) -> str:
            """Read a supported file while resolving it inside the workspace.

            Args:
                file_path: Workspace-relative or absolute file path.
                start_line: First line to read for line-addressable text files.
                end_line: Last line to read, or ``-1`` for the remaining file.

            Returns:
                Text, formatted CSV/Excel contents, or a file-not-found message.
            """
            return read_file(
                _resolve_workspace_path(file_path, workspace),
                start_line,
                end_line,
            )

        def unzip_workspace_file(zip_path: str, extract_to: str | None = None) -> str:
            """Extract a workspace archive to a workspace-contained destination.

            Args:
                zip_path: Workspace-relative or absolute ZIP path.
                extract_to: Optional workspace destination; defaults to a folder
                    named after the archive.

            Returns:
                An extraction summary and up to twenty archive member names.

            Raises:
                ValueError: If the archive or destination escapes the workspace.
            """
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
