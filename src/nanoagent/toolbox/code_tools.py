import ast
import os
import re
import signal
import subprocess
import sys
import threading
import tokenize
from pathlib import Path
from typing import Any, BinaryIO, Literal

# add git style del, edit indicator with color code
TEST_TIMEOUT_SECONDS = 30
MAX_TEST_OUTPUT_LENGTH = 10_000
PYTHON_TIMEOUT_SECONDS = 30
MAX_PYTHON_OUTPUT_BYTES = 10_000
IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", "node_modules"}


def _require_python_file(path: Path) -> None:
    """Reject paths whose suffix is not the supported Python source extension.

    Args:
        path: Resolved path to validate.

    Returns:
        None when the suffix is ``.py`` (case-insensitive).

    Raises:
        ValueError: If ``path`` does not end in ``.py``.
    """
    if path.suffix.lower() != ".py":
        raise ValueError(f"Only Python source files (.py) are supported: '{path}'.")


def _safe_path(path: str, workspace: str = ".") -> Path:
    """Resolve a workspace-relative path and prevent escaping the workspace.

    Args:
        path: Relative or absolute candidate path.
        workspace: Root directory that bounds the candidate.

    Returns:
        The resolved target path.

    Raises:
        ValueError: If the resolved target is outside the workspace root.
    """
    root = Path(workspace).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path '{path}' is outside the workspace.") from error
    return target


def list_files(directory: str = ".", workspace: str = ".") -> str:
    """List Python files and subdirectories in one workspace directory.

    Args:
        directory: Directory to list, relative to ``workspace`` unless absolute.
        workspace: Root directory that bounds file access.

    Returns:
        A sorted newline-separated listing. Directories end with ``/``; an empty
        directory returns ``(empty)``.

    Raises:
        ValueError: If the path escapes the workspace.
        NotADirectoryError: If the target is not a directory.
    """
    target = _safe_path(directory, workspace)
    if not target.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory.")
    entries = sorted(
        entry
        for entry in target.iterdir()
        if entry.is_dir() or entry.suffix.lower() == ".py"
    )
    return (
        "\n".join(entry.name + ("/" if entry.is_dir() else "") for entry in entries)
        or "(empty)"
    )


def read_file(path: str, workspace: str = ".") -> str:
    """Read a UTF-8 Python source file from inside the workspace.

    Args:
        path: Workspace-relative or absolute path to a ``.py`` file.
        workspace: Root directory that bounds file access.

    Returns:
        The complete source text.

    Raises:
        ValueError: If the path escapes the workspace or is not a Python file.
        FileNotFoundError: If the target does not exist as a file.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    target = _safe_path(path, workspace)
    _require_python_file(target)
    if not target.is_file():
        raise FileNotFoundError(f"'{path}' is not a file.")
    return target.read_text(encoding="utf-8")


def read_lines(
    path: str,
    start_line: int = 1,
    end_line: int = 50,
    workspace: str = ".",
) -> str:
    """Read an inclusive, 1-based line range from a Python source file.

    Args:
        path: Workspace-relative or absolute path to a ``.py`` file.
        start_line: First 1-based line to include.
        end_line: Last 1-based line to include.
        workspace: Root directory that bounds file access.

    Returns:
        Selected lines prefixed with their source line numbers, or
        ``(empty range)`` when the requested start is beyond end-of-file.

    Raises:
        ValueError: If line bounds are invalid or the path escapes the workspace.
        FileNotFoundError: If the target does not exist as a file.
    """
    if start_line < 1:
        raise ValueError("start_line must be at least 1.")
    if end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line.")

    target = _safe_path(path, workspace)
    _require_python_file(target)
    if not target.is_file():
        raise FileNotFoundError(f"'{path}' is not a file.")

    lines = target.read_text(encoding="utf-8").splitlines()
    selected = lines[start_line - 1 : end_line]
    return (
        "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        or "(empty range)"
    )


def find_files(pattern: str, workspace: str = ".") -> str:
    """Recursively find Python files matching a workspace-relative glob.

    Args:
        pattern: Glob matched against each relative Python file path.
        workspace: Root directory to search.

    Returns:
        Sorted relative paths, at most 30 entries, with a truncation notice when
        more matches exist; otherwise ``No files found.``.

    Raises:
        NotADirectoryError: If the workspace is not a directory.
        OSError: If directory traversal fails.
    """
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Workspace '{workspace}' is not a directory.")

    def raise_walk_error(error: OSError) -> None:
        """Propagate an ``os.walk`` traversal error to the caller.

        Args:
            error: Filesystem error reported during traversal.

        Returns:
            None. Always re-raises ``error``.
        """
        raise error

    matches = []
    for current, directories, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=raise_walk_error
    ):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in IGNORED_DIRECTORIES
            and not (current_path / name).is_symlink()
        ]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                continue
            relative_path = path.relative_to(root)
            if relative_path.suffix.lower() == ".py" and relative_path.match(pattern):
                matches.append(relative_path.as_posix())

    matches.sort()
    if len(matches) > 30:
        return "\n".join(matches[:30]) + f"\n(truncated; {len(matches)} files found)"
    return "\n".join(matches) or "No files found."


def _matches_file_glob(path: Path, pattern: str) -> bool:
    """Match a relative path, including a root-level ``**/`` pattern.

    Args:
        path: Relative file path to test.
        pattern: Glob pattern supplied by the caller.

    Returns:
        ``True`` if the path matches the pattern or its leading-``**/`` variant.
    """
    return path.match(pattern) or (
        pattern.startswith("**/") and path.match(pattern[3:])
    )


def search_text(
    query: str,
    file_glob: str = "**/*",
    use_regex: bool = False,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_results: int = 50,
    workspace: str = ".",
) -> str:
    """Search workspace text files and return bounded, contextual match blocks.

    Args:
        query: Non-empty literal text or regular expression.
        file_glob: Glob used to select files; generated and symlinked paths are
            skipped.
        use_regex: Interpret ``query`` as a regular expression when true.
        case_sensitive: Preserve case when matching if true.
        context_lines: Number of lines before and after each match to include.
        max_results: Maximum number of matching lines, not output context lines.
        workspace: Root directory to search.

    Returns:
        Formatted file/line blocks, a no-match message, or a truncation notice
        after reaching ``max_results``.

    Raises:
        ValueError: If query/bounds are invalid or the regular expression is bad.
        NotADirectoryError: If the workspace is not a directory.
    """
    if not query:
        raise ValueError("query must not be empty.")
    if context_lines < 0:
        raise ValueError("context_lines must be non-negative.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    expression = None
    if use_regex:
        try:
            expression = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"Invalid regular expression: {error}") from error
    else:
        normalized_query = query if case_sensitive else query.casefold()

    root = Path(workspace).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Workspace '{workspace}' is not a directory.")

    results = []
    result_count = 0
    truncated = False
    for current, directories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name not in IGNORED_DIRECTORIES
            and not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink():
                continue
            relative_path = path.relative_to(root)
            if not _matches_file_glob(relative_path, file_glob):
                continue

            try:
                with path.open("rb") as source_file:
                    prefix = source_file.read(8192)
                if b"\x00" in prefix:
                    continue
                prefix.decode("utf-8-sig")
                content = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            if "\x00" in content:
                continue

            lines = content.splitlines()
            matching_lines = []
            for line_number, line in enumerate(lines, start=1):
                if expression is not None:
                    matches = expression.search(line) is not None
                else:
                    searchable_line = line if case_sensitive else line.casefold()
                    matches = normalized_query in searchable_line
                if matches:
                    matching_lines.append(line_number)

            remaining = max_results - result_count
            if len(matching_lines) > remaining:
                matching_lines = matching_lines[:remaining]
                truncated = True

            if matching_lines:
                ranges = []
                for line_number in matching_lines:
                    start = max(1, line_number - context_lines)
                    end = min(len(lines), line_number + context_lines)
                    if ranges and start <= ranges[-1][1] + 1:
                        ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
                    else:
                        ranges.append((start, end))

                blocks = [relative_path.as_posix()]
                for start, end in ranges:
                    for line_number in range(start, end + 1):
                        marker = ">" if line_number in matching_lines else " "
                        line = lines[line_number - 1]
                        if len(line) > 1000:
                            line = line[:1000] + " [line truncated]"
                        blocks.append(f"{marker} {line_number}: {line}")
                    blocks.append("")
                results.append("\n".join(blocks).rstrip())
                result_count += len(matching_lines)

            if truncated:
                break
        if truncated:
            break

    if not results:
        return "No matches found."
    output = "\n\n".join(results)
    if truncated:
        output += f"\n\n[Results truncated at {max_results} matching lines]"
    return output


def search_symbols(
    file_path: str,
    query: str = "",
    symbol_type: Literal["all", "variable", "function", "class"] = "all",
    workspace: str = ".",
) -> str:
    """Search a Python AST for matching variable, function, and class names.

    Args:
        file_path: Workspace-relative or absolute Python file to inspect.
        query: Case-insensitive substring filter; an empty string matches all.
        symbol_type: Restrict results to ``variable``, ``function``, ``class``,
            or ``all``.
        workspace: Root directory that bounds file access.

    Returns:
        Up to 50 unique symbol locations in source order, or ``No symbols found.``.

    Raises:
        ValueError: If the type filter is invalid, the path escapes the workspace,
            or the file is not Python source.
        FileNotFoundError: If the target is not a file.
        SyntaxError: If the source cannot be parsed.
    """
    if symbol_type not in {"all", "variable", "function", "class"}:
        raise ValueError("symbol_type must be all, variable, function, or class.")

    root = Path(workspace).resolve()
    target = _safe_path(file_path, str(root))
    _require_python_file(target)
    if not target.is_file():
        raise FileNotFoundError(f"'{file_path}' is not a file.")

    with tokenize.open(target) as source_file:
        tree = ast.parse(source_file.read(), filename=str(target))

    query = query.casefold()
    matches = []
    for node in ast.walk(tree):
        symbol = None
        if isinstance(node, ast.ClassDef):
            symbol = ("class", node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol = ("function", node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            symbol = ("variable", node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            symbol = ("variable", node.attr)
        elif isinstance(node, ast.arg):
            symbol = ("variable", node.arg)

        if (
            symbol is not None
            and symbol_type in {"all", symbol[0]}
            and query in symbol[1].casefold()
        ):
            matches.append((node.lineno, node.col_offset, symbol[0], symbol[1]))

    matches.sort()
    unique_matches = {}
    for match in matches:
        unique_matches.setdefault((match[2], match[3]), match)
    matches = sorted(unique_matches.values())

    file_name = target.relative_to(root).as_posix()
    lines = [
        f"{file_name}:{line_number}: {kind} {name}"
        for line_number, _, kind, name in matches[:50]
    ]
    if len(matches) > 50:
        lines.append(f"(truncated; {len(matches)} matches found)")
    return "\n".join(lines) or "No symbols found."


def write_file(path: str, content: str, workspace: str = ".") -> str:
    """Create or replace a Python source file inside the workspace.

    Args:
        path: Workspace-relative or absolute destination ending in ``.py``.
        content: Complete source text to write as UTF-8.
        workspace: Root directory that bounds the destination.

    Returns:
        A status string naming the written path.

    Raises:
        ValueError: If the destination escapes the workspace or is not Python.
        OSError: If its parent directory or file cannot be created/written.
    """
    target = _safe_path(path, workspace)
    _require_python_file(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written to '{path}'."


def edit_file(path: str, old_text: str, new_text: str, workspace: str = ".") -> str:
    """Replace exactly one occurrence in a workspace Python source file.

    Args:
        path: Workspace-relative or absolute target ending in ``.py``.
        old_text: Non-empty exact text expected to occur once.
        new_text: Replacement text.
        workspace: Root directory that bounds the target.

    Returns:
        A status string naming the updated path.

    Raises:
        ValueError: If the target escapes the workspace, is not Python, the old
            text is empty, or it occurs zero or multiple times.
        FileNotFoundError: If the target is not a file.
        OSError: If reading or writing the file fails.
    """
    target = _safe_path(path, workspace)
    _require_python_file(target)
    if not target.is_file():
        raise FileNotFoundError(f"'{path}' is not a file.")
    if not old_text:
        raise ValueError("old_text must not be empty.")

    content = target.read_text(encoding="utf-8")
    matches = content.count(old_text)
    if matches != 1:
        raise ValueError(f"Expected one match for old_text, found {matches}.")

    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Updated '{path}'."


def _execution_environment() -> dict[str, str]:
    """Copy the small set of host environment variables needed by child Python.

    Inputs:
        None. Reads the current process environment.

    Returns:
        A new mapping containing only path, Windows, and temporary/home variables.
    """
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _capture_limited_output(stream: BinaryIO, result: dict[str, Any]) -> None:
    """Drain a child-process pipe while retaining only the configured byte limit.

    Args:
        stream: Binary stdout or stderr pipe to read until EOF.
        result: Mutable dictionary populated with ``output`` bytes and a
            ``truncated`` flag, or an ``error`` on read failure.

    Returns:
        None. The stream is consumed by the current thread.
    """
    output = bytearray()
    truncated = False
    try:
        while chunk := stream.read(8192):
            remaining = MAX_PYTHON_OUTPUT_BYTES - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated = True
        result["output"] = bytes(output)
        result["truncated"] = truncated
    except OSError as error:
        result["error"] = error


def _read_captured_output(result: dict[str, Any], stream_name: str) -> str:
    """Decode a captured process stream and annotate truncated output.

    Args:
        result: Capture state populated by ``_capture_limited_output``.
        stream_name: Label used in the truncation message, usually stdout/stderr.

    Returns:
        UTF-8-decoded output with replacement characters for invalid bytes.

    Raises:
        OSError: If the capture thread recorded a read error.
    """
    if "error" in result:
        raise OSError(f"Could not capture Python {stream_name}.") from result["error"]
    output = result["output"].decode("utf-8", errors="replace")
    if result["truncated"]:
        output += f"\n[{stream_name} truncated at {MAX_PYTHON_OUTPUT_BYTES} bytes]"
    return output


def _execute_python(command: list[str], workspace: str) -> str:
    """Run a Python child process with bounded runtime and captured output.

    Args:
        command: Executable and arguments passed to ``subprocess.Popen``.
        workspace: Working directory for the process.

    Returns:
        Trimmed stdout, stdout plus stderr, or a formatted non-zero exit result.

    Raises:
        TimeoutError: If execution exceeds the configured timeout or a descendant
            keeps an output pipe open.
        OSError: If process startup, pipe capture, or cleanup fails.
    """
    process_options = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True

    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=_execution_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **process_options,
    )
    stdout_result: dict = {}
    stderr_result: dict = {}
    stdout_thread = threading.Thread(
        target=_capture_limited_output,
        args=(process.stdout, stdout_result),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_capture_limited_output,
        args=(process.stderr, stderr_result),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=PYTHON_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise TimeoutError("A Python child process kept an output pipe open.")
    process.stdout.close()
    process.stderr.close()

    stdout = _read_captured_output(stdout_result, "stdout")
    stderr = _read_captured_output(stderr_result, "stderr")
    if timed_out:
        raise TimeoutError(
            f"Python execution timed out after {PYTHON_TIMEOUT_SECONDS} seconds.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if process.returncode:
        return (
            f"Exit code {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ).strip()
    if stderr:
        return f"{stdout}\nSTDERR:\n{stderr}".strip()
    return stdout.strip() or "(no output)"


def run_python(code: str, workspace: str = ".") -> str:
    """Execute a Python source snippet in a workspace child process.

    Args:
        code: Python source passed to the interpreter's ``-c`` option.
        workspace: Working directory for execution.

    Returns:
        Captured output or a formatted non-zero exit result.

    Raises:
        TimeoutError: If the process exceeds its time or output-pipe limits.
        OSError: If the child process cannot be started or captured.
    """
    root = Path(workspace).resolve()
    return _execute_python([sys.executable, "-E", "-s", "-c", code], str(root))


def run_python_file(file_path: str, workspace: str = ".") -> str:
    """Execute a workspace Python file with bounded runtime and output.

    Args:
        file_path: Workspace-relative or absolute path to the script.
        workspace: Root and working directory for the process.

    Returns:
        Captured output or a formatted non-zero exit result.

    Raises:
        ValueError: If the path escapes the workspace or is not a ``.py`` file.
        FileNotFoundError: If the target is not a file.
        TimeoutError: If execution exceeds its configured limits.
        OSError: If the child process cannot be started or captured.
    """
    root = Path(workspace).resolve()
    script = _safe_path(file_path, str(root))
    if not script.is_file():
        raise FileNotFoundError(f"'{file_path}' is not a file.")
    if script.suffix.lower() != ".py":
        raise ValueError("file_path must point to a .py file.")
    return _execute_python(
        [sys.executable, "-E", "-s", str(script)],
        str(root),
    )


def run_tests(test_path: str = "tests", workspace: str = ".") -> str:
    """Run Python ``unittest`` discovery in a workspace directory.

    Args:
        test_path: Workspace-relative or absolute directory containing tests.
        workspace: Root directory that bounds test discovery and sets the cwd.

    Returns:
        Combined test output, truncated to the configured tail when necessary.
        Non-zero test exits are returned as a failure string.

    Raises:
        ValueError: If the test path escapes the workspace.
        NotADirectoryError: If the test directory does not exist as a directory.
        TimeoutError: If discovery exceeds the configured time limit.
        OSError: If the test process cannot be started.
    """
    root = Path(workspace).resolve()
    test_directory = _safe_path(test_path, str(root))
    if not test_directory.is_dir():
        raise NotADirectoryError(f"Test directory '{test_path}' does not exist.")

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(test_directory),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = "\n".join(
            part.decode("utf-8", errors="replace")
            if isinstance(part, bytes)
            else part or ""
            for part in (error.stdout, error.stderr)
        )
        raise TimeoutError(
            f"Tests timed out after {TEST_TIMEOUT_SECONDS} seconds.\n"
            f"{output[-MAX_TEST_OUTPUT_LENGTH:]}"
        ) from error

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if len(output) > MAX_TEST_OUTPUT_LENGTH:
        output = output[-MAX_TEST_OUTPUT_LENGTH:]
        output = f"[Earlier test output omitted]\n{output}"
    if result.returncode:
        return f"Tests failed (exit code {result.returncode}).\n{output}"
    return output or "Tests passed with no output."


def _get_function_arguments(node: ast.FunctionDef) -> str:
    """Render a function's arguments, annotations, and defaults from its AST.

    Args:
        node: Parsed synchronous function definition.

    Returns:
        A comma-separated argument signature fragment suitable for an outline.
    """
    args = []

    positional = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(positional) - len(node.args.defaults))
    defaults += node.args.defaults

    for arg, default in zip(positional, defaults):
        value = arg.arg

        if arg.annotation:
            value += f": {ast.unparse(arg.annotation)}"

        if default is not None:
            value += f" = {ast.unparse(default)}"

        args.append(value)

    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)

    if node.args.kwonlyargs:
        if not node.args.vararg:
            args.append("*")

        for arg, default in zip(
            node.args.kwonlyargs,
            node.args.kw_defaults,
        ):
            value = arg.arg

            if arg.annotation:
                value += f": {ast.unparse(arg.annotation)}"

            if default is not None:
                value += f" = {ast.unparse(default)}"

            args.append(value)

    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)

    return ", ".join(args)


def _get_decorators(node: ast.AST) -> list[str]:
    """Render decorator expressions attached to an AST node.

    Args:
        node: Function or other AST node that may have ``decorator_list``.

    Returns:
        Decorator expressions as source-like strings, or an empty list.
    """
    return [ast.unparse(decorator) for decorator in getattr(node, "decorator_list", [])]


def _get_nested_functions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Find nested functions while excluding definitions inside nested classes.

    Args:
        node: Function body whose nested function definitions are inspected.
        qualified_name: Parent name used to qualify nested function names.

    Returns:
        Ordered pairs of qualified names and nested function AST nodes.
    """
    nested = []

    def visit_scope(
        scope: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_name: str,
    ) -> None:
        """Visit statements directly belonging to one function scope.

        Args:
            scope: Function or async-function AST node to traverse.
            parent_name: Qualified name of the current function scope.

        Returns:
            None. Adds discovered descendants to the enclosing ``nested`` list.
        """
        for statement in scope.body:
            visit(statement, parent_name)

    def visit(statement: ast.AST, parent_name: str) -> None:
        """Walk a statement and record nested function definitions.

        Args:
            statement: AST statement to inspect.
            parent_name: Qualified name used for any child function.

        Returns:
            None. Nested classes are intentionally not traversed.
        """
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{parent_name}.{statement.name}"
            nested.append((name, statement))
            visit_scope(statement, name)
        elif isinstance(statement, ast.ClassDef):
            return
        else:
            for child in ast.iter_child_nodes(statement):
                visit(child, parent_name)

    visit_scope(node, qualified_name)
    return nested


def _get_target_names(target: ast.AST) -> list[str]:
    """Extract names from a simple or unpacked assignment target.

    Args:
        target: AST target expression from an assignment.

    Returns:
        Name identifiers found in ``Name``, tuple, or list targets; unsupported
        target forms produce an empty list.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _get_target_names(item)]
    return []


def _get_class_attributes(node: ast.ClassDef) -> list[str]:
    """Describe direct class-body assignments and annotations.

    Args:
        node: Class definition whose immediate statements are inspected.

    Returns:
        Attribute strings containing names, annotations, and unparsed assigned
        values for supported assignment forms.
    """
    attributes = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            annotation = None
            value = statement.value
            operator = None
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            annotation = statement.annotation
            value = statement.value
            operator = None
        elif isinstance(statement, ast.AugAssign):
            targets = [statement.target]
            annotation = None
            value = statement.value
            operator = ast.unparse(statement.op)
        else:
            continue

        for target in targets:
            for name in _get_target_names(target):
                attribute = name
                if annotation is not None:
                    attribute += f": {ast.unparse(annotation)}"
                if operator is not None:
                    attribute += f" {operator}= {ast.unparse(value)}"
                elif value is not None:
                    attribute += f" = {ast.unparse(value)}"
                attributes.append(attribute)

    return attributes


def _append_function_details(
    output: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    indent: str,
) -> None:
    """Append return annotation, decorators, source range, and docstring details.

    Args:
        output: Mutable outline lines to extend.
        node: Function AST node being described.
        indent: Prefix applied to each appended line.

    Returns:
        None. Mutates ``output`` in place.
    """
    if node.returns:
        output.append(f"{indent}Returns: {ast.unparse(node.returns)}")

    decorators = _get_decorators(node)
    if decorators:
        output.append(f"{indent}Decorators: {', '.join(decorators)}")

    output.append(f"{indent}Lines: {node.lineno}-{node.end_lineno}")

    docstring = ast.get_docstring(node)
    if docstring:
        output.append(f"{indent}Docstring: {docstring}")


def _append_nested_functions(
    output: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
    indent: str,
) -> None:
    """Append signatures and details for a function's nested definitions.

    Args:
        output: Mutable outline lines to extend.
        node: Parent function AST node.
        qualified_name: Qualified parent name for child entries.
        indent: Prefix used for nested-function headings.

    Returns:
        None. Mutates ``output`` in place; adds nothing if there are no children.
    """
    nested_functions = _get_nested_functions(node, qualified_name)
    if not nested_functions:
        return

    output.append(f"{indent}Nested functions:")
    for name, nested in nested_functions:
        nested_indent = indent + "  "
        output.append(f"{nested_indent}{name}({_get_function_arguments(nested)})")
        _append_function_details(output, nested, nested_indent + "  ")


def inspect_file(
    file_path: str,
    workspace: str = ".",
) -> str:
    """Build a structural outline of a workspace Python source file.

    Args:
        file_path: Workspace-relative or absolute path to inspect.
        workspace: Root directory that bounds file access.

    Returns:
        Text containing top-level assigned variable names, function and class
        signatures, nested functions, source line ranges, docstrings, class
        attributes, and raise statements. It does not include full function bodies.

    Raises:
        ValueError: If the path escapes the workspace or is not a ``.py`` file.
        FileNotFoundError: If the target does not exist as a file.
        SyntaxError: If the source cannot be parsed into a Python AST.
        OSError: If the source file cannot be read.
    """

    root = Path(workspace).resolve()
    target = _safe_path(file_path, str(root))

    _require_python_file(target)

    if not target.is_file():
        raise FileNotFoundError(f"'{file_path}' is not a file.")

    with tokenize.open(target) as source_file:
        source = source_file.read()

    tree = ast.parse(source, filename=str(target))

    output = [f"FILE: {target.relative_to(root).as_posix()}"]

    globals_found = []
    functions = []
    classes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)

        elif isinstance(node, ast.ClassDef):
            classes.append(node)

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]

            for target_node in targets:
                if isinstance(target_node, ast.Name):
                    globals_found.append(target_node.id)

    output.append("\nGLOBAL VARIABLES:")
    output.extend(f"  {name}" for name in globals_found)

    output.append("\nFUNCTIONS:")

    for node in functions:
        output.append(f"  {node.name}({_get_function_arguments(node)})")
        _append_function_details(output, node, "    ")
        _append_nested_functions(output, node, node.name, "    ")

    output.append("\nCLASSES:")

    for node in classes:
        bases = [ast.unparse(base) for base in node.bases]

        output.append(f"{node.name}")

        if bases:
            output.append(f"Inherits: {', '.join(bases)}")

        output.append(f"Lines: {node.lineno}-{node.end_lineno}")

        docstring = ast.get_docstring(node)
        if docstring:
            output.append(f"Docstring: {docstring}")

        attributes = _get_class_attributes(node)
        if attributes:
            output.append("CLASS ATTRIBUTES:")
            output.extend(f"{attribute}" for attribute in attributes)

        output.append("METHODS:")

        for method in node.body:
            if isinstance(
                method,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                args = _get_function_arguments(method)
                output.append(f"      {method.name}({args})")
                _append_function_details(output, method, "        ")
                _append_nested_functions(
                    output,
                    method,
                    f"{node.name}.{method.name}",
                    "        ",
                )

    raises = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Raise)),
        key=lambda node: node.lineno,
    )
    if raises:
        output.append("\nRAISE STATEMENTS:")
        for node in raises:
            exception = ast.unparse(node.exc) if node.exc is not None else "(bare)"
            output.append(f"  Line {node.lineno}: raise {exception}")

    return "\n".join(output)
