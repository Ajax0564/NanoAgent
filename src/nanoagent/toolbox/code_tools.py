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
    if path.suffix.lower() != ".py":
        raise ValueError(f"Only Python source files (.py) are supported: '{path}'.")


def _safe_path(path: str, workspace: str = ".") -> Path:
    """Resolve a path and keep it inside the workspace."""
    root = Path(workspace).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path '{path}' is outside the workspace.") from error
    return target


def list_files(directory: str = ".", workspace: str = ".") -> str:
    """List Python files and folders in a workspace directory."""
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
    """Read a Python source file from the workspace."""
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
    """Read an inclusive range of numbered lines from a Python source file."""
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
    """Find matching Python files, skipping symlinks and generated folders."""
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Workspace '{workspace}' is not a directory.")

    def raise_walk_error(error: OSError) -> None:
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
    """Search workspace text files and return matching lines with context."""
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
    """Find variable, function, or class names in one Python file."""
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
    """Create or replace a Python source file."""
    target = _safe_path(path, workspace)
    _require_python_file(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written to '{path}'."


def edit_file(path: str, old_text: str, new_text: str, workspace: str = ".") -> str:
    """Replace one exact text occurrence in a Python source file."""
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
    if "error" in result:
        raise OSError(f"Could not capture Python {stream_name}.") from result["error"]
    output = result["output"].decode("utf-8", errors="replace")
    if result["truncated"]:
        output += f"\n[{stream_name} truncated at {MAX_PYTHON_OUTPUT_BYTES} bytes]"
    return output


def _execute_python(command: list[str], workspace: str) -> str:
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
    """Run a Python snippet in the workspace with bounded time and output."""
    root = Path(workspace).resolve()
    return _execute_python([sys.executable, "-E", "-s", "-c", code], str(root))


def run_python_file(file_path: str, workspace: str = ".") -> str:
    """Run a Python file inside the workspace with bounded time and output."""
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
    """Run unittest discovery for a workspace test directory."""
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
    return [ast.unparse(decorator) for decorator in getattr(node, "decorator_list", [])]


def _get_nested_functions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    nested = []

    def visit_scope(
        scope: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_name: str,
    ) -> None:
        for statement in scope.body:
            visit(statement, parent_name)

    def visit(statement: ast.AST, parent_name: str) -> None:
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
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _get_target_names(item)]
    return []


def _get_class_attributes(node: ast.ClassDef) -> list[str]:
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
    """Inspect classes, functions, methods and globals in a Python file."""

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

        output.append(f"  {node.name}")

        if bases:
            output.append(f"    Inherits: {', '.join(bases)}")

        output.append(f"    Lines: {node.lineno}-{node.end_lineno}")

        docstring = ast.get_docstring(node)
        if docstring:
            output.append(f"    Docstring: {docstring}")

        attributes = _get_class_attributes(node)
        if attributes:
            output.append("    CLASS ATTRIBUTES:")
            output.extend(f"      {attribute}" for attribute in attributes)

        output.append("    METHODS:")

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
