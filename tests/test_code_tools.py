import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoagent.models import Response
from nanoagent.tools import ToolFailure
from nanoagent.toolbox import create_native_code_tools
from nanoagent.toolbox.code_tools import (
    MAX_PYTHON_OUTPUT_BYTES,
    edit_file,
    find_files,
    inspect_file,
    list_files,
    read_file,
    read_lines,
    run_python,
    run_python_file,
    search_text,
    search_symbols,
    run_tests,
    write_file,
)


class CodeToolTests(unittest.TestCase):
    def test_file_tools_only_list_find_read_and_write_python_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text("value = 1\n", encoding="utf-8")
            (root / "trace.jsonl").write_text('{"event": "run"}\n', encoding="utf-8")

            self.assertIn("sample.py", list_files(workspace=str(root)))
            self.assertNotIn("trace.jsonl", list_files(workspace=str(root)))
            self.assertEqual(find_files("*", workspace=str(root)), "sample.py")
            with self.assertRaisesRegex(ValueError, r"Only Python source files"):
                read_file("trace.jsonl", workspace=str(root))
            with self.assertRaisesRegex(ValueError, r"Only Python source files"):
                read_lines("trace.jsonl", workspace=str(root))
            with self.assertRaisesRegex(ValueError, r"Only Python source files"):
                write_file("new.jsonl", "{}", workspace=str(root))
            with self.assertRaisesRegex(ValueError, r"Only Python source files"):
                edit_file(
                    "trace.jsonl",
                    '{"event": "run"}',
                    '{"event": "changed"}',
                    workspace=str(root),
                )

    def test_workspace_tools_read_write_and_edit_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_file("src/example.py", "answer = 1\n", workspace=str(root))

            self.assertIn("src/", list_files(workspace=str(root)))
            self.assertEqual(
                read_file("src/example.py", workspace=str(root)), "answer = 1\n"
            )
            edit_file(
                "src/example.py",
                "answer = 1",
                "answer = 2",
                workspace=str(root),
            )
            self.assertEqual(
                read_file("src/example.py", workspace=str(root)), "answer = 2\n"
            )

            with self.assertRaisesRegex(ValueError, "outside the workspace"):
                read_file("../outside.py", workspace=str(root))
            write_file(
                "src/example.py",
                "answer = 2\nanswer = 2\n",
                workspace=str(root),
            )
            with self.assertRaisesRegex(ValueError, "found 2"):
                edit_file(
                    "src/example.py",
                    "answer",
                    "result",
                    workspace=str(root),
                )

    def test_run_tests_runs_unittest_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_example.py").write_text(
                "import unittest\n"
                "class ExampleTests(unittest.TestCase):\n"
                "    def test_answer(self):\n"
                "        self.assertEqual(2 + 2, 4)\n",
                encoding="utf-8",
            )

            result = run_tests(workspace=str(root))

        self.assertIn("Ran 1 test", result)
        self.assertIn("OK", result)

    def test_run_python_executes_snippets_with_bounded_output(self):
        with patch.dict(os.environ, {"NANOAGENT_CODE_TOOL_SECRET": "hidden"}):
            output = run_python(
                "import os\n"
                "print(os.environ.get('NANOAGENT_CODE_TOOL_SECRET'))\n"
                f"print('x' * {MAX_PYTHON_OUTPUT_BYTES * 2})"
            )

        self.assertIn("None", output)
        self.assertIn("stdout truncated", output)
        self.assertLess(len(output), MAX_PYTHON_OUTPUT_BYTES + 100)

    def test_run_python_reports_nonzero_exit_codes(self):
        result = run_python("raise SystemExit(3)")

        self.assertIn("Exit code 3", result)

    def test_run_python_times_out(self):
        with patch(
            "nanoagent.toolbox.code_tools.PYTHON_TIMEOUT_SECONDS", 0.1
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                run_python("import time; time.sleep(2)")

    def test_run_python_file_runs_only_workspace_python_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "example.py"
            script.write_text("print('file ran')\n", encoding="utf-8")

            self.assertEqual(
                run_python_file("example.py", workspace=str(root)), "file ran"
            )
            with self.assertRaisesRegex(ValueError, "outside the workspace"):
                run_python_file("../outside.py", workspace=str(root))
            (root / "notes.txt").write_text("not Python\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\.py"):
                run_python_file("notes.txt", workspace=str(root))

    def test_read_lines_returns_numbered_inclusive_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_file("sample.py", "one\ntwo\nthree\n", workspace=str(root))

            result = read_lines(
                "sample.py", start_line=2, end_line=3, workspace=str(root)
            )

            self.assertEqual(result, "2: two\n3: three")
            with self.assertRaisesRegex(ValueError, "start_line"):
                read_lines("sample.py", start_line=0, workspace=str(root))
            with self.assertRaisesRegex(ValueError, "end_line"):
                read_lines(
                    "sample.py", start_line=3, end_line=2, workspace=str(root)
                )

    def test_find_files_skips_common_generated_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "visible.py").write_text("", encoding="utf-8")
            for name in (".git", ".venv", "__pycache__", "node_modules"):
                ignored = root / name
                ignored.mkdir()
                (ignored / "hidden.py").write_text("", encoding="utf-8")

            self.assertEqual(find_files("*.py", workspace=str(root)), "src/visible.py")

    def test_find_files_skips_paths_reported_as_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "file_link.py").write_text("", encoding="utf-8")
            directory_link = workspace / "directory_link"
            directory_link.mkdir()
            (directory_link / "nested.py").write_text("", encoding="utf-8")

            def is_symlink(path):
                return path.name in {"file_link.py", "directory_link"}

            with patch.object(
                type(workspace), "is_symlink", autospec=True, side_effect=is_symlink
            ):
                result = find_files("*.py", workspace=str(workspace))

            self.assertEqual(result, "No files found.")

    def test_search_text_searches_non_python_files_with_context_and_globs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "src"
            source.mkdir()
            (source / "settings.toml").write_text(
                "before = true\nToken = 'Needle'\nafter = false\n",
                encoding="utf-8",
            )
            (source / "notes.md").write_text("needle in markdown\n", encoding="utf-8")

            result = search_text(
                "needle",
                file_glob="src/*.toml",
                context_lines=1,
                workspace=str(root),
            )

        self.assertIn("src/settings.toml", result)
        self.assertIn("> 2: Token = 'Needle'", result)
        self.assertIn("  1: before = true", result)
        self.assertIn("  3: after = false", result)
        self.assertNotIn("notes.md", result)

    def test_search_text_supports_regex_case_sensitivity_and_result_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one.py").write_text(
                "Alpha = 1\nalpha = 2\nother = 3\n", encoding="utf-8"
            )

            case_sensitive = search_text(
                "alpha",
                case_sensitive=True,
                context_lines=0,
                workspace=str(root),
            )
            regex = search_text(
                r"^alpha\s*=",
                use_regex=True,
                case_sensitive=False,
                context_lines=0,
                max_results=1,
                workspace=str(root),
            )

        self.assertEqual(case_sensitive.count(">"), 1)
        self.assertIn("> 2: alpha = 2", case_sensitive)
        self.assertIn("> 1: Alpha = 1", regex)
        self.assertIn("[Results truncated at 1 matching lines]", regex)

    def test_search_text_applies_result_limit_across_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("target\n", encoding="utf-8")
            (root / "b.txt").write_text("target\n", encoding="utf-8")

            result = search_text(
                "target",
                context_lines=0,
                max_results=1,
                workspace=str(root),
            )

        self.assertIn("a.txt", result)
        self.assertNotIn("b.txt", result)
        self.assertIn("[Results truncated at 1 matching lines]", result)

    def test_search_text_validates_query_regex_and_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "query"):
                search_text("", workspace=temp_dir)
            with self.assertRaisesRegex(ValueError, "regular expression"):
                search_text("[", use_regex=True, workspace=temp_dir)
            with self.assertRaisesRegex(ValueError, "context_lines"):
                search_text("text", context_lines=-1, workspace=temp_dir)
            with self.assertRaisesRegex(ValueError, "max_results"):
                search_text("text", max_results=0, workspace=temp_dir)

    def test_code_agent_search_text_tool_is_callable_and_has_argument_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "settings.toml").write_text(
                "mode = 'local'\n", encoding="utf-8"
            )
            with patch("nanoagent.toolbox.Path.cwd", return_value=root):
                tools = create_native_code_tools()
                result = tools.execute(
                    Response(
                        tool_call={
                            "tool": "search_text",
                            "kwargs": {
                                "query": "local",
                                "file_glob": "*.toml",
                                "context_lines": 0,
                            },
                        }
                    )
                )

        self.assertIn("> 1: mode = 'local'", result)
        search_schema = next(
            schema["function"]["parameters"]
            for schema in tools.schemas
            if schema["function"]["name"] == "search_text"
        )
        self.assertEqual(search_schema["required"], ["query"])
        self.assertEqual(
            search_schema["properties"]["use_regex"], {"type": "boolean"}
        )
        self.assertEqual(
            search_schema["properties"]["max_results"], {"type": "integer"}
        )

    def test_search_text_skips_generated_and_binary_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "visible.txt").write_text("target\n", encoding="utf-8")
            generated = root / ".venv"
            generated.mkdir()
            (generated / "hidden.txt").write_text("target\n", encoding="utf-8")
            (root / "binary.dat").write_bytes(b"target\x00data")

            result = search_text("target", workspace=str(root))

        self.assertIn("visible.txt", result)
        self.assertNotIn("hidden.txt", result)
        self.assertNotIn("binary.dat", result)

    def test_search_symbols_finds_variables_functions_and_classes_in_one_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text(
                "global_value = 1\n"
                "global_value = 2\n"
                "class Widget:\n"
                "    kind = 'basic'\n"
                "    def render(self, label):\n"
                "        local = label\n"
                "        self.value = 2\n"
                "        self.value = 3\n"
                "        return local\n"
                "    def resize(self):\n"
                "        self.value = 4\n"
                "async def refresh():\n"
                "    result = 3\n",
                encoding="utf-8",
            )
            (root / "other.py").write_text(
                "def hidden_from_search():\n    pass\n", encoding="utf-8"
            )

            all_symbols = search_symbols("sample.py", workspace=str(root))
            matching_variable = search_symbols(
                "sample.py",
                query="local",
                symbol_type="variable",
                workspace=str(root),
            )

        self.assertIn("sample.py:1: variable global_value", all_symbols)
        self.assertEqual(all_symbols.count("variable global_value"), 1)
        self.assertIn("sample.py:3: class Widget", all_symbols)
        self.assertIn("sample.py:5: function render", all_symbols)
        self.assertIn("sample.py:6: variable local", all_symbols)
        self.assertIn("sample.py:7: variable value", all_symbols)
        self.assertEqual(all_symbols.count("variable value"), 1)
        self.assertIn("sample.py:12: function refresh", all_symbols)
        self.assertNotIn("other.py", all_symbols)
        self.assertEqual(matching_variable, "sample.py:6: variable local")

    def test_inspect_file_reports_python_structure_and_rejects_non_python_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text(
                "VALUE: int = 3\n"
                "def greet(name: str, enabled: bool = False, retries: int = 0, "
                "optional=None, label: str = '') -> str:\n"
                "    \"\"\"Return a greeting.\"\"\"\n"
                "    def normalize(value: str = '') -> str:\n"
                "        return value\n"
                "    raise ValueError(name)\n"
                "    return name\n"
                "class Widget:\n"
                "    KIND: str = 'basic'\n"
                "    count = 0\n"
                "    def render(self) -> None:\n"
                "        def helper(value=0):\n"
                "            return value\n"
                "        raise RuntimeError('failed')\n"
                "        pass\n",
                encoding="utf-8",
            )
            (root / "notes.md").write_text("not Python\n", encoding="utf-8")

            result = inspect_file("sample.py", workspace=str(root))

            with self.assertRaisesRegex(ValueError, r"Only Python source files"):
                inspect_file("notes.md", workspace=str(root))
            with self.assertRaisesRegex(ValueError, "outside the workspace"):
                inspect_file("../outside.py", workspace=str(root))

        self.assertIn("GLOBAL VARIABLES:\n  VALUE", result)
        self.assertIn("greet(name: str,", result)
        self.assertIn("enabled: bool = False", result)
        self.assertIn("retries: int = 0", result)
        self.assertIn("optional = None", result)
        self.assertIn("label: str = ''", result)
        self.assertIn("greet.normalize(value: str", result)
        self.assertIn("Returns: str", result)
        self.assertIn("Docstring: Return a greeting.", result)
        self.assertIn("Widget", result)
        self.assertIn("CLASS ATTRIBUTES:", result)
        self.assertIn("KIND: str", result)
        self.assertIn("count = 0", result)
        self.assertIn("render(self)", result)
        self.assertIn("Widget.render.helper(value = 0)", result)
        self.assertIn("Returns: None", result)
        self.assertIn("raise ValueError(name)", result)
        self.assertIn("raise RuntimeError('failed')", result)

    def test_inspect_file_is_available_through_the_code_agent_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text(
                "def calculate(value: int) -> int:\n    return value * 2\n",
                encoding="utf-8",
            )
            with patch("nanoagent.toolbox.Path.cwd", return_value=root):
                tools = create_native_code_tools()
                result = tools.execute(
                    Response(
                        tool_call={
                            "tool": "inspect_file",
                            "kwargs": {"file_path": "sample.py"},
                        }
                    )
                )

        self.assertIn("calculate(value: int)", result)
        self.assertIn("Returns: int", result)
        self.assertNotIn("inspect_file", tools.requires_approval)

    def test_code_agent_exposes_workspace_search_and_execution_tools(self):
        tools = create_native_code_tools()

        self.assertEqual(
            set(tools.registry),
            {
                "list_files",
                "read_file",
                "read_lines",
                "find_files",
                "search_text",
                "inspect_file",
                "search_symbols",
                "write_file",
                "edit_file",
                "run_python",
                "run_python_file",
                "run_tests",
            },
        )
        self.assertEqual(
            set(tools.requires_approval),
            {
                "write_file",
                "edit_file",
                "run_python",
                "run_python_file",
                "run_tests",
            },
        )
        read_lines_schema = next(
            schema["function"]["parameters"]
            for schema in tools.schemas
            if schema["function"]["name"] == "read_lines"
        )
        self.assertEqual(
            read_lines_schema["properties"]["start_line"], {"type": "integer"}
        )
        symbol_schema = next(
            schema["function"]["parameters"]
            for schema in tools.schemas
            if schema["function"]["name"] == "search_symbols"
        )
        self.assertEqual(
            symbol_schema["properties"]["symbol_type"]["enum"],
            ["all", "variable", "function", "class"],
        )
        python_schema = next(
            schema["function"]["parameters"]
            for schema in tools.schemas
            if schema["function"]["name"] == "run_python"
        )
        self.assertEqual(python_schema["required"], ["code"])
        self.assertEqual(python_schema["properties"]["code"], {"type": "string"})

    def test_run_tests_requires_approval_and_describes_the_test_directory(self):
        tools = create_native_code_tools()
        with patch("builtins.input", return_value="no") as approval:
            with self.assertRaises(ToolFailure) as raised:
                tools.execute(
                    Response(
                        tool_call={
                            "tool": "run_tests",
                            "kwargs": {"test_path": "tests"},
                        }
                    )
                )

        self.assertEqual(raised.exception.code, "permission_denied")
        self.assertIn("unittest discovery", approval.call_args.args[0])
        self.assertIn("tests", approval.call_args.args[0])
        self.assertIn("not a security sandbox", approval.call_args.args[0])

    def test_python_execution_requires_approval_and_warns_about_permissions(self):
        tools = create_native_code_tools()
        calls = [
            {"tool": "run_python", "kwargs": {"code": "print('not run')"}},
            {
                "tool": "run_python_file",
                "kwargs": {"file_path": "example.py"},
            },
        ]
        for call in calls:
            with self.subTest(tool=call["tool"]):
                with patch("builtins.input", return_value="no") as approval:
                    with self.assertRaises(ToolFailure) as raised:
                        tools.execute(Response(tool_call=call))

                self.assertEqual(raised.exception.code, "permission_denied")
                self.assertIn("not a security sandbox", approval.call_args.args[0])

    def test_approved_python_snippet_runs_through_the_tool_registry(self):
        tools = create_native_code_tools()
        with patch("builtins.input", return_value="yes") as approval:
            result = tools.execute(
                Response(
                    tool_call={
                        "tool": "run_python",
                        "kwargs": {"code": "print('snippet ran')"},
                    }
                )
            )

        self.assertEqual(result, "snippet ran")
        self.assertIn("Runtime and output are limited", approval.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
