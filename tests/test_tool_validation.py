import unittest

from nanoagent.agent import NanoAgent
from nanoagent.memory import Memory
from nanoagent.models import Response
from nanoagent.planning import NativeReAct
from nanoagent.tools import NativeTools, ToolFailure, Tools, tool_to_schema


class TypedTool:
    def __init__(self):
        self.calls = []

    def get_schema(self):
        return {
            "description": "Look up a value.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        }

    def execute(self, query: str, limit: int = 2):
        self.calls.append((query, limit))
        return f"{query}:{limit}"


class SequenceLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, messages, tools=None):
        return self.response


class ToolValidationTests(unittest.TestCase):
    def setUp(self):
        self.tool = TypedTool()
        self.tools = NativeTools()
        self.tools.add_tool("lookup", self.tool.execute)

    def _call(self, kwargs):
        return Response(tool_call={"tool": "lookup", "kwargs": kwargs})

    def test_native_schema_is_shared_and_rejects_invalid_arguments(self):
        advertised_schema = self.tools.schemas[0]["function"]["parameters"]
        self.assertFalse(advertised_schema["additionalProperties"])
        self.assertEqual(
            self.tools.schemas[0]["function"]["description"], "Look up a value."
        )

        invalid_arguments = [
            {},
            {"query": 42},
            {"query": "valid", "limit": "many"},
            {"query": "valid", "limit": True},
            {"query": "valid", "unexpected": True},
        ]
        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ToolFailure) as raised:
                    self.tools.execute(self._call(kwargs))
                self.assertEqual(raised.exception.code, "invalid_arguments")

        self.assertEqual(self.tool.calls, [])

    def test_valid_arguments_execute_with_optional_defaults(self):
        result = self.tools.execute(self._call({"query": "example"}))

        self.assertEqual(result, "example:2")
        self.assertEqual(self.tool.calls, [("example", 2)])

    def test_inferred_schema_rejects_wrong_types(self):
        def add_count(count: int):
            return count

        tools = Tools()
        tools.add_tool("add_count", add_count)
        response = Response(
            tool_call={"tool": "add_count", "kwargs": {"count": "five"}}
        )

        with self.assertRaises(ToolFailure) as raised:
            tools.execute(response)

        self.assertEqual(raised.exception.code, "invalid_arguments")

    def test_modern_annotations_generate_nested_and_optional_schemas(self):
        def search(query: str, tags: list[str] | None = None):
            return query, tags

        schema = tool_to_schema(search)["function"]["parameters"]

        self.assertEqual(schema["properties"]["query"], {"type": "string"})
        self.assertEqual(
            schema["properties"]["tags"],
            {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
        )
        tools = NativeTools()
        tools.add_tool("search", search)
        with self.assertRaises(ToolFailure) as raised:
            tools.execute(
                Response(
                    tool_call={
                        "tool": "search",
                        "kwargs": {"query": "test", "tags": [1]},
                    }
                )
            )
        self.assertEqual(raised.exception.code, "invalid_arguments")

    def test_registration_rejects_invalid_and_duplicate_tools(self):
        def existing_tool():
            """Tool description from its docstring."""

        tools = Tools()
        tools.add_tool("existing", existing_tool)

        with self.assertRaises(ValueError):
            tools.add_tool("existing", lambda: None)
        with self.assertRaises(ValueError):
            tools.add_tool("  ", existing_tool)
        with self.assertRaises(TypeError):
            tools.add_tool("not_callable", object())

        self.assertIs(tools.registry["existing"]["function"], existing_tool)
        self.assertEqual(
            tools.registry["existing"]["description"],
            "Tool description from its docstring.",
        )

    def test_text_tool_calls_require_standalone_json(self):
        tools = Tools()
        prose = Response(content='The example contains {"tool": "not_a_call"}.')
        fenced = Response(content='```json\n{"tool": "not_a_call"}\n```')
        malformed = Response(content='{"tool": "broken",}')

        self.assertIs(tools.parse(prose), prose)
        self.assertIs(tools.parse(fenced), fenced)
        self.assertIs(tools.parse(malformed), malformed)

        parsed = tools.parse(
            Response(content='{"tool": "lookup", "kwargs": {"query": "value"}}')
        )
        self.assertEqual(
            parsed.tool_call,
            {"tool": "lookup", "kwargs": {"query": "value"}},
        )

    def test_text_tool_call_rejects_unexpected_fields(self):
        tools = Tools()
        response = Response(content='{"tool": "lookup", "extra": true}')

        with self.assertRaises(ToolFailure) as raised:
            tools.parse(response)

        self.assertEqual(raised.exception.code, "invalid_tool_call")

    def test_python_tool_exception_is_normalized_and_handled_by_agent(self):
        def broken_tool(value: str):
            raise RuntimeError("backend unavailable")

        tools = Tools()
        tools.add_tool("broken", broken_tool)
        agent = NanoAgent(
            SequenceLLM(
                Response(tool_call={"tool": "broken", "kwargs": {"value": "test"}})
            ),
            Memory(),
            tools,
            NativeReAct(max_steps=2),
        )

        result = agent.run("Run the broken tool")

        self.assertIn("tool_execution_failed", result)
        self.assertIn("backend unavailable", result)

    def test_missing_or_directory_file_arguments_are_recoverable(self):
        def read_file(file_path: str):
            raise IsADirectoryError(f"'{file_path}' is not a file.")

        tools = Tools()
        tools.add_tool("read_file", read_file)

        with self.assertRaises(ToolFailure) as raised:
            tools.execute(
                Response(
                    tool_call={
                        "tool": "read_file",
                        "kwargs": {"file_path": "src/nanoagent"},
                    }
                )
            )

        self.assertEqual(raised.exception.code, "invalid_arguments")
        self.assertIn("not a file", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
