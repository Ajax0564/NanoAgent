import unittest
from typing import Literal
from unittest.mock import patch

from nanoagent.agent import NanoAgent
from nanoagent.memory import Memory
from nanoagent.models import Response
from nanoagent.planning import NativeReAct
from nanoagent.tools import (
    NativeTools,
    ToolFailure,
    Tools,
    annotation_to_schema,
    tool_to_schema,
)


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

    def test_nullable_literal_and_untyped_annotations_have_valid_schemas(self):
        def find(
            query: str | None,
            mode: Literal["fast", "exact"] = "fast",
            extra=None,
        ):
            return query, mode, extra

        schema = tool_to_schema(find)["function"]["parameters"]

        self.assertEqual(
            schema["properties"]["query"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )
        self.assertEqual(
            schema["properties"]["mode"],
            {"enum": ["fast", "exact"], "type": "string"},
        )
        self.assertEqual(schema["properties"]["extra"], {})
        self.assertEqual(annotation_to_schema(dict[str, int])["type"], "object")

        def choose(value: int | str):
            return value

        union_schema = tool_to_schema(choose)["function"]["parameters"]
        self.assertEqual(
            union_schema["properties"]["value"],
            {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        )
        tools = NativeTools()
        tools.add_tool("choose", choose)
        with self.assertRaises(ToolFailure) as raised:
            tools.execute(
                Response(tool_call={"tool": "choose", "kwargs": {"value": None}})
            )
        self.assertEqual(raised.exception.code, "invalid_arguments")

    def test_schema_definitions_are_cached_and_invalidated_on_update(self):
        def search(query: str):
            return query

        tools = NativeTools()
        tools.add_tool("search", search)
        with patch.object(
            tools, "_build_tool_definition", wraps=tools._build_tool_definition
        ) as build:
            first = tools.schemas
            tools.schemas
            self.assertEqual(build.call_count, 1)

            first[0]["function"]["description"] = "corrupted"
            self.assertNotEqual(
                tools.schemas[0]["function"]["description"], "corrupted"
            )

            def search_by_id(item_id: int):
                return item_id

            tools.update_tool(
                "search",
                func=search_by_id,
                description="Search by id.",
                metadata={"group": "lookup"},
            )
            updated = tools.schemas[0]["function"]

        self.assertEqual(build.call_count, 2)
        self.assertEqual(updated["description"], "Search by id.")
        self.assertEqual(updated["parameters"]["required"], ["item_id"])
        self.assertEqual(
            updated["parameters"]["properties"]["item_id"], {"type": "integer"}
        )
        self.assertEqual(tools.registry["search"]["metadata"], {"group": "lookup"})
        self.assertEqual(
            tools.execute(
                Response(
                    tool_call={"tool": "search", "kwargs": {"item_id": 42}}
                )
            ),
            42,
        )
        with self.assertRaises(ToolFailure):
            tools.execute(
                Response(
                    tool_call={"tool": "search", "kwargs": {"item_id": "42"}}
                )
            )

    def test_tool_metadata_is_json_safe_and_not_exposed_in_schema(self):
        def lookup(query: str):
            return query

        metadata = {"category": "lookup", "tags": ["read-only"]}
        tools = NativeTools()
        tools.add_tool("lookup", lookup, metadata=metadata)
        metadata["tags"].append("mutated")

        self.assertEqual(
            tools.registry["lookup"]["metadata"],
            {"category": "lookup", "tags": ["read-only"]},
        )
        self.assertNotIn("metadata", tools.schemas[0]["function"])
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            tools.add_tool("unsafe", lookup, metadata={"callback": object()})

    def test_native_schemas_normalize_nullable_fields_and_reject_invalid_shapes(self):
        class NullableTool:
            def get_schema(self):
                return {
                    "description": "Accept an optional query.",
                    "input_schema": {
                        "properties": {
                            "query": {
                                "type": "string",
                                "nullable": True,
                                "default": {"nullable": True},
                            }
                        },
                        "required": ["query"],
                    },
                }

            def execute(self, query):
                return query

        tools = NativeTools()
        tools.add_tool("nullable", NullableTool().execute)

        parameters = tools.schemas[0]["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(
            parameters["properties"]["query"],
            {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": {"nullable": True},
            },
        )
        self.assertEqual(
            tools.execute(
                Response(tool_call={"tool": "nullable", "kwargs": {"query": None}})
            ),
            None,
        )

        class InvalidSchemaTool:
            def get_schema(self):
                return {
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": ["missing"],
                    }
                }

            def execute(self):
                return None

        invalid = NativeTools()
        invalid.add_tool("invalid", InvalidSchemaTool().execute)
        with self.assertRaises(ToolFailure) as raised:
            invalid.schemas
        self.assertEqual(raised.exception.code, "invalid_tool_schema")

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
        with self.assertRaises(ToolFailure) as raised:
            tools.parse(malformed)
        self.assertEqual(raised.exception.code, "invalid_tool_call")

        parsed = tools.parse(
            Response(content='{"tool": "lookup", "kwargs": {"query": "value"}}')
        )
        self.assertEqual(
            parsed.tool_call,
            {"tool": "lookup", "kwargs": {"query": "value"}},
        )
        with self.assertRaises(ToolFailure) as raised:
            tools.parse(
                Response(content='{"tool":"lookup","kwargs":{"query":1,"query":2}}')
            )
        self.assertEqual(raised.exception.code, "invalid_tool_call")

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

    def test_execution_errors_use_specific_failure_categories(self):
        failures = [
            (ValueError("bad value"), "invalid_arguments"),
            (TimeoutError("too slow"), "tool_timeout"),
            (PermissionError("not allowed"), "tool_permission_denied"),
            (RuntimeError("unexpected"), "tool_execution_failed"),
        ]
        for error, expected_code in failures:
            with self.subTest(expected_code=expected_code):

                def fail():
                    raise error

                tools = Tools()
                tools.add_tool("fail", fail)
                with self.assertRaises(ToolFailure) as raised:
                    tools.execute(
                        Response(tool_call={"tool": "fail", "kwargs": {}})
                    )

                self.assertEqual(raised.exception.code, expected_code)

    def test_native_tool_calls_reject_invalid_types_and_non_object_arguments(self):
        invalid_calls = [
            {
                "type": "other",
                "function": {"name": "lookup", "arguments": {}},
            },
            {
                "function": {"name": "lookup", "arguments": "[]"},
            },
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"query":"one","query":"two"}',
                },
            },
        ]
        for tool_call in invalid_calls:
            with self.subTest(tool_call=tool_call):
                with self.assertRaises(ToolFailure):
                    self.tools.parse(Response(tool_call=tool_call))

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
