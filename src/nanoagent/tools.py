import json
import inspect
import types

from pathlib import Path
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from jsonschema import SchemaError, ValidationError, validate

from .models import Response


def annotation_to_schema(annotation: Any) -> dict:
    """Convert a Python annotation into its JSON Schema representation."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return {"enum": list(args)}

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        schemas = [annotation_to_schema(arg) for arg in non_none]
        if not non_none:
            return {"type": "null"}
        if len(non_none) == 1:
            return {"anyOf": [schemas[0], {"type": "null"}]}
        return {"anyOf": [*schemas, {"type": "null"}]}

    if origin is list or annotation is list:
        return {
            "type": "array",
            "items": annotation_to_schema(args[0]) if args else {},
        }

    if origin is dict or annotation is dict:
        return {
            "type": "object",
            "additionalProperties": (
                annotation_to_schema(args[1]) if len(args) > 1 else True
            ),
        }

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    return {"type": type_map.get(annotation, "string")}


class ToolFailure(Exception):
    """A tool call failed with a machine-readable category."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def tool_to_schema(
    function: Callable,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)
    try:
        type_hints = get_type_hints(function)
    except (NameError, TypeError):
        type_hints = {}
    tool_name = name or function.__name__

    properties, required = {}, []
    for parameter_name, parameter in signature.parameters.items():
        annotation = type_hints.get(parameter_name, parameter.annotation)
        properties[parameter_name] = annotation_to_schema(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                description if description is not None else inspect.getdoc(function)
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

    return schema


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] = []):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = requires_approval

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if name in self.registry:
            raise ValueError(f"Tool '{name}' is already registered.")
        if not callable(func):
            raise TypeError("Tool must be callable.")

        self.registry[name] = {
            "function": func,
            "description": description or inspect.getdoc(func) or "",
        }

    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools

If needed, you can only use the following tools to assist you in completing tasks:

{self.descriptions}

To use a tool, respond with JSON: {{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """Parse a JSON tool call from text."""
        text = response.content.strip()
        if not text.startswith("{") or not text.endswith("}"):
            return response

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return response

        if not isinstance(payload, dict) or "tool" not in payload:
            return response
        if set(payload) - {"tool", "kwargs"}:
            raise ToolFailure("invalid_tool_call", "Unexpected tool call fields.")
        if not isinstance(payload.get("kwargs", {}), dict):
            raise ToolFailure(
                "invalid_arguments", "Tool arguments must be an object."
            )

        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call={
                "tool": payload["tool"],
                "kwargs": payload.get("kwargs", {}),
            },
        )

    def execute(self, response: Response) -> Any:
        """Run a registered tool.

        Arguments:
            tool_call: A parsed tool call dict with "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call
        if not isinstance(tool_call, dict):
            raise ToolFailure("invalid_tool_call", "Tool call must be an object.")

        name = tool_call.get("tool")
        kwargs = tool_call.get("kwargs", {})
        if not isinstance(name, str) or not name:
            raise ToolFailure("invalid_tool", "Tool name must be a non-empty string.")
        if not isinstance(kwargs, dict):
            raise ToolFailure("invalid_arguments", "Tool arguments must be an object.")

        if name not in self.registry:
            raise ToolFailure("invalid_tool", f"Unknown tool: {name}.")

        self._validate_arguments(name, kwargs)
        tool_func = self.registry[name]["function"]
        try:
            inspect.signature(tool_func).bind(**kwargs)
        except TypeError as error:
            raise ToolFailure("invalid_arguments", str(error)) from error

        # Human-in-the-loop: ask before running dangerous tools
        if name in self.requires_approval:
            if name == "write_file":
                details = (
                    f"File: {kwargs.get('file_path')!r}\n"
                    "This will create or overwrite the file with:\n"
                    f"Content: {kwargs.get('content')!r}\n"
                )
            elif name == "edit_file":
                details = (
                    f"File: {kwargs.get('file_path')!r}\n"
                    "Exactly one matching occurrence will be replaced.\n"
                    f"Old text: {kwargs.get('old_text')!r}\n"
                    f"New text: {kwargs.get('new_text')!r}\n"
                )
            elif name == "run_python":
                code = kwargs.get("code", "")
                preview = code[:4000]
                if len(code) > len(preview):
                    preview += "\n[remaining code omitted]"
                details = (
                    "This Python snippet runs with this account's permissions. "
                    "It is not a security sandbox. Runtime and output are limited.\n"
                    f"Code: {preview!r}\n"
                )
            elif name == "run_python_file":
                details = (
                    "This Python file runs with this account's permissions. "
                    "It is not a security sandbox. Runtime and output are limited.\n"
                    f"File: {kwargs.get('file_path')!r}\n"
                )
            elif name == "run_tests":
                details = (
                    "This will run Python unittest discovery for:\n"
                    f"Test directory: {kwargs.get('test_path', 'tests')!r}\n"
                    "Test code runs with this account's permissions; this is not "
                    "a security sandbox.\n"
                )
            elif name == "unzip_file":
                details = (
                    f"Archive: {kwargs.get('zip_path')!r}\n"
                    f"Extract to: {kwargs.get('extract_to') or '<archive folder>'!r}\n"
                )
            else:
                details = f"Arguments: {json.dumps(kwargs, ensure_ascii=True)}\n"
            response = input(
                f"Approve {name}?\n{details}Type yes to approve [y/N]: "
            ).strip().lower()
            if response not in ("y", "yes"):
                raise ToolFailure(
                    "permission_denied", f"Execution of '{name}' was denied."
                )

        try:
            return tool_func(**kwargs)
        except ToolFailure:
            raise
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as error:
            raise ToolFailure("invalid_arguments", f"Tool '{name}': {error}") from error
        except Exception as error:
            raise ToolFailure(
                "tool_execution_failed", f"Tool '{name}' failed: {error}"
            ) from error

    def _tool_definition(self, name: str) -> dict:
        """Return the function definition used for model requests and validation."""
        function = self.registry[name]["function"]
        return tool_to_schema(
            function,
            name=name,
            description=self.registry[name]["description"] or None,
        )["function"]

    def _validate_arguments(self, name: str, kwargs: dict) -> None:
        """Reject arguments that do not match the registered tool schema."""
        schema = self._tool_definition(name)["parameters"]
        try:
            validate(instance=kwargs, schema=schema)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path) or "arguments"
            raise ToolFailure(
                "invalid_arguments", f"{path}: {error.message}"
            ) from error
        except SchemaError as error:
            raise ToolFailure(
                "invalid_tool_schema",
                f"Tool '{name}' has an invalid argument schema: {error.message}",
            ) from error

    def observation(self, result: str) -> tuple[str, str]:
        """Return the observation as a user."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The `TinyAgent`'s stopping mechanism."""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    @property
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None


class NativeTools(Tools):
    """Tool registry using native function calling."""

    def _tool_definition(self, name: str) -> dict:
        function = self.registry[name]["function"]
        owner = getattr(function, "__self__", None)
        get_schema = getattr(owner, "get_schema", None)
        if callable(get_schema):
            tool_schema = get_schema()
            if not isinstance(tool_schema, dict):
                raise ToolFailure(
                    "invalid_tool_schema", f"Tool '{name}' returned an invalid schema."
                )
            input_schema = tool_schema.get("input_schema")
            if input_schema is not None:
                if not isinstance(input_schema, dict):
                    raise ToolFailure(
                        "invalid_tool_schema",
                        f"Tool '{name}' input_schema must be an object.",
                    )
                input_schema = dict(input_schema)
                if input_schema.get("type") == "object" or "properties" in input_schema:
                    input_schema.setdefault("additionalProperties", False)
                return {
                    "name": name,
                    "description": tool_schema.get(
                        "description", self.registry[name]["description"]
                    )
                    or None,
                    "parameters": input_schema,
                }

        return super()._tool_definition(name)

    @property
    def schemas(self) -> list[dict]:
        """Return tool functions for native function calling."""
        schemas = []
        for name in self.registry:
            schemas.append(
                {
                    "type": "function",
                    "function": self._tool_definition(name),
                }
            )
        return schemas

    @property
    def prompt(self) -> str:
        """Return guidance for native tool use."""
        if "calculator" not in self.registry:
            return ""
        return (
            "\n## Calculator use\n\n"
            "For arithmetic expressions and numerical calculations, always call "
            "the `calculator` tool with operation `calculate` and the expression. "
            "Do not calculate these mentally or give the result before the tool "
            "returns.\n"
        )

    def parse(self, response: Response) -> Response:
        """Parse a tool call."""
        # If there's no tool call, return the response as is
        if response.tool_call is None:
            return response

        if not isinstance(response.tool_call, dict):
            raise ToolFailure("invalid_tool_call", "Tool call must be an object.")

        # Extract the tool name and arguments from the tool call
        function = response.tool_call.get("function")
        if not isinstance(function, dict):
            raise ToolFailure("invalid_tool_call", "Tool function is missing.")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ToolFailure("invalid_tool", "Tool name must be a non-empty string.")

        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as error:
                raise ToolFailure(
                    "invalid_arguments", "Tool arguments contain invalid JSON."
                ) from error
        if not isinstance(args, dict):
            raise ToolFailure("invalid_arguments", "Tool arguments must be an object.")

        tool_call = {
            "tool": name,
            "kwargs": args,
        }

        # Add the parsed tool call to the response
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
        )

    def observation(self, result: str) -> tuple[str, str]:
        """Native tool results use the 'tool' role."""
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """No tool call means the `TinyAgent` is done."""
        return response.tool_call is None


def _parse_frontmatter(text: str) -> dict:
    """Parse simple `key: value` YAML frontmatter without external deps."""
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class Skills(NativeTools):
    """A `Tools` and `Skills` registry

    Skills are recipes. When you activate one, you get instructions
    in return on how to approach a given task. Although it is not
    a tool in the same way a calculator is one, we can still approach
    it as such since the Agent has to decide when to activate it.

    The skills are loaded progressively. As such, the name and description are
    available in the system prompt, but the full instructions are only injected
    when the agent **activates** a skill (uses it as a tool).
    """

    def add_skill(self, path: str):
        """Load a SKILL.md file and register it as a callable tool."""
        content = Path(path).read_text(encoding="utf-8")

        # Split on YAML delimiters and extract the frontmatter and instructions
        parts = content.split("---", 2)
        frontmatter = _parse_frontmatter(parts[1])
        name = frontmatter["name"]
        description = frontmatter["description"]
        instructions = parts[2].strip()

        # Register the skill
        def skill(**kwargs):
            return instructions

        skill.__name__ = name
        skill.__doc__ = f"A skill that when activated provides the following context: '{description}'"
        skill.__signature__ = (
            inspect.Signature()
        )  # schema sees no params; lambda still tolerates any
        self.add_tool(name, skill, skill.__doc__)

    @property
    def prompt(self):
        return """You have specialized skills available. To use a skill,
call it like a tool by referencing their name.

The skill will provide detailed instructions for completing the task."""
