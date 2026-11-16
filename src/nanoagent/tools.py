import json
import inspect

from pathlib import Path
from typing import Any, Callable

from .models import Response


# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


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
    tool_name = name or function.__name__

    # Extract meatadata
    properties, required = {}, []
    for parameter_name, parameter in signature.parameters.items():
        properties[parameter_name] = {
            "type": TYPE_MAP.get(parameter.annotation, "string")
        }
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
        self.registry[name] = {"function": func, "description": description}

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
        text = response.content

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            try:
                tool_call = json.loads(text[start:end])
            except json.JSONDecodeError as error:
                raise ToolFailure(
                    "invalid_arguments", "Tool call contains invalid JSON."
                ) from error
            if not isinstance(tool_call, dict):
                raise ToolFailure("invalid_tool_call", "Tool call must be an object.")
            if not isinstance(tool_call.get("kwargs", {}), dict):
                raise ToolFailure(
                    "invalid_arguments", "Tool arguments must be an object."
                )

            # Add the parsed tool call to the response
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
            )

        return response

    def execute(self, response: Response) -> Any:
        """Run a registered tool.

        Arguments:
            tool_call: A parsed tool call dict with "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call
        if not isinstance(tool_call, dict):
            return ToolFailure("invalid_tool_call", "Tool call must be an object.")

        name = tool_call.get("tool")
        kwargs = tool_call.get("kwargs", {})
        if not isinstance(name, str) or not name:
            return ToolFailure("invalid_tool", "Tool name must be a non-empty string.")
        if not isinstance(kwargs, dict):
            return ToolFailure("invalid_arguments", "Tool arguments must be an object.")

        if name not in self.registry:
            return ToolFailure("invalid_tool", f"Unknown tool: {name}.")

        # Human-in-the-loop: ask before running dangerous tools
        if name in self.requires_approval:
            response = input(f"Allow {name}? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                return ToolFailure(
                    "permission_denied", f"Execution of '{name}' was denied."
                )

        tool_func = self.registry[name]["function"]
        try:
            inspect.signature(tool_func).bind(**kwargs)
        except TypeError as error:
            return ToolFailure("invalid_arguments", str(error))

        return tool_func(**kwargs)

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

    @property
    def schemas(self) -> list[dict]:
        """Return tool functions for native function calling."""
        schemas = []
        for name, tool in self.registry.items():
            function = tool["function"]
            owner = getattr(function, "__self__", None)
            get_schema = getattr(owner, "get_schema", None)
            if callable(get_schema):
                tool_schema = get_schema()
                input_schema = tool_schema.get("input_schema")
                if input_schema is not None:
                    schemas.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": tool_schema.get(
                                    "description", tool["description"]
                                ),
                                "parameters": input_schema,
                            },
                        }
                    )
                    continue

            schemas.append(
                tool_to_schema(
                    function,
                    name=name,
                    description=tool["description"] or None,
                )
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
        if not response.tool_call:
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
        return not response.tool_call


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
