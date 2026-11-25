import copy
import inspect
import json
import math
import re
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .models import Response


def annotation_to_schema(annotation: Any) -> dict:
    """Translate a supported Python type annotation to a JSON Schema fragment.

    Args:
        annotation: A built-in type, generic list/mapping/sequence, union,
            ``Literal``, ``Any``, or an absent parameter annotation.

    Returns:
        A JSON Schema mapping. Unsupported or absent annotations become ``{}``;
        optional unions include a null alternative.
    """
    if annotation is inspect.Parameter.empty:
        return {}
    if annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        if not all(_is_json_value(value) for value in values):
            return {}
        schema = {"enum": values}
        value_types = {type(value) for value in values}
        if len(value_types) == 1:
            literal_type = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                type(None): "null",
            }.get(next(iter(value_types)))
            if literal_type is not None:
                schema["type"] = literal_type
        return schema

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        schemas = [annotation_to_schema(arg) for arg in non_none]
        if not non_none:
            return {"type": "null"}
        if type(None) in args:
            schemas.append({"type": "null"})
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}

    if origin in (list, Sequence) or annotation in (list, Sequence):
        return {
            "type": "array",
            "items": annotation_to_schema(args[0]) if args else {},
        }

    if origin in (dict, Mapping) or annotation in (dict, Mapping):
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
    json_type = type_map.get(annotation)
    return {"type": json_type} if json_type else {}


def _is_json_value(value: Any, ancestors: set[int] | None = None) -> bool:
    """Check that a value is composed only of finite, acyclic JSON primitives.

    Args:
        value: Candidate metadata value to check recursively.
        ancestors: Object identities on the current recursive path; used to
            reject cycles while allowing shared but acyclic substructures.

    Returns:
        ``True`` when the value can be represented safely as JSON, otherwise
        ``False``.
    """
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) not in (list, dict):
        return False

    ancestors = ancestors if ancestors is not None else set()
    value_id = id(value)
    if value_id in ancestors:
        return False
    ancestors.add(value_id)
    try:
        if type(value) is list:
            return all(_is_json_value(item, ancestors) for item in value)
        return all(
            type(key) is str and _is_json_value(item, ancestors)
            for key, item in value.items()
        )
    finally:
        ancestors.remove(value_id)


def _copy_metadata(metadata: dict | None) -> dict:
    """Validate metadata and return an independent copy of the JSON-safe mapping.

    Args:
        metadata: Optional mapping of string keys to JSON-compatible values.

    Returns:
        A deep copy of the metadata, or an empty dict when ``None`` is supplied.

    Raises:
        TypeError: If metadata is not a plain dict or contains unsupported values.
    """
    if metadata is None:
        return {}
    if type(metadata) is not dict or not _is_json_value(metadata):
        raise TypeError("Tool metadata must contain only JSON-compatible values.")
    return copy.deepcopy(metadata)


_SCHEMA_MAP_KEYS = {"$defs", "definitions", "properties", "patternProperties"}
_SCHEMA_CHILD_KEYS = {
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SCHEMA_LIST_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}


def _normalize_json_schema(schema: Any) -> Any:
    """Convert OpenAPI-style nullable keywords in schema positions to JSON Schema.

    Args:
        schema: A schema fragment or a non-schema value that should remain intact.

    Returns:
        A deep-copied schema with nested ``nullable: true`` constraints expressed
        as a null type or an ``anyOf`` branch. Annotation data such as ``default``
        is preserved as data.
    """
    if not isinstance(schema, dict):
        return schema

    normalized = copy.deepcopy(schema)
    for key in _SCHEMA_MAP_KEYS:
        if isinstance(normalized.get(key), dict):
            normalized[key] = {
                name: _normalize_json_schema(value)
                for name, value in normalized[key].items()
            }
    for key in _SCHEMA_CHILD_KEYS:
        if isinstance(normalized.get(key), dict):
            normalized[key] = _normalize_json_schema(normalized[key])
    for key in _SCHEMA_LIST_KEYS:
        if isinstance(normalized.get(key), list):
            normalized[key] = [
                _normalize_json_schema(value) for value in normalized[key]
            ]

    nullable = normalized.pop("nullable", False)
    if nullable is not True:
        return normalized

    schema_type = normalized.get("type")
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            normalized["type"] = [*schema_type, "null"]
        return normalized

    alternatives = normalized.get("anyOf")
    if isinstance(alternatives, list) and any(
        isinstance(option, dict) and option.get("type") == "null"
        for option in alternatives
    ):
        return normalized
    annotations = {
        key: normalized.pop(key)
        for key in _SCHEMA_ANNOTATION_KEYS
        if key in normalized
    }
    return {**annotations, "anyOf": [normalized, {"type": "null"}]}


def _normalize_parameters_schema(schema: dict, tool_name: str) -> dict:
    """Normalize and validate a tool's root JSON Schema parameter object.

    Args:
        schema: Parameter schema supplied by a tool or inferred from annotations.
        tool_name: Registered tool name used in validation error messages.

    Returns:
        A validated object schema with properties, required fields, and a
        default-deny ``additionalProperties`` setting.

    Raises:
        ToolFailure: If the root shape, required list, or JSON Schema definition
            is invalid.
    """
    if not isinstance(schema, dict):
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' parameters must be a JSON Schema object.",
        )
    normalized = _normalize_json_schema(schema)
    if not isinstance(normalized, dict):
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' parameters must be a JSON Schema object.",
        )
    if "type" not in normalized and "properties" in normalized:
        normalized["type"] = "object"
    if normalized.get("type") != "object":
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' parameters must describe an object.",
        )

    properties = normalized.setdefault("properties", {})
    required = normalized.setdefault("required", [])
    if not isinstance(properties, dict):
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' parameter properties must be an object.",
        )
    if (
        not isinstance(required, list)
        or any(not isinstance(item, str) for item in required)
        or len(required) != len(set(required))
        or any(item not in properties for item in required)
    ):
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' has an invalid required parameter list.",
        )
    normalized.setdefault("additionalProperties", False)
    try:
        Draft202012Validator.check_schema(normalized)
    except SchemaError as error:
        raise ToolFailure(
            "invalid_tool_schema",
            f"Tool '{tool_name}' has an invalid argument schema: {error.message}",
        ) from error
    return normalized


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    """Build a JSON object while rejecting repeated member names.

    Args:
        pairs: Ordered key/value pairs supplied by ``json.loads``.

    Returns:
        A dictionary containing the unique pairs.

    Raises:
        ValueError: If the input contains the same key more than once.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity.

    Args:
        value: Non-standard constant token encountered by the JSON parser.

    Returns:
        None. Always raises ``ValueError``.
    """
    raise ValueError(f"Invalid JSON constant: {value}.")


def _load_json(text: str) -> Any:
    """Parse strict JSON with duplicate-key and non-finite-number checks.

    Args:
        text: JSON document to decode.

    Returns:
        The decoded JSON value.

    Raises:
        JSONDecodeError: If the document is not syntactically valid JSON.
        ValueError: If it contains a duplicate key or non-standard constant.
    """
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


class ToolFailure(Exception):
    """A tool call failed with a machine-readable category."""

    def __init__(self, code: str, message: str):
        """Create a classified tool failure.

        Args:
            code: Stable machine-readable failure category.
            message: Human-readable error details.

        Returns:
            None. Stores ``code`` and initializes the exception message.
        """
        super().__init__(message)
        self.code = code


def tool_to_schema(
    function: Callable,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    """Build an OpenAI function-tool schema from a callable's signature.

    Args:
        function: Callable whose parameters and type hints define tool inputs.
        name: Optional model-facing name; defaults to the callable's name.
        description: Optional tool description; otherwise uses the callable's
            docstring when present.

    Returns:
        A tool definition containing the function name, description, parameter
        schemas, required names, and a closed root object schema.

    Raises:
        TypeError: If the function is not callable or the description is not text.
        ValueError: If an explicitly supplied name is empty.
    """
    if not callable(function):
        raise TypeError("Tool must be callable.")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("Tool name must be a non-empty string.")
    if description is not None and not isinstance(description, str):
        raise TypeError("Tool description must be a string.")

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

    def __init__(self, requires_approval: list[str] | None = None):
        """Create an empty registry with an optional approval policy.

        Args:
            requires_approval: Tool names that must receive interactive approval
                before execution.

        Returns:
            None. Initializes the registry and empty schema/validator caches.
        """
        self.registry: dict[str, dict[str, Any]] = {}
        self.requires_approval = list(requires_approval or [])
        self._tool_definition_cache: dict[str, dict] = {}
        self._validator_cache: dict[str, Draft202012Validator] = {}

    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Register a callable under a unique name.

        Args:
            name: Non-empty identifier used in model tool calls.
            func: Callable invoked with validated keyword arguments. Returned
                awaitables are not automatically awaited.
            description: Human-readable description; falls back to the callable's
                docstring when empty.
            metadata: JSON-compatible application metadata stored locally and
                excluded from model-facing schemas.

        Returns:
            None. Stores the callable, description, and a defensive metadata copy.

        Raises:
            ValueError: If the name is empty or already registered.
            TypeError: If the callable, description, or metadata is invalid.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if name in self.registry:
            raise ValueError(f"Tool '{name}' is already registered.")
        if not callable(func):
            raise TypeError("Tool must be callable.")
        if not isinstance(description, str):
            raise TypeError("Tool description must be a string.")

        self.registry[name] = {
            "function": func,
            "description": description or inspect.getdoc(func) or "",
            "metadata": _copy_metadata(metadata),
        }
        self._tool_definition_cache.pop(name, None)
        self._validator_cache.pop(name, None)

    def update_tool(
        self,
        name: str,
        *,
        func: Callable | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Update selected registration fields and clear derived caches.

        Args:
            name: Existing tool identifier.
            func: Optional replacement callable. Returned awaitables are not
                automatically awaited.
            description: Optional replacement description; an empty string falls
                back to the replacement callable's docstring.
            metadata: Optional replacement JSON-safe metadata mapping. Pass an
                empty dict to clear existing metadata.

        Returns:
            None. Replaces supplied fields and invalidates the schema and
            validator caches for this tool.

        Raises:
            ValueError: If ``name`` is not registered.
            TypeError: If the replacement callable, description, or metadata is
                invalid.
        """
        if name not in self.registry:
            raise ValueError(f"Unknown tool: {name}.")
        if func is not None and not callable(func):
            raise TypeError("Tool must be callable.")
        if description is not None and not isinstance(description, str):
            raise TypeError("Tool description must be a string.")

        current = self.registry[name]
        updated_func = func if func is not None else current["function"]
        updated_description = (
            current["description"]
            if description is None
            else description or inspect.getdoc(updated_func) or ""
        )
        updated_metadata = (
            current["metadata"] if metadata is None else _copy_metadata(metadata)
        )
        self.registry[name] = {
            "function": updated_func,
            "description": updated_description,
            "metadata": copy.deepcopy(updated_metadata),
        }
        self._tool_definition_cache.pop(name, None)
        self._validator_cache.pop(name, None)

    @property
    def descriptions(self) -> str:
        """Return registered tool names and descriptions for text prompts.

        Inputs:
            None.

        Returns:
            A newline-separated string in registration order. Empty registries
            produce an empty string.
        """
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        """Render text-mode instructions for calling registered tools.

        Inputs:
            None.

        Returns:
            Prompt text containing tool descriptions and the expected JSON call
            shape.
        """
        return f"""
# Tools

If needed, you can only use the following tools to assist you in completing tasks:

{self.descriptions}

To use a tool, respond with JSON: {{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """Parse a standalone JSON tool call from a text response.

        Args:
            response: Model response whose content may be a ``tool``/``kwargs``
                JSON object.

        Returns:
            A new response with a normalized internal tool call when the content
            is a valid call; otherwise returns the original response unchanged.

        Raises:
            ToolFailure: If a tool-shaped JSON object has malformed JSON, unknown
                fields, an empty tool name, or non-object arguments.
        """
        text = response.content.strip()
        if not text.startswith("{") or not text.endswith("}"):
            return response

        try:
            payload = _load_json(text)
        except (json.JSONDecodeError, ValueError) as error:
            if re.match(r'^\{\s*"tool"\s*:', text):
                raise ToolFailure(
                    "invalid_tool_call",
                    f"Tool call contains invalid JSON: {error}",
                ) from error
            return response

        if not isinstance(payload, dict) or "tool" not in payload:
            return response
        if set(payload) - {"tool", "kwargs"}:
            raise ToolFailure("invalid_tool_call", "Unexpected tool call fields.")
        name = payload["tool"]
        if not isinstance(name, str) or not name.strip():
            raise ToolFailure("invalid_tool", "Tool name must be a non-empty string.")
        kwargs = payload.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ToolFailure(
                "invalid_arguments", "Tool arguments must be an object."
            )

        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call={"tool": name, "kwargs": kwargs},
            metadata=response.metadata,
        )

    def execute(self, response: Response) -> Any:
        """Validate, authorize, and synchronously execute a registered tool call.

        Args:
            response: Response containing an internal ``tool`` name and ``kwargs``.

        Returns:
            The callable's result, which is later converted to an observation by
            the agent.

        Raises:
            ToolFailure: For malformed calls, unknown tools, invalid arguments,
                denied approval, known execution failures, or unexpected tool
                exceptions.
        """
        tool_call = response.tool_call
        if not isinstance(tool_call, dict):
            raise ToolFailure("invalid_tool_call", "Tool call must be an object.")

        name = tool_call.get("tool")
        kwargs = tool_call.get("kwargs", {})
        if not isinstance(name, str) or not name.strip():
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
        except TimeoutError as error:
            raise ToolFailure(
                "tool_timeout", f"Tool '{name}' timed out: {error}"
            ) from error
        except PermissionError as error:
            raise ToolFailure(
                "tool_permission_denied", f"Tool '{name}' was denied: {error}"
            ) from error
        except ValueError as error:
            raise ToolFailure(
                "invalid_arguments", f"Tool '{name}': {error}"
            ) from error
        except Exception as error:
            raise ToolFailure(
                "tool_execution_failed", f"Tool '{name}' failed: {error}"
            ) from error

    def _build_tool_definition(self, name: str) -> dict:
        """Construct one function definition from its registered callable.

        Args:
            name: Registered tool name.

        Returns:
            Function schema with a normalized parameter object.

        Raises:
            ToolFailure: If the callable's generated parameter schema is invalid.
        """
        function = self.registry[name]["function"]
        definition = tool_to_schema(
            function,
            name=name,
            description=self.registry[name]["description"] or None,
        )["function"]
        definition["parameters"] = _normalize_parameters_schema(
            definition["parameters"], name
        )
        return definition

    def _tool_definition(self, name: str) -> dict:
        """Return a defensive copy of a cached or newly built tool definition.

        Args:
            name: Registered tool name whose schema is requested.

        Returns:
            A deep copy of the cached function schema so callers cannot mutate the
            registry's cached definition.
        """
        if name not in self._tool_definition_cache:
            self._tool_definition_cache[name] = self._build_tool_definition(name)
        return copy.deepcopy(self._tool_definition_cache[name])

    def _validate_arguments(self, name: str, kwargs: dict) -> None:
        """Validate tool arguments with the cached JSON Schema validator.

        Args:
            name: Registered tool name.
            kwargs: Parsed keyword arguments supplied to the tool.

        Returns:
            None when validation succeeds.

        Raises:
            ToolFailure: With code ``invalid_arguments`` and the failing JSON
                path when the arguments do not match the tool schema.
        """
        if name not in self._validator_cache:
            schema = self._tool_definition_cache.get(name)
            if schema is None:
                self._tool_definition(name)
                schema = self._tool_definition_cache[name]
            self._validator_cache[name] = Draft202012Validator(
                schema["parameters"]
            )
        try:
            self._validator_cache[name].validate(kwargs)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path) or "arguments"
            raise ToolFailure(
                "invalid_arguments", f"{path}: {error.message}"
            ) from error

    def observation(self, result: str) -> tuple[str, str]:
        """Format a tool result as a user-role observation for text-mode agents.

        Args:
            result: String representation of the tool result.

        Returns:
            A ``(role, content)`` tuple with role ``user`` and an
            ``OBSERVATION:`` prefix.
        """
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """Determine whether a text-mode response completes the current run.

        Args:
            response: Parsed response to inspect.

        Returns:
            ``True`` when no tool call remains or the call is ``final_answer``;
            otherwise ``False``. A final-answer call copies its ``kwargs`` into
            ``response.content``.
        """
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    @property
    def schemas(self) -> None:
        """Indicate that this registry does not expose native tool schemas.

        Inputs:
            None.

        Returns:
            ``None``; text-mode agents use ``prompt`` instead.
        """
        return None


class NativeTools(Tools):
    """Tool registry using native function calling."""

    def _build_tool_definition(self, name: str) -> dict:
        """Build a model-facing definition, preferring a callable's custom schema.

        Args:
            name: Registered tool name.

        Returns:
            A function definition with a normalized, validated parameter schema.

        Raises:
            ToolFailure: If the callable returns malformed schema or its inferred
                parameter schema is invalid.
        """
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
                description = tool_schema.get(
                    "description", self.registry[name]["description"]
                )
                if description is not None and not isinstance(description, str):
                    raise ToolFailure(
                        "invalid_tool_schema",
                        f"Tool '{name}' description must be a string.",
                    )
                return {
                    "name": name,
                    "description": description or None,
                    "parameters": _normalize_parameters_schema(input_schema, name),
                }

        return super()._build_tool_definition(name)

    @property
    def schemas(self) -> list[dict]:
        """Return defensive native function definitions in registration order.

        Inputs:
            None.

        Returns:
            A new list of OpenAI-compatible function tool objects. Definitions
            are backed by the registry's cache but returned as independent copies.

        Raises:
            ToolFailure: If any registered tool exposes an invalid schema.
        """
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
        """Return calculator-specific guidance when a calculator is registered.

        Inputs:
            None.

        Returns:
            A short instruction string for calculator use, or an empty string
            when the registry has no ``calculator`` tool.
        """
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
        """Normalize a provider-native function call to the internal call format.

        Args:
            response: Model response with an optional native ``function`` call.

        Returns:
            The original response when it has no tool call; otherwise a new
            response whose tool call contains ``tool`` and object-valued ``kwargs``.

        Raises:
            ToolFailure: If the call structure, function name, call type, or
                argument JSON is invalid.
        """
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
        if not isinstance(name, str) or not name.strip():
            raise ToolFailure("invalid_tool", "Tool name must be a non-empty string.")

        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = _load_json(args)
            except (json.JSONDecodeError, ValueError) as error:
                raise ToolFailure(
                    "invalid_arguments", f"Tool arguments contain invalid JSON: {error}"
                ) from error
        if not isinstance(args, dict):
            raise ToolFailure("invalid_arguments", "Tool arguments must be an object.")
        if response.tool_call.get("type", "function") != "function":
            raise ToolFailure(
                "invalid_tool_call", "Tool call type must be 'function'."
            )

        tool_call = {
            "tool": name,
            "kwargs": args,
        }

        # Add the parsed tool call to the response
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
            metadata=response.metadata,
        )

    def observation(self, result: str) -> tuple[str, str]:
        """Format a tool result using the native chat API's tool role.

        Args:
            result: Tool output to pass back to the model.

        Returns:
            A ``(role, content)`` tuple with role ``tool`` and stringified content.
        """
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """Check whether a native-tool response contains another tool call.

        Args:
            response: Parsed model response.

        Returns:
            ``True`` when no tool call is present; otherwise ``False``.
        """
        return response.tool_call is None


def _parse_frontmatter(text: str) -> dict:
    """Parse basic single-line ``key: value`` frontmatter without a YAML library.

    Args:
        text: Frontmatter body, typically between ``---`` delimiters.

    Returns:
        A dictionary of stripped keys and values. Lines without a colon are
        ignored; repeated keys keep the last value.

    Notes:
        This intentionally handles only simple scalar lines, not full YAML syntax.
    """
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
        """Load a skill file and register its instructions as a tool.

        Args:
            path: Markdown file containing ``name`` and ``description`` frontmatter
                followed by the skill instructions.

        Returns:
            None. Registers a callable that returns the instruction text when
            activated.

        Raises:
            OSError: If the file cannot be read.
            IndexError: If the expected frontmatter delimiters are absent.
            KeyError: If a required frontmatter key is missing.
            ValueError: If a tool with the same name is already registered.
        """
        content = Path(path).read_text(encoding="utf-8")

        # Split on YAML delimiters and extract the frontmatter and instructions
        parts = content.split("---", 2)
        frontmatter = _parse_frontmatter(parts[1])
        name = frontmatter["name"]
        description = frontmatter["description"]
        instructions = parts[2].strip()

        # Register the skill
        def skill(**kwargs):
            """Return the loaded instructions; tool arguments are ignored.

            Args:
                **kwargs: Accepted for tool-call compatibility and not inspected.

            Returns:
                The skill instruction text loaded from the Markdown file.
            """
            return instructions

        skill.__name__ = name
        skill.__doc__ = f"A skill that when activated provides the following context: '{description}'"
        skill.__signature__ = (
            inspect.Signature()
        )  # schema sees no params; lambda still tolerates any
        self.add_tool(name, skill, skill.__doc__)

    @property
    def prompt(self):
        """Return guidance for activating the available progressive skills.

        Inputs:
            None.

        Returns:
            Static instructions directing the model to activate a skill by its
            registered tool name.
        """
        return """You have specialized skills available. To use a skill,
call it like a tool by referencing their name.

The skill will provide detailed instructions for completing the task."""
