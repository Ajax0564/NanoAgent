#  llama-server -m .\LFM3-1.7B-Q4_K_M.gguf --port 8080 -c 16384 -np 1  -t 4 -b 512 -ub 128 --metrics # 16k -> 16384
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class SlotContext:
    """Context information reported for one llama.cpp server slot."""

    slot_id: int
    context_size: int
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    is_processing: bool | None


class Trajectory:
    """Records agent execution as a sequence of runs."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        api_key: str = "no_key",
        think: bool = False,
        temperature: float | None = None,
    ):
        """Initialize the LLM with the given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.temperature = temperature

    def _monitoring_url(self, endpoint: str) -> str:
        parsed_url = urlsplit(self.base_url)
        path = parsed_url.path.rstrip("/")
        for endpoint_suffix in ("/v1/chat/completions", "/chat/completions"):
            if path.endswith(endpoint_suffix):
                path = path[: -len(endpoint_suffix)]
                break
        else:
            raise ValueError(
                "Cannot derive the llama.cpp monitoring URL from the chat endpoint."
            )
        return urlunsplit(
            (parsed_url.scheme, parsed_url.netloc, f"{path}/{endpoint}", "", "")
        )

    def slot_context(self) -> list[SlotContext] | None:
        """Return per-slot prompt/context details when llama.cpp exposes them."""
        request = urllib.request.Request(
            self._monitoring_url("slots"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                data = json.loads(response.read())
        except HTTPError as error:
            if error.code in (404, 501):
                return None
            raise

        if not isinstance(data, list):
            raise ValueError("The llama.cpp /slots response must be a JSON array.")

        slots = []
        for index, slot in enumerate(data):
            if not isinstance(slot, dict):
                raise ValueError(f"Invalid llama.cpp slot entry at index {index}.")

            def optional_nonnegative_int(field: str) -> int | None:
                value = slot.get(field)
                if value is None:
                    return None
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"Invalid {field} value in llama.cpp slot {index}."
                    )
                return value

            slot_id = optional_nonnegative_int("id")
            context_size = optional_nonnegative_int("n_ctx")
            if slot_id is None or context_size is None or context_size == 0:
                raise ValueError(
                    f"llama.cpp slot {index} is missing its id or context size."
                )

            is_processing = slot.get("is_processing")
            if is_processing is not None and not isinstance(is_processing, bool):
                raise ValueError(
                    f"Invalid is_processing value in llama.cpp slot {index}."
                )

            slots.append(
                SlotContext(
                    slot_id=slot_id,
                    context_size=context_size,
                    prompt_tokens=optional_nonnegative_int("n_prompt_tokens"),
                    cached_prompt_tokens=optional_nonnegative_int(
                        "n_prompt_tokens_cache"
                    ),
                    is_processing=is_processing,
                )
            )
        return slots

    def kv_cache_usage_ratio(self) -> float | None:
        """Return llama.cpp's current KV-cache usage ratio when metrics are enabled."""
        request = urllib.request.Request(
            self._monitoring_url("metrics"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                metrics = response.read().decode("utf-8")
        except HTTPError as error:
            if error.code in (404, 501):
                return None
            raise

        metric_pattern = re.compile(
            r"^llamacpp:kv_cache_usage_ratio(?:\{[^}]*\})?\s+(\S+)"
        )
        ratios = []
        for line in metrics.splitlines():
            match = metric_pattern.match(line)
            if match:
                ratio = float(match.group(1))
                if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                    raise ValueError(
                        f"Invalid llama.cpp KV-cache usage ratio: {ratio}."
                    )
                ratios.append(ratio)

        if not ratios:
            return None
        if len(ratios) > 1:
            raise ValueError(
                "The llama.cpp metrics endpoint returned multiple KV-cache usage "
                "ratios; a single total cannot be determined."
            )
        return ratios[0]

    def generate(self, messages: list[dict], tools: list | None = None) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"
        else:
            body["reasoning_effort"] = "medium"
        if self.temperature is not None:
            body["temperature"] = self.temperature

        # POST to the configured OpenAI-compatible chat completions endpoint
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=reasoning,
            tool_call=tool_call,
            metadata=metadata,
        )


class EmbeddingModel:
    """Generate embeddings."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
    ):
        """Initialize the embedding model with the given model."""
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        # POST to the OpenAI-compatible /embeddings endpoint
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # Extract and return the embedding
        return response["data"][0]["embedding"]
