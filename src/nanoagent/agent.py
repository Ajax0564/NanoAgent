from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from .models import LLM, Response, Trajectory
from .memory import Memory
from .planning import ReAct
from .tools import ToolFailure, Tools


MAX_WEB_SEARCHES_PER_TURN = 2


def _ignore_display(*_args) -> None:
    """Provide the no-op progress callback used when display is not configured.

    Args:
        *_args: Progress event and payload values, accepted and ignored.

    Returns:
        None.
    """


class NanoAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools,
        planner: ReAct,
        display: Callable[..., None] | None = None,
        log_file: str | Path | None = None,
        instructions: str | None = None,
    ):
        """Configure the model, memory, planner, tools, display, and system prompt.

        Args:
            llm: Synchronous language-model client used to generate each step.
            memory: Conversation store; receives the system prompt immediately.
            tools: Tool registry used to parse, validate, and execute tool calls.
            planner: Response parser and maximum-step policy for the run loop.
            display: Optional callback receiving progress event names and payloads.
                When omitted, progress events are ignored.
            log_file: Optional JSONL path for recording run and tool events.
            instructions: Optional profile-specific instructions appended to the
                system prompt.

        Returns:
            None. Initializes per-agent trajectory and system-prompt state.
        """
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.display = display or _ignore_display
        self.log_file = Path(log_file) if log_file is not None else None

        self.trajectory = Trajectory()

        system_prompt = "You are a helpful assistant.\n\n"
        if "web_search" in self.tools.registry:
            system_prompt += (
                "For time-sensitive questions, use the application date/time provided "
                "with each user request as today's date. For current events, prices, "
                "scores, or schedules, use web_search and verify the source date. "
                "Never present an older or undated result as current; if freshness "
                "cannot be verified, say so and give the source date when available. "
                "Make one focused web_search call and reuse its results. Make at "
                "most two distinct searches per user request, only when needed; "
                "never repeat a query. If the search budget is reached, answer from "
                "the available results and state any limitations. "
                "If web_search reports a failure, do not repeat the same query; explain "
                "that current information could not be retrieved and include the error "
                "category when useful.\n\n"
            )
        if instructions:
            system_prompt += f"{instructions}\n\n"
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str, image_data: str = None) -> str:
        """Run a user request through the bounded model/tool loop.

        Args:
            task: Text request to answer.
            image_data: Optional image payload forwarded to conversation memory.

        Returns:
            The final answer, a tool-failure message, or a max-steps message.

        Raises:
            Exception: Propagates unexpected model, memory, logging, or callback
                failures after recording them when logging is enabled.
        """
        self._run_id = str(uuid4())
        self._step_number = 0
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        runtime_context = (
            f"Application date/time (Asia/Kolkata): "
            f"{now.strftime('%A, %B %d, %Y at %H:%M %Z')}\n\n"
            f"User request:\n{task}"
        )
        self.memory.begin_turn()
        self._tool_corrections_remaining = 1
        self._web_search_count = 0
        self._web_search_results: dict[str, Any] = {}
        try:
            self._log_event(
                "run_started",
                task=task,
                runtime_context=runtime_context,
                image_data=image_data,
            )
            self.memory.add("user", runtime_context, image_data=image_data)
            self.trajectory.initialize(task)

            # Autonomy loop
            for step in range(self.planner.max_steps):
                result = self._step()
                if result is not None:
                    return result

            answer = "Max steps reached without completion."
            self._log_event("final_answer", answer=answer, reason="max_steps")
            return answer
        except Exception as error:
            self._log_event(
                "run_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            self.memory.end_turn()

    def _step(self) -> str | None:
        """Generate, parse, and handle one model response.

        Inputs:
            None. Reads current memory and uses the configured model, planner,
            tools, and display callback.

        Returns:
            A final answer when the run is complete or fails; otherwise ``None``
            after scheduling another model step.
        """
        # THOUGHT: Generate response and add to memory
        self._step_number += 1
        messages = self.memory.get_messages()
        self._log_event(
            "model_request",
            step=self._step_number,
            model=getattr(self.llm, "model", None),
            messages=messages,
            tools=self.tools.schemas,
        )
        self.display("thinking")
        response = self.llm.generate(messages, tools=self.tools.schemas)
        self._log_event(
            "model_response",
            step=self._step_number,
            response={
                "content": response.content,
                "reasoning": response.reasoning,
                "tool_call": response.tool_call,
                "metadata": response.metadata,
            },
        )
        self.memory.add("assistant", response.content, tool_call=response.tool_call)
        self.display("response", response)

        # Tool parsing
        response = self.planner.parse(response)
        try:
            response = self.tools.parse(response)
        except ToolFailure as failure:
            return self._handle_tool_failure(response, failure)

        # ANSWER: Stopping mechanism
        if self.tools.is_done(response):
            self.trajectory.add(response)
            self.display("final", response.content)
            self._log_event("final_answer", answer=response.content)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str | None:
        """Execute one parsed tool call and record its result.

        Args:
            response: Parsed response whose ``tool_call`` names a registered tool.

        Returns:
            A final failure message when execution cannot continue, or ``None``
            after adding a successful observation to memory and trajectory.
        """

        # ACTION: execute tools
        self._log_event(
            "tool_start",
            step=self._step_number,
            tool_call=response.tool_call,
        )
        self.display("tool_start", response)
        tool_name = self._tool_name(response.tool_call)
        query_key = (
            self._web_search_query_key(response) if tool_name == "web_search" else None
        )
        if query_key is not None and query_key in self._web_search_results:
            result = self._web_search_results[query_key]
        elif (
            query_key is not None
            and self._web_search_count >= MAX_WEB_SEARCHES_PER_TURN
        ):
            result = (
                "Web search limit reached (two distinct searches per request). "
                "Use the results already in the conversation and state any "
                "remaining uncertainty."
            )
        else:
            if tool_name == "web_search" and query_key is not None:
                self._web_search_count += 1
            try:
                result = self.tools.execute(response)
            except ToolFailure as failure:
                return self._handle_tool_failure(response, failure)
            if query_key is not None and not isinstance(result, ToolFailure):
                self._web_search_results[query_key] = result
        failure = self._tool_failure(result)
        if failure is not None:
            return self._handle_tool_failure(response, failure)

        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        self._log_event(
            "tool_result",
            step=self._step_number,
            tool_call=response.tool_call,
            result=result,
            observation=observation,
        )
        self.display("tool_result", result)

        return None

    def _handle_tool_failure(
        self, response: Response, failure: ToolFailure
    ) -> str | None:
        """Record a tool failure and decide whether one correction is allowed.

        Args:
            response: Model response containing the failed tool call.
            failure: Classified failure raised while parsing or executing it.

        Returns:
            ``None`` for the single recoverable correction attempt; otherwise a
            final user-facing failure message.
        """
        recoverable_codes = {"invalid_tool", "invalid_arguments", "invalid_tool_call"}
        can_correct = (
            failure.code in recoverable_codes and self._tool_corrections_remaining > 0
        )
        if can_correct:
            self._tool_corrections_remaining -= 1
            instruction = (
                "Correct the tool name or arguments and try once more. "
                "This is the only correction attempt."
            )
        else:
            instruction = "Stopped this run without retrying."

        observation = f"Tool call failed ({failure.code}): {failure}. {instruction}"
        role, _ = self.tools.observation(str(failure))
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        self._log_event(
            "tool_result",
            step=self._step_number,
            tool_call=response.tool_call,
            failure={"code": failure.code, "message": str(failure)},
            observation=observation,
        )
        self.display("tool_result", failure)

        if can_correct:
            return None

        tool_name = self._tool_name(response.tool_call)
        final_answer = (
            f"Tool '{tool_name}' failed ({failure.code}): {failure}\n"
            "Stopped this run. You can enter another prompt when ready."
        )
        self.memory.add("assistant", final_answer)
        self.trajectory.add(Response(content=final_answer))
        self.display("final", final_answer)
        self._log_event("final_answer", answer=final_answer, reason="tool_failure")
        return final_answer

    def _log_event(self, event: str, **data: Any) -> None:
        """Append a timestamped event record when JSONL logging is configured.

        Args:
            event: Event type, such as ``model_request`` or ``tool_result``.
            **data: JSON-serializable event-specific fields.

        Returns:
            None. Does nothing when ``log_file`` is unset.
        """
        if self.log_file is None:
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "event": event,
            **data,
        }
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _tool_name(tool_call: dict | None) -> str:
        """Extract a tool name from either internal or native call structure.

        Args:
            tool_call: Tool-call mapping, or ``None`` for a non-tool response.

        Returns:
            The non-empty tool name, or ``"tool"`` when it cannot be determined.
        """
        if not isinstance(tool_call, dict):
            return "tool"
        name = tool_call.get("tool")
        if not name and isinstance(tool_call.get("function"), dict):
            name = tool_call["function"].get("name")
        return name if isinstance(name, str) and name else "tool"

    @staticmethod
    def _web_search_query_key(response: Response) -> str | None:
        """Normalize a web-search query for per-run duplicate-result reuse.

        Args:
            response: Model response whose internal tool call may contain
                ``kwargs.query``.

        Returns:
            A whitespace-collapsed, case-folded query, or ``None`` if there is
            no non-empty string query.
        """
        tool_call = response.tool_call
        if not isinstance(tool_call, dict):
            return None
        kwargs = tool_call.get("kwargs")
        query = kwargs.get("query") if isinstance(kwargs, dict) else None
        if not isinstance(query, str):
            return None
        normalized_query = " ".join(query.split()).casefold()
        return normalized_query or None

    @staticmethod
    def _tool_failure(result) -> ToolFailure | None:
        """Convert a structured unsuccessful tool result to ``ToolFailure``.

        Args:
            result: Tool result object, or JSON string, optionally containing
                ``success: false``, an error code, and error details.

        Returns:
            The existing or newly constructed failure, or ``None`` when the
            result does not represent a structured failure.
        """
        if isinstance(result, ToolFailure):
            return result

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return None

        if not isinstance(result, dict) or result.get("success") is not False:
            return None

        code = result.get("code")
        if not code:
            code = {
                "TimeoutException": "timeout",
                "RatelimitException": "rate_limit",
            }.get(result.get("error_type"), "provider_unavailable")
        error = str(result.get("error") or "Tool returned an unsuccessful result.")
        details = result.get("details")
        if details:
            error += f" ({details})"
        return ToolFailure(str(code), error)
