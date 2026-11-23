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
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.display = display or (lambda *_args: None)
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
        """Run the agent on a task."""
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
        """Perform a single step."""
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
        """Execute a tool action."""

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
        """Correct invalid calls once; stop on other or repeated failures."""
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
        """Append one structured agent event to the configured JSONL log."""
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
        if not isinstance(tool_call, dict):
            return "tool"
        name = tool_call.get("tool")
        if not name and isinstance(tool_call.get("function"), dict):
            name = tool_call["function"].get("name")
        return name if isinstance(name, str) and name else "tool"

    @staticmethod
    def _web_search_query_key(response: Response) -> str | None:
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
        """Normalize structured tool errors to a classified failure."""
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
