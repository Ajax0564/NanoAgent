from collections.abc import Callable
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from .models import LLM, Response, Trajectory
from .memory import Memory
from .planning import ReAct
from .tools import ToolFailure, Tools


class NanoAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools,
        planner: ReAct,
        display: Callable[..., None] | None = None,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.display = display or (lambda *_args: None)

        self.trajectory = Trajectory()

        # Build system prompt with all components
        system_prompt = (
            "You are a helpful assistant.\n\n"
            "For time-sensitive questions, use the application date/time provided "
            "with each user request as today's date. For current events, prices, "
            "scores, or schedules, use web_search and verify the source date. "
            "Never present an older or undated result as current; if freshness "
            "cannot be verified, say so and give the source date when available. "
            "If web_search reports a failure, do not repeat the same query; explain "
            "that current information could not be retrieved and include the error "
            "category when useful.\n\n"
        )
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str, image_data: str = None) -> str:
        """Run the agent on a task."""
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        runtime_context = (
            f"Application date/time (Asia/Kolkata): "
            f"{now.strftime('%A, %B %d, %Y at %H:%M %Z')}\n\n"
            f"User request:\n{task}"
        )
        self.memory.begin_turn()
        self._tool_corrections_remaining = 1
        try:
            self.memory.add("user", runtime_context, image_data=image_data)
            self.trajectory.initialize(task)

            # *Autonomy* loop
            for step in range(self.planner.max_steps):
                result = self._step()
                if result is not None:
                    return result

            return "Max steps reached without completion."
        finally:
            self.memory.end_turn()

    def _step(self) -> str | None:
        """Perform a single step."""
        # THOUGHT: Generate response and add to memory
        self.display("thinking")
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )
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
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str | None:
        """Execute a tool action."""

        # ACTION: execute tools
        self.display("tool_start", response)
        result = self.tools.execute(response)
        failure = self._tool_failure(result)
        if failure is not None:
            return self._handle_tool_failure(response, failure)

        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
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
        return final_answer

    @staticmethod
    def _tool_name(tool_call: dict | None) -> str:
        if not isinstance(tool_call, dict):
            return "tool"
        name = tool_call.get("tool")
        if not name and isinstance(tool_call.get("function"), dict):
            name = tool_call["function"].get("name")
        return name if isinstance(name, str) and name else "tool"

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