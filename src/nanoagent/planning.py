import re
from .models import Response, LLM
from .memory import Memory
from .tools import Tools


class ReAct:
    """ReAct module."""

    def __init__(self, max_steps: int = 10):
        """Set the maximum number of model/tool iterations per request.

        Args:
            max_steps: Upper bound on iterations before the agent returns its
                max-steps response.

        Returns:
            None. Stores the step limit.
        """
        self.max_steps = max_steps

    @property
    def prompt(self) -> str:
        """Return instructions describing the text-based ReAct protocol.

        Inputs:
            None.

        Returns:
            The system-prompt text that specifies THOUGHT/ACTION formatting and
            the ``final_answer`` completion action.
        """
        return """
# ReAct (Reason and Act)

You are a ReAct agent that performs exactly ONE step per turn.

## ReAct Format

You use the following format for each step:

THOUGHT: [Your reasoning about what to do next]
ACTION:
{
    "tool": "a_tool_name",
    "kwargs": {"param": "value"},
}

An observation will be provided after each action. You do not generate the observation yourself.

## ReAct Completion

To provide the final answer to the task, use an action blob with "tool": "final_answer" tool. 
It is the only way to complete the task, else you will be stuck on a loop. 
So your final output should look like this:

ACTION:
{
    "tool": "final_answer",
    "kwargs": "insert your final answer here"
}

Use the `final_answer` tool when you are completely done with all subtasks and have the final answer ready.
You can also use `final_answer` to directly reply to a user's question without using any other tools.

"""

    def parse(self, response: Response) -> Response:
        """Extract THOUGHT and ACTION sections from a text response.

        Args:
            response: Response whose content may contain ReAct section labels.

        Returns:
            The same response object with ``content`` replaced by the extracted
            ACTION and ``reasoning`` replaced by THOUGHT. Missing sections become
            empty strings.
        """
        text = response.content

        # The patterns for each section
        patterns = {
            "THOUGHT": r"THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)",
            "ACTION": r"ACTION:\s*(.+?)(?=THOUGHT:|OBSERVATION:|$)",
        }

        # Extract each section using regex
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            result[key] = match.group(1).strip() if match else ""

        # Update Response and extract only the action
        response.content = result["ACTION"]
        response.reasoning = result["THOUGHT"]
        return response


class NativeReAct(ReAct):
    """ReAct using native LLM reasoning instead of text-based parsing."""

    @property
    def prompt(self) -> str:
        """Return no extra instructions for native tool-calling models.

        Inputs:
            None.

        Returns:
            An empty string because the model API supplies tool schemas directly.
        """
        return ""

    def parse(self, response: Response) -> Response:
        """Pass a native tool-calling response through unchanged.

        Args:
            response: Model response containing its native tool-call structure.

        Returns:
            The same response object.
        """
        return response


class PlanAndSolve(ReAct):
    """ReAct planner that gives the LLM tools to create and update its own plan."""

    def __init__(self, tools: Tools, max_steps: int = 10):
        """Initialize a planner and register its plan-management tools.

        Args:
            tools: Registry that receives the ``update_plan`` and ``mark_done``
                callable tools.
            max_steps: Maximum model/tool iterations for the request.

        Returns:
            None. Initializes an empty plan and attaches both tools to ``tools``.
        """
        super().__init__(max_steps)
        self.plan: list[dict] = []

        tools.add_tool(
            "update_plan",
            self.update_plan,
            "Create or overwrite the plan with a list of sub-tasks: update_plan(items: list[str])",
        )
        tools.add_tool(
            "mark_done",
            self.mark_done,
            "Mark a sub-task complete by 1-based index: mark_done(index: int)",
        )

    @property
    def prompt(self) -> str:
        """Return ReAct instructions augmented with planning guidance.

        Inputs:
            None.

        Returns:
            The base ReAct prompt plus instructions to create, update, and mark
            plan items complete.
        """
        return super().prompt + (
            "\n## Planning\n\n"
            "For non-trivial tasks, call `update_plan` first to outline your approach, "
            "then `mark_done` as you complete each sub-task. Revise the plan with "
            "`update_plan` whenever new information changes what is needed.\n"
        )

    def update_plan(self, items: list[str]) -> str:
        """Replace the current plan with unfinished sub-tasks.

        Args:
            items: Ordered task descriptions for the plan.

        Returns:
            A status string containing the rendered plan. Every new task starts
            with ``done=False``.
        """
        self.plan = [{"task": item, "done": False} for item in items]
        return f"Plan updated:\n{self._render()}"

    def mark_done(self, index: int) -> str:
        """Mark one plan item complete using its 1-based position.

        Args:
            index: Position of the task in the current plan, starting at one.

        Returns:
            A status string containing the updated rendered plan.

        Notes:
            The valid caller range is ``1`` through the current plan length.
        """
        self.plan[index - 1]["done"] = True
        return f"Plan:\n{self._render()}"

    def _render(self) -> str:
        """Format all plan items as numbered checkbox lines.

        Inputs:
            None.

        Returns:
            A newline-separated string using ``[x]`` for completed tasks and
            ``[ ]`` for pending tasks; an empty plan produces an empty string.
        """
        return "\n".join(
            f"{'[x]' if item['done'] else '[ ]'} {i + 1}. {item['task']}"
            for i, item in enumerate(self.plan)
        )


class Reflexion:
    """Reflector that periodically critiques recent steps and injects a course correction."""

    def __init__(self, llm: LLM, interval: int = 3):
        """Configure periodic critique generation.

        Args:
            llm: Model client used to produce critiques.
            interval: Positive step interval at which ``reflect`` requests a
                critique. Zero causes ``ZeroDivisionError`` on nonzero steps.

        Returns:
            None. Stores the model and reflection interval.
        """
        self.llm = llm
        self.interval = interval

    def reflect(self, memory: Memory, step: int) -> None:
        """Ask the model for a critique on scheduled steps and store it.

        Args:
            memory: Conversation memory supplying the recent messages and receiving
                the corrective reflection message.
            step: Current zero-based step number.

        Returns:
            None. Does nothing for step zero or steps not divisible by ``interval``;
            otherwise appends a critique as a user message.
        """
        if step == 0 or step % self.interval:
            return
        critique = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "Critique the agent's recent steps and suggest a course correction.",
                },
                *memory.get_messages()[-6:],
            ]
        )
        memory.add("user", f"[Reflection] {critique.content}")
