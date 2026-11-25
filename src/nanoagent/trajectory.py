from dataclasses import dataclass
from .models import Response


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent execution as a sequence of runs.

    Each run is a dict with keys "query" (str) and "steps" (list[Step]).
    """

    def __init__(self) -> None:
        """Create an empty trajectory history.

        Inputs:
            None.

        Returns:
            None. Initializes ``runs`` as an empty list.
        """
        self.runs: list[dict] = []

    def new_run(self, query: str) -> None:
        """Start a run and initialize its step list.

        Args:
            query: User request associated with the new run.

        Returns:
            None. Appends a run dictionary containing the query and no steps.
        """
        self.runs.append({"query": query, "steps": []})

    def add_step(self, response: Response, observation: str | None = None) -> None:
        """Append a model answer or tool interaction to the current run.

        Args:
            response: Model response containing reasoning and, when applicable,
                the tool call or final answer.
            observation: Tool result to record. If omitted, the response content
                is recorded as the answer.

        Returns:
            None. Adds a ``Step`` to the most recently initialized run.

        Raises:
            IndexError: If no run has been started with ``new_run``.
        """
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content
        self.runs[-1]["steps"].append(step)

    @property
    def steps(self) -> list[Step]:
        """Return a flattened snapshot of steps from every recorded run.

        Inputs:
            None.

        Returns:
            A new list of ``Step`` objects in run and insertion order.
        """
        return [step for run in self.runs for step in run["steps"]]

    @property
    def last_run(self) -> dict | None:
        """Return the latest run record, or ``None`` before the first run.

        Inputs:
            None.

        Returns:
            The final run dictionary, containing ``query`` and ``steps``, or
            ``None`` when the trajectory is empty.
        """
        return self.runs[-1] if self.runs else None