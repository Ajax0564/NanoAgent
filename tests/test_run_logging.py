import json
import tempfile
import unittest
from pathlib import Path

from nanoagent.agent import NanoAgent
from nanoagent.memory import Memory
from nanoagent.models import Response
from nanoagent.planning import NativeReAct
from nanoagent.tools import NativeTools


class SequenceLLM:
    model = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, tools=None):
        return next(self.responses)


class RunLoggingTests(unittest.TestCase):
    def test_logs_each_prompt_response_tool_result_and_final_answer(self):
        tools = NativeTools()
        tools.add_tool("lookup", lambda query: f"result for {query}")
        llm = SequenceLLM(
            [
                Response(
                    content="",
                    tool_call={
                        "function": {
                            "name": "lookup",
                            "arguments": {"query": "example"},
                        }
                    },
                ),
                Response(content="The answer is example."),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "nested" / "agent.jsonl"
            agent = NanoAgent(
                llm,
                Memory(),
                tools,
                NativeReAct(max_steps=3),
                log_file=log_file,
            )

            result = agent.run("Find an example")

            events = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result, "The answer is example.")
        event_names = [event["event"] for event in events]
        self.assertEqual(
            event_names,
            [
                "run_started",
                "model_request",
                "model_response",
                "tool_start",
                "tool_result",
                "model_request",
                "model_response",
                "final_answer",
            ],
        )
        self.assertEqual(events[1]["messages"][0]["role"], "system")
        self.assertIn("Find an example", events[1]["messages"][-1]["content"])
        self.assertEqual(
            events[4]["result"],
            "result for example",
        )
        self.assertEqual(
            events[5]["messages"][-1]["content"],
            "result for example",
        )
        self.assertEqual(events[-1]["answer"], "The answer is example.")
        self.assertTrue(all(event["run_id"] == events[0]["run_id"] for event in events))

    def test_appends_new_runs_without_overwriting_previous_logs(self):
        llm = SequenceLLM(
            [Response(content="First answer"), Response(content="Second answer")]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "agent.jsonl"
            agent = NanoAgent(
                llm,
                Memory(),
                NativeTools(),
                NativeReAct(max_steps=1),
                log_file=log_file,
            )

            agent.run("First request")
            agent.run("Second request")

            events = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").splitlines()
            ]

        run_ids = {event["run_id"] for event in events}
        self.assertEqual(len(run_ids), 2)
        self.assertEqual(
            [event["answer"] for event in events if event["event"] == "final_answer"],
            ["First answer", "Second answer"],
        )


if __name__ == "__main__":
    unittest.main()
