import json
import unittest
from unittest.mock import Mock, patch

from nanoagent.agent import NanoAgent
from nanoagent.memory import Memory
from nanoagent.models import Response
from nanoagent.planning import NativeReAct
from nanoagent.tools import NativeTools
from nanoagent.toolbox.websearch_tools import (
    DEFAULT_MAX_RESULTS,
    SEARCH_TIMEOUT_SECONDS,
    WebSearchTool,
)


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, tools=None):
        return next(self.responses)


class WebSearchBudgetTests(unittest.TestCase):
    def test_reuses_duplicate_queries_and_caps_distinct_searches_per_turn(self):
        queries = []

        def search(query):
            queries.append(query)
            return json.dumps({"success": True, "query": query})

        tools = NativeTools()
        tools.add_tool("web_search", search)
        responses = [
            Response(
                tool_call={
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": query},
                    }
                }
            )
            for query in (
                "today's weather",
                " TODAY'S   WEATHER ",
                "local news",
                "another topic",
            )
        ]
        responses.append(Response(content="Answer"))
        memory = Memory()
        agent = NanoAgent(
            SequenceLLM(responses),
            memory,
            tools,
            NativeReAct(max_steps=10),
        )

        self.assertEqual(agent.run("Find current information"), "Answer")

        self.assertEqual(queries, ["today's weather", "local news"])
        observations = [
            message["content"]
            for message in memory.messages
            if message["role"] == "tool"
        ]
        self.assertEqual(observations[0], observations[1])
        self.assertIn("Web search limit reached", observations[3])

    def test_search_uses_explicit_timeout_and_smaller_default(self):
        ddgs = Mock()
        ddgs.__enter__ = Mock(return_value=ddgs)
        ddgs.__exit__ = Mock(return_value=None)
        ddgs.text.return_value = []

        with patch(
            "nanoagent.toolbox.websearch_tools.DDGS", return_value=ddgs
        ) as factory:
            WebSearchTool().execute("sample query")

        factory.assert_called_once_with(timeout=SEARCH_TIMEOUT_SECONDS)
        ddgs.text.assert_called_once_with(
            "sample query", max_results=DEFAULT_MAX_RESULTS
        )
        self.assertEqual(DEFAULT_MAX_RESULTS, 3)
        self.assertEqual(SEARCH_TIMEOUT_SECONDS, 5)


if __name__ == "__main__":
    unittest.main()
