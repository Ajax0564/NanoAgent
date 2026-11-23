import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from nanoagent import cli
from nanoagent.models import SlotContext


class CliTests(unittest.TestCase):
    def test_rich_progress_pauses_while_tools_may_request_approval(self):
        progress = Mock()
        used_tools = []
        payload = SimpleNamespace(tool_call={"tool": "write_file"})

        with patch.object(cli.console, "print") as console_print:
            cli._display_rich_progress(
                "tool_start", payload, "test-model", progress, used_tools
            )

        self.assertEqual(used_tools, ["write_file"])
        progress.stop.assert_called_once_with()
        progress.update.assert_not_called()
        console_print.assert_called_once_with("  Running tool: write_file")

    def test_rich_progress_restarts_after_tool_result(self):
        progress = Mock()

        cli._display_rich_progress(
            "tool_result", None, "test-model", progress, []
        )

        progress.start.assert_called_once_with()
        progress.update.assert_called_once_with("Tool finished; continuing...")

    def test_elapsed_time_is_printed_after_final_result(self):
        with patch.object(cli.console, "print") as console_print:
            cli._display_run_result(
                "Final answer",
                ["write_file"],
                2.5,
                "Current KV cache usage: 37.5%",
            )

        self.assertEqual(
            console_print.call_args_list,
            [
                call("\n  [dim]Tools used:[/] write_file"),
                call("\nFinal answer\n"),
                call("  [dim]Elapsed time: 2.50 seconds.[/]\n"),
                call("  [dim]Current KV cache usage: 37.5%[/]\n"),
            ],
        )

    def test_kv_cache_status_reports_disabled_metrics(self):
        llm = Mock()
        llm.slot_context.return_value = None
        llm.kv_cache_usage_ratio.return_value = None

        self.assertIn("does not expose", cli._kv_cache_status(llm))

    def test_kv_cache_status_reports_current_ratio(self):
        llm = Mock()
        llm.slot_context.return_value = None
        llm.kv_cache_usage_ratio.return_value = 0.375

        self.assertEqual(
            cli._kv_cache_status(llm), "Current KV cache usage: 37.5%"
        )

    def test_kv_cache_status_reports_per_slot_context(self):
        llm = Mock()
        llm.slot_context.return_value = [
            SlotContext(0, 16384, 4978, 1200, False),
            SlotContext(1, 8192, 100, 0, False),
        ]

        self.assertEqual(
            cli._kv_cache_status(llm),
            "KV cache context: 5,078 / 24,576 tokens",
        )
        llm.kv_cache_usage_ratio.assert_not_called()


if __name__ == "__main__":
    unittest.main()
