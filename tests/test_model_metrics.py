import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from nanoagent.models import LLM, SlotContext


class ModelMetricsTests(unittest.TestCase):
    def test_reads_per_slot_context_details(self):
        response = Mock()
        response.read.return_value = (
            b'[{"id":0,"n_ctx":16384,"n_prompt_tokens":4978,'
            b'"n_prompt_tokens_cache":1200,"is_processing":false}]'
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        llm = LLM(model="test", api_key="test-key")

        with patch(
            "nanoagent.models.urllib.request.urlopen", return_value=response
        ) as urlopen:
            slots = llm.slot_context()

        self.assertEqual(
            slots,
            [SlotContext(0, 16384, 4978, 1200, False)],
        )
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "http://127.0.0.1:8080/slots",
        )

    def test_reads_current_kv_cache_ratio_from_llama_metrics(self):
        response = Mock()
        response.read.return_value = (
            b"# HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio\n"
            b"llamacpp:kv_cache_usage_ratio 0.375\n"
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        llm = LLM(model="test", api_key="test-key")

        with patch(
            "nanoagent.models.urllib.request.urlopen", return_value=response
        ) as urlopen:
            ratio = llm.kv_cache_usage_ratio()

        self.assertEqual(ratio, 0.375)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/metrics")
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-key"
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 2)

    def test_returns_none_when_metrics_endpoint_is_unavailable(self):
        llm = LLM(model="test")

        for status_code in (404, 501):
            with self.subTest(status_code=status_code):
                error = HTTPError(
                    "http://127.0.0.1:8080/metrics",
                    status_code,
                    "Metrics unavailable",
                    {},
                    None,
                )
                with patch(
                    "nanoagent.models.urllib.request.urlopen", side_effect=error
                ):
                    self.assertIsNone(llm.kv_cache_usage_ratio())

    def test_rejects_multiple_cache_usage_series(self):
        response = Mock()
        response.read.return_value = (
            b'llamacpp:kv_cache_usage_ratio{slot="0"} 0.2\n'
            b'llamacpp:kv_cache_usage_ratio{slot="1"} 0.3\n'
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        llm = LLM(model="test")

        with patch("nanoagent.models.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "multiple"):
                llm.kv_cache_usage_ratio()


if __name__ == "__main__":
    unittest.main()
