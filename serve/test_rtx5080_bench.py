"""CPU-only tests for tools/rtx5080_openclaw_bench.py."""
import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rtx5080_openclaw_bench", ROOT / "tools" / "rtx5080_openclaw_bench.py")
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


class TinySSE(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ]
        for obj in chunks:
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class HarnessHelpers(unittest.TestCase):
    def test_build_prompt_short_and_long(self):
        self.assertEqual(BENCH.build_prompt("abc", 0, "A"), "abc")
        p = BENCH.build_prompt("abc", 1000, "A")
        self.assertGreaterEqual(len(p), 3900)
        self.assertIn("BEGIN BENCHMARK HISTORY LANE A", p)
        self.assertTrue(p.endswith("abc"))

    def test_compact_metrics_drops_history(self):
        src = {
            "engine": {"model": "m", "max_context": 131072, "batch_slots": 2, "secret": "drop"},
            "live": {"parallel": 2, "running": 2},
            "hardware": {"gpu": "x"},
            "conversation_cache": {"slots": 2},
            "requests": [{"huge": "history"}],
            "history": [{"huge": "series"}],
        }
        got = BENCH.compact_metrics(src)
        self.assertNotIn("requests", got)
        self.assertNotIn("history", got)
        self.assertNotIn("secret", got["engine"])
        self.assertEqual(got["engine"]["batch_slots"], 2)
        self.assertEqual(got["live"]["running"], 2)

    def test_slot_observations(self):
        samples = [
            {"data": {"live": {"state": "generating", "running": 2, "waiting": 1,
                               "slots": [{"state": "decoding"}, {"state": "idle"}]}}},
            {"data": {"live": {"state": "generating", "running": 2, "waiting": 0,
                               "slots": [{"state": "decoding"}, {"state": "reading"}]}}},
        ]
        got = BENCH.useful_slot_observations(samples)
        self.assertEqual(got["max_running"], 2)
        self.assertEqual(got["max_waiting_or_queued"], 1)
        self.assertEqual(got["max_nonidle_slots"], 2)

    def test_stream_parser_records_ttft_and_text(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), TinySSE)
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            result = {}
            gate = threading.Barrier(2)
            worker = threading.Thread(
                target=BENCH.post_stream,
                args=(f"http://127.0.0.1:{httpd.server_address[1]}", "", "prompt", 8, gate, result),
            )
            worker.start()
            gate.wait(timeout=5)
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(result["ok"])
            self.assertEqual(result["content"], "hello world")
            self.assertEqual(result["finish_reason"], "stop")
            self.assertIsNotNone(result["ttft_s"])
            self.assertGreaterEqual(result["ttft_s"], 0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_markdown_report(self):
        report = {
            "arm": "c2",
            "started_at": "now",
            "base_url": "http://127.0.0.1:8080",
            "expect_serving": 2,
            "target_prompt_tokens": 0,
            "max_tokens": 32,
            "campaign_wall_s": 1.5,
            "status_before": {"concurrency": {"serving": 2}},
            "requests": [{"lane": "A", "ok": True, "ttft_s": 0.1, "wall_s": 1.0,
                          "finish_reason": "stop", "content": "x"}],
            "gpu_summary": {},
            "metrics_observations": {"max_running": 2, "max_nonidle_slots": 2,
                                     "max_waiting_or_queued": 0},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.md"
            BENCH.write_markdown(p, report)
            text = p.read_text(encoding="utf-8")
            self.assertIn("RTX 5080 C1/C2 benchmark", text)
            self.assertIn("Max non-idle batch slots: **2**", text)


if __name__ == "__main__":
    unittest.main()
