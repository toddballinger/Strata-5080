"""CPU-only tests for the Issue #2 Pi C1/C2 harness: no GPU, no Strata engine, no downloads.

    python -m unittest discover -s tools -p 'test_pi_c1c2_bench.py'   (or: python tools/test_pi_c1c2_bench.py)

A fake Strata server on 127.0.0.1 answers /v1/status, /metrics and /api/requests and streams scripted OpenAI-shaped
SSE, so the lane loop, the tool-call guard, the scoring, the queue-delay separation, the evidence files and the
comparator's gate are all exercised without a model.  Nothing here starts or restarts a real server.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi_c1c2_bench as bench                          # noqa: E402
import pi_c1c2_compare as compare                      # noqa: E402

SESSIONS_USED = "server-option,expert-cache"     # two independent tasks, both over this checkout


def anchor_from(rel: str) -> str:
    """A line of that file, quoted verbatim - the anchor check has to pass on real bytes, not on a copy."""
    lines = [x.strip() for x in (ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines()]
    for line in sorted(lines, key=len, reverse=True):
        if 40 <= len(line) <= 120:
            return line
    raise AssertionError(f"no usable anchor line in {rel}")


ANCHOR = anchor_from("serve/server.py")
GOOD_ARGS = {"path": "serve/server.py", "anchor": ANCHOR,
             "summary": "Add a bounded queue in do_POST and answer 503 when it is full, with a test in serve/."}
# each lane submits against a file its own task is about, quoting a line of it
SUBMITS = {"A": ("serve/server.py", ANCHOR), "B": ("src/core/expert_cache.cpp", anchor_from("src/core/expert_cache.cpp"))}


def chunk(delta, finish=None, **extra):
    out = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    out.update(extra)
    return out


TIMINGS = {"cache_n": 0, "prompt_n": 1200, "prompt_ms": 600.0, "prompt_per_second": 2000.0,
           "predicted_n": 40, "predicted_ms": 1250.0, "predicted_per_second": 32.0,
           "draft_n": 30, "draft_n_accepted": 21}
USAGE = {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
         "prompt_tokens_details": {"cached_tokens": 0}}


def tool_call_chunks(name: str, arguments: dict, cid: str) -> list:
    """A tool call split the way a real stream splits it: the name and the arguments arrive in pieces."""
    blob = json.dumps(arguments)
    return [chunk({"role": "assistant", "content": ""}),
            chunk({"tool_calls": [{"index": 0, "id": cid, "type": "function",
                                   "function": {"name": name[:5]}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": name[5:],
                                                            "arguments": blob[:12]}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": blob[12:]}}]}),
            chunk({}, "tool_calls", usage=USAGE, timings=TIMINGS)]


class _Fake(BaseHTTPRequestHandler):
    server_version = "FakeStrata/1"

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        f = self.server.fake
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/status":
            return self._json({"model": "fake-model", "engine": {"loaded": True, "max_context": 131072},
                               "concurrency": {"serving": f.serving},
                               "vram": {"gpu_total_mib": 16303, "gpu_used_mib": 13000},
                               "machine": {"gpu": {"used_mib": 13000, "total_mib": 16303, "util_pct": 97,
                                                   "power_w": 205.0}, "ram": {"used_gib": 21.5}},
                               "activity": {"requests": 0, "in_flight": 0}})
        if path == "/metrics":
            return self._json({"engine": {"model": "fake-model", "max_context": 131072, "expert_slots": 1564},
                               "live": {"state": "idle", "queued": 0, "parallel": f.serving, "running": 0,
                                        "slots": [{"state": "idle", "tok_s": None} for _ in range(f.serving)]},
                               "requests": [{"prompt_tokens": 1200, "reused": 0, "prompt_read": 1200,
                                             "decode_tok_s": 32.0, "hit_rate": 0.41, "pcie_share": 0.6,
                                             "drafts_offered": 30, "drafts_accepted": 21, "duration_s": 1.5,
                                             "finish": "stop"}],
                               "conversation_cache": {"entries": 0}, "totals": {"requests": 2},
                               "hardware": {"gpu_mem_used": 13000, "gpu_mem_total": 16303, "gpu_util": 97,
                                            "gpu_power_w": 205.0, "ram_used_gib": 21.5}, "time": time.time()})
        if path == "/api/requests":
            if not f.api_requests:
                return self._json({"error": {"message": "the API monitor is off"}}, 404)
            return self._json({"requests": f.records, "retention": 100, "persistent": False, "loaded": True})
        return self._json({"error": {"message": "no such path"}}, 404)

    def do_POST(self):
        f = self.server.fake
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        text = json.dumps(body.get("messages") or [])
        nonce = next((x for x in ("A", "B") if f"[session {x}]" in text), "A")
        i = f.counts[nonce]
        f.counts[nonce] += 1
        mode = f.modes.get(nonce, f.mode)
        f.records.append({"id": f"r{len(f.records)}", "started_at": time.time(), "queue_s": f.queue_s.get(nonce, 0.0),
                          "first_token_s": 0.02, "state": "done", "model": "fake-model"})
        try:
            if mode == "boom":
                return self._json({"error": {"message": f"engine failed under {os.path.expanduser('~')}"}}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": keep-alive\n\n")
            if i == 0 and f.delays.get(nonce):
                time.sleep(f.delays[nonce])             # a lane that waited: its first token comes later
            for c in self.script(mode, nonce, i, body):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.flush()
                if mode == "slow":
                    time.sleep(0.12)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):    # the client hung up: the cancellation fixture
            pass

    def script(self, mode: str, nonce: str, i: int, body: dict) -> list:
        f = self.server.fake
        path, anchor = f.submits.get(nonce, SUBMITS["A"])
        good = {"path": path, "anchor": anchor,
                "summary": "Add a bounded queue in do_POST and answer 503 when it is full, with a test in serve/."}
        if mode == "silent":
            return [chunk({"content": "I think the fix is obvious."}), chunk({}, "stop", usage=USAGE,
                                                                             timings=TIMINGS)]
        if mode == "undeclared":
            return tool_call_chunks("write_file", {"path": path}, f"call_{nonce}_{i}")
        if mode == "anchor_bad":
            args = dict(good, anchor="a line nobody wrote in this file anywhere")
            return tool_call_chunks("submit_solution", args, f"call_{nonce}_{i}") if i else \
                tool_call_chunks("read_file", {"path": path}, f"call_{nonce}_{i}")
        if mode == "path_bad":
            args = dict(good, path="setup.py", anchor=anchor_from("setup.py"))
            return tool_call_chunks("submit_solution", args, f"call_{nonce}_{i}") if i else \
                tool_call_chunks("read_file", {"path": path}, f"call_{nonce}_{i}")
        if mode == "slow":
            return [chunk({"content": "thinking " * 20}), chunk({"content": "and reading " * 20}),
                    chunk({"content": "and again " * 20}), chunk({}, "stop", usage=USAGE, timings=TIMINGS)]
        if i == 0:
            return tool_call_chunks("read_file", {"path": path}, f"call_{nonce}_{i}")
        return tool_call_chunks("submit_solution", good, f"call_{nonce}_{i}")


class FakeStrata:
    def __init__(self, serving=2, mode="good", modes=None, api_requests=True, queue_s=None, submits=None,
                 delays=None):
        self.serving, self.mode, self.modes = serving, mode, modes or {}
        self.api_requests, self.submits = api_requests, submits or SUBMITS
        self.delays = delays or {}
        self.queue_s = queue_s or {}
        self.counts, self.records = {"A": 0, "B": 0}, []
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
        self.srv.fake = self
        self.srv.daemon_threads = True
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


def run_main(argv: list) -> tuple:
    out = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        old = sys.argv
        sys.argv = ["pi_c1c2_bench.py"] + argv
        try:
            code = bench.main()
        except SystemExit as e:
            if isinstance(e.code, int):
                code = e.code
            else:                       # a message: the harness refused to run, or was told something wrong
                code = 3
                out.write(f"\n{e}\n")
        finally:
            sys.argv = old
    return code, out.getvalue()


def campaign_args(fake: FakeStrata, tmp: str, *extra: str) -> list:
    return ["--url", fake.url, "--arm", "c2", "--expect-serving", "2", "--sessions", SESSIONS_USED,
            "--sample-interval", "0.05", "--output-dir", tmp, "--run", *extra]


class LaneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_dry_run_sends_nothing(self):
        """Without --run the harness must not open a connection at all."""
        real = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run touched the network"))
        try:
            code, text = run_main(["--arm", "c2", "--expect-serving", "2", "--context-class", "32k"])
        finally:
            urllib.request.urlopen = real
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", text)
        self.assertIn("15700 filler words", text)
        self.assertIn("submit_solution", text)

    def test_guard_refuses_the_wrong_arm(self):
        fake = FakeStrata(serving=1)
        try:
            code, text = run_main(campaign_args(fake, self.tmp))
            self.assertEqual(code, 3)
            self.assertIn("refusing to measure", text)
            self.assertIn("concurrency.serving=1", text)
            self.assertEqual(fake.counts, {"A": 0, "B": 0})       # no traffic was sent
        finally:
            fake.stop()

    def test_arm_and_expect_serving_must_agree(self):
        code, text = run_main(["--arm", "c1", "--expect-serving", "2"])
        self.assertEqual(code, 3)
        self.assertIn("contradicts", text)

    def test_two_lanes_complete_and_are_scored(self):
        fake = FakeStrata(serving=2)
        try:
            code, text = run_main(campaign_args(fake, self.tmp))
            self.assertEqual(code, 0, text)
            self.assertEqual(fake.counts, {"A": 2, "B": 2})       # read, then submit, per lane
            report = load_one(self.tmp)
            self.assertEqual(report["schema"], bench.SCHEMA)
            self.assertEqual(report["run"]["arm"], "c2")
            self.assertEqual(report["decision_metric"]["tasks_completed"], 2)
            self.assertEqual(report["decision_metric"]["useful_work"], 2.0)
            self.assertGreater(report["decision_metric"]["useful_work_per_minute"], 0)
            self.assertEqual(report["guard"]["serving_observed"], 2)
            for lane in report["lanes"]:
                self.assertTrue(lane["task_complete"], lane["checks"])
                self.assertEqual(lane["reads"], 1)
                self.assertEqual(lane["tool_calls"], 2)
                self.assertEqual(lane["decode_tok_s_server"], 32.0)
                self.assertGreater(lane["client_tok_s"], 0)          # the client's own whole-lane rate
                self.assertEqual(lane["draft_n_accepted"], 21)
                self.assertLessEqual(lane["model_ttft_s"], lane["ttft_s"])
        finally:
            fake.stop()

    def test_queue_delay_comes_from_the_server_when_available(self):
        fake = FakeStrata(serving=2, queue_s={"A": 0.0, "B": 3.5}, delays={"B": 0.6})
        try:
            code, _ = run_main(campaign_args(fake, self.tmp))
            self.assertEqual(code, 0)
            report = load_one(self.tmp)
            self.assertEqual(report["server_evidence"]["queue_delay_source"], "api_requests")
            self.assertTrue(report["server_evidence"]["api_requests_available"])
            by_lane = {l["lane"]: l for l in report["lanes"]}
            self.assertGreater(by_lane["B"]["ttft_s"], by_lane["A"]["ttft_s"])   # B is the one that waited
            self.assertEqual(by_lane["A"]["queue_delay_s"], 0.0)
            self.assertEqual(by_lane["B"]["queue_delay_s"], 3.5)
            self.assertEqual(by_lane["B"]["queue_delay_source"], "api_requests")
            self.assertGreaterEqual(by_lane["B"]["model_ttft_s"], 0.0)  # TTFT minus the queue, never negative
            self.assertLessEqual(by_lane["B"]["model_ttft_s"], by_lane["B"]["ttft_s"])
        finally:
            fake.stop()

    def test_undeclared_tool_is_a_correctness_failure(self):
        fake = FakeStrata(serving=2, mode="undeclared")
        try:
            run_main(campaign_args(fake, self.tmp))
            report = load_one(self.tmp)
            for lane in report["lanes"]:
                self.assertEqual(lane["stopped"], "undeclared_tool")
                self.assertFalse(lane["checks"]["declared_tools_only"])
                self.assertEqual(lane["undeclared_tools"], ["write_file"])
                self.assertFalse(lane["task_complete"])
            self.assertEqual(report["decision_metric"]["tasks_completed"], 0)
            self.assertLess(report["decision_metric"]["useful_work"], 2.0)
        finally:
            fake.stop()

    def test_anchor_must_be_verbatim(self):
        fake = FakeStrata(serving=2, mode="anchor_bad")
        try:
            run_main(campaign_args(fake, self.tmp))
            for lane in load_one(self.tmp)["lanes"]:
                self.assertTrue(lane["checks"]["read_before_submit"])
                self.assertFalse(lane["checks"]["anchor_verbatim"])
                self.assertFalse(lane["task_complete"])
        finally:
            fake.stop()

    def test_target_path_must_be_one_the_task_is_about(self):
        fake = FakeStrata(serving=2, mode="path_bad")
        try:
            run_main(campaign_args(fake, self.tmp))
            for lane in load_one(self.tmp)["lanes"]:
                self.assertFalse(lane["checks"]["target_path_expected"])   # setup.py is not what these tasks change
                self.assertTrue(lane["checks"]["anchor_verbatim"])          # the anchor is real in the file it names
                self.assertFalse(lane["task_complete"])
        finally:
            fake.stop()

    def test_prose_without_a_tool_call_is_nudged_once_then_stops(self):
        fake = FakeStrata(serving=2, mode="silent")
        try:
            run_main(campaign_args(fake, self.tmp))
            for lane in load_one(self.tmp)["lanes"]:
                self.assertEqual(lane["stopped"], "no_tool_call")
                self.assertTrue(lane["nudged"])
                self.assertEqual(lane["tool_calls"], 0)
                self.assertFalse(lane["task_complete"])
            self.assertEqual(fake.counts["A"], 2)
        finally:
            fake.stop()

    def test_request_errors_are_recorded_and_redacted(self):
        fake = FakeStrata(serving=2, mode="boom")
        try:
            run_main(campaign_args(fake, self.tmp, "--api-key", "SECRETKEY123"))
            report = load_one(self.tmp)
            self.assertTrue(any("HTTP 500" in e for l in report["lanes"] for e in l["errors"]))
            self.assertEqual(report["decision_metric"]["tasks_completed"], 0)
            raw = json.dumps(report)
            self.assertNotIn(os.path.expanduser("~"), raw)
            self.assertIn("<user>", raw)
        finally:
            fake.stop()

    def test_cancellation_lets_the_other_lane_finish(self):
        fake = FakeStrata(serving=2, modes={"B": "slow"})
        try:
            code, _ = run_main(campaign_args(fake, self.tmp, "--cancel-lane", "B", "--cancel-after", "0.3"))
            self.assertEqual(code, 0)
            report = load_one(self.tmp)
            c = report["cancellation"]
            self.assertEqual(c["cancelled_lane"], "B")
            self.assertTrue(c["victim_aborted"])
            self.assertEqual(c["victim_finish"], "client_aborted")
            self.assertTrue(c["survivor_ok"])
            self.assertTrue(c["server_still_serving"])
            by_lane = {l["lane"]: l for l in report["lanes"]}
            self.assertTrue(by_lane["A"]["task_complete"])
            self.assertFalse(by_lane["B"]["task_complete"])
        finally:
            fake.stop()

    def test_markdown_carries_the_sections_and_the_numbers(self):
        fake = FakeStrata(serving=2)
        try:
            run_main(campaign_args(fake, self.tmp))
            report = load_one(self.tmp)
            md = bench.render_markdown(report)
            for heading in ("## Decision metric", "## Lanes", "## Server evidence", "## Correctness",
                            "## Not measured by this run"):
                self.assertIn(heading, md)
            self.assertIn("useful work per wall-clock minute", md)
            self.assertIn("queue delay from `api_requests`", md)
            self.assertNotIn(os.path.expanduser("~"), md)
        finally:
            fake.stop()

    def test_single_lane_needs_no_barrier(self):
        fake = FakeStrata(serving=2)
        try:
            code, _ = run_main(campaign_args(fake, self.tmp, "--lanes", "1"))
            self.assertEqual(code, 0)
            report = load_one(self.tmp)
            self.assertEqual([l["lane"] for l in report["lanes"]], ["A"])
            self.assertEqual(fake.counts["B"], 0)
        finally:
            fake.stop()


class PureTests(unittest.TestCase):
    def test_tool_call_fragments_are_folded(self):
        acc = {}
        bench.merge_tool_calls([{"index": 0, "id": "c1", "function": {"name": "read_"}}], acc)
        bench.merge_tool_calls([{"index": 0, "function": {"name": "file", "arguments": '{"pa'}}], acc)
        bench.merge_tool_calls([{"index": 0, "function": {"arguments": 'th":"x"}'}}], acc)
        self.assertEqual(acc[0]["name"], "read_file")
        self.assertEqual(json.loads(acc[0]["arguments"]), {"path": "x"})

    def test_sse_comments_and_done_are_handled(self):
        class Resp:
            def __init__(self, lines):
                self.lines = iter(lines)

            def readline(self):
                return next(self.lines, b"")

            def close(self):
                pass
        lines = [b": keep-alive\n\n", b"\n",
                 b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
                 b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                 b'"usage":{"completion_tokens":2},"timings":{"predicted_per_second":9.5}}\n\n',
                 b"data: [DONE]\n\n"]
        out = bench.read_sse(Resp(lines))
        self.assertEqual(out["content"], "hi")
        self.assertEqual(out["comments"], 1)
        self.assertEqual(out["finish"], "stop")
        self.assertEqual(out["timings"]["predicted_per_second"], 9.5)
        self.assertIsNotNone(out["first_useful_s"])

    def test_abort_closes_the_stream(self):
        class Resp:
            closed = False

            def readline(self):
                time.sleep(0.01)
                return b": ping\n\n"

            def close(self):
                Resp.closed = True
        out = bench.read_sse(Resp(), abort_at=time.monotonic() + 0.05)
        self.assertTrue(out["aborted"])
        self.assertTrue(Resp.closed)

    def test_context_classes_are_deterministic_and_sized(self):
        spec = {"task": "do it"}
        short = bench.build_lane_messages(spec, "short", "A")
        big = bench.build_lane_messages(spec, "32k", "A")
        self.assertEqual(len(short), 2)
        self.assertEqual(len(big), 4)
        self.assertIn("[session A]", big[3]["content"])
        self.assertGreater(len(big[1]["content"]), 60000)
        self.assertEqual(bench.build_lane_messages(spec, "32k", "A"), big)     # the same bytes every run

    def test_slot_overlap_and_hardware_peak(self):
        rows = [{"ts": 0.0, "nonidle_slots": 1, "queued": 0, "vram_used_mib": 12000, "ram_used_gib": 20.0,
                 "gpu_util_pct": 90, "gpu_power_w": 190, "vram_total_mib": 16303},
                {"ts": 1.0, "nonidle_slots": 2, "queued": 1, "vram_used_mib": 13500, "ram_used_gib": 22.0,
                 "gpu_util_pct": 99, "gpu_power_w": 220, "vram_total_mib": 16303},
                {"ts": 3.0, "nonidle_slots": 2, "queued": 0, "vram_used_mib": 13900, "ram_used_gib": 23.0,
                 "gpu_util_pct": 99, "gpu_power_w": 230, "vram_total_mib": 16303},
                {"ts": 4.0, "nonidle_slots": 1, "queued": 0, "vram_used_mib": 12500, "ram_used_gib": 21.0,
                 "gpu_util_pct": 95, "gpu_power_w": 200, "vram_total_mib": 16303},
                {"ts": 5.0, "nonidle_slots": 1, "queued": 0, "vram_used_mib": 12400, "ram_used_gib": 20.5,
                 "gpu_util_pct": 0, "gpu_power_w": 60, "vram_total_mib": 16303}]
        # two slots were non-idle in both samples of the 1-3 s and 3-4 s intervals only, and one lane was left alone
        self.assertEqual(bench.slot_overlap(rows), {"max_nonidle_slots": 2, "slot_overlap_s": 2.0,
                                                    "solo_fallback_observed": True})
        self.assertEqual(bench.queued_seconds(rows), 2.0)          # live.queued was 1 between 1 s and 3 s
        peak = bench.hardware_peak(rows)
        self.assertEqual(peak["vram_used_mib"]["peak"], 13900)
        self.assertEqual(peak["vram_used_mib"]["n"], 5)
        self.assertEqual(peak["vram_total_mib"], 16303)

    def test_queue_delay_estimate_without_the_api_monitor(self):
        rows = [{"ts": 0.0, "queued": 0}, {"ts": 1.0, "queued": 1}, {"ts": 3.0, "queued": 0}]
        lanes = [{"lane": "A", "ttft_s": 0.4, "started_at": 1.0}, {"lane": "B", "ttft_s": 5.0, "started_at": 1.0}]
        self.assertEqual(bench.attribute_queue_delay(lanes, rows, None), "metrics_estimate")
        by = {l["lane"]: l for l in lanes}
        self.assertEqual(by["A"]["queue_delay_s"], 0.0)
        self.assertEqual(by["B"]["queue_delay_s"], 2.0)          # the seconds live.queued was >= 1
        self.assertEqual(by["B"]["model_ttft_s"], 3.0)           # TTFT 5.0 minus the queue
        self.assertEqual(by["A"]["model_ttft_s"], 0.4)

    def test_read_is_confined_to_the_checkout(self):
        self.assertIn("not a file of this repository", bench.safe_read("read_file", "../etc/passwd"))
        self.assertIn("not a file of this repository", bench.safe_read("read_file", "/etc/passwd"))
        self.assertTrue(bench.safe_read("read_file", "serve/server.py").startswith('"""serve/server.py'))

    def test_scoring_needs_every_check(self):
        spec = bench.task_specs()["server-option"]
        calls = [{"name": "read_file", "executed": True}]
        self.assertFalse(bench.score_solution(spec, [], None)["task_complete"])
        self.assertFalse(bench.score_solution(spec, calls, None)["checks"]["submitted"])
        full = bench.score_solution(spec, calls, dict(GOOD_ARGS))
        self.assertTrue(full["task_complete"], full["checks"])
        self.assertEqual(full["useful_work"], 1.0)
        partial = bench.score_solution(spec, calls, dict(GOOD_ARGS, anchor="not in the file at all"))
        self.assertFalse(partial["task_complete"])
        self.assertGreater(partial["useful_work"], 0.0)           # partial credit is reported, not counted as done


class ComparatorTests(unittest.TestCase):
    def make(self, arm="c2", ctx="short", *, work=2.0, wall=60.0, tasks=2, lane_decode=30.0, agg=60.0,
             ttft=1.0, queue=0.0, vram=13000, overlap=20.0, slots=2, complete=True, undeclared=(), errors=(),
             bad_args=0, cancel=None, repeat=1, serving=None):
        ok = bool(complete)
        return {"schema": compare.SCHEMA,
                "run": {"arm": arm, "context_class": ctx, "campaign_wall_s": wall, "repeat": repeat,
                        "started_at": 1000.0 + repeat, "lanes": 2, "tag": ""},
                "guard": {"serving_observed": serving if serving is not None else {"c1": 1}.get(arm, 2)},
                "decision_metric": {"useful_work_per_minute": work, "tasks_completed": tasks, "useful_work": tasks,
                                    "aggregate_decode_tok_s": agg, "median_per_lane_decode_tok_s": lane_decode,
                                    "last_lane_first_useful_s": ttft},
                "lanes": [{"lane": x, "task_complete": ok, "ttft_s": ttft, "queue_delay_s": queue,
                           "model_ttft_s": max(0.0, ttft - queue), "undeclared_tools": list(undeclared),
                           "errors": list(errors), "calls": [{"bad_arguments": bool(bad_args)} if x == "B" else
                           {"bad_arguments": False}]} for x in ("A", "B")],
                "server_evidence": {"hardware_peak": {"vram_used_mib": {"peak": vram}}, "slot_overlap_s": overlap,
                                    "max_nonidle_slots": slots, "queued_seconds": queue},
                "cancellation": cancel}

    def compare(self, runs, *argv) -> tuple:
        tmp = tempfile.mkdtemp()
        for i, r in enumerate(runs):
            Path(tmp, f"run{i}.json").write_text(json.dumps(r), encoding="utf-8")
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["pi_c1c2_compare.py", str(Path(tmp) / "*.json"), *argv]
        try:
            with contextlib.redirect_stdout(out):
                code = compare.main()
        finally:
            sys.argv = old
        return code, out.getvalue()

    def test_clears_every_gate_and_promotes(self):
        runs = [self.make("c1", work=1.0, lane_decode=30.0, slots=1, overlap=0.0) for _ in range(3)]
        runs += [self.make("c2", work=1.6, lane_decode=26.0, cancel={"survivor_ok": True}) for _ in range(3)]
        code, text = self.compare(runs)
        self.assertEqual(code, 0)
        self.assertIn("Recommendation: PROMOTE", text)
        self.assertIn("x1.6", text)

    def test_too_few_repetitions_defers(self):
        runs = [self.make("c1", work=1.0), self.make("c2", work=1.6, cancel={"survivor_ok": True})]
        _, text = self.compare(runs)
        self.assertIn("Recommendation: DEFER", text)
        self.assertIn("the gate wants 3", text)

    def test_undeclared_tool_rejects(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.6, undeclared=["write_file"], cancel={"survivor_ok": True})
                 for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("Recommendation: REJECT", text)
        self.assertIn("undeclared tools", text)

    def test_starved_lanes_reject(self):
        runs = [self.make("c1", work=1.0, lane_decode=30.0) for _ in range(3)]
        runs += [self.make("c2", work=1.1, lane_decode=10.0, cancel={"survivor_ok": True}) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("Recommendation: REJECT", text)
        self.assertIn("both workers got slower", text)

    def test_no_gain_defers(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.01, cancel={"survivor_ok": True}) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("Recommendation: DEFER", text)
        self.assertIn("inside the", text)

    def test_missing_cancellation_and_no_overlap_defer(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.6) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("no cancellation fixture", text)
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.6, slots=1, overlap=0.0, cancel={"survivor_ok": True}) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("never held two slots non-idle", text)

    def test_lost_survivor_rejects(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.6, cancel={"survivor_ok": False, "survivor_task_complete": False})
                 for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("Recommendation: REJECT", text)
        self.assertIn("lost the surviving lane", text)

    def test_no_c1_baseline_defers(self):
        _, text = self.compare([self.make("c2", cancel={"survivor_ok": True}) for _ in range(3)])
        self.assertIn("no C1 baseline", text)

    def test_batch_mtp_is_compared_separately(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)]
        runs += [self.make("c2", work=1.2, lane_decode=20.0, cancel={"survivor_ok": True}) for _ in range(3)]
        runs += [self.make("c2mtp", work=1.6, lane_decode=30.0, cancel={"survivor_ok": True}) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("c2mtp against c2: 1.5x per-lane decode", text)
        self.assertIn("the MTP drafts, not the batching", text)

    def test_missing_mtp_arm_is_called_out(self):
        runs = [self.make("c1", work=1.0) for _ in range(3)] + [self.make("c2", work=1.6) for _ in range(3)]
        _, text = self.compare(runs)
        self.assertIn("No `c2mtp` group", text)

    def test_foreign_files_are_skipped(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "other.json").write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
        (Path(tmp) / "run0.json").write_text(json.dumps(self.make("c2")), encoding="utf-8")
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["pi_c1c2_compare.py", str(Path(tmp) / "*.json")]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                compare.main()
        finally:
            sys.argv = old
        self.assertIn("skipped", out.getvalue())
        self.assertIn("arm/context group(s)", out.getvalue())

    def test_ttft_percentiles_are_reported(self):
        tmp = tempfile.mkdtemp()
        dest = Path(tmp) / "c.json"
        runs = [self.make("c1", work=1.0, ttft=1.0 + 0.1 * i) for i in range(3)]
        runs += [self.make("c2", work=1.6, ttft=0.4 + 0.1 * i, cancel={"survivor_ok": True}) for i in range(3)]
        _, text = self.compare(runs, "--json", str(dest))
        self.assertIn("TTFT p50 / p90 s", text)
        self.assertIn("queue delay p50 / p90 s", text)
        obj = json.loads(dest.read_text(encoding="utf-8"))
        c2 = next(g for g in obj["groups"] if g["arm"] == "c2")
        self.assertEqual(c2["ttft_p50_p90"]["n"], 6)               # two lanes over three repetitions
        self.assertLess(c2["ttft_p50_p90"]["p50"], c2["ttft_p50_p90"]["p90"])
        self.assertEqual(c2["model_ttft_p50_p90"]["n"], 6)

    def test_json_output_is_written(self):
        tmp = tempfile.mkdtemp()
        dest = Path(tmp) / "compare.json"
        runs = [self.make("c1", work=1.0) for _ in range(3)] + [self.make("c2", work=1.6) for _ in range(3)]
        self.compare(runs, "--json", str(dest))
        obj = json.loads(dest.read_text(encoding="utf-8"))
        self.assertIn(obj["recommendation"], ("PROMOTE", "DEFER", "REJECT"))
        self.assertEqual(len(obj["groups"]), 2)


def load_one(tmp: str) -> dict:
    files = sorted(Path(tmp).glob("run-*.json"))
    assert len(files) == 1, [f.name for f in files]
    return json.loads(files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
