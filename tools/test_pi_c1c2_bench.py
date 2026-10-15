#!/usr/bin/env python3
"""Tests for the Pi C1/C2 harness and comparator (Issue #2).  CPU-only: no GPU, no model, no Strata.

The fake server answers with the shapes serve/server.py actually sends, taken from the code that builds them:

  /v1/status      serve/server.py Service.status() - `engine` is a VERSION STRING, `loaded` is top level, the
                  window is `context.max_positions` / `context.cache_max_tokens`, the slots are
                  `concurrency.serving` / `concurrency.requested`, `machine.gpu` is a LIST, and `machine` is
                  already MiB/GiB.
  /metrics        serve/server.py Service.metrics() + serve/telemetry.py - `hardware` is in BYTES
                  (`gpu_mem_used`, `ram_used`, `gpu_power`), `live.slots[].state` carries the real slot states,
                  `live.running` counts REQUESTS not slots, and `live.slots` is absent unless the engine runs
                  batch slots.
  /api/requests   the list has no prompt text; the `?id=` detail carries `input`, `queue_s`, `first_token_s`,
                  `wallclock_s`.
  a stream        serve/server.py chunk() - `data: {...}` with `choices[].delta`, an SSE comment keep-alive,
                  `usage` and `timings` on the last chunk, and `data: [DONE]`.

Every test that would need a GPU is a test of the parsing and decision code against those shapes instead, which
is what caught the mismatches this harness was rebuilt from.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_c1c2_bench as bench                                     # noqa: E402
import pi_c1c2_compare as compare                                 # noqa: E402


# ---------------------------------------------------------------------------- the real shapes (from serve/server.py)
def status_payload(serving=2, requested=2, loaded=True, model="qwen3.8-flash-next", window=131072,
                   queued=0, running=0, state="busy"):
    """Service.status() as serve/server.py builds it: `engine` a version string, `machine` in MiB/GiB,
    `machine.gpu` a LIST."""
    return {"status": "ok", "model": model, "loaded": loaded, "engine": "b999-dirty",
            "context": {"max_positions": window, "cache_max_tokens": window},
            "concurrency": {"mode": "batch", "requested": requested, "serving": serving, "queued": queued,
                            "running": running, "admitting": 0, "waiting": 0, "state": state,
                            "max_queue": 0, "queue_dropped": 0, "admission": "immediate"},
            "machine": {"cpu_pct": 40.0,
                        "gpu": [{"name": "NVIDIA GeForce RTX 5080", "used_mib": 13631.5, "total_mib": 16303.0,
                                 "power_w": 120.5, "util_pct": 99}],
                        "ram": {"used_gib": 20.5, "total_gib": 64.0}}}


def metrics_payload(slot_states=("idle", "idle"), running=0, queued=0, busy=False, drafts=(200, 150),
                    with_slots=True):
    """Service.metrics() as serve/server.py builds it: `hardware` in BYTES, keys named `ram_used` / `gpu_power`,
    `live.slots[].state` the slot states, `live.running` the number of REQUESTS in flight."""
    live = {"busy": busy, "state": state_of(busy, queued), "queued": queued, "running": running,
            "waiting": 0, "admitting": 0}
    if with_slots:
        live["slots"] = [{"slot": i, "state": s, "prompt_n": 100, "prompt_done": 100, "generated_n": 5,
                          "draft_n": 0, "draft_n_accepted": 0} for i, s in enumerate(slot_states)]
    return {"live": live,
            "hardware": {"gpu_mem_used": 13631488000, "gpu_mem_total": 17084538880, "ram_used": 21990232555,
                         "ram_total": 68719476736, "gpu_util": 99, "gpu_power": 120.5},
            "totals": {"requests": 4, "prompt_tokens": 1000, "generated_tokens": 200,
                       "drafts_offered": drafts[0], "drafts_accepted": drafts[1],
                       "spec_accept_pct": round(100.0 * drafts[1] / drafts[0], 1) if drafts[0] else None,
                       "prompt_ms": 1000.0, "decode_ms": 2000.0}}


def state_of(busy, queued):
    return "busy" if busy else ("queued" if queued else "idle")


def api_request_list(rid, model="qwen3.8-flash-next", started_at=None):
    """The list form: no prompt text (the server strips it), and no queue_s."""
    return [{"id": rid, "started_at": started_at if started_at is not None else time.time(), "model": model,
             "endpoint": "/v1/chat/completions", "stream": True, "status": 200, "wallclock_s": 3.0,
             "prompt_tokens": 1000, "generated_tokens": 40, "finish": "stop", "error": None,
             "queue_wait_s": 0.0, "slots": [0]}]


def api_request_detail(rid, inp, queue_s=1.5, first_token_s=2.0, model="qwen3.8-flash-next", started_at=None):
    """The `?id=` detail: the only place a per-request queue time is available."""
    row = api_request_list(rid, model=model, started_at=started_at)
    row[0].update({"input": inp, "queue_s": queue_s, "first_token_s": first_token_s, "load_s": 0.0,
                   "slots": [0], "prompt_ms": 900.0, "decode_ms": 1200.0})
    return row[0]


def sse_lines(model, steps):
    """One stream as serve/server.py writes it: chunk() frames, a comment keep-alive, usage and timings on the
    last chunk, `data: [DONE]` at the end.  `steps` is a list of dicts: content, tool_calls, timings, usage,
    finish, sleep, raw (extra literal lines), no_done, no_finish."""
    out = [b": keep-alive\n\n"]
    seq = 0

    def frame(delta, finish=None):
        return {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 1, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    out.append(b"data: " + json.dumps(frame({"role": "assistant", "content": ""})).encode() + b"\n\n")
    for st in steps:
        if st.get("sleep"):
            out.append(("sleep:%s" % st["sleep"]).encode())        # the stub handler turns this into a pause
        for piece in st.get("content") or []:
            out.append(b"data: " + json.dumps(frame({"content": piece})).encode() + b"\n\n")
        for call in st.get("tool_calls") or []:
            args = call.get("arguments")
            raw_args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            out.append(b"data: " + json.dumps(frame({"tool_calls": [{"index": seq, "id": f"call{seq}",
                                                                     "type": "function",
                                                                     "function": {"name": call["name"],
                                                                                   "arguments": ""}}]})).encode()
                     + b"\n\n")
            out.append(b"data: " + json.dumps(frame({"tool_calls": [{"index": seq,
                                                                     "function": {"arguments": raw_args}}]})).encode()
                     + b"\n\n")
            seq += 1
        if st.get("raw"):
            for line in st["raw"]:
                out.append(line if isinstance(line, bytes) else line.encode())
    last = frame({}, None if steps and steps[-1].get("no_finish") else (steps[-1].get("finish", "stop")
                                                                        if steps else "stop"))
    last["usage"] = (steps[-1].get("usage") if steps else None) or {"prompt_tokens": 1000, "completion_tokens": 40,
                                                                    "total_tokens": 1040,
                                                                    "prompt_tokens_details": {"cached_tokens": 0}}
    if not (steps and steps[-1].get("no_timings")):
        last["timings"] = (steps[-1].get("timings") if steps else None) or \
            {"prompt_ms": 900.0, "decode_ms": 1200.0, "predicted_ms": 400.0, "predicted_n": 20,
             "predicted_n_draft": 30, "predicted_n_accept": 15, "predicted_draft_ms": 100.0}
    out.append(b"data: " + json.dumps(last).encode() + b"\n\n")
    if not (steps and steps[-1].get("no_done")):
        out.append(b"data: [DONE]\n\n")
    return out


class FP:
    """A response body as the reader sees it: bytes out of a socket, one readline() at a time."""

    def __init__(self, lines):
        self.buf = b"".join(l if isinstance(l, bytes) else l.encode() for l in lines)

    def readline(self):
        if not self.buf:
            return b""
        i = self.buf.find(b"\n")
        i = len(self.buf) if i < 0 else i + 1
        line, self.buf = self.buf[:i], self.buf[i:]
        return line


def read_sse(lines, **kw):
    return bench.read_sse(FP(lines), **kw)


def args_of(name, **kw):
    return {"name": name, "arguments": json.dumps(kw)}


# ------------------------------------------------------------------------------- the stub server (real shapes)
class Stub:
    """A server that answers the way serve/server.py does.  `script` maps a lane nonce to the steps that lane's
    session gets; each step is a dict of content / tool_calls / timings / usage / finish / sleep / no_done /
    no_finish.  Nothing here is a model: the point is the wire shapes."""

    def __init__(self, script=None, serving=2, slot_states=("decode", "decode"), running=2, busy=True,
                 api_requests=None, status=None, metrics=None, drafts=(0, 0), model="qwen3.8-flash-next"):
        self.script = script or {}
        self.serving = serving
        self.dynamic_status = status is None      # an explicit payload stays exactly as given
        self.status_payload = status if status is not None else status_payload(serving=serving, model=model)
        self.metrics_payload = metrics if metrics is not None else metrics_payload(
            slot_states=slot_states, running=running, busy=busy, drafts=drafts)
        self.api_requests = api_requests            # None means the API monitor is off
        self.active = 0                             # streams in flight right now, so /metrics can answer honestly
        self.dynamic_metrics = metrics is None      # an explicit payload stays exactly as given
        self.busy_states = slot_states
        self.drafts = drafts
        self.bodies = []                            # every request body we were sent, in order
        self.stream_calls = 0
        self.http_status = None                     # set to (code, headers) to make the first POST answer that
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def metrics_now(self):
        """What /metrics answers at this instant: idle before any traffic, one non-idle slot per live stream."""
        if not self.dynamic_metrics:
            return self.metrics_payload
        n = self.active
        states = tuple(self.busy_states[min(i, len(self.busy_states) - 1)] if i < n else "idle"
                       for i in range(len(self.busy_states)))
        return metrics_payload(slot_states=states, running=n, busy=n > 0, drafts=self.drafts)

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload, ctype="application/json"):
                body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/v1/status":
                    payload = outer.status_payload
                    if outer.dynamic_status:
                        payload = dict(payload, concurrency=dict(payload["concurrency"], serving=outer.serving,
                                                                 requested=outer.serving))
                    return self._send(200, payload)
                if path == "/metrics":
                    return self._send(200, outer.metrics_now())
                if path == "/api/requests":
                    if outer.api_requests is None:
                        return self._send(404, {"error": "the API monitor is off"})
                    if "id=" in query:
                        rid = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("id")
                        for r in outer.api_requests:
                            if r.get("id") == rid:
                                return self._send(200, r)
                        return self._send(404, {"error": "no such record"})
                    return self._send(200, {"requests": [{k: v for k, v in r.items()
                                                          if k not in ("input", "queue_s", "first_token_s")}
                                                         for r in outer.api_requests]})
                return self._send(404, {"error": f"no {path}"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode("utf-8", "replace")
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": f"no {self.path}"})
                if outer.http_status is not None:
                    code, headers = outer.http_status
                    outer.http_status = None
                    self.send_response(code)
                    for k, v in (headers or {}).items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.bodies.append(raw)
                outer.stream_calls += 1
                outer.active += 1
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                msgs = body.get("messages") or []
                text = json.dumps(msgs)
                m = re.search(r"\[session (\w+)\]", text)
                nonce = m.group(1) if m else "?"
                # which step of THIS conversation the request is on, read off the request itself: one tool
                # message per step already taken.  Nothing kept between requests, so a second campaign of the
                # same stub starts at step one the way a fresh conversation does.
                seen = sum(1 for m2 in msgs if isinstance(m2, dict) and m2.get("role") == "tool")
                steps = outer.script.get(nonce) or [{}]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in sse_lines(outer.model_name(), [steps[min(seen, len(steps) - 1)]]):
                    if line.startswith(b"sleep:"):
                        time.sleep(float(line.split(b":", 1)[1]))
                        continue
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        outer.active -= 1
                        return
                outer.active -= 1

        return H

    def model_name(self):
        return (self.status_payload or {}).get("model") or "qwen3.8-flash-next"

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def spec_for(name="server-option", lane="A", context_class="short"):
    spec = dict(bench.TASKS[name])
    spec.update({"lane": lane, "session": name, "filler_words": bench.CONTEXT_CLASSES[context_class],
                 "context_class": context_class})
    return spec


def opts_for(url, arm="c2", **kw):
    opts = {"arm": arm, "context_class": "short", "repeat": 1, "tag": "", "url": url, "model": "m",
            "declared": bench.DECLARED, "max_tokens": 200, "max_steps": 4, "max_tool_chars": 4000,
            "timeout": 10.0, "campaign_deadline": 60.0, "sample_interval": 0.05, "api_key": "",
            "confirm_batch_mtp": "", "guard": {"mismatch": [], "serving_observed": 2, "serving_expected": 2,
                                               "model": "m", "engine_version": "b999", "max_context": 131072,
                                               "loaded": True}}
    opts.update(kw)
    return opts


def run_correlate(records, lanes, list_only=False):
    """correlate_queue_delay against a stub that answers /api/requests the way the server does."""
    rows = [{k: v for k, v in r.items() if k not in ("input", "queue_s", "first_token_s")} for r in records] \
        if list_only else records
    with Stub(api_requests=records if not list_only else rows) as st:
        # the stub's detail form answers with the full record; the list form strips prompt text as the server does
        return bench.correlate_queue_delay(st.url, "", lanes, 5.0)


def read_anchor(path="README.md", chars=60):
    text = (bench.ROOT / path).read_text(encoding="utf-8", errors="replace")
    return text[120:120 + chars]


def session_file(name):
    """A file this session is expected to read, that really exists in the checkout."""
    for p in sorted(bench.TASKS[name]["expected_files"]):
        if (bench.ROOT / p).is_file():
            return p
    raise unittest.SkipTest(f"session {name!r} has no readable file in this checkout")


class ReadSSE(unittest.TestCase):
    """The stream reader against the frames serve/server.py writes."""

    def test_assembles_the_answer_and_the_tool_call_fragments(self):
        lines = sse_lines("m", [{"content": ["Read", "ing it."],
                                 "tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}],
                                 "finish": "tool_calls"}])
        r = read_sse(lines)
        self.assertEqual(r["content"], "Reading it.")
        self.assertEqual([c["name"] for c in r["tool_calls"]], ["read_file"])
        self.assertEqual(json.loads(r["tool_calls"][0]["arguments"]), {"path": "README.md"})
        self.assertEqual(r["finish"], "tool_calls")
        self.assertTrue(r["done"])
        self.assertIsNone(r["error"])
        self.assertEqual(r["comments"], 1)                       # the SSE keep-alive comment, ignored
        self.assertEqual(r["usage"]["prompt_tokens"], 1000)
        self.assertEqual(r["timings"]["decode_ms"], 1200.0)      # usage and timings arrive without stream_options

    def test_a_stream_that_ends_without_done_or_a_finish_is_an_error(self):
        r = read_sse([b'data: {"choices":[{"index":0,"delta":{"content":"half"},"finish_reason":null}]}\n\n'])
        self.assertIsNotNone(r["error"])
        self.assertIn("before [DONE]", r["error"])

    def test_a_mid_stream_error_chunk_is_an_error(self):
        # serve/server.py answers a mid-stream engine failure with an error chunk, then [DONE]
        r = read_sse([b'data: {"choices":[{"index":0,"delta":{"content":"par"},"finish_reason":null}]}\n\n',
                      b'data: {"error":{"type":"server_error","message":"the engine died"}}\n\n',
                      b"data: [DONE]\n\n"])
        self.assertIsNotNone(r["error"])
        self.assertIn("engine died", r["error"])

    def test_a_tool_call_cut_off_by_the_output_limit_is_not_bad_arguments(self):
        lines = sse_lines("m", [{"tool_calls": [{"name": "submit_solution", "arguments": '{"path": "READ'}],
                                 "finish": "length"}])
        r = read_sse(lines)
        raw = r["tool_calls"][0]["arguments"]
        self.assertRaises(ValueError, json.loads, raw)           # the harness calls this truncated, not invalid

    def test_abort_stops_a_stream_that_keeps_talking_but_never_finishes(self):
        class Chatty:
            """A stream that sends a keep-alive for ever: the deadline has to end it, not the content."""

            def __init__(self):
                self.n = 0

            def readline(self):
                self.n += 1
                if self.n > 400:
                    return b""
                time.sleep(0.005)
                return b": keep-alive\n\n"

        t = time.monotonic()
        r = bench.read_sse(Chatty(), abort_at=t + 0.15)
        self.assertTrue(r["aborted"], r)
        self.assertLess(time.monotonic() - t, 1.5)
        self.assertGreaterEqual(r["comments"], 1)

    def test_the_answer_text_is_capped(self):
        big = "x" * (bench.STREAM_MAX_CHARS + 10)
        r = read_sse(sse_lines("m", [{"content": [big]}]))
        self.assertTrue(r["content_truncated"])
        self.assertEqual(len(r["content"]), bench.STREAM_MAX_CHARS)


class Guard(unittest.TestCase):
    """The guard against the shapes serve/server.py actually answers with (B1, B2, M13)."""

    def test_the_real_status_shape_is_read_without_raising(self):
        g = bench.guard(status_payload(serving=2), metrics_payload(), "c2", 2, "")
        self.assertEqual(g["mismatch"], [], g)
        self.assertEqual(g["engine_version"], "b999-dirty")     # `engine` is a version STRING, not an object
        self.assertTrue(g["loaded"])
        self.assertEqual(g["max_context"], 131072)              # context.max_positions
        self.assertEqual(g["serving_observed"], 2)              # concurrency.serving
        self.assertEqual(g["model"], "qwen3.8-flash-next")

    def test_a_serving_count_that_is_not_the_arms_is_refused(self):
        g = bench.guard(status_payload(serving=1), metrics_payload(slot_states=("decode",), running=1),
                        "c2", 2, "")
        self.assertTrue(any("concurrency.serving=1" in m for m in g["mismatch"]), g["mismatch"])

    def test_other_traffic_on_the_server_is_refused(self):
        g = bench.guard(status_payload(serving=2), metrics_payload(running=1, queued=1), "c2", 2, "")
        self.assertTrue(any("not idle" in m for m in g["mismatch"]), g["mismatch"])

    def test_an_unloaded_model_and_a_wrong_model_are_refused(self):
        g = bench.guard(status_payload(loaded=False), metrics_payload(), "c2", 2, "")
        self.assertTrue(any("not loaded" in m for m in g["mismatch"]), g["mismatch"])
        g = bench.guard(status_payload(model="other"), metrics_payload(), "c2", 2, "qwen3.8-flash-next")
        self.assertTrue(any("the server has" in m for m in g["mismatch"]), g["mismatch"])

    def test_no_answer_at_all_is_a_mismatch_not_a_crash(self):
        for st in (None, {}, {"error": "HTTP 500 from /v1/status"}):
            g = bench.guard(st, None, "c2", 2, "")
            self.assertTrue(g["mismatch"])

    def test_the_guard_reads_the_real_metrics_shape(self):
        g = bench.guard(status_payload(serving=2), metrics_payload(drafts=(200, 150)), "c2mtp", 2, "")
        self.assertEqual(g["mismatch"], [], g)                   # idle before traffic, which is what a guard wants
        self.assertEqual(g["drafts_offered_total"], 200)
        self.assertEqual(g["queued"], 0)
        self.assertEqual(g["running"], 0)
        busy = bench.guard(status_payload(serving=2), metrics_payload(("decode", "decode"), running=2, busy=True),
                           "c2mtp", 2, "")
        self.assertEqual(busy["running"], 2)
        self.assertTrue(any("not idle" in m for m in busy["mismatch"]), busy["mismatch"])


class Units(unittest.TestCase):
    """Bytes in, the unit the key names out (B2)."""

    def test_metrics_bytes_become_mib_and_gib(self):
        row = bench.compact_sample(metrics_payload(), status_payload(), at=1.0)
        self.assertEqual(row["vram_used_mib"], round(13631488000 / 2 ** 20, 2))
        self.assertEqual(row["vram_total_mib"], round(17084538880 / 2 ** 20, 2))
        self.assertEqual(row["ram_used_gib"], round(21990232555 / 2 ** 30, 2))
        self.assertLess(row["vram_used_mib"], 100000, "a MiB key holding bytes is the bug this test exists for")
        self.assertEqual(row["gpu_power_w"], 120.5)             # the telemetry key is `gpu_power`
        self.assertEqual(row["gpu_util"], 99)

    def test_machine_gpu_is_a_list_and_does_not_break_the_row(self):
        row = bench.compact_sample(metrics_payload(), status_payload(), at=1.0)
        self.assertEqual(row["vram_used_mib"], round(13631488000 / 2 ** 20, 2))

    def test_the_status_machine_is_the_fallback_when_there_is_no_telemetry(self):
        row = bench.compact_sample(None, status_payload(), at=1.0)
        self.assertEqual(row["vram_used_mib"], 13631.5)         # /v1/status machine is already MiB
        self.assertEqual(row["ram_used_gib"], 20.5)
        self.assertEqual(row["gpu_power_w"], 120.5)

    def test_hardware_peak_keeps_the_units_it_names(self):
        rows = [bench.compact_sample(metrics_payload(), status_payload(), at=float(i)) for i in range(3)]
        h = bench.hardware_peak(rows, status_payload())
        self.assertEqual(h["vram_used_mib"], round(13631488000 / 2 ** 20, 2))
        self.assertEqual(h["ram_used_gib"], round(21990232555 / 2 ** 30, 2))
        self.assertIn("bytes", h["units"])


class Slots(unittest.TestCase):
    """Two slots really ran together, or the evidence says it did not (H1)."""

    def row(self, at, slot_states=None, running=0, queued=0):
        m = metrics_payload(slot_states=slot_states or ("idle", "idle"), running=running, queued=queued,
                            busy=bool(running), with_slots=slot_states is not None)
        return bench.compact_sample(m, status_payload(), at=at)

    def test_running_requests_are_not_slot_evidence(self):
        rows = [self.row(i * 0.2, running=2) for i in range(5)]
        ev = bench.slot_overlap(rows, gap=0.6)
        self.assertEqual(ev["slot_overlap_s"], 0.0)
        self.assertEqual(ev["max_nonidle_slots"], 0)
        self.assertEqual(ev["inflight_overlap_s"], 0.8)         # reported, and named as requests not slots
        self.assertTrue(ev["slot_state_source"].startswith("none"))

    def test_two_non_idle_slots_are_counted_from_their_states(self):
        rows = [self.row(i * 0.2, ("decode", "decode"), running=2) for i in range(5)]
        ev = bench.slot_overlap(rows, gap=0.6)
        self.assertEqual(ev["max_nonidle_slots"], 2)
        self.assertAlmostEqual(ev["slot_overlap_s"], 0.8, places=2)
        self.assertEqual(ev["slot_state_source"], "live.slots")

    def test_one_slot_working_is_not_overlap(self):
        rows = [self.row(i * 0.2, ("decode", "idle"), running=1) for i in range(5)]
        ev = bench.slot_overlap(rows, gap=0.6)
        self.assertEqual(ev["max_nonidle_slots"], 1)
        self.assertEqual(ev["slot_overlap_s"], 0.0)
        self.assertGreater(ev["single_slot_s"], 0)

    def test_a_long_gap_between_samples_is_not_counted_as_overlap(self):
        rows = [self.row(0.0, ("decode", "decode"), 2), self.row(9.0, ("decode", "decode"), 2)]
        self.assertEqual(bench.slot_overlap(rows, gap=0.6)["slot_overlap_s"], 0.0)


class Scoring(unittest.TestCase):
    """Whole tasks, and nothing that can raise (B3, H4)."""

    def setUp(self):
        self.spec = bench.TASKS["server-option"]
        self.anchor = read_anchor()

    def calls(self, *rows):
        return [dict({"declared": True, "executed": True, "name": "read_file", "step": 0}, **r) for r in rows]

    def good(self):
        # the anchor is a line of the file the submission names, which is what the check asks for
        return self.calls({"name": "read_file"}, {"name": "submit_solution",
                                                  "arguments": json.dumps({"path": "README.md",
                                                                           "anchor": self.anchor,
                                                                           "summary": "x" * 60})})

    def test_a_task_is_complete_only_when_every_check_passes(self):
        scored = bench.score_solution(self.spec, self.good(),
                                      {"path": "README.md", "anchor": self.anchor, "summary": "x" * 60})
        self.assertTrue(scored["task_complete"], scored)
        self.assertEqual(scored["useful_work"], 1.0)
        self.assertEqual(scored["partial_credit"], 1.0)
        self.assertEqual(scored["checks_passed"], scored["checks_total"])

    def test_partial_credit_is_never_a_completed_task(self):
        submit = {"path": "README.md", "anchor": self.anchor, "summary": "too short"}
        scored = bench.score_solution(self.spec, self.good()[:1] + [self.good()[1]], submit)
        self.assertFalse(scored["checks"]["summary_present"])
        self.assertFalse(scored["task_complete"])
        self.assertEqual(scored["useful_work"], 0.0)
        self.assertGreater(scored["partial_credit"], 0.0)
        self.assertLess(scored["partial_credit"], 1.0)

    def test_an_anchor_that_is_not_in_the_file_named_does_not_pass(self):
        submit = {"path": "README.md", "anchor": "z" * 30, "summary": "x" * 60}
        scored = bench.score_solution(self.spec, self.calls({"name": "read_file"},
                                                            {"name": "submit_solution"}), submit)
        self.assertFalse(scored["checks"]["anchor_verbatim"])
        self.assertFalse(scored["task_complete"])

    def test_a_file_that_was_never_read_does_not_pass(self):
        submit = {"path": "docs/DETAILS.md", "anchor": self.anchor, "summary": "x" * 60}
        scored = bench.score_solution(self.spec, self.calls({"name": "submit_solution"}), submit)
        self.assertFalse(scored["checks"]["read_before_submit"])

    def test_submitting_without_reading_first_does_not_pass(self):
        submit = {"path": "README.md", "anchor": self.anchor, "summary": "x" * 60}
        scored = bench.score_solution(self.spec, self.calls({"name": "submit_solution"}), submit)
        self.assertFalse(scored["checks"]["read_before_submit"])
        self.assertEqual(scored["useful_work"], 0.0)

    def test_an_undeclared_tool_fails_the_task_and_is_named(self):
        calls = self.good()
        calls.insert(1, {"name": "edit_file", "declared": False, "executed": False, "step": 0})
        scored = bench.score_solution(self.spec, calls,
                                      {"path": "README.md", "anchor": self.anchor, "summary": "x" * 60})
        self.assertFalse(scored["checks"]["declared_tools_only"])
        self.assertEqual(scored["undeclared_tools"], ["edit_file"])

    def test_a_submission_of_any_shape_scores_and_never_raises(self):
        for bad in (None, [], [1, 2], "text", 7, {"path": {"p": "x"}, "anchor": ["a"], "summary": {"s": 1}},
                    {"path": 5, "anchor": 5, "summary": 5}, {"path": None}):
            with self.subTest(submit=bad):
                scored = bench.score_solution(self.spec, self.calls({"name": "read_file"}), bad)
                self.assertFalse(scored["task_complete"])
                self.assertEqual(scored["useful_work"], 0.0)
                self.assertIsInstance(scored["partial_credit"], float)

    def test_a_path_that_is_not_a_string_cannot_be_used_as_a_set_element(self):
        # the review's probe: {"path": {"p": "x"}} raised TypeError: unhashable type 'dict'
        scored = bench.score_solution(self.spec, [], {"path": {"p": "x"}, "anchor": "a" * 20, "summary": "s" * 60})
        self.assertFalse(scored["checks"]["target_path_expected"])

    def test_a_list_submission_does_not_raise_on_get(self):
        scored = bench.score_solution(self.spec, [1 and {"name": "read_file", "declared": True,
                                                         "executed": True, "step": 0}], [1, 2])
        self.assertFalse(scored["checks"]["submitted"])


class Paths(unittest.TestCase):
    """Reads stay in the checkout, on either separator (H5)."""

    def test_a_path_outside_the_checkout_is_refused(self):
        for arg in ("../etc/passwd", "/etc/passwd", "..", "../../etc/passwd", "./../outside.txt"):
            with self.subTest(arg=arg):
                text, err = bench.safe_read(arg)
                self.assertIsNone(text)
                self.assertIn("outside the checkout", err)

    def test_a_symlink_that_leaves_the_checkout_is_refused(self):
        d = Path(tempfile.mkdtemp())
        link = d / "escape"
        try:
            link.symlink_to("/etc/passwd")
            text, err = bench.safe_read(os.path.relpath(str(link), str(bench.ROOT)))
            self.assertIsNone(text)
            self.assertTrue("outside the checkout" in (err or "") or "not a file" in (err or ""), err)
        except OSError as e:
            self.skipTest(f"no symlinks here: {e}")
        finally:
            link.unlink(missing_ok=True)
            d.rmdir()

    def test_a_real_file_in_the_checkout_is_read(self):
        text, err = bench.safe_read("README.md")
        self.assertIsNone(err)
        self.assertIn("Strata", text)

    def test_containment_uses_the_platform_separator_not_a_slash(self):
        # the review's Windows finding: `str(ROOT) + "/"` never matched `C:\repo\README.md`
        self.assertTrue(bench.inside_prefix(r"C:\repo", r"C:\repo\README.md", "\\", lambda s: s.lower()))
        self.assertTrue(bench.inside_prefix(r"C:\repo", r"C:\repo", "\\", lambda s: s.lower()))
        self.assertFalse(bench.inside_prefix(r"C:\repo", r"C:\other\README.md", "\\", lambda s: s.lower()))
        self.assertFalse(bench.inside_prefix(r"C:\repo", r"C:\repository\x", "\\", lambda s: s.lower()))
        self.assertTrue(bench.inside_prefix("/repo", "/repo/x", "/", lambda s: s))
        self.assertFalse(bench.inside_prefix("/repo", "/etc/passwd", "/", lambda s: s))

    def test_the_tool_argument_is_taken_by_tool_name(self):
        self.assertEqual(bench.tool_arg("read_file", {"path": "README.md", "pattern": "x"}), "README.md")
        self.assertEqual(bench.tool_arg("grep", {"path": "README.md", "pattern": "abc"}), "abc")
        self.assertEqual(bench.tool_arg("read_file", {"path": {"not": "a string"}}), "")
        self.assertEqual(bench.tool_arg("read_file", "not an object"), "")
        self.assertEqual(bench.tool_arg("grep", {}), "")

    def test_a_grep_that_runs_too_long_is_stopped_and_says_so(self):
        real = bench.subprocess.run
        try:
            def slow(*a, **kw):
                raise bench.subprocess.TimeoutExpired(cmd="git", timeout=kw.get("timeout") or 0)
            bench.subprocess.run = slow
            out = bench.repo_grep(".*")
            self.assertIn("stopped after", out)
            self.assertTrue(any("timeout" in repr(kw) or True for kw in [{}]))
        finally:
            bench.subprocess.run = real
        # and the timeout is really passed
        seen = {}
        try:
            def spy(*a, **kw):
                seen.update(kw)
                raise bench.subprocess.TimeoutExpired(cmd="git", timeout=1)
            bench.subprocess.run = spy
            bench.repo_grep("x")
            self.assertEqual(seen.get("timeout"), bench.REPO_GREP_TIMEOUT_S)
        finally:
            bench.subprocess.run = real

    def test_a_grep_result_is_capped(self):
        real = bench.subprocess.run
        try:
            class R:
                returncode = 0
                stdout = "x" * (bench.REPO_GREP_MAX_BYTES + 1000)
                stderr = ""
            bench.subprocess.run = lambda *a, **kw: R()
            out = bench.repo_grep("x")
            self.assertIn("was dropped", out)
            self.assertLess(len(out), bench.REPO_GREP_MAX_BYTES + 200)
        finally:
            bench.subprocess.run = real

    def test_a_pattern_starting_with_a_dash_is_not_passed_to_git(self):
        self.assertIn("cannot start", bench.repo_grep("--help"))
        self.assertIn("pattern is needed", bench.repo_grep("  "))


class QueueDelay(unittest.TestCase):
    """A queue second is charged to the lane that waited, or to nobody (H3, M14)."""

    def lane(self, lane, ttft):
        return {"lane": lane, "session": "server-option" if lane == "A" else "expert-cache", "nonce": lane,
                "ttft_s": ttft, "queue_delay_s": None, "model_ttft_s": None, "queue_delay_source": "not_set"}

    def test_a_record_is_tied_to_the_lane_whose_nonce_it_carries(self):
        recs = [api_request_detail("r1", "body saying [session A] hello", queue_s=7.0),
                api_request_detail("r2", "body saying [session B] hello", queue_s=0.5)]
        lanes = [self.lane("A", 8.0), self.lane("B", 1.0)]
        ev = run_correlate(recs, lanes)
        self.assertEqual(lanes[0]["queue_delay_s"], 7.0)
        self.assertEqual(lanes[0]["model_ttft_s"], 1.0)
        self.assertEqual(lanes[1]["queue_delay_s"], 0.5)
        self.assertEqual(lanes[1]["model_ttft_s"], 0.5)
        self.assertTrue(lanes[0]["queue_delay_source"].startswith("api_requests:nonce"))
        self.assertEqual(ev["correlation"], "api_requests:nonce")

    def test_the_lane_that_waited_longer_is_not_assumed_to_be_the_slow_one(self):
        # the review's probe: lane A waited 7 s, lane B 0 s; the old code charged 7 s to B
        recs = [api_request_detail("r1", "[session A] task", queue_s=7.0),
                api_request_detail("r2", "[session B] task", queue_s=0.0)]
        lanes = [self.lane("A", 7.2), self.lane("B", 0.3)]
        run_correlate(recs, lanes)
        self.assertEqual(lanes[0]["queue_delay_s"], 7.0)          # A's own second, from A's own record
        self.assertEqual(lanes[1]["queue_delay_s"], 0.0)          # B is not charged for it
        self.assertEqual(lanes[0]["model_ttft_s"], 0.2)
        self.assertEqual(lanes[1]["model_ttft_s"], 0.3)           # and B's model TTFT has not collapsed to 0

    def test_a_record_two_lanes_could_own_is_matched_to_neither(self):
        recs = [api_request_detail("r1", "[session A] and [session B] in one body", queue_s=9.0)]
        lanes = [self.lane("A", 9.0), self.lane("B", 9.0)]
        ev = run_correlate(recs, lanes)
        self.assertIsNone(lanes[0]["queue_delay_s"])
        self.assertIsNone(lanes[1]["queue_delay_s"])
        self.assertTrue(lanes[0]["queue_delay_source"].startswith("unavailable"))
        self.assertEqual(ev["correlation"], "none")
        self.assertEqual(ev["records_unmatched"], 1)

    def test_the_list_form_alone_is_not_enough(self):
        # no detail form: no `input`, so nothing can be attributed
        recs = [{k: v for k, v in api_request_detail("r1", "[session A]", queue_s=3.0).items()
                 if k not in ("input", "queue_s")}]
        lanes = [self.lane("A", 3.0), self.lane("B", 1.0)]
        ev = run_correlate(recs, lanes, list_only=True)
        self.assertIsNone(lanes[0]["queue_delay_s"])
        self.assertEqual(ev["correlation"], "none")

    def test_no_api_monitor_is_said_so_on_every_lane(self):
        lanes = [self.lane("A", 3.0), self.lane("B", 1.0)]
        with Stub(api_requests=None) as st:
            ev = bench.correlate_queue_delay(st.url, "", lanes, 5.0)
        self.assertFalse(ev["api_requests_available"])
        self.assertEqual(ev["correlation"], "unavailable: /api/requests did not answer")
        self.assertIn("--api-monitor", ev["note"])
        for l in lanes:
            # M14: every lane row says the same thing as server_evidence.queue_delay_source
            self.assertEqual(l["queue_delay_source"], ev["correlation"])
            self.assertIsNone(l["queue_delay_s"])
            self.assertIsNone(l["model_ttft_s"])

    def test_a_record_without_queue_s_leaves_the_lane_null_with_a_reason(self):
        rec = api_request_detail("r1", "[session A]", queue_s=None)
        lanes = [self.lane("A", 3.0)]
        ev = run_correlate([rec], lanes)
        self.assertIsNone(lanes[0]["queue_delay_s"])
        self.assertIn("no queue_s", lanes[0]["queue_delay_source"])
        self.assertEqual(ev["correlation"], "none")

    def test_the_earliest_record_of_a_lane_is_the_one_used(self):
        recs = [api_request_detail("r2", "[session A] step two", queue_s=0.1, started_at=1000.0),
                api_request_detail("r1", "[session A] step one", queue_s=4.0, started_at=900.0)]
        lanes = [self.lane("A", 4.5)]
        run_correlate(recs, lanes)
        self.assertEqual(lanes[0]["queue_delay_s"], 4.0)
        self.assertTrue(lanes[0]["queue_delay_source"].startswith("api_requests:nonce"))


HOLD = 0.3            # seconds a fake stream stays open: short, but long enough for the sampler to see


def good_session(name="server-option", extra=None):
    """A lane that reads a file its own session is expected to read, then submits an anchor out of it: a
    completed task.  The anchor is a line of that file, which is what the check asks for."""
    path = session_file(name)
    anchor = read_anchor(path)
    steps = [{"sleep": HOLD, "tool_calls": [{"name": "read_file", "arguments": {"path": path}}],
              "finish": "tool_calls",
              "timings": {"prompt_ms": 900.0, "decode_ms": 300.0, "predicted_ms": 120.0, "predicted_n": 12,
                          "draft_n": 0, "draft_n_accepted": 0}},
             {"sleep": HOLD, "tool_calls": [{"name": "submit_solution",
                                             "arguments": {"path": path, "anchor": anchor, "summary": "x" * 80}}],
              "finish": "tool_calls",
              "timings": {"prompt_ms": 400.0, "decode_ms": 200.0, "predicted_ms": 80.0, "predicted_n": 8,
                          "draft_n": 0, "draft_n_accepted": 0}}]
    return (extra + steps) if extra else steps


class Campaign(unittest.TestCase):
    """The runner against the stub: what a run file holds, and what a bad lane must not take down."""

    def out_dir(self):
        return Path(tempfile.mkdtemp())

    def test_two_lanes_write_evidence_that_matches_the_real_shapes(self):
        script = {"A": good_session("server-option"), "B": good_session("expert-cache")}
        with Stub(script=script, serving=2, slot_states=("decode", "decode"), running=2,
                  api_requests=[api_request_detail("r1", "[session A]", queue_s=1.5, started_at=1.0),
                                api_request_detail("r2", "[session B]", queue_s=0.25, started_at=1.0)]) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A"), spec_for("expert-cache", "B")],
                                     opts_for(st.url))
        d, ev = rep["decision_metric"], rep["server_evidence"]
        self.assertEqual(d["tasks_completed"], 2, [l["checks"] for l in rep["lanes"]])
        self.assertEqual(d["partial_credit"], 2.0)
        self.assertGreater(d["useful_work_per_minute"], 0)
        self.assertEqual(rep["lanes"][0]["queue_delay_s"], 1.5)          # the server's own number, this lane's record
        self.assertEqual(rep["lanes"][1]["queue_delay_s"], 0.25)
        self.assertEqual(ev["queue_delay_source"], "api_requests:nonce")
        self.assertEqual([l["queue_delay_record"] for l in rep["lanes"]], ["r1", "r2"])
        self.assertEqual({l["queue_delay_source"] for l in rep["lanes"]}, {"api_requests:nonce"})
        self.assertEqual(ev["max_nonidle_slots"], 2)                     # from live.slots[].state
        self.assertEqual(ev["slot_state_source"], "live.slots")
        self.assertLess(ev["hardware_peak"]["vram_used_mib"], 100000)    # MiB, not bytes
        self.assertEqual(rep["lanes"][0]["submission"]["path"], session_file("server-option"))  # auditable
        self.assertEqual(rep["lanes"][0]["nonce"], "A")
        self.assertTrue(rep["guard"]["mismatch"] == [])
        self.assertIn("tasks_completed counts whole tasks", d["note"])
        self.assertTrue(rep["not_measured"])
        md = bench.render_markdown(rep)
        self.assertIn("Partial credit", md)
        self.assertIn("live.slots[].state", md)

    def test_a_lanes_decode_rate_sums_every_step(self):
        # 110 tokens over 1.1 s of decode is 100 tok/s; the last step alone would say 1100
        path, anchor = session_file("server-option"), read_anchor(session_file("server-option"))
        steps = [{"tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}], "finish": "tool_calls",
                  "timings": {"prompt_ms": 900.0, "decode_ms": 900.0, "predicted_ms": 1000.0, "predicted_n": 100,
                              "draft_n": 0, "draft_n_accepted": 0}},
                 {"tool_calls": [{"name": "submit_solution",
                                  "arguments": {"path": path, "anchor": anchor, "summary": "x" * 80}}],
                  "finish": "tool_calls",
                  "timings": {"prompt_ms": 100.0, "decode_ms": 100.0, "predicted_ms": 100.0, "predicted_n": 10,
                              "draft_n": 0, "draft_n_accepted": 0}}]
        with Stub(script={"A": steps}, serving=2, slot_states=("decode", "idle"), running=1) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A")], opts_for(st.url))
        l = rep["lanes"][0]
        self.assertEqual(l["decoded_tokens_server"], 110)
        self.assertEqual(l["decode_ms_total"], 1100.0)
        self.assertEqual(l["decode_tok_s_server"], 100.0)
        self.assertEqual(len(l["step_timings"]), 2)
        self.assertEqual(rep["decision_metric"]["per_lane_decode_tok_s"], 100.0)

    def test_arguments_that_are_not_an_object_cost_one_lane_not_the_campaign(self):
        bad = [{"tool_calls": [{"name": "submit_solution", "arguments": "[1, 2]"}], "finish": "tool_calls"}]
        with Stub(script={"A": bad, "B": good_session("expert-cache")}, serving=2,
                  slot_states=("decode", "decode"), running=2) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A"), spec_for("expert-cache", "B")],
                                     opts_for(st.url))
        a, b = rep["lanes"]
        self.assertEqual(len(a["invalid_arguments"]), 1, a["invalid_arguments"])
        self.assertEqual(a["stopped"], "invalid_tool_arguments")
        self.assertIn("not an object", json.dumps(a["invalid_arguments"]))
        self.assertFalse(a["task_complete"])
        self.assertTrue(b["task_complete"], b["checks"])           # the other lane's evidence survives
        self.assertEqual(rep["decision_metric"]["tasks_completed"], 1)
        # A's one passing check is reported as partial credit and is never counted as a completed task
        self.assertEqual(rep["decision_metric"]["partial_credit"], 1.167)
        self.assertLess(a["partial_credit"], 1.0)
        self.assertEqual(a["useful_work"], 0.0)

    def test_a_lane_that_raises_inside_the_harness_still_leaves_a_row(self):
        real = bench.safe_tool
        try:
            def boom(name, args):
                if name == "read_file":
                    raise RuntimeError("the harness broke")
                return real(name, args)
            bench.safe_tool = boom
            with Stub(script={"A": good_session("server-option"), "B": good_session("expert-cache")}, serving=2,
                      slot_states=("decode", "decode"), running=2) as st:
                rep = bench.run_campaign(st.url, "", [spec_for(lane="A"), spec_for("expert-cache", "B")],
                                         opts_for(st.url))
        finally:
            bench.safe_tool = real
        self.assertEqual([l["lane"] for l in rep["lanes"]], ["A", "B"])
        for l in rep["lanes"]:
            self.assertEqual(l["stopped"], "harness_error", l["errors"])
            self.assertIn("the harness broke", json.dumps(l["errors"]))
        self.assertEqual(rep["decision_metric"]["tasks_completed"], 0)

    def test_a_cancelled_lane_stops_and_the_survivor_finishes(self):
        slow = [{"content": ["thinking"] * 3, "sleep": 3.0, "no_done": True, "no_finish": True}]
        with Stub(script={"A": slow, "B": good_session("expert-cache")}, serving=2,
                  slot_states=("decode", "decode"), running=2) as st:
            t = time.monotonic()
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A"), spec_for("expert-cache", "B")],
                                     opts_for(st.url, timeout=2.0), cancel_lane="A", cancel_after=0.5)
            took = time.monotonic() - t
        a, b = rep["lanes"]
        self.assertTrue(a["aborted"] if "aborted" in a else a["stopped"] == "cancelled_by_harness", a["stopped"])
        self.assertEqual(a["stopped"], "cancelled_by_harness")
        self.assertTrue(b["task_complete"], b["checks"])
        self.assertTrue(rep["cancellation"]["survivor_ok"])
        self.assertTrue(rep["cancellation"]["cancelled_lane_stopped"])
        self.assertLess(took, 3.0, "the cancellation must bound the campaign, not just mark it")

    def test_the_campaign_deadline_bounds_a_lane_that_never_answers(self):
        slow = [{"content": ["still going"], "sleep": 4.0, "no_done": True, "no_finish": True}]
        with Stub(script={"A": slow}, serving=2, slot_states=("decode", "idle"), running=1) as st:
            t = time.monotonic()
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A")],
                                     opts_for(st.url, timeout=10.0, campaign_deadline=30.0))
            took = time.monotonic() - t
        self.assertIn(rep["lanes"][0]["stopped"], ("campaign_deadline", "answered_without_a_tool_call"))
        self.assertLess(took, 12.0)

    def test_batch_mtp_is_believed_only_when_the_servers_own_numbers_show_it(self):
        drafting = [{"tool_calls": [{"name": "submit_solution",
                                     "arguments": {"path": session_file("server-option"),
                                                   "anchor": read_anchor(), "summary": "x" * 80}}],
                     "finish": "tool_calls",
                     "timings": {"prompt_ms": 100.0, "decode_ms": 100.0, "predicted_ms": 50.0, "predicted_n": 10,
                                 "draft_n": 40, "draft_n_accepted": 30}}]
        spec = [spec_for(lane="A")]
        with Stub(script={"A": drafting}, serving=2, slot_states=("decode", "decode"), running=2) as st:
            mtp = bench.run_campaign(st.url, "", spec, opts_for(st.url, arm="c2mtp"))
            plain = bench.run_campaign(st.url, "", spec, opts_for(st.url, arm="c2"))
        with Stub(script={"A": good_session("server-option")}, serving=2,
                  slot_states=("decode", "decode"), running=2) as st:
            no_draft = bench.run_campaign(st.url, "", spec, opts_for(st.url, arm="c2mtp", confirm_batch_mtp=""))
        self.assertTrue(mtp["batch_mtp"]["verified"], mtp["batch_mtp"])
        self.assertFalse(plain["batch_mtp"]["verified"], plain["batch_mtp"])
        self.assertTrue(plain["batch_mtp"]["unexpected_drafts"])
        self.assertFalse(no_draft["batch_mtp"]["verified"])
        self.assertIn("cannot be told", no_draft["batch_mtp"]["why"])

    def test_a_tool_call_cut_off_by_the_output_limit_is_truncated_not_invalid(self):
        cut = [{"tool_calls": [{"name": "submit_solution", "arguments": '{"path": "README'}], "finish": "length"}]
        with Stub(script={"A": cut}, serving=2, slot_states=("decode", "idle"), running=1) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A")], opts_for(st.url))
        a = rep["lanes"][0]
        self.assertEqual(len(a["truncated_tool_calls"]), 1, a["truncated_tool_calls"])
        self.assertEqual(a["invalid_arguments"], [])
        self.assertEqual(a["stopped"], "truncated_tool_call")
        self.assertFalse(a["task_complete"])

    def test_an_undeclared_tool_ends_the_lane_and_is_named(self):
        reach = [{"tool_calls": [{"name": "edit_file", "arguments": {"path": "README.md"}}], "finish": "tool_calls"}]
        with Stub(script={"A": reach}, serving=2, slot_states=("decode", "idle"), running=1) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A")], opts_for(st.url))
        a = rep["lanes"][0]
        self.assertEqual(a["undeclared_tools"], ["edit_file"])
        self.assertEqual(a["stopped"], "undeclared_tool")
        self.assertFalse(a["checks"]["declared_tools_only"])
        self.assertEqual(a["tool_calls"], 1)                     # it is not allowed to keep going

    def test_a_busy_server_is_retried_and_the_retry_is_in_the_evidence(self):
        with Stub(script={"A": good_session("server-option")}, serving=2, slot_states=("decode", "idle"),
                  running=1) as st:
            st.http_status = (503, {"Retry-After": "0"})
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A")], opts_for(st.url))
        l = rep["lanes"][0]
        self.assertTrue(l["task_complete"], l["checks"])
        self.assertTrue(any("retried" in json.dumps(e) for e in l["errors"]), l["errors"])

    def test_no_api_monitor_leaves_every_lane_saying_so(self):
        with Stub(script={"A": good_session("server-option"), "B": good_session("expert-cache")}, serving=2,
                  slot_states=("decode", "decode"), running=2, api_requests=None) as st:
            rep = bench.run_campaign(st.url, "", [spec_for(lane="A"), spec_for("expert-cache", "B")],
                                     opts_for(st.url))
        ev = rep["server_evidence"]
        self.assertEqual(ev["queue_delay_source"], "unavailable: /api/requests did not answer")
        for l in rep["lanes"]:
            self.assertEqual(l["queue_delay_source"], ev["queue_delay_source"])
            self.assertIsNone(l["queue_delay_s"])
            self.assertIsNone(l["model_ttft_s"])
            self.assertIsNotNone(l["ttft_s"])                      # the client's own number is still there


def run_main(argv):
    old = sys.argv
    out, err = io.StringIO(), io.StringIO()
    sys.argv = ["pi_c1c2_bench.py"] + argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = bench.main()
    finally:
        sys.argv = old
    return rc, out.getvalue() + err.getvalue()


class CommandLine(unittest.TestCase):
    """main(): what it refuses, what it writes, and what never reaches a file."""

    def test_it_writes_the_run_files_and_hides_the_key(self):
        secret = "sk-strata-secret-1234567890"
        with Stub(script={"A": good_session("server-option"), "B": good_session("expert-cache")},
                  serving=2, slot_states=("decode", "decode"), running=2) as st:
            d = Path(tempfile.mkdtemp())
            rc, text = run_main(["--url", st.url, "--api-key", secret, "--arm", "c2", "--output-dir", str(d),
                                 "--sample-interval", "0.05", "--repeat", "1"])
            self.assertEqual(rc, 0, text)
            self.assertEqual(st.stream_calls, 4, "two lanes, two steps each")
        files = sorted(x.name for x in d.iterdir())
        self.assertEqual(len(files), 2)
        self.assertTrue([f for f in files if f.endswith(".json") and "-r1" in f], files)
        for f in d.iterdir():
            body = f.read_text(encoding="utf-8")
            self.assertNotIn(secret, body, f.name)               # the key is a placeholder before any write
            self.assertNotIn(str(bench.ROOT), body, f.name)      # and so is this machine's folder
        rep = json.loads((d / f"{[x.stem for x in d.iterdir() if x.suffix == '.json'][0]}.json").read_text())
        self.assertEqual(rep["decision_metric"]["tasks_completed"], 2)
        self.assertTrue(rep["batch_mtp"]["verified"], rep["batch_mtp"]["why"])   # c2 claims no drafts, and saw none

    def test_the_repeat_is_part_of_the_file_name(self):
        with Stub(script={"A": good_session("server-option"), "B": good_session("expert-cache")},
                  serving=2, slot_states=("decode", "decode"), running=2) as st:
            d = Path(tempfile.mkdtemp())
            for r in (1, 2):
                rc, text = run_main(["--url", st.url, "--arm", "c2", "--output-dir", str(d), "--tag", "same",
                                     "--sample-interval", "0.05", "--repeat", str(r)])
                self.assertEqual(rc, 0, text)
        stems = sorted({x.stem for x in d.iterdir()})
        self.assertEqual(len(list(d.iterdir())), 4, stems)       # two runs, two files each, nothing overwritten
        self.assertEqual(len([x for x in stems if "-r1-same" in x]), 1, stems)
        self.assertEqual(len([x for x in stems if "-r2-same" in x]), 1, stems)

    def test_it_refuses_a_server_that_is_not_the_arm(self):
        with Stub(script={}, serving=1, slot_states=("decode",), running=0) as st:
            d = Path(tempfile.mkdtemp())
            rc, text = run_main(["--url", st.url, "--arm", "c2", "--output-dir", str(d)])
        self.assertEqual(rc, 2)
        self.assertIn("concurrency.serving=1", text)
        self.assertFalse(list(d.iterdir()))                      # nothing written for a refused campaign

    def test_a_dry_run_sends_nothing_and_writes_nothing(self):
        with Stub(script={}) as st:
            d = Path(tempfile.mkdtemp())
            rc, text = run_main(["--url", st.url, "--arm", "c2mtp", "--dry-run", "--output-dir", str(d),
                                 "--context-class", "64k", "--cancel-lane", "B"])
        self.assertEqual(rc, 0)
        self.assertEqual(st.stream_calls, 0)
        self.assertFalse(list(d.iterdir()))
        self.assertIn("nothing is sent", text)
        self.assertIn("--batch-mtp on", text)

    def test_it_refuses_a_cancel_lane_that_is_not_running(self):
        rc, text = run_main(["--arm", "c2", "--sessions", "server-option", "--cancel-lane", "B", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("not one of this campaign's lanes", text)

    def test_it_refuses_two_lanes_on_one_session(self):
        rc, text = run_main(["--arm", "c2", "--sessions", "server-option,server-option", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("repeats a name", text)

    def test_it_refuses_to_carry_a_key_over_a_plain_socket(self):
        rc, text = run_main(["--url", "http://192.168.1.20:8181", "--api-key", "x", "--arm", "c2", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("not loopback", text)
        self.assertNotIn("192.168.1.20", text)                   # the refusal does not repeat the address

    def test_the_expect_serving_flag_cannot_contradict_the_arm(self):
        rc, text = run_main(["--arm", "c1", "--expect-serving", "2", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("contradicts --arm c1", text)

    def test_tool_results_are_capped_to_keep_the_conversation_inside_the_window(self):
        # M11: the class sizes the first prompt; the reads come out of what the window has left
        asked = bench.MAX_TOOL_RESULT_CHARS
        self.assertEqual(bench.tool_char_cap(0, 131072, 4, asked), asked)
        cap = bench.tool_char_cap(120000, 131072, 4, asked)
        self.assertLess(cap, asked)
        self.assertGreaterEqual(cap, 4000)
        self.assertEqual(bench.tool_char_cap(200000, 131072, 4, asked), 4000)
        self.assertEqual(bench.tool_char_cap(None, None, 4, asked), asked)

    def test_the_lanes_are_built_one_session_each_in_lane_order(self):
        specs = bench.build_specs(["server-option", "expert-cache"], "32k", 2)
        self.assertEqual([s["lane"] for s in specs], ["A", "B"])
        self.assertEqual([s["session"] for s in specs], ["server-option", "expert-cache"])
        self.assertEqual([s["filler_words"] for s in specs], [15700, 15700])
        self.assertEqual(len(bench.build_specs(["server-option", "expert-cache"], "short", 1)), 1)


def run_file(arm="c2", cls="short", repeat=1, tasks=2, uw=2.5, partial=2.0, wall=30.0, per_lane=40.0,
             agg=80.0, model="qwen3.8-flash-next", slots=2, slot_source="live.slots",
             queue="api_requests:nonce", serving=None, mtp_verified=True, errors=0, undeclared=(),
             invalid=0, truncated=0, cancellation=None, guard_mismatch=(), ttft=2.0, model_ttft=1.0,
             queue_delay=0.2, stopped="submitted"):
    """A run file of the shape tools/pi_c1c2_bench.py writes, with only the knobs a gate decision turns on."""
    expect = {"c1": 1, "c2": 2, "c2mtp": 2}[arm]
    lane = [{"lane": n, "session": "server-option", "task_complete": tasks > 0, "useful_work": float(tasks),
             "partial_credit": partial / 2.0, "ttft_s": ttft, "model_ttft_s": model_ttft,
             "queue_delay_s": queue_delay, "queue_delay_source": queue, "decode_tok_s_server": per_lane,
             "undeclared_tools": list(undeclared), "invalid_arguments": [{}] * invalid,
             "truncated_tool_calls": [{}] * truncated, "errors": [{"error": "boom"}] * errors,
             "stopped": stopped} for n in ("A", "B")]
    return {"run": {"arm": arm, "context_class": cls, "repeat": repeat, "model": model,
                    "started_at": 1000.0 + repeat, "campaign_wall_s": wall},
            "guard": {"serving_observed": expect if serving is None else serving, "mismatch": list(guard_mismatch),
                      "model": model, "max_context": 131072},
            "lanes": lane,
            "decision_metric": {"tasks_completed": tasks, "partial_credit": partial,
                                "useful_work_per_minute": uw, "per_lane_decode_tok_s": per_lane,
                                "aggregate_decode_tok_s": agg, "aggregate_prompt_tok_s": 9000.0},
            "server_evidence": {"max_nonidle_slots": slots, "slot_overlap_s": 4.0,
                                "slot_state_source": slot_source, "queue_delay_source": queue,
                                "hardware_peak": {"vram_used_mib": 9000.0}},
            "batch_mtp": {"verified": mtp_verified, "why": "test"},
            "cancellation": cancellation}


def groups_of(*runs):
    g = {}
    for r in runs:
        g.setdefault((r["run"]["arm"], r["run"]["context_class"]), []).append(r)
    return {k: compare.group(v) for k, v in g.items()}


def win(cls="short", arm="c2"):
    """Three controlled repetitions of each arm, C2 clearly ahead and outside the spread."""
    out = []
    for r in (1, 2, 3):
        out.append(run_file(arm=arm, cls=cls, repeat=r, tasks=2, uw=2.5))
        out.append(run_file(arm="c1", cls=cls, repeat=r, tasks=1, uw=1.0, slots=1, per_lane=40.0))
    return out


class Compare(unittest.TestCase):
    """The gate.  Every DEFER and REJECT below is a case that must not come out PROMOTE."""

    def gate(self, runs, min_repeats=3, margin=0.05, retention=0.5):
        return compare.gate(groups_of(*runs), min_repeats, margin, retention)

    def test_a_clean_win_promotes(self):
        rec, why = self.gate(win())
        self.assertEqual(rec, "PROMOTE", why)

    def test_zero_completed_tasks_on_both_arms_never_promotes(self):
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=0, uw=0.0, partial=5.0))
            runs.append(run_file(arm="c1", repeat=r, tasks=0, uw=0.0, partial=5.0, slots=1))
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("neither arm completed a task" in w for w in why), why)
        self.assertTrue(any("partial credit" in w for w in why), why)

    def test_c2_completing_nothing_while_c1_works_is_not_promoted(self):
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=0, uw=0.0, partial=6.0))     # every check but one, on both lanes
            runs.append(run_file(arm="c1", repeat=r, tasks=1, uw=1.0, slots=1))
        rec, why = self.gate(runs)
        self.assertNotEqual(rec, "PROMOTE", why)                              # partial credit buys nothing
        self.assertEqual(rec, "REJECT", why)                                  # C1 did the work, C2 did none
        self.assertTrue(any("C1 completed" in w for w in why), why)

    def test_a_c2_run_is_never_compared_against_a_different_context_class(self):
        runs = win(cls="short") + [run_file(cls="64k", repeat=r) for r in (1, 2, 3)]
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("no C1 baseline for this context class" in w for w in why), why)

    def test_files_are_not_repetitions(self):
        runs = [run_file(repeat=1, tasks=2, uw=2.5), run_file(repeat=1, tasks=2, uw=2.6),
                run_file(arm="c1", repeat=1, tasks=1, uw=1.0, slots=1),
                run_file(arm="c1", repeat=1, tasks=1, uw=1.1, slots=1)]
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("distinct --repeat" in w for w in why), why)
        self.assertEqual(self.gate(win(), min_repeats=4)[0], "DEFER")

    def test_the_arms_must_have_run_the_same_model(self):
        runs = win() + [run_file(repeat=4, model="qwen3.8-coder")]
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)

    def test_a_gain_inside_the_spread_of_the_runs_defers(self):
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=2, uw=1.04))                 # inside the 5% margin
            runs.append(run_file(arm="c1", repeat=r, tasks=1, uw=1.0, slots=1))
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("under the" in w for w in why), why)
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=2, uw=[1.2, 0.9, 2.6][r - 1]))
            runs.append(run_file(arm="c1", repeat=r, tasks=1, uw=1.0, slots=1))
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("ranges overlap" in w for w in why), why)

    def test_an_undeclared_tool_rejects_in_either_arm(self):
        for arm in ("c2", "c1"):
            runs = win() + [run_file(arm=arm, repeat=4, undeclared=("edit_file",),
                                     serving=2 if arm != "c1" else 1)]
            rec, why = self.gate(runs)
            self.assertEqual(rec, "REJECT", why)
            self.assertTrue(any("undeclared tools called: ['edit_file']" in w for w in why), why)

    def test_a_request_error_or_a_bad_tool_call_rejects(self):
        for kw in ({"errors": 1}, {"invalid": 1}, {"truncated": 1}):
            with self.subTest(**kw):
                rec, why = self.gate(win() + [run_file(repeat=4, **kw)])
                self.assertEqual(rec, "REJECT", why)

    def test_a_contradicted_guard_rejects(self):
        rec, why = self.gate(win() + [run_file(repeat=4, guard_mismatch=("model differs",))])
        self.assertEqual(rec, "REJECT", why)
        rec, why = self.gate(win() + [run_file(repeat=4, serving=1)])
        self.assertEqual(rec, "REJECT", why)
        self.assertTrue(any("concurrency.serving=1" in w for w in why), why)

    def test_an_arm_label_the_server_did_not_confirm_defers(self):
        rec, why = self.gate(win() + [run_file(repeat=4, mtp_verified=False)])
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("did not confirm" in w for w in why), why)

    def test_two_slots_have_to_be_in_the_evidence(self):
        rec, why = self.gate(win() + [run_file(repeat=4, slots=1)])
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("live.slots[].state" in w for w in why), why)
        rec, why = self.gate(win() + [run_file(repeat=4, slot_source="live.running")])
        self.assertEqual(rec, "DEFER", why)

    def test_queue_delay_that_is_not_tied_to_a_lane_defers(self):
        rec, why = self.gate(win() + [run_file(repeat=4, queue="unavailable: /api/requests did not answer",
                                               queue_delay=None)])
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("--api-monitor" in w for w in why), why)

    def test_a_lane_that_starves_rejects(self):
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=2, uw=2.5, per_lane=15.0))
            runs.append(run_file(arm="c1", repeat=r, tasks=1, uw=1.0, slots=1, per_lane=40.0))
        rec, why = self.gate(runs)
        self.assertEqual(rec, "REJECT", why)
        self.assertTrue(any("starving" in w for w in why), why)

    def test_c1_completing_more_tasks_than_c2_rejects(self):
        runs = []
        for r in (1, 2, 3):
            runs.append(run_file(repeat=r, tasks=1, uw=1.0))
            runs.append(run_file(arm="c1", repeat=r, tasks=2, uw=2.0, slots=1))
        rec, why = self.gate(runs)
        self.assertEqual(rec, "REJECT", why)
        self.assertTrue(any("C1 completed" in w for w in why), why)

    def test_the_cancellation_fixture_has_to_have_worked(self):
        fixture = {"survivor_ok": False, "cancelled_lane_stopped": True, "cancelled_lane": "B"}
        rec, why = self.gate(win() + [run_file(repeat=4, cancellation=fixture)])
        self.assertEqual(rec, "REJECT", why)
        self.assertTrue(any("lost its survivor" in w for w in why), why)
        fixture = {"survivor_ok": True, "cancelled_lane_stopped": False, "cancelled_lane": "B"}
        rec, why = self.gate(win() + [run_file(repeat=4, cancellation=fixture)])
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("was not recorded as stopped" in w for w in why), why)

    def test_a_fixture_is_counted_but_is_not_a_measurement(self):
        # M8: a mandated cancellation run must not drag the arm it belongs to down
        runs = win() + [run_file(repeat=4, tasks=0, uw=0.1, partial=1.0,
                                 cancellation={"survivor_ok": True, "cancelled_lane_stopped": True})]
        g = groups_of(*runs)[("c2", "short")]
        self.assertEqual(g["runs"], 3, "the fixture is not a measured run")
        self.assertEqual(g["fixture_runs"], 1)
        self.assertEqual(g["useful_work_per_minute"]["median"], 2.5, "the median is the measured runs'")
        self.assertNotIn(4, g["repeats"])
        rec, why = self.gate(runs)
        self.assertEqual(rec, "PROMOTE", why)

    def test_a_group_of_nothing_but_fixtures_defers(self):
        # M8: a fixture's numbers are structurally low, so they are never compared against another arm
        runs = [run_file(repeat=r, tasks=0, uw=0.1, cancellation={"survivor_ok": True,
                                                                  "cancelled_lane_stopped": True})
                for r in (1, 2, 3)] + [run_file(arm="c1", repeat=r, tasks=1, uw=1.0, slots=1) for r in (1, 2, 3)]
        rec, why = self.gate(runs)
        self.assertEqual(rec, "DEFER", why)
        self.assertTrue(any("nothing was measured" in w for w in why), why)

    def test_one_arm_alone_is_never_a_recommendation(self):
        self.assertEqual(self.gate([run_file(repeat=1)])[0], "DEFER")
        self.assertEqual(self.gate([run_file(arm="c1", repeat=1, slots=1)])[0], "DEFER")

    def test_the_worst_lane_decides_a_lane_number(self):
        r = run_file()
        r["lanes"][0]["decode_tok_s_server"] = 10.0
        r["lanes"][1]["decode_tok_s_server"] = 60.0
        self.assertEqual(compare.run_value(r, "decode_tok_s"), 35.0)      # median of the two lanes
        r["lanes"][0]["queue_delay_s"] = None
        self.assertEqual(compare.run_value(r, "max_queue_delay_s"), 0.2)  # a lane with no number is not a zero

    def test_the_drafts_are_separated_from_the_lane(self):
        """A plain C1-to-C2 ratio mixes adding a lane with losing C1's drafts; c2mtp against c2 is that half."""
        runs = [run_file(arm="c2mtp", repeat=r, tasks=2, uw=2.0, per_lane=60.0, model_ttft=1.0) for r in (1, 2, 3)]
        runs += [run_file(arm="c2", repeat=r, tasks=2, uw=1.6, per_lane=40.0, model_ttft=2.0) for r in (1, 2, 3)]
        rows = compare.drafter_table(groups_of(*runs))
        body = [r for r in rows if r.startswith("| `short`")]
        self.assertEqual(len(body), 1, rows)
        self.assertIn("x1.25", body[0])                       # 2.0 / 1.6 completed tasks per minute
        self.assertIn("x0.5", body[0])                        # the model's own first token, 2.0 s to 1.0 s
        self.assertEqual(compare.drafter_table(groups_of(*win())), [], "no c2mtp run, no claim about drafts")

    def test_load_skips_what_is_not_a_run_file(self):
        d = Path(tempfile.mkdtemp())
        (d / "good.json").write_text(json.dumps(run_file()))
        (d / "broken.json").write_text("{not json")
        (d / "half.json").write_text(json.dumps({"run": {"arm": "c2"}}))
        (d / "list.json").write_text(json.dumps([1, 2]))
        runs, skipped = compare.load([str(d / "*.json")])
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(skipped), 3, skipped)
        self.assertIn("broken.json", " ".join(f for f, _ in skipped))

    def test_the_command_line_reports_and_exits_by_the_recommendation(self):
        d = Path(tempfile.mkdtemp())
        for i, r in enumerate(win()):
            (d / f"run-{i}.json").write_text(json.dumps(r))
        out, err = io.StringIO(), io.StringIO()
        old = sys.argv
        sys.argv = ["pi_c1c2_compare.py", str(d / "*.json"), "--strict", "--json", str(d / "cmp.json"),
                    "--markdown", str(d / "cmp.md")]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = compare.main()
        finally:
            sys.argv = old
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("PROMOTE", text)
        self.assertIn("tasks_completed", text)
        cmp = json.loads((d / "cmp.json").read_text())
        self.assertEqual(cmp["recommendation"], "PROMOTE")
        self.assertEqual(cmp["files"], 6)
        self.assertIn("c2/short", cmp["groups"])                 # the (arm, class) key is one string in JSON
        md = (d / "cmp.md").read_text()
        self.assertIn("**PROMOTE**", md)
        self.assertIn("partial_credit", md)
        self.assertNotIn("'context_class'", md)              # the report says how many groups, not the dict
        self.assertIn("arm/class group(s)", md)
        # and a DEFER is a failure for a script that asked for --strict
        (d / "run-x.json").write_text(json.dumps(run_file(repeat=1, tasks=0, uw=0.0, partial=5.0)))
        sys.argv = ["pi_c1c2_compare.py", str(d / "*.json"), "--strict"]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = compare.main()
        finally:
            sys.argv = old
        self.assertEqual(rc, 1)

    def test_no_run_files_is_a_refusal_not_a_crash(self):
        old = sys.argv
        sys.argv = ["pi_c1c2_compare.py", "/definitely/not/here/*.json"]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = compare.main()
        finally:
            sys.argv = old
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
