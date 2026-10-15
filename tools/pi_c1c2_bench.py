"""Issue #2 Phase 1: two synchronized Pi coding-agent lanes against one running Strata server (C1/C2 evidence).

    python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2 --run            # dry-run without --run
    python tools/pi_c1c2_bench.py --arm c1 --expect-serving 1 --context-class 32k --run
    python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2 --cancel-lane B --cancel-after 20 --run

Two independent coding-agent sessions run at the same instant against one server.  Each lane gets a real task from
`agent_turns_bench.SESSIONS` and the four-step loop a coding agent runs: read a file, search, propose, submit.  The
harness executes only read-only tools (`read_file`, `grep` return this checkout's real bytes); a lane finishes by
calling `submit_solution`, which is scored by fixed checks - the target path is one the task is about, the anchor the
lane quotes is verbatim in that file, the file was read before submitting, every tool called was declared.  Nothing is
written to disk by a lane, and no model judgement is used in the score.

What is separated, because Issue #2 asks for it and they are different quantities:

  queue delay  how long a request waits before the engine starts it.  The server's own per-request `queue_s` from
               /api/requests when the API monitor is enabled; otherwise estimated from /metrics `live.queued`
               sampled at --sample-interval and given to the lane with the larger client TTFT.
  TTFT         the client's: request sent -> first useful streamed token (content, reasoning or a tool-call
               argument).  Under C1 the queued lane's TTFT contains its queue delay; `model_ttft_s` is TTFT minus
               the queue delay, which is what the engine itself took.
  throughput   the server's own decode rate per lane (the `timings` of the last streamed chunk) and the aggregate
               over lanes, both reported, because a C2 run can raise the aggregate while lowering each lane.
  useful work  tasks whose checks all passed, per wall-clock minute of the campaign - the decision metric.

C1 and C2 are **startup configuration**: `parallel` becomes `--batch` on the engine's command line and the server
reads the slot count back once at engine start (`serve/server.py`, `INFO batch_slots=`).  There is no runtime
resize, so an arm change is a server restart and "C1 -> C2 transition latency" is not measurable inside one server.
What this harness measures instead, inside one C2 server: how long a lane waits for a slot (`admission_s`), whether
two slots were ever non-idle together (`slot_overlap_s`), and how the surviving lane behaves when a lane ends or is
cancelled.  Batch-MTP is a third arm (`c2mtp`) for the same reason: batch slots decode without MTP drafts unless
`--batch-mtp` is on, so C1-vs-C2 alone conflates batching with losing speculation.

Dry-run by default: without `--run` it prints the plan and sends nothing, not even /v1/status.  It never starts,
stops or reconfigures a server - restarting between arms is `tools/ab_engine.py`'s job and needs a GPU window.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_engine import long_prompt                      # noqa: E402  the same deterministic filler as every other suite
from agent_turns_bench import SESSIONS, SYSTEM, tool_result  # noqa: E402  the same tasks, system prompt, real file reads
from strata_bench import redact_obj, redactor          # noqa: E402  the same placeholders before anything is written

SCHEMA = "strata-pi-c1c2-v1"
# arm -> (serving slots the guard expects, batch-MTP expected).  The arm names the engine's start-up configuration.
ARMS = {"c1": {"serving": 1, "batch_mtp": False},
        "c2": {"serving": 2, "batch_mtp": False},
        "c2mtp": {"serving": 2, "batch_mtp": True}}
# Filler words chosen to land near the token target (about 3.2 characters per token, about 6.7 characters per filler
# word).  The label is a class, not a measurement: the server's own `prompt_tokens` is what the evidence records.
CONTEXT_CLASSES = {"short": 0, "32k": 15700, "64k": 31400, "118k": 56600}
SUMMARY_MIN_CHARS = 40
ANCHOR_MIN_CHARS = 12

PI_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "path from the repository root"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "submit_solution",
        "description": "Finish the task: name the file you would change, quote one line of it verbatim as the anchor "
                       "of your edit, and summarize the change and how to test it.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "anchor": {"type": "string"},
                                                        "summary": {"type": "string"}},
                       "required": ["path", "anchor", "summary"]}}},
]
DECLARED = {t["function"]["name"] for t in PI_TOOLS}
NUDGE = "Call a tool or finish with submit_solution. Do not answer in prose without a tool call."


def task_specs() -> dict:
    """{name: {task, expected_files, required_reads}} from the shared agent-turn sessions (no second copy of them)."""
    out = {}
    for name, task, steps in SESSIONS:
        reads = [arg for tool, arg, _ in steps if tool == "read_file"]
        out[name] = {"task": task, "expected_files": set(reads), "required_reads": set(reads[:1])}
    return out


def build_lane_messages(spec: dict, context_class: str, nonce: str) -> list:
    """The lane's first request: a background block sized to the context class, then the task.  The nonce marks the
    lane in the prompt so a run's requests can be told apart in the server's own records."""
    msgs = [{"role": "system", "content": SYSTEM}]
    words = CONTEXT_CLASSES[context_class]
    if words:
        msgs.append({"role": "user", "content": f"[session {nonce}] Background from earlier in this session:\n"
                                                f"{long_prompt(words)}"})
        msgs.append({"role": "assistant", "content": "Understood. I have the background."})
    msgs.append({"role": "user", "content": f"[session {nonce}] {spec['task']}"})
    return msgs


def merge_tool_calls(deltas: list, acc: dict) -> dict:
    """Fold OpenAI `delta.tool_calls` fragments into {index: {id, name, arguments}}; calls arrive piece by piece."""
    for d in deltas or []:
        i = d.get("index", 0)
        slot = acc.setdefault(i, {"id": None, "name": "", "arguments": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["name"] += fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]
    return acc


def read_sse(resp, abort_at=None, t0=None) -> dict:
    """Read one streamed chat completion.  Returns the text, the tool calls, the server's own usage/timings and the
    client's own first-event / first-useful-token times.  Closes the socket at `abort_at` (the cancellation fixture)."""
    t0 = t0 if t0 is not None else time.monotonic()
    out = {"content": "", "reasoning": "", "calls": {}, "usage": None, "timings": None, "finish": None,
           "events": 0, "comments": 0, "first_event_s": None, "first_useful_s": None, "error": None,
           "aborted": False}
    while True:
        if abort_at is not None and time.monotonic() >= abort_at:
            out["aborted"] = True
            resp.close()
            return out
        line = resp.readline()
        if not line:
            return out
        text = line.decode("utf-8", "replace").strip()
        if not text:
            continue
        if text.startswith(":"):                        # an SSE comment (keep-alive): ignored by clients
            out["comments"] += 1
            if out["first_event_s"] is None:
                out["first_event_s"] = round(time.monotonic() - t0, 3)
            continue
        if not text.startswith("data:"):
            continue
        payload = text[5:].strip()
        if payload == "[DONE]":
            return out
        try:
            chunk = json.loads(payload)
        except ValueError:
            out["error"] = f"stream carried something that is not JSON: {payload[:120]}"
            return out
        out["events"] += 1
        if chunk.get("error"):
            out["error"] = str(chunk["error"])[:300]
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        if chunk.get("timings"):
            out["timings"] = chunk["timings"]
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
            calls = delta.get("tool_calls") or []
            if not out["first_useful_s"] and (piece or think or calls):
                out["first_useful_s"] = round(time.monotonic() - t0, 3)
            out["content"] += piece
            out["reasoning"] += think
            if calls:
                merge_tool_calls(calls, out["calls"])


def call_stream(base: str, key: str, body: dict, timeout: float, abort_at=None) -> dict:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=data, headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = read_sse(resp, abort_at=abort_at, t0=t0)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:                               # noqa: BLE001
            pass
        return {"error": f"HTTP {e.code}: {detail}", "content": "", "reasoning": "", "calls": {}, "usage": None,
                "timings": None, "finish": None, "events": 0, "comments": 0, "first_event_s": None,
                "first_useful_s": None, "aborted": False}
    except Exception as e:                              # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "content": "", "reasoning": "", "calls": {}, "usage": None,
                "timings": None, "finish": None, "events": 0, "comments": 0, "first_event_s": None,
                "first_useful_s": None, "aborted": False}
    out["wall_s"] = round(time.monotonic() - t0, 3)
    return out


def get_json(url: str, key: str, timeout: float = 5.0):
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:                                   # noqa: BLE001
        return None


def serving_of(status: dict):
    return ((status or {}).get("concurrency") or {}).get("serving")


def guard(base: str, key: str, arm: dict, expect_serving: int, expect_model: str = "") -> tuple:
    """Refuse to measure the wrong arm: the server must report the serving count the caller declared."""
    st = get_json(f"{base}/v1/status", key)
    if st is None:
        return False, f"no /v1/status answer from {base}: is the server up and the API key right?"
    got = serving_of(st)
    if got != expect_serving:
        return False, (f"{base} reports concurrency.serving={got!r}, this run is labelled {arm['serving']}-slot "
                       f"and expects --expect-serving {expect_serving}. Restart the server on the arm's configuration "
                       f"(tools/ab_engine.py) before measuring; no traffic was sent.")
    model = (st.get("model") or "")
    if expect_model and expect_model not in model:
        return False, f"{base} serves {model!r}, not the expected model {expect_model!r}; no traffic was sent."
    return True, {"serving": got, "model": model, "loaded": (st.get("engine") or {}).get("loaded"),
                  "max_context": (st.get("engine") or {}).get("max_context"),
                  "vram": st.get("vram"), "machine": st.get("machine"),
                  "activity": st.get("activity")}


# ------------------------------------------------------------------------------------------------- lane execution
def safe_read(tool: str, arg: str) -> str:
    """Execute one read-only tool.  `read_file` is confined to this checkout; nothing here writes or runs anything."""
    if tool == "read_file":
        target = (ROOT / arg).resolve()
        if not str(target).startswith(str(ROOT) + "/") or not target.is_file():
            return "error: that path is not a file of this repository"
        return tool_result("read_file", arg)
    if tool == "grep":
        return tool_result("grep", arg)
    return "error: that tool is not available"


def score_solution(spec: dict, calls: list, submit: dict | None) -> dict:
    """Fixed checks, no model judgement: a task counts as useful work only when every one of them passes."""
    checks = {
        "declared_tools_only": bool(calls) and all(c["name"] in DECLARED for c in calls),
        "read_before_submit": any(c["name"] in ("read_file", "grep") and c.get("executed") for c in calls),
        "target_path_expected": bool(submit) and submit.get("path") in spec["expected_files"],
        "anchor_verbatim": False,
        "summary_present": bool(submit) and len(str(submit.get("summary") or "").strip()) >= SUMMARY_MIN_CHARS,
        "submitted": bool(submit),
    }
    if submit and isinstance(submit.get("anchor"), str) and len(submit["anchor"].strip()) >= ANCHOR_MIN_CHARS:
        target = (ROOT / str(submit.get("path") or "")).resolve()
        if str(target).startswith(str(ROOT) + "/") and target.is_file():
            try:
                checks["anchor_verbatim"] = submit["anchor"] in target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                checks["anchor_verbatim"] = False
    passed = sum(1 for v in checks.values() if v)
    return {"checks": checks, "checks_passed": passed, "checks_total": len(checks),
            "task_complete": passed == len(checks), "useful_work": 1.0 if passed == len(checks) else
            round(passed / len(checks), 3),
            "undeclared_tools": sorted({c["name"] for c in calls if c["name"] not in DECLARED})}


def run_lane(lane: str, spec: dict, base: str, key: str, args, barrier, samples) -> dict:
    """One coding-agent session: task, then read/search steps the harness answers, until submit_solution or the step
    budget.  Returns the lane's record: client timings, the server's timings, and the score."""
    msgs = build_lane_messages(spec, args.context_class, lane)
    calls, submit, steps, errors = [], None, [], []
    reads = 0
    result = {"lane": lane, "session": spec["name"], "nonce": lane, "ok": False, "steps": 0,
              "stopped": None, "tool_calls": 0, "generated_tokens": 0, "prompt_tokens": 0,
              "ttft_s": None, "first_event_s": None, "model_ttft_s": None, "queue_delay_s": None,
              "queue_delay_source": None, "admission_s": None, "decode_tok_s_server": None,
              "client_tok_s": None, "prompt_tok_s_server": None, "draft_n": None, "draft_n_accepted": None,
              "finish": None, "server_timings": None, "started_at": None, "ended_at": None, "wall_s": 0.0}
    lane_start = None
    for step in range(args.max_steps):
        body = {"messages": msgs, "tools": PI_TOOLS, "max_tokens": args.max_tokens_per_step, "temperature": 0.0,
                "seed": args.seed, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
        if barrier is not None and step == 0:
            try:
                barrier.wait(timeout=args.start_timeout)
            except threading.BrokenBarrierError:
                errors.append("the other lane never reached the start line")
                break
        if lane_start is None:
            lane_start = time.monotonic()
            result["started_at"] = time.time()
            result["admission_s"] = 0.0                 # replaced from the samples once the campaign is over
        abort_at = None
        if args.cancel_lane == lane and args.cancel_after > 0:
            abort_at = lane_start + args.cancel_after
        r = call_stream(base, key, body, args.timeout, abort_at=abort_at)
        result["wall_s"] = round(result["wall_s"] + float(r.get("wall_s") or 0.0), 3)
        if result["ttft_s"] is None and r.get("first_useful_s") is not None:
            result["ttft_s"] = r["first_useful_s"]
            result["first_event_s"] = r.get("first_event_s")
        if r.get("usage"):
            result["generated_tokens"] += int(r["usage"].get("completion_tokens") or 0)
            result["prompt_tokens"] = int(r["usage"].get("prompt_tokens") or result["prompt_tokens"])
        if r.get("timings"):
            result["server_timings"] = r["timings"]
        if r.get("error"):
            errors.append(str(r["error"])[:300])
        if r.get("aborted"):
            result["finish"] = "client_aborted"
            steps.append({"step": step, "aborted": True, "ttft_s": r.get("first_useful_s")})
            break
        result["finish"] = r.get("finish") or result["finish"]
        got = [dict(v, index=i) for i, v in sorted(r["calls"].items())]
        for c in got:
            result["tool_calls"] += 1
            c["arguments_raw"] = (c.get("arguments") or "")[:400]
            try:
                c["args"] = json.loads(c.get("arguments") or "{}")
            except ValueError:
                c["args"], c["bad_arguments"] = {}, True
            calls.append(c)
        step_row = {"step": step, "ttft_s": r.get("first_useful_s"), "finish": r.get("finish"),
                    "events": r.get("events"), "comments": r.get("comments"),
                    "tools": [c["name"] for c in got], "error": r.get("error")}
        steps.append(step_row)
        result["steps"] = step + 1
        if any(c["name"] not in DECLARED for c in got):
            result["stopped"] = "undeclared_tool"
            break
        submit_call = next((c for c in got if c["name"] == "submit_solution"), None)
        if submit_call:
            submit = submit_call.get("args") or {}
            result["stopped"] = "submitted"
            break
        answered = [c for c in got if c["name"] in ("read_file", "grep") and not c.get("bad_arguments")]
        if not answered:
            if not got and not result.get("nudged"):
                result["nudged"] = True
                msgs.append({"role": "assistant", "content": (r.get("content") or "")[:2000] or "I have nothing."})
                msgs.append({"role": "user", "content": NUDGE})
                continue
            result["stopped"] = "no_tool_call" if not got else "bad_tool_arguments"
            break
        msgs.append({"role": "assistant", "content": r.get("content") or "",
                     "tool_calls": [{"id": c["id"] or f"call_{step}_{c['index']}", "type": "function",
                                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                    for c in answered]})
        for c in answered:
            arg = (c.get("args") or {}).get("path") or (c.get("args") or {}).get("pattern") or ""
            text = safe_read(c["name"], arg)
            c["executed"], reads = True, reads + 1
            msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{step}_{c['index']}",
                         "content": text[:args.max_tool_result_chars]})
    else:
        result["stopped"] = "step_budget"
    result["ended_at"] = time.time()
    result["reads"] = reads
    result["calls"] = [{"name": c["name"], "executed": bool(c.get("executed")),
                        "bad_arguments": bool(c.get("bad_arguments"))} for c in calls]
    result.update(score_solution(spec, calls, submit))
    t = result["server_timings"] or {}
    result["decode_tok_s_server"] = t.get("predicted_per_second")
    result["prompt_tok_s_server"] = t.get("prompt_per_second")
    result["draft_n"], result["draft_n_accepted"] = t.get("draft_n"), t.get("draft_n_accepted")
    if result["wall_s"] > 0 and result["generated_tokens"]:
        # the client's own rate over the whole lane (prefill and gaps included) - a different quantity from the
        # server's decode-only rate, which is why both are kept
        result["client_tok_s"] = round(result["generated_tokens"] / result["wall_s"], 1)
    result["ok"] = bool(result["task_complete"]) and not errors
    result["errors"] = errors
    result["steps_detail"] = steps
    return result


# ------------------------------------------------------------------------------------------- server-side sampling
class Sampler(threading.Thread):
    """One thread, both endpoints, --sample-interval apart: the server's own /metrics and /v1/status.  VRAM comes from
    /metrics `hardware.gpu_mem_used`, falling back to /v1/status `machine.gpu.used_mib` - no nvidia-smi subprocess."""

    def __init__(self, base: str, key: str, interval: float):
        super().__init__(daemon=True)
        self.base, self.key, self.interval = base, key, max(0.05, interval)
        self.samples, self.stop, self.failures = [], threading.Event(), 0
        self._lock = threading.Lock()

    def run(self):
        while not self.stop.is_set():
            m = get_json(f"{self.base}/metrics?requests=all", self.key)
            s = get_json(f"{self.base}/v1/status", self.key)
            if m is not None or s is not None:
                row = compact_sample(m, s)
                with self._lock:
                    self.samples.append(row)
            else:
                self.failures += 1
            self.stop.wait(self.interval)
        return

    def rows(self) -> list:
        with self._lock:
            return list(self.samples)


def compact_sample(metrics: dict | None, status: dict | None) -> dict:
    now = time.monotonic()
    live = (metrics or {}).get("live") or {}
    slots = live.get("slots") or []
    hw = (metrics or {}).get("hardware") or {}
    gpu = ((status or {}).get("machine") or {}).get("gpu") or {}
    ram = ((status or {}).get("machine") or {}).get("ram") or {}
    nonidle = sum(1 for x in slots if (x.get("state") or "idle") != "idle")
    return {"ts": round(now, 3), "state": live.get("state"), "queued": int(live.get("queued") or 0),
            "running": int(live.get("running") or 0) if live.get("running") is not None else None,
            "parallel": live.get("parallel"), "nonidle_slots": nonidle,
            "slot_states": [x.get("state") for x in slots] or None,
            "slot_tok_s": [x.get("tok_s") for x in slots] or None,
            "tok_s": live.get("tok_s"), "prompt_read": live.get("prompt_read"),
            "prompt_total": live.get("prompt_total"),
            "vram_used_mib": hw.get("gpu_mem_used") if hw.get("gpu_mem_used") is not None else gpu.get("used_mib"),
            "vram_total_mib": hw.get("gpu_mem_total") if hw.get("gpu_mem_total") is not None else gpu.get("total_mib"),
            "gpu_util_pct": hw.get("gpu_util") if hw.get("gpu_util") is not None else gpu.get("util_pct"),
            "gpu_power_w": hw.get("gpu_power_w") if hw.get("gpu_power_w") is not None else gpu.get("power_w"),
            "ram_used_gib": hw.get("ram_used_gib") if hw.get("ram_used_gib") is not None else ram.get("used_gib")}


def nonidle_at(sample: dict) -> int:
    if sample.get("nonidle_slots"):
        return int(sample["nonidle_slots"])
    return int(sample.get("running") or 0)


def slot_overlap(samples: list) -> dict:
    """Seconds during which two slots were non-idle at once - the evidence that a C2 run really ran two lanes."""
    peak = max([nonidle_at(s) for s in samples], default=0)
    overlap = 0.0
    solo_after_pair = False
    saw_pair = False
    for a, b in zip(samples, samples[1:]):
        dt = max(0.0, b["ts"] - a["ts"])
        lo, hi = min(nonidle_at(a), nonidle_at(b)), max(nonidle_at(a), nonidle_at(b))
        if lo >= 2:                                     # counted only when both samples saw two slots
            overlap += dt
            saw_pair = True
        elif saw_pair and hi <= 1:
            solo_after_pair = True
    return {"max_nonidle_slots": peak, "slot_overlap_s": round(overlap, 2),
            "solo_fallback_observed": solo_after_pair}


def queued_seconds(samples: list) -> float:
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        if int(a.get("queued") or 0) >= 1:
            total += max(0.0, b["ts"] - a["ts"])
    return round(total, 2)


def hardware_peak(samples: list) -> dict:
    out = {}
    for key, label in (("vram_used_mib", "vram_used_mib"), ("ram_used_gib", "ram_used_gib"),
                       ("gpu_util_pct", "gpu_util_pct"), ("gpu_power_w", "gpu_power_w")):
        vals = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        if vals:
            out[label] = {"peak": round(max(vals), 2), "median": round(statistics.median(vals), 2),
                          "n": len(vals)}
    for s in reversed(samples):
        if s.get("vram_total_mib"):
            out["vram_total_mib"] = s["vram_total_mib"]
            break
    return out


def attribute_queue_delay(lanes: list, samples: list, records: list | None) -> str:
    """Fill in each lane's queue delay and subtract it from its TTFT.

    The server's own per-request `queue_s` (from /api/requests, which needs the API monitor) when it is available,
    otherwise the campaign's queued seconds from /metrics `live.queued`.  The server does not echo a client-side
    request id, so a record is tied to a lane by waiting rather than by identity: the longest queue of the campaign
    goes to the lane with the largest client TTFT, the others to the shortest.  The raw records are kept in the run
    file, so this attribution can be checked by hand.
    """
    live = [l for l in lanes if l.get("ttft_s") is not None]
    waits = [float(r["queue_s"]) for r in (records or []) if isinstance(r.get("queue_s"), (int, float))]
    if waits:
        low, high = min(waits), max(waits)
        slowest = max(live, key=lambda l: l["ttft_s"]) if len(live) > 1 else None
        for lane in live:
            lane["queue_delay_s"] = round(high if lane is slowest or slowest is None else low, 3)
            lane["queue_delay_source"] = "api_requests"
    else:
        waited = queued_seconds(samples)
        source = "metrics_estimate" if waited > 0 else "none_observed"
        for lane in live:
            lane["queue_delay_s"] = 0.0
            lane["queue_delay_source"] = source
        if waited > 0 and len(live) > 1:
            max(live, key=lambda l: l["ttft_s"])["queue_delay_s"] = waited
    for lane in live:
        lane["model_ttft_s"] = round(max(0.0, lane["ttft_s"] - lane["queue_delay_s"]), 3)
    return "api_requests" if waits else ("metrics_estimate" if any(l["queue_delay_s"] for l in live) else
                                        "none_observed")


# ---------------------------------------------------------------------------------------------------- the campaign
def run_campaign(args, specs: list) -> dict:
    base = args.url.rstrip("/")
    arm = ARMS[args.arm]
    started = time.time()
    ok, info = guard(base, args.api_key, arm, args.expect_serving, args.expect_model)
    if not ok:
        raise SystemExit(f"[pi_c1c2] refusing to measure: {info}")
    sampler = Sampler(base, args.api_key, args.sample_interval)
    sampler.start()
    time.sleep(max(0.3, args.sample_interval * 2))      # a couple of samples of the server at rest
    barrier = threading.Barrier(len(specs), timeout=args.start_timeout) if len(specs) > 1 else None
    lanes, out = [], {}

    def go(spec):
        out[spec["lane"]] = run_lane(spec["lane"], spec, base, args.api_key, args, barrier, sampler)

    threads = [threading.Thread(target=go, args=(s,)) for s in specs]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = round(time.monotonic() - t0, 2)
    time.sleep(args.sample_interval * 2)
    sampler.stop.set()
    sampler.join(timeout=5)
    samples = sampler.rows()

    lanes = [out[s["lane"]] for s in specs]
    records = None
    req = get_json(f"{base}/api/requests", args.api_key)
    if isinstance(req, dict) and isinstance(req.get("requests"), list):
        records = [r for r in req["requests"] if isinstance(r, dict) and r.get("started_at", 0) >= started - 2]
    queue_source = attribute_queue_delay(lanes, samples, records)

    metrics_after = get_json(f"{base}/metrics?requests=all", args.api_key) or {}
    status_after = get_json(f"{base}/v1/status", args.api_key) or {}
    completed = sum(1 for l in lanes if l["task_complete"])
    useful = round(sum(float(l["useful_work"] or 0) for l in lanes), 3)
    decode_rates = [l["decode_tok_s_server"] for l in lanes if l.get("decode_tok_s_server")]
    ttfts = [l["ttft_s"] for l in lanes if l.get("ttft_s") is not None]
    gen = sum(int(l["generated_tokens"] or 0) for l in lanes)
    decode_s = sum(((l["server_timings"] or {}).get("predicted_ms") or 0) / 1000.0 for l in lanes)
    hit_rates = [r.get("hit_rate") for r in (metrics_after.get("requests") or [])[:8]
                 if isinstance(r.get("hit_rate"), (int, float))]
    report = {
        "schema": SCHEMA,
        "run": {"arm": args.arm, "expected_serving": args.expect_serving, "batch_mtp_expected": arm["batch_mtp"],
                "context_class": args.context_class, "repeat": args.repeat, "tag": args.tag,
                "started_at": started, "campaign_wall_s": wall, "dry_run": False,
                "sessions": [s["session"] for s in specs], "lanes": len(specs),
                "max_steps": args.max_steps, "max_tokens_per_step": args.max_tokens_per_step,
                "cancel_lane": args.cancel_lane, "cancel_after_s": args.cancel_after,
                "sample_interval_s": args.sample_interval, "seed": args.seed},
        "guard": {"serving_observed": serving_of(status_after) if status_after else info["serving"],
                  "model": info["model"], "engine_loaded": info["loaded"],
                  "max_context": info["max_context"], "expect_model": args.expect_model or None},
        "decision_metric": {
            "tasks_completed": completed, "useful_work": useful,
            "useful_work_per_minute": round(useful / (wall / 60.0), 3) if wall > 0 else None,
            "two_worker_wall_time_s": wall,
            "last_lane_first_useful_s": round(max(ttfts), 3) if ttfts else None,
            "aggregate_decode_tok_s": round(gen / decode_s, 1) if decode_s > 0 else None,
            "per_lane_decode_tok_s": [round(x, 1) for x in decode_rates],
            "median_per_lane_decode_tok_s": round(statistics.median(decode_rates), 1) if decode_rates else None,
            "generated_tokens": gen},
        "lanes": lanes,
        "server_evidence": {
            "queue_delay_source": queue_source, "queued_seconds": queued_seconds(samples),
            "api_requests_available": bool(records), "api_requests": (records or [])[:8],
            "samples": len(samples), "sampler_failures": sampler.failures,
            **slot_overlap(samples), "hardware_peak": hardware_peak(samples),
            "expert_hit_rates_recent": hit_rates,
            "engine": metrics_after.get("engine"), "totals": metrics_after.get("totals"),
            "conversation_cache": metrics_after.get("conversation_cache"),
            "requests_recent": [{k: r.get(k) for k in ("prompt_tokens", "reused", "prompt_read", "decode_tok_s",
                                                       "hit_rate", "pcie_share", "drafts_offered", "drafts_accepted",
                                                       "duration_s", "finish")}
                                for r in (metrics_after.get("requests") or [])[:8]],
            "vram": status_after.get("vram"), "machine": status_after.get("machine")},
        "cancellation": None,
        "not_measured": ["before/after READY and C1/C2 peak VRAM (needs a restart per arm: tools/ab_engine.py)",
                         "C1<->C2 transition latency (parallel is fixed at engine start; no runtime resize)",
                         "batch-MTP lane/row association (needs the engine-side gate, see the doc)",
                         "expert-cache displacement attribution (needs the same-memory-budget C1 arm)"],
    }
    if args.cancel_lane:
        victim = next((l for l in lanes if l["lane"] == args.cancel_lane), None)
        survivor = next((l for l in lanes if l["lane"] != args.cancel_lane), None)
        report["cancellation"] = {
            "cancelled_lane": args.cancel_lane, "after_s": args.cancel_after,
            "victim_finish": (victim or {}).get("finish"), "victim_aborted": bool((victim or {}).get("finish") ==
                                                                                  "client_aborted"),
            "survivor_lane": (survivor or {}).get("lane"), "survivor_ok": bool((survivor or {}).get("ok")),
            "survivor_task_complete": bool((survivor or {}).get("task_complete")),
            "survivor_decode_tok_s": (survivor or {}).get("decode_tok_s_server"),
            "server_still_serving": bool(status_after)}
    return report


def render_markdown(r: dict) -> str:
    red = r  # already redacted by the caller
    d, s = red["decision_metric"], red["server_evidence"]
    L = [f"# Pi C1/C2 lane run - arm `{red['run']['arm']}`, context `{red['run']['context_class']}`",
         "",
         f"Schema `{red['schema']}`.  Guard: server reported `concurrency.serving="
         f"{red['guard']['serving_observed']}` before any traffic.  Model `{red['guard']['model']}`, "
         f"max_context {red['guard']['max_context']}.  Repeat {red['run']['repeat']}"
         f"{(' (' + red['run']['tag'] + ')') if red['run']['tag'] else ''}.",
         "",
         "## Decision metric",
         "",
         "| quantity | value |",
         "|---|---|",
         f"| campaign wall (both lanes) | {red['run']['campaign_wall_s']} s |",
         f"| tasks completed (all checks passed) | {d['tasks_completed']} of {red['run']['lanes']} |",
         f"| useful work | {d['useful_work']} |",
         f"| **useful work per wall-clock minute** | **{d['useful_work_per_minute']}** |",
         f"| last lane's first useful token | {d['last_lane_first_useful_s']} s |",
         f"| aggregate decode | {d['aggregate_decode_tok_s']} tok/s |",
         f"| per-lane decode (server) | {d['per_lane_decode_tok_s']} |",
         "",
         "## Lanes",
         "",
         "| lane | session | TTFT s | queue s | model TTFT s | tok/s (server decode) | tok/s (client, whole lane) "
         "| prompt tok | "
         "gen tok | drafts | tool calls | reads | checks | complete | stopped |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in red["lanes"]:
        L.append(f"| {l['lane']} | {l['session']} | {l['ttft_s']} | {l['queue_delay_s']} ({l['queue_delay_source']}) "
                 f"| {l['model_ttft_s']} | {l['decode_tok_s_server']} | {l['client_tok_s']} "
                 f"| {l['prompt_tokens']} | {l['generated_tokens']} | {l['draft_n_accepted']}/{l['draft_n']} "
                 f"| {l['tool_calls']} | {l['reads']} | {l['checks_passed']}/{l['checks_total']} "
                 f"| {'yes' if l['task_complete'] else 'no'} | {l['stopped']} |")
    L += ["", "## Server evidence", "",
          f"- queue delay from `{s['queue_delay_source']}`; `/api/requests` available: {s['api_requests_available']}",
          f"- seconds with a request queued: {s['queued_seconds']}",
          f"- peak non-idle slots: {s['max_nonidle_slots']}; two slots non-idle together for {s['slot_overlap_s']} s; "
          f"solo fallback seen: {s['solo_fallback_observed']}",
          f"- samples: {s['samples']} at {red['run']['sample_interval_s']} s ({s['sampler_failures']} failed)",
          f"- hardware peak: {json.dumps(s['hardware_peak'], sort_keys=True)}",
          f"- expert cache hit rates of recent requests: {s['expert_hit_rates_recent']}",
          f"- recent requests: {json.dumps(s['requests_recent'][:4])}",
          ""]
    if red.get("cancellation"):
        c = red["cancellation"]
        L += [f"## Cancellation (lane {c['cancelled_lane']} closed at {c['after_s']} s)", "",
              f"- cancelled lane finished as `{c['victim_finish']}` (aborted: {c['victim_aborted']})",
              f"- surviving lane {c['survivor_lane']}: ok {c['survivor_ok']}, task complete "
              f"{c['survivor_task_complete']}, {c['survivor_decode_tok_s']} tok/s",
              f"- server still serving afterwards: {c['server_still_serving']}", ""]
    L += ["## Correctness", "",
          "| lane | undeclared tools | bad arguments | errors |",
          "|---|---|---|---|"]
    for l in red["lanes"]:
        bad = sum(1 for c in l["calls"] if c["bad_arguments"])
        L.append(f"| {l['lane']} | {', '.join(l['undeclared_tools']) or 'none'} | {bad} | "
                 f"{'; '.join(l['errors']) or 'none'} |")
    L += ["", "## Not measured by this run", ""] + [f"- {x}" for x in red["not_measured"]] + [""]
    return "\n".join(L)


def plan_text(args, specs: list) -> str:
    arm = ARMS[args.arm]
    words = CONTEXT_CLASSES[args.context_class]
    L = [f"DRY RUN - nothing is sent, no server is started or restarted.  Schema {SCHEMA}.",
         f"  server            {args.url}",
         f"  arm               {args.arm}: {arm['serving']} batch slot(s), batch-MTP "
         f"{'expected' if arm['batch_mtp'] else 'off'} (a start-up configuration: an arm change is a restart)",
         f"  guard             refuse unless /v1/status reports concurrency.serving={args.expect_serving}"
         + (f" and the model contains {args.expect_model!r}" if args.expect_model else ""),
         f"  context class     {args.context_class} ({words} filler words, about "
         f"{int(words * 6.7 / 3.2 / 1000)}K tokens; the server's own prompt_tokens is the evidence)",
         f"  lanes             {len(specs)} started at one instant"
         f"{' (a barrier; both send their first request together)' if len(specs) > 1 else ''}",
         f"  steps per lane    at most {args.max_steps}, {args.max_tokens_per_step} new tokens each, greedy, "
         f"seed {args.seed}",
         f"  tools             {', '.join(sorted(DECLARED))} - read_file and grep are executed read-only against this "
         f"checkout; submit_solution is scored, nothing is written",
         f"  scoring           declared tools only, read before submitting, expected target path, anchor verbatim in "
         f"that file, summary present",
         f"  sampling          /metrics?requests=all and /v1/status every {args.sample_interval} s "
         f"(VRAM from hardware.gpu_mem_used, no nvidia-smi)",
         f"  queue delay       the server's queue_s from /api/requests when the API monitor is on, else "
         f"/metrics live.queued integrated over the campaign"]
    if args.cancel_lane:
        L.append(f"  cancellation      lane {args.cancel_lane} closes its connection at {args.cancel_after} s; the "
                 f"other lane must finish")
    L.append("  lanes:")
    for s in specs:
        L.append(f"    {s['lane']}: {s['session']} - reads expected from "
                 f"{', '.join(sorted(task_specs()[s['session']]['expected_files'])[:2])} ...")
    L.append(f"  writes            {args.output_dir}/run-{args.arm}-{args.context_class}-<stamp>.json and .md")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--expect-serving", type=int, required=True,
                    help="concurrency.serving this run must find; the run stops if the server says otherwise")
    ap.add_argument("--expect-model", default="", help="refuse unless /v1/status serves a model containing this")
    ap.add_argument("--context-class", default="short", choices=sorted(CONTEXT_CLASSES))
    ap.add_argument("--sessions", default="", help=f"comma-separated session names (default: the first two of "
                                                   f"{', '.join(n for n, _, _ in SESSIONS)})")
    ap.add_argument("--lanes", type=int, default=2, help="1 or 2 (2 is what the issue asks for)")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-tokens-per-step", type=int, default=256)
    ap.add_argument("--max-tool-result-chars", type=int, default=60000)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--start-timeout", type=float, default=120.0)
    ap.add_argument("--sample-interval", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=1, help="which repetition of the campaign this is (the gate wants 3)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cancel-lane", default="", choices=["", "A", "B"])
    ap.add_argument("--cancel-after", type=float, default=20.0)
    ap.add_argument("--output-dir", default="bench/results/rtx5080-pi-c1c2")
    ap.add_argument("--run", action="store_true", help="actually send traffic (without this, only the plan is printed)")
    a = ap.parse_args()
    if a.lanes not in (1, 2):
        raise SystemExit("--lanes is 1 or 2")
    if a.expect_serving != ARMS[a.arm]["serving"]:
        raise SystemExit(f"arm {a.arm} is a {ARMS[a.arm]['serving']}-slot configuration; --expect-serving "
                         f"{a.expect_serving} contradicts it")
    if a.cancel_lane and a.lanes == 1:
        raise SystemExit("the cancellation fixture needs two lanes")
    names = [x.strip() for x in a.sessions.split(",") if x.strip()] or [n for n, _, _ in SESSIONS]
    specs_all = task_specs()
    unknown = [n for n in names if n not in specs_all]
    if unknown:
        raise SystemExit(f"unknown session(s) {unknown}; known: {', '.join(specs_all)}")
    specs = [{"lane": "AB"[i], "session": names[i], "name": names[i], **specs_all[names[i]]}
             for i in range(min(a.lanes, len(names)))]
    if not a.run:
        print(plan_text(a, specs))
        return 0
    red = redactor(key=a.api_key)
    report = run_campaign(a, specs)
    report = redact_obj(report, red)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(report["run"]["started_at"]))
    stem = f"run-{a.arm}-{a.context_class}-{stamp}" + (f"-{a.tag}" if a.tag else "")
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    d = report["decision_metric"]
    print(red(f"[pi_c1c2] arm {a.arm} {a.context_class}: wall {report['run']['campaign_wall_s']} s, tasks "
              f"{d['tasks_completed']}/{len(specs)}, useful work {d['useful_work_per_minute']}/min, "
              f"per-lane decode {d['per_lane_decode_tok_s']}, aggregate {d['aggregate_decode_tok_s']}, "
              f"peak slots {report['server_evidence']['max_nonidle_slots']}, "
              f"overlap {report['server_evidence']['slot_overlap_s']} s"))
    for lane in report["lanes"]:
        print(red(f"[pi_c1c2]   lane {lane['lane']} {lane['session']}: TTFT {lane['ttft_s']} s "
                  f"(queue {lane['queue_delay_s']} s, model {lane['model_ttft_s']} s), "
                  f"{lane['decode_tok_s_server']} tok/s, checks {lane['checks_passed']}/{lane['checks_total']}, "
                  f"complete {lane['task_complete']}, stopped {lane['stopped']}"))
    print(f"[pi_c1c2] wrote {out_dir / stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
