#!/usr/bin/env python3
"""Pi C1/C2 two-lane harness (Issue #2): measure what two independent Pi sessions running at once are worth.

One campaign = one arm, one context class, two lanes.  Both lanes start at one instant and work the same fixed
task a Pi agent would - read a file of this checkout, find an anchor in it, submit a change plan - each in its
own session, so the two prompts share nothing but the system block.  Every quantity the decision uses is
measured, not inferred:

  client TTFT           the seconds from the request leaving this process to the first answer token arriving,
                        timed here, over the streaming path a Pi client uses
  queue delay           the server's own per-request `queue_s`, tied to the lane that waited by that lane's own
                        nonce in the request body - and marked unavailable when the tie is not unambiguous
  model-side TTFT       client TTFT minus that queue delay, only when the queue delay is known
  prompt / decode rate  the server's own per-request `timings`, summed over every step of the lane
  drafts                the same timings' `draft_n` / `draft_n_accepted`: a draft offered on a batch slot, which
                        is how a `c2mtp` run proves the server really runs --batch-mtp instead of trusting the
                        arm label (no endpoint answers with the server's own command line)
  slot overlap          the seconds two slots were non-idle in two /metrics samples in a row, from
                        `live.slots[].state` only - `live.running` counts requests, not slots, and is reported
                        separately as in-flight overlap
  VRAM / RAM peak       /metrics `hardware`, which is in BYTES, converted to MiB / GiB here
  useful work           whole tasks finished: every fixed check passed.  Partial credit is recorded separately
                        and is never counted as a finished task

What is NOT measured, because the server cannot say it: the server's slot-transition latency (no timestamp per
transition) and a per-lane admission wait (no per-lane admission clock).  The run file carries both as null with
the reason, and `queue_delay_source` says which queue number a lane's row is, or `unavailable:<why>`.

The task set is this repo's, so the numbers mean work on Strata itself: the same sessions, system prompt and
long-prompt filler `tools/agent_turns_bench.py` uses, with the same conversation-cache property (one shared
system block, a different user turn per session).  The system prompt is extended here to name only the three
tools this harness answers, because the shared one advertises edit and run: a lane told it has tools it has not
gets an undeclared call, and that would be the harness's fault, not the model's.

Safety, and why:
  - GET only for the evidence, one POST per step.  Nothing here starts, stops or reconfigures a server;
    `--parallel` and `--batch-mtp` are start-up configuration (serve/server.py answers `concurrency.serving`
    from the engine's batch slots), so the operator starts the two servers and this harness only refuses to
    measure one whose count does not match the arm.
  - read_file is confined to this checkout by resolving the path first, on either path separator, so a symlink
    cannot leave it and Windows does not reject every file; grep runs `git grep` with a timeout and a byte cap.
  - a lane may only call the three declared tools; an undeclared call ends that lane and is reported.  A tool
    call whose arguments never arrived whole is recorded as truncated (the server's own client tolerates this,
    issue #211) and ends the lane as an incomplete task, not as a violation.
  - every read is capped, every stream is capped, the sampler keeps a bounded number of rows and `git grep` has
    a timeout, so a model asking for a huge file, a huge answer or a huge search cannot grow the harness.
  - every request has a socket deadline, every lane has the campaign's deadline, and a cancelled lane's read is
    cut off at its own deadline even when the stream has gone quiet - a stalled stream must not hang a campaign.
  - a 429 or 503 from the server's own busy path is retried a bounded number of times, honouring Retry-After,
    because that answer is expected under concurrency and is not a model failure.
  - a model's tool arguments are untrusted input: they are type-checked before use, and a lane that raises is
    recorded as a failed lane instead of taking the campaign's evidence down with it.
  - a non-loopback --url over http:// is refused unless --allow-remote says you meant it, and the key can come
    from the environment so it does not appear in a command line.
  - Home, user and host names, IP addresses and the API key are replaced by placeholders before anything is
    written, with the same redactor tools/strata_bench.py uses.

Exit status: 0 clean, 3 a lane had an error, 4 a c2mtp run that could not show a draft on a batch slot (or a c2
run that showed one it did not expect), 2 a refused request.  Nothing is written when the request is refused.

CPU-only: no CUDA, no model, no GPU.  Run --dry-run first; it prints the plan and writes nothing.
Read the contract first: docs/RTX5080_PI_C1C2.md
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "serve"), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from ab_engine import long_prompt                           # noqa: E402  the same filler every suite uses
from agent_turns_bench import SESSIONS, SYSTEM, tool_result  # noqa: E402  the same tasks, prompt, file reads
from strata_bench import redact_obj, redactor                # noqa: E402  the same placeholders on write

# The bounds the harness holds itself to.  Each one is here because a model, a slow stream or a busy server can
# otherwise make the harness the thing that grows without limit.
MAX_TOOL_RESULT_CHARS = 60000        # the default cap on one tool result handed back to the model
READ_MAX_BYTES = 4_000_000           # a file bigger than this is not read at all
REPO_GREP_TIMEOUT_S = 20.0           # `git grep` with a model's regex must not run forever
REPO_GREP_MAX_BYTES = 200_000        # what of the search result is kept
STREAM_MAX_CHARS = 400_000           # answer + reasoning kept per step; the rest is dropped and flagged
SAMPLE_MAX_ROWS = 20_000             # /metrics rows per campaign (5 Hz for about an hour)
RECORD_DETAIL_MAX = 100              # /api/requests details fetched per campaign (the server's retention)
RETRIES_MAX = 2                      # retries of a 429/503 before the step is called an error
RETRY_WAIT_MAX_S = 30.0
# the filler's own estimate: about 6.7 characters per filler word, about 3.2 characters per token
TOKENS_PER_FILLER_WORD = 6.7 / 3.2
CHARS_PER_TOKEN = 3.2

# The three tools a lane may use.  `submit_solution` is the lane's last act and is answered by the harness, not
# the model; it is declared so a lane that calls it has called only declared tools.
PI_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the Strata checkout. Path is relative to the "
                                            "checkout root.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the checkout with a regular expression.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "submit_solution", "description": "Finish: name the file you would change, quote an anchor line "
                                                  "that is really in it, and give the change plan in a few "
                                                  "sentences.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "anchor": {"type": "string"},
                                                        "summary": {"type": "string"}},
                       "required": ["path", "anchor", "summary"]}}},
]
DECLARED = [t["function"]["name"] for t in PI_TOOLS]
# The shared system prompt says the assistant can "edit files and run commands".  This harness answers three
# tools, so say so: otherwise a lane that believes the prompt makes an undeclared call and the arm is rejected
# for the harness's own wording.
LANE_SYSTEM = (SYSTEM.rstrip() + " In this session exactly three tools exist: read_file, grep and "
                                 "submit_solution. There is no edit, write or run tool, so never ask for one: "
                                 "put the change you would make into submit_solution.")

# The tasks are the shared agent-turn sessions themselves (tools/agent_turns_bench.py SESSIONS), so the
# numbers mean work on Strata and there is no second copy of a task set to drift.  `expected_files` is the
# files that session has to read; `required_reads` is how many reads a lane must have made before it submits
# and may still be counted as having read before submitting.
def task_specs() -> dict:
    out = {}
    for name, task, steps in SESSIONS:
        reads = [arg for tool, arg, _ in steps if tool == "read_file"]
        out[name] = {"name": name, "task": task, "expected_files": set(reads), "required_reads": 1}
    return out


TASKS = task_specs()

# Filler words chosen to land near the token target (about 3.2 characters per token, about 6.7 characters per
# filler word).  The label is a class, not a measurement: the server's own `prompt_tokens` is what the evidence
# records.
CONTEXT_CLASSES = {"short": 0, "32k": 15700, "64k": 31400, "118k": 56600}
ARMS = {"c1": {"serving": 1, "batch_mtp": False},
        "c2": {"serving": 2, "batch_mtp": False},
        "c2mtp": {"serving": 2, "batch_mtp": True}}
# what a batch slot's `state` looks like in /metrics `live.slots` when the slot is not working
IDLE_SLOT = (None, "", "idle")
# the six fixed checks, in the order the report prints them (the same order score_solution builds)
CHECKS_ORDER = ["submitted", "target_path_expected", "anchor_verbatim", "read_before_submit",
                "summary_present", "declared_tools_only"]


# `redactor()` answers with the replacing function itself (tools/strata_bench.py), not an object
_RED = redactor()


def red(s: str) -> str:
    return _RED(s)


def use_key_for_redaction(key: str) -> None:
    """Rebuild the redactor knowing this run's API key, so the key can never be written or printed."""
    global _RED
    _RED = redactor(key=key or "")


def build_lane_messages(spec: dict, context_class: str, nonce: str) -> list:
    """The lane's first request: a background block sized to the context class, then the task.  The nonce
    marks the lane in the prompt, which is how a request of this campaign can be tied back to the lane that
    sent it in the server's own records (see correlate_queue_delay)."""
    msgs = [{"role": "system", "content": LANE_SYSTEM}]
    words = CONTEXT_CLASSES[context_class]
    if words:
        msgs.append({"role": "user", "content": f"[session {nonce}] Background from earlier in this "
                                                f"session:\n{long_prompt(words)}"})
        msgs.append({"role": "assistant", "content": "Understood. I have the background."})
    msgs.append({"role": "user", "content": f"[session {nonce}] {spec['task']}"})
    return msgs


def inside_prefix(root_abs: str, target_abs: str, sep: str, normcase) -> bool:
    """The containment test on two already-absolute paths, as a pure function of the separator and the
    case-folding of the platform doing the measuring.  A `str(ROOT) + "/"` prefix test is wrong on Windows,
    where `str(Path)` uses "\\": every read and every anchor check would fail there and no task could be
    completed, which is why this takes the separator instead of assuming one."""
    t, r = normcase(target_abs), normcase(root_abs)
    return t == r or t.startswith(r + sep)


def inside_root(target) -> bool:
    """True when `target` (already resolved) is the checkout or inside it.  The path is resolved before this is
    called, so a symlink that leaves the checkout resolves outside it and is refused."""
    return inside_prefix(os.path.abspath(str(ROOT)), os.path.abspath(str(target)), os.sep, os.path.normcase)


def safe_read(arg):
    """(text, None) or (None, why-not).  Confined to this checkout: the path is resolved first, so a symlink
    that leaves it resolves outside and is refused, and the containment test works on either separator."""
    if not isinstance(arg, str) or not arg.strip():
        return None, "(read_file: a path is needed)"
    target = (ROOT / arg)
    try:
        target = target.resolve()
    except OSError as e:
        return None, f"(read_file: {arg!r} could not be resolved: {type(e).__name__})"
    if not inside_root(target):
        return None, f"(read_file: {arg!r} is outside the checkout and was not read)"
    if not target.is_file():
        return None, f"(read_file: {arg!r} is not a file in the checkout)"
    try:
        if target.stat().st_size > READ_MAX_BYTES:
            return None, (f"(read_file: {arg!r} is bigger than {READ_MAX_BYTES // 1_000_000} MB and was not "
                          "read; ask for a smaller file)")
    except OSError as e:
        return None, f"(read_file: {arg!r} could not be opened: {type(e).__name__})"
    return tool_result("read_file", os.path.relpath(str(target), str(ROOT)).replace(os.sep, "/")), None


def repo_grep(pattern):
    """`git grep` over the checkout, with a timeout and a byte cap: the pattern is the model's, so neither the
    run time nor the output may be left unbounded."""
    if not isinstance(pattern, str) or not pattern.strip():
        return "(grep: a pattern is needed)"
    if pattern.startswith("-"):
        return "(grep: the pattern cannot start with '-')"
    try:
        p = subprocess.run(["git", "grep", "-in", pattern], cwd=str(ROOT), capture_output=True,
                           encoding="utf-8", errors="replace", timeout=REPO_GREP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"(grep: `git grep` was stopped after {REPO_GREP_TIMEOUT_S:.0f} s and returned nothing)"
    except OSError as e:
        return f"(grep: `git grep` could not run: {type(e).__name__})"
    out = (p.stdout or "")
    if len(out) > REPO_GREP_MAX_BYTES:
        out = out[:REPO_GREP_MAX_BYTES] + "\n(grep: the rest of the search result was dropped)"
    if p.returncode == 1 and not out.strip():
        return "(grep: no match)"
    if p.returncode not in (0, 1) and not out.strip():
        return f"(grep: `git grep` said rc={p.returncode}: {(p.stderr or '').strip()[:200]})"
    return out


def tool_arg(name: str, args) -> str:
    """The one argument of the tool the model asked for, taken by tool name.  `args` is model output: anything
    that is not an object of strings yields "" rather than an exception."""
    if not isinstance(args, dict):
        return ""
    key = "pattern" if name == "grep" else "path"
    v = args.get(key)
    return v if isinstance(v, str) else ""


def safe_tool(name: str, args) -> str:
    if name == "read_file":
        text, err = safe_read(tool_arg("read_file", args))
        return text if err is None else err
    if name == "grep":
        return repo_grep(tool_arg("grep", args))
    return f"(the harness has no tool {name!r})"


def score_solution(spec: dict, calls: list, submit) -> dict:
    """The six fixed checks.  A task is COMPLETED only when all six pass, and `useful_work` is then 1 and 0
    otherwise: partial credit is reported on its own as `partial_credit` and never counted as a finished task.

    `submit` is model output and may be anything at all - a list, a string, an object whose "path" is an object -
    so every field is type-checked before it is used.  A malformed submission scores zero; it must not raise."""
    submit = submit if isinstance(submit, dict) else None
    path = submit.get("path") if submit and isinstance(submit.get("path"), str) else None
    anchor = submit.get("anchor") if submit and isinstance(submit.get("anchor"), str) else None
    reads = [c for c in calls if c["name"] == "read_file"]
    submit_at = next((i for i, c in enumerate(calls) if c["name"] == "submit_solution"), len(calls))
    anchor_ok = False
    if anchor and path:
        text, err = safe_read(path)
        anchor_ok = bool(text) and err is None and len(anchor.strip()) >= 12 and anchor.strip() in text
    reads_before = sum(1 for c in reads[:submit_at] if c.get("executed"))
    try:
        needed = int(spec.get("required_reads", 1))
    except (TypeError, ValueError):
        needed = 1
    undeclared = sorted({c["name"] for c in calls if not c.get("declared")})
    checks = {
        "submitted": bool(submit),
        "target_path_expected": bool(path) and path in set(spec["expected_files"]),
        "anchor_verbatim": anchor_ok,
        "read_before_submit": bool(submit) and reads_before >= max(1, needed),
        "summary_present": bool(submit) and len(str(submit.get("summary") or "").strip()) >= 40,
        "declared_tools_only": not undeclared,
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    return {"checks": checks, "checks_passed": passed, "checks_total": total,
            "task_complete": passed == total, "useful_work": 1.0 if passed == total else 0.0,
            "partial_credit": round(passed / total, 3), "undeclared_tools": undeclared}


def guard(st, mt, arm: str, expect_serving: int, model_want: str) -> dict:
    """Refuse to measure a server that is not what the arm claims.

    Reads the shapes the server really answers with (serve/server.py `status()` and `metrics()`): `engine` is a
    version STRING, `loaded` is top level, the window is `context.max_positions` / `cache_max_tokens`, the slot
    count is `concurrency.serving`, and what is running now comes from /metrics `live` - /v1/status carries no
    queue depth at all.  Nothing assumes `engine` is an object: a build that answers a dict is read for what it
    has instead of raising."""
    out = {"status_ok": bool(st), "metrics_ok": bool(mt), "loaded": None, "model": None, "engine_version": None,
           "max_context": None, "serving_expected": expect_serving, "serving_observed": None,
           "serving_requested": None, "queued": None, "running": None, "waiting": None, "admitting": None,
           "drafts_offered_total": None, "drafts_accepted_total": None, "mismatch": []}
    if not st:
        out["mismatch"].append("no /v1/status answer (as JSON) - is the server up and the API key right?")
        return out
    eng = st.get("engine")
    out["engine_version"] = eng if isinstance(eng, str) else None
    out["loaded"] = bool(st.get("loaded"))
    out["model"] = st.get("model")
    ctx = st.get("context") if isinstance(st.get("context"), dict) else {}
    out["max_context"] = ctx.get("max_positions") or st.get("cache_max_tokens")
    conc = st.get("concurrency") if isinstance(st.get("concurrency"), dict) else {}
    out["serving_observed"] = conc.get("serving")
    out["serving_requested"] = conc.get("requested")
    if mt:
        live = mt.get("live") if isinstance(mt.get("live"), dict) else {}
        tot = mt.get("totals") if isinstance(mt.get("totals"), dict) else {}
        eng_m = mt.get("engine") if isinstance(mt.get("engine"), dict) else {}
        out["queued"] = live.get("queued")
        out["running"] = live.get("running") if live.get("running") is not None else (1 if live.get("busy") else 0)
        out["waiting"], out["admitting"] = live.get("waiting"), live.get("admitting")
        out["drafts_offered_total"], out["drafts_accepted_total"] = tot.get("drafts_offered"), \
            tot.get("drafts_accepted")
        out["max_context"] = out["max_context"] or eng_m.get("max_context")
    if not out["loaded"]:
        out["mismatch"].append("the model is not loaded: load it before measuring (a load would be read as a "
                               "queue wait)")
    if out["serving_observed"] != expect_serving:
        out["mismatch"].append(f"--expect-serving {expect_serving} but the server reports "
                               f"concurrency.serving={out['serving_observed']}: start the server for this arm "
                               "with the right --parallel (it is start-up configuration and cannot be changed "
                               "while measuring)")
    if model_want and out["model"] != model_want:
        out["mismatch"].append(f"--model {model_want} but the server has {out['model']!r}: the two arms must run "
                               "the same model")
    if out["queued"] or out["running"]:
        out["mismatch"].append(f"the server is not idle ({out['queued'] or 0} queued, {out['running'] or 0} "
                               "running): someone else is using it, and their queue time would be read as ours")
    return out


def machine_part(st, key: str) -> dict:
    """One part of /v1/status `machine`.  `machine.gpu` is a LIST of cards (serve/server.py), `machine.ram` a
    dict; either may be missing, and the values are already MiB / GiB / W / %."""
    mach = st.get("machine") if isinstance(st, dict) else {}
    node = (mach or {}).get(key)
    if isinstance(node, list):
        node = next((x for x in node if isinstance(x, dict)), {})
    return node if isinstance(node, dict) else {}


def compact_sample(metrics: dict, status: dict, at: float) -> dict:
    """The few /metrics fields the evidence needs, in the units the report names.

    /metrics `hardware` is in BYTES (serve/telemetry.py; the Prometheus exporter publishes the same numbers as
    strata_gpu_mem_used_bytes), while /v1/status `machine` is already MiB and GiB - so the bytes are converted
    here and the two are never mixed.  The telemetry key names are `ram_used` and `gpu_power`, not
    `ram_used_gib` / `gpu_power_w`."""
    hw = (metrics or {}).get("hardware") or {}
    gpu, ram = machine_part(status, "gpu"), machine_part(status, "ram")

    def to_unit(value, scale, fallback):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(value / scale, 2)
        return fallback

    live = (metrics or {}).get("live") or {}
    row = {"at": round(at, 3), "state": live.get("state"), "queued": live.get("queued") or 0,
           "running": live.get("running") if live.get("running") is not None else (1 if live.get("busy") else 0),
           "waiting": live.get("waiting") or 0, "admitting": live.get("admitting") or 0,
           "vram_used_mib": to_unit(hw.get("gpu_mem_used"), 2 ** 20, (gpu or {}).get("used_mib")),
           "vram_total_mib": to_unit(hw.get("gpu_mem_total"), 2 ** 20, (gpu or {}).get("total_mib")),
           "ram_used_gib": to_unit(hw.get("ram_used"), 2 ** 30, (ram or {}).get("used_gib")),
           "ram_total_gib": to_unit(hw.get("ram_total"), 2 ** 30, (ram or {}).get("total_gib")),
           "gpu_power_w": hw.get("gpu_power") if hw.get("gpu_power") is not None else (gpu or {}).get("power_w"),
           "gpu_util": hw.get("gpu_util") if hw.get("gpu_util") is not None else (gpu or {}).get("util_pct"),
           "nonidle_slots": None, "slot_states": [], "slot_state_source": "none"}
    # `live.slots` is there only when the engine runs batch slots, and its absence is NOT "zero non-idle slots",
    # so the row says which evidence it carries instead of falling back to a different quantity.
    slots = live.get("slots")
    if isinstance(slots, list) and slots:
        rows = [s for s in slots if isinstance(s, dict)]
        row["nonidle_slots"] = sum(1 for s in rows if s.get("state") not in IDLE_SLOT)
        row["slot_states"] = [f"{s.get('slot')}:{s.get('state')}" for s in rows]
        row["slot_state_source"] = "live.slots"
    return row


class Sampler(threading.Thread):
    """GET /metrics (and /v1/status for the machine's MiB figures, only when /metrics has no hardware) on an
    interval while the campaign runs.

    Plain /metrics, not `?requests=all`: the sampler needs `live` and `hardware`, and asking for the whole
    request history makes the server copy that history under its status lock at 5 Hz - the sampler would be
    loading the thing it measures."""

    daemon = True

    def __init__(self, url: str, api_key: str, interval: float, timeout: float):
        super().__init__(daemon=True)
        self.url, self.key, self.interval, self.timeout = url, api_key, interval, timeout
        self.samples, self.stop, self.errors, self.dropped = [], threading.Event(), 0, 0

    def run(self):
        while not self.stop.is_set():
            try:
                m = get_json(self.url + "/metrics", self.key, self.timeout)
                s = None
                if isinstance(m, dict) and m and not (m.get("hardware") or {}):
                    s = get_json(self.url + "/v1/status", self.key, self.timeout)
                if isinstance(m, dict) and m and not m.get("error"):
                    if len(self.samples) >= SAMPLE_MAX_ROWS:
                        self.dropped += 1
                    else:
                        self.samples.append(compact_sample(m, s, time.time()))
            except Exception:                                   # noqa: BLE001 - a lost sample is not a failure
                self.errors += 1
            self.stop.wait(self.interval)

    def rows(self) -> list:
        return list(self.samples)


def _socket_of(resp):
    """The socket behind an urllib HTTPResponse, so a read can be given a deadline.  Plain HTTP gives
    `fp.raw._sock`; an https connection gives the SSL socket itself."""
    fp = getattr(resp, "fp", None)
    for cand in (fp, getattr(fp, "raw", None)):
        s = getattr(cand, "_sock", None)
        if isinstance(s, socket.socket):
            return s
    return None


def read_empty():
    return {"content": "", "reasoning": "", "tool_calls": [], "finish": None, "usage": None, "timings": None,
            "events": 0, "comments": 0, "error": None, "aborted": False, "done": False,
            "content_truncated": False}


def read_sse(resp, on_event=None, abort_at=None, deadline=None):
    """Read one OpenAI-style SSE stream, assembling the answer and the tool-call fragments.

    Three bounds, and each is the reason this function exists: `abort_at` is a real deadline (the socket's own
    timeout is pulled in before every line, so a stream that has gone quiet is still cut off), `deadline` caps
    the whole step, and the answer text is capped so a runaway stream cannot grow the harness.  A stream that
    ends without [DONE] is an error: half an answer is not a clean step."""
    out = read_empty()
    sock = _socket_of(resp)
    while True:
        lefts = [x for x in ((None if abort_at is None else abort_at - time.monotonic()),
                             (None if deadline is None else deadline - time.monotonic())) if x is not None]
        if lefts and min(lefts) <= 0:
            out["aborted"] = True
            return out
        if sock is not None and lefts:
            try:
                sock.settimeout(max(0.05, min(lefts)))
            except (OSError, ValueError):
                pass
        try:
            line = resp.readline()
        except (socket.timeout, TimeoutError):
            # the socket waited its whole allowance, so say which bound was reached: a cancellation (`abort_at`)
            # or the campaign's deadline is an abort, a plain socket timeout with neither is an error
            left_now = [x for x in ((None if abort_at is None else abort_at - time.monotonic()),
                                    (None if deadline is None else deadline - time.monotonic()))
                        if x is not None]
            if left_now and min(left_now) <= 0.05:
                out["aborted"] = True
            else:
                out["error"] = "the stream stopped answering inside its deadline"
            return out
        except OSError as e:
            out["error"] = f"the stream ended on {type(e).__name__}"
            return out
        if not line:
            if not out["done"] and out["finish"] is None:
                out["error"] = "the stream ended before [DONE]: the answer is not whole"
            return out
        text = line.decode("utf-8", "replace").strip()
        if not text:
            continue
        if text.startswith(":"):                        # an SSE comment (keep-alive): ignored by clients
            out["comments"] += 1
            continue
        if not text.startswith("data:"):
            continue
        payload = text[5:].strip()
        if payload == "[DONE]":
            out["done"] = True
            return out
        try:
            chunk = json.loads(payload)
        except ValueError:
            out["error"] = "an SSE data line was not JSON"
            return out
        if chunk.get("error"):
            out["error"] = str(chunk["error"])[:300]
            continue
        out["events"] += 1
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        if chunk.get("timings"):
            out["timings"] = chunk["timings"]

        def keep(field, piece):
            """Append an answer fragment up to the cap, and say when the cap was hit."""
            if not isinstance(piece, str) or not piece:
                return
            room = STREAM_MAX_CHARS - len(out[field])
            if room <= 0:
                out["content_truncated"] = True
                return
            if len(piece) > room:
                out[field] += piece[:room]
                out["content_truncated"] = True
            else:
                out[field] += piece

        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            keep("content", delta.get("content"))
            for key in ("reasoning_content", "reasoning"):
                keep("reasoning", delta.get(key))
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(out["tool_calls"]))
                while len(out["tool_calls"]) <= idx:
                    out["tool_calls"].append({"id": None, "name": "", "arguments": ""})
                slot = out["tool_calls"][idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
        if on_event and out["events"] == 1:
            on_event()


def get_json(url: str, api_key: str, timeout: float):
    return post_json(url, None, api_key, timeout)


def post_json(url: str, body, api_key: str, timeout: float, retries: int = 0):
    """A JSON GET (body None) or POST, with the same bounded 429/503 retry as a stream: the server answers those
    when it is busy, and a busy answer is not a model failure."""
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8", "replace")) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                attempt += 1
                time.sleep(min(RETRY_WAIT_MAX_S, 2.0 * attempt))
                continue
            return {"error": f"HTTP {e.code} from {url.split('?')[0]}"}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            return {"error": f"{type(e).__name__} from {url.split('?')[0]}"}


def retry_after(e) -> float:
    """How long the server asked us to wait (Retry-After in seconds), bounded; a sane default when it did not say."""
    try:
        said = float((e.headers or {}).get("Retry-After") or 0)
    except (TypeError, ValueError):
        said = 0.0
    return max(0.0, min(RETRY_WAIT_MAX_S, said or 2.0))


def call_stream(url: str, body: dict, api_key: str, timeout: float, on_event=None, abort_at=None, deadline=None,
                retries: int = RETRIES_MAX):
    """POST one streaming request and read it.  Returns the read_sse result plus `retries` (the 429/503 answers
    that were retried, so the evidence shows a busy server instead of hiding it)."""
    attempt, tried = 0, []
    while True:
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            return dict(read_empty(), error="the campaign's deadline passed before this request finished",
                        retries=tried)
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = read_sse(resp, on_event, abort_at=abort_at, deadline=deadline)
            out["retries"] = tried
            return out
        except urllib.error.HTTPError as e:
            said = ""
            try:
                said = e.read().decode("utf-8", "replace")[:200]
            except Exception:                                           # noqa: BLE001
                pass
            if e.code in (429, 503) and attempt < retries:
                wait = retry_after(e)
                left_now = None if deadline is None else deadline - time.monotonic()
                if left_now is not None:
                    wait = min(wait, max(0.0, left_now - 1.0))
                tried.append({"attempt": attempt + 1, "http_status": e.code, "waited_s": round(wait, 1)})
                attempt += 1
                if wait > 0:
                    time.sleep(wait)
                continue
            return dict(read_empty(), error=f"HTTP {e.code}: {said}", retries=tried)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            return dict(read_empty(), error=f"{type(e).__name__}: {e}", retries=tried)


def tool_char_cap(prompt_tokens, max_context, max_steps, requested) -> int:
    """How much of one tool result to hand back, so the conversation cannot grow past the window.

    The context class sizes the FIRST prompt; every read can add tens of thousands of tokens after it.  The cap
    is shrunk to keep `max_steps` reads inside what the window has left, and never falls below 4 000 chars."""
    if not max_context or not prompt_tokens:
        return requested
    left = max_context - prompt_tokens
    if left <= 0:
        return 4000
    per_step = int(0.6 * left * CHARS_PER_TOKEN / max(1, max_steps))
    return max(4000, min(requested, per_step))


def failed_lane(spec: dict, campaign_id: str, err: str) -> dict:
    """A lane row for a lane that raised, so one bad lane never costs the campaign its evidence."""
    return {"lane": spec["lane"], "session": spec["session"], "context_class": spec["context_class"],
            "filler_words": spec["filler_words"], "campaign_id": campaign_id,
            "started_at": round(time.time(), 3), "start_offset_ms": None, "ttft_s": None, "client_send_ms": None,
            "queue_delay_s": None, "model_ttft_s": None, "queue_delay_source": "unavailable: the lane stopped",
            "admission_s": None, "admission_source": "not_measured", "steps": 0, "tool_calls": 0,
            "undeclared_tools": [], "invalid_arguments": [], "truncated_tool_calls": [], "reads": [], "calls": [],
            "submission": None, "errors": [{"error": err}], "stopped": "harness_error", "generated_tokens": 0,
            "prompt_tokens": 0, "reused_tokens": 0, "drafts_offered": 0, "drafts_accepted": 0,
            "step_timings": [], "decode_ms_total": 0.0, "decoded_tokens_server": 0, "decode_tok_s_server": None,
            "prompt_ms_total": 0.0, "prompt_tok_s_server": None, "tool_char_cap": None, "wall_s": None,
            "checks": {}, "checks_passed": 0, "checks_total": 6, "task_complete": False, "useful_work": 0.0,
            "partial_credit": 0.0, "undeclared_tools_seen": []}


def run_lane(url: str, spec: dict, opts: dict, start_at: float, cancel_lane: str, cancel_after: float,
             campaign_id: str, deadline: float) -> dict:
    """One lane: wait for the shared start, then read / find / submit, one tool call per step.

    The row is returned whatever happens, because a campaign that loses a lane still has to write evidence for
    the other one."""
    messages = build_lane_messages(spec, spec["context_class"], spec["lane"])
    result = {"lane": spec["lane"], "session": spec["session"], "context_class": spec["context_class"],
              "filler_words": spec["filler_words"], "campaign_id": campaign_id, "nonce": spec["lane"],
              "started_at": round(time.time(), 3),
              "start_offset_ms": round((start_at - time.monotonic()) * 1000, 1),
              "ttft_s": None, "client_send_ms": None, "queue_delay_s": None, "model_ttft_s": None,
              "queue_delay_source": "not_set", "admission_s": None,
              "admission_source": "not_measured: the server keeps no per-lane admission clock",
              "steps": 0, "tool_calls": 0, "undeclared_tools": [], "invalid_arguments": [],
              "truncated_tool_calls": [], "reads": [], "calls": [], "submission": None, "errors": [],
              "stopped": None, "generated_tokens": 0, "prompt_tokens": 0, "reused_tokens": 0,
              "drafts_offered": 0, "drafts_accepted": 0, "step_timings": [], "decode_ms_total": 0.0,
              "decoded_tokens_server": 0, "decode_tok_s_server": None, "prompt_ms_total": 0.0,
              "prompt_tok_s_server": None, "tool_char_cap": opts["max_tool_chars"], "wall_s": None}
    t0 = time.monotonic()
    while time.monotonic() < start_at:                          # the barrier: both lanes release together
        time.sleep(min(0.002, max(0.0, start_at - time.monotonic())))
    body = {"model": opts["model"], "messages": messages, "tools": PI_TOOLS, "stream": True,
            "max_tokens": opts["max_tokens"], "temperature": 0.0,
            "user": f"pi-c1c2-{campaign_id}-{spec['lane']}"}
    submit = None
    for step in range(opts["max_steps"]):
        if deadline is not None and time.monotonic() >= deadline:
            result["stopped"] = "campaign_deadline"
            break
        abort_at = None
        if cancel_lane == spec["lane"]:
            abort_at = start_at + cancel_after
            if deadline is not None:
                abort_at = min(abort_at, deadline)
        send_at = time.monotonic()
        r = call_stream(url, body, opts["api_key"], opts["timeout"],
                        on_event=lambda: result.__setitem__("ttft_s", round(time.monotonic() - send_at, 3)),
                        abort_at=abort_at, deadline=deadline)
        if result["ttft_s"] is None:
            result["ttft_s"] = round(time.monotonic() - send_at, 3)
        result["steps"] += 1
        if r.get("retries"):
            result["errors"].append({"step": step, "retried": r["retries"]})
        if r.get("error"):
            result["errors"].append({"step": step, "error": r["error"]})
        if r.get("content_truncated"):
            result["errors"].append({"step": step, "error": "the answer was longer than the harness keeps"})
        usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
        result["generated_tokens"] += usage.get("completion_tokens") or 0
        result["prompt_tokens"] = usage.get("prompt_tokens") or result["prompt_tokens"]
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        result["reused_tokens"] = details.get("cached_tokens") or result["reused_tokens"]
        tim = r.get("timings") if isinstance(r.get("timings"), dict) else {}
        if tim:
            # the server's own per-request numbers, kept per step: a lane's rate is the sum of its steps' tokens
            # over the sum of their seconds, never one step's seconds under another step's tokens
            result["step_timings"].append(dict(tim, step=step))
            try:
                result["decode_ms_total"] = round(result["decode_ms_total"] + float(tim.get("predicted_ms") or 0), 1)
                result["prompt_ms_total"] = round(result["prompt_ms_total"] + float(tim.get("prompt_ms") or 0), 1)
            except (TypeError, ValueError):
                pass
            result["decoded_tokens_server"] += int(tim.get("predicted_n") or 0)
            result["drafts_offered"] += int(tim.get("draft_n") or 0)
            result["drafts_accepted"] += int(tim.get("draft_n_accepted") or 0)
        if r.get("aborted"):
            result["stopped"] = "cancelled_by_harness"
            break
        calls = r.get("tool_calls") or []
        if not calls:
            result["stopped"] = "answered_without_a_tool_call"
            break
        messages.append({"role": "assistant", "content": r.get("content") or None,
                         "tool_calls": [{"id": c["id"] or f"call_{step}_{i}", "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for i, c in enumerate(calls)]})
        stop_after = None
        for i, c in enumerate(calls):
            result["tool_calls"] += 1
            name, call_id = c["name"], c["id"] or f"call_{step}_{i}"
            declared = name in opts["declared"]
            row = {"name": name, "declared": declared, "step": step}
            args = None
            try:
                parsed = json.loads(c["arguments"]) if c["arguments"] else {}
                if not isinstance(parsed, dict):
                    result["invalid_arguments"].append({"step": step, "tool": name,
                                                        "why": f"the arguments are a "
                                                               f"{type(parsed).__name__}, not an object"})
                else:
                    args = parsed
            except ValueError:
                # the stream ended inside the call: the server's own client tolerates this (issue #211), so it is
                # a truncated call - an incomplete task, not a boundary violation
                row["truncated"] = True
                result["truncated_tool_calls"].append({"step": step, "tool": name,
                                                       "arguments_chars": len(c["arguments"] or "")})
            if not declared:
                result["undeclared_tools"].append(name)
                row["executed"] = False
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": f"(the harness has no tool {name!r}; only "
                                            f"{', '.join(opts['declared'])} exist)"})
                result["calls"].append(row)
                stop_after = "undeclared_tool"
                break
            if args is None:
                row["executed"] = False
                result["calls"].append(row)
                stop_after = "truncated_tool_call" if row.get("truncated") else "invalid_tool_arguments"
                break
            row["executed"] = True
            result["calls"].append(row)
            if name == "submit_solution":
                submit = args
                # kept so a check can be audited by hand, the way the queue records are
                result["submission"] = {
                    "path": args.get("path")[:200] if isinstance(args.get("path"), str) else None,
                    "anchor": args.get("anchor")[:200] if isinstance(args.get("anchor"), str) else None,
                    "anchor_chars": len(args.get("anchor")) if isinstance(args.get("anchor"), str) else 0,
                    "summary_chars": len(args.get("summary")) if isinstance(args.get("summary"), str) else 0}
                stop_after = "submitted"
                break
            if name == "read_file":
                result["reads"].append({"tool_call_index": len(result["calls"]) - 1, "step": step,
                                        "path": tool_arg("read_file", args), "executed": True})
            out_txt = safe_tool(name, args)
            cap = result["tool_char_cap"]
            if len(out_txt) > cap:
                out_txt = out_txt[:cap] + "\n(the harness keeps only this much of a tool result)"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": out_txt})
        if stop_after:
            result["stopped"] = stop_after
            break
    result["wall_s"] = round(time.monotonic() - t0, 3)
    if result["decode_ms_total"] > 0 and result["decoded_tokens_server"]:
        result["decode_tok_s_server"] = round(result["decoded_tokens_server"] / (result["decode_ms_total"] / 1000),
                                              1)
    if result["prompt_ms_total"] > 0:
        read = max(0, (result["prompt_tokens"] or 0) - (result["reused_tokens"] or 0))
        result["prompt_tok_s_server"] = round(read / (result["prompt_ms_total"] / 1000), 1) or None
    scored = score_solution(spec, result["calls"], submit)
    result.update(scored)
    result["undeclared_tools"] = sorted(set(result["undeclared_tools"]))
    return result


def correlate_queue_delay(url: str, api_key: str, lanes: list, timeout: float) -> dict:
    """Tie the server's per-request `queue_s` to the lane that waited, and only when the tie is unambiguous.

    The list form of /api/requests carries no prompt text, so it cannot say which lane a record belongs to.
    The detail form (?id=) carries `input`, the raw request body (serve/server.py `begin_request`), and every
    lane's first request carries that lane's own nonce - `[session <name>]`, the first line of its background
    block, which no other client on this server can be sending.  A record is attributed only when exactly one
    lane's nonce is in it, and a lane's queue delay is the `queue_s` of the earliest such record: a lane's steps
    are sequential, so its earliest start is its first step, the one the client TTFT was timed on.

    Anything else - the API monitor off, no detail, no nonce, two lanes' nonces in one body - leaves that lane's
    queue delay null with the reason, rather than guessing.  A queue second charged to the lane that did not wait
    moves seconds between the two arms being compared, which is worse than no second."""
    evidence = {"api_requests_available": False, "records_seen": 0, "records_matched": 0,
                "records_unmatched": 0, "details_fetched": 0, "campaign_queued_seconds": None,
                "correlation": "none"}
    payload = get_json(url + "/api/requests", api_key, timeout)
    records = payload.get("requests") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        evidence["correlation"] = "unavailable: /api/requests did not answer"
        evidence["note"] = ("the API monitor is off by default: start the server with --api-monitor for "
                            "per-request queue seconds")
        for l in lanes:
            l["queue_delay_source"] = evidence["correlation"]
        return evidence
    evidence["api_requests_available"] = True
    evidence["records_seen"] = len(records)
    nonces = {l["lane"]: f"[session {l.get('nonce') or l['lane']}]" for l in lanes if l.get("lane")}
    by_lane = {}
    for rec in [r for r in records if isinstance(r, dict)][:RECORD_DETAIL_MAX]:
        rid = rec.get("id")
        if not rid:
            evidence["records_unmatched"] += 1
            continue
        detail = get_json(f"{url}/api/requests?id={rid}", api_key, timeout)
        evidence["details_fetched"] += 1
        if not isinstance(detail, dict) or detail.get("error"):
            evidence["records_unmatched"] += 1
            continue
        body = detail.get("input") if isinstance(detail.get("input"), str) else ""
        hits = [lane for lane, nonce in nonces.items() if nonce in body]
        if len(hits) != 1:
            evidence["records_unmatched"] += 1
            continue
        by_lane.setdefault(hits[0], []).append(detail)
    for lane, recs in by_lane.items():
        recs.sort(key=lambda r: r.get("started_at") or 0)
        first, row = recs[0], next((l for l in lanes if l["lane"] == lane), None)
        if row is None or row.get("ttft_s") is None:
            continue
        q = first.get("queue_s")
        if not isinstance(q, (int, float)) or isinstance(q, bool):
            row["queue_delay_source"] = f"unavailable: record {first.get('id')} carries no queue_s"
            continue
        row["queue_delay_s"] = round(float(q), 3)
        row["model_ttft_s"] = round(max(0.0, row["ttft_s"] - float(q)), 3)
        # the source names the METHOD (the gate looks for it); which record it came from is its own field
        row["queue_delay_source"] = "api_requests:nonce"
        row["queue_delay_record"] = first.get("id")
        row["server_first_token_s"] = first.get("first_token_s")
        row["server_wallclock_s"] = first.get("wallclock_s")
        evidence["records_matched"] += 1
    for l in lanes:
        if l.get("queue_delay_source") == "not_set":
            l["queue_delay_s"] = None
            l["model_ttft_s"] = None
            l["queue_delay_source"] = "unavailable: no request of this campaign carried this lane's nonce"
    matched = [r for recs in by_lane.values() for r in recs]
    if evidence["records_matched"]:
        evidence["campaign_queued_seconds"] = round(sum(float(r.get("queue_s") or 0) for r in matched), 3)
        evidence["correlation"] = "api_requests:nonce"
    return evidence


def batch_mtp_evidence(lanes: list, arm: str, confirm: str) -> dict:
    """Did the server really run a drafter on the batch slots?  The arm label is not evidence.

    No endpoint answers with the server's own command line, so --batch-mtp cannot be read back.  What can be
    seen is the per-request `timings` of our own streams: `draft_n` is there only when the engine reported
    drafts, and in batch mode a draft comes from a slot's own drafter, which exists only with --batch-mtp.  So a
    c2mtp run must SHOW a draft, and a c2 run that shows one is not the arm it is labelled.  An engine that
    reports no drafts at all needs the operator's --confirm-batch-mtp, which is recorded in the run file."""
    offered = sum(l.get("drafts_offered") or 0 for l in lanes)
    accepted = sum(l.get("drafts_accepted") or 0 for l in lanes)
    steps = sum(1 for l in lanes for t in l.get("step_timings") or [] if (t.get("draft_n") or 0) > 0)
    claims = ARMS.get(arm, {}).get("batch_mtp")
    out = {"arm_claims_batch_mtp": claims, "drafts_offered": offered, "drafts_accepted": accepted,
           "steps_reporting_drafts": steps, "operator_confirmation": confirm or None,
           "verified": claims is None, "unexpected_drafts": False, "why": ""}
    if claims is None:
        out["why"] = f"arm {arm!r} says nothing about drafts"
    elif offered > 0 or steps > 0:
        out["verified"] = bool(claims)
        out["unexpected_drafts"] = bool(claims is False and (offered or steps))
        out["why"] = (f"{offered} drafts offered over {steps} steps of our own requests, which only a batch "
                      "slot's own drafter can do" if out["verified"] else
                      f"{offered} drafts offered: the server is drafting on batch slots, so this is not the "
                      f"{arm!r} arm it is labelled")
    elif claims is False:
        out["verified"] = True
        out["why"] = ("no draft was reported on any request of this campaign, which is what the "
                      f"{arm!r} arm claims: batch slots without their own drafter")
    elif confirm:
        out["verified"] = True
        out["why"] = f"the engine reported no drafts; the operator confirmed --batch-mtp: {confirm}"
    else:
        out["why"] = ("no draft was reported on any request of this campaign, so --batch-mtp cannot be told from "
                      "a plain --batch 2 server from outside; start the server with --batch-mtp and --mtp, or "
                      "pass --confirm-batch-mtp \"how you checked\"")
    return out


def slot_overlap(samples: list, gap: float) -> dict:
    """The seconds two or more slots were non-idle in two samples in a row, from `live.slots[].state` only.

    `live.running` counts requests in flight, and a request can be in flight without holding a slot (waiting, or
    alone on a solo path), so it is reported as `inflight_overlap_s` and never used as slot evidence.  A sample
    with no `live.slots` at all (a server running no batch slots) is neither: it is not evidence of zero."""
    def span(rows, want, key):
        """Seconds between two samples in a row that both saw `want` or more, with the clock of the samples.
        A sample that does not see it breaks the run, and a gap wider than `gap` is a hole in the evidence,
        not time to count."""
        total, prev_at = 0.0, None
        for s in rows:
            v = s.get(key)
            hit = isinstance(v, (int, float)) and v >= want
            if hit and prev_at is not None and 0 < s["at"] - prev_at <= gap:
                total += s["at"] - prev_at
            prev_at = s["at"] if hit else None
        return round(total, 3)

    slot_rows = [s for s in samples if s.get("slot_state_source") == "live.slots"]
    return {"slot_samples": len(slot_rows),
            "max_nonidle_slots": max([s["nonidle_slots"] or 0 for s in slot_rows], default=0),
            "slot_overlap_s": span(slot_rows, 2, "nonidle_slots"),
            "single_slot_s": span(slot_rows, 1, "nonidle_slots"),
            "slot_state_source": "live.slots" if slot_rows else "none: the server answered no live.slots",
            "inflight_samples": len(samples),
            "inflight_overlap_s": span(samples, 2, "running"),
            "max_queued": max([s.get("queued") or 0 for s in samples], default=0),
            "max_running": max([s.get("running") or 0 for s in samples], default=0)}


def hardware_peak(samples: list, status_after: dict) -> dict:
    """The peaks of the sampled rows.  Every number is already in the unit its key names: /metrics bytes were
    converted in compact_sample, /v1/status `machine` arrives in MiB and GiB."""
    def peak(key):
        vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float))]
        return round(max(vals), 2) if vals else None

    gpu, ram = machine_part(status_after, "gpu"), machine_part(status_after, "ram")
    return {"vram_used_mib": peak("vram_used_mib") or gpu.get("used_mib"),
            "vram_total_mib": peak("vram_total_mib") or gpu.get("total_mib"),
            "ram_used_gib": peak("ram_used_gib") or ram.get("used_gib"),
            "ram_total_gib": peak("ram_total_gib") or ram.get("total_gib"),
            "gpu_power_w": peak("gpu_power_w") or gpu.get("power_w"),
            "gpu_util": peak("gpu_util"), "samples": len(samples),
            "units": "MiB and GiB: /metrics hardware is bytes and is converted here; /v1/status machine is "
                     "already MiB and GiB"}


def median(vals: list):
    vals = sorted(v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else round((vals[n // 2 - 1] + vals[n // 2]) / 2, 3)


def run_campaign(url: str, api_key: str, specs: list, opts: dict, cancel_lane: str = "",
                 cancel_after: float = 0.0, campaign_id: str = "") -> dict:
    """Start every lane at one instant, run them, and gather the evidence.

    A lane that raises is a failed lane, not a failed campaign: its row is written with the error and the other
    lane's evidence survives.  A model's tool arguments are untrusted input and have crashed a lane before."""
    started_wall = time.time()
    t0 = time.monotonic()
    deadline = t0 + opts["campaign_deadline"]
    sampler = Sampler(url, api_key, opts["sample_interval"], opts["timeout"])
    sampler.start()
    barrier = t0 + 0.35                                   # both threads release together, after any body building
    lanes, threads = {}, {}

    def go(spec):
        try:
            lanes[spec["lane"]] = run_lane(url, spec, opts, barrier, cancel_lane, cancel_after, campaign_id,
                                           deadline)
        except BaseException as e:                      # noqa: BLE001  a model's tool arguments are untrusted input
            print(red(f"[pi_c1c2] lane {spec['lane']} raised {type(e).__name__}: {e}"), file=sys.stderr)
            lanes[spec["lane"]] = failed_lane(spec, campaign_id, f"{type(e).__name__}: {e}")

    for s in specs:
        t = threading.Thread(target=go, args=(s,), daemon=True)
        threads[s["lane"]] = t
        t.start()
    for lane, t in threads.items():
        t.join(timeout=max(5.0, deadline - time.monotonic()) + 10.0)
        if t.is_alive():
            print(red(f"[pi_c1c2] lane {lane} is still running past the campaign deadline "
                      f"({opts['campaign_deadline']:.0f} s); its row is left as it stands"), file=sys.stderr)
    sampler.stop.set()
    time.sleep(0.05)
    samples = sampler.rows()
    status_after = get_json(url + "/v1/status", api_key, opts["timeout"])
    metrics_after = get_json(url + "/metrics", api_key, opts["timeout"])
    lanes_list = [lanes.get(s["lane"]) or failed_lane(s, campaign_id, "the lane produced no row") for s in specs]
    queue_evidence = correlate_queue_delay(url, api_key, lanes_list, opts["timeout"])

    # the decision metric: whole tasks per wall-clock minute.  Partial credit is summed beside it, never into it.
    completed = sum(1 for l in lanes_list if l["task_complete"])
    partial = round(sum(l["partial_credit"] for l in lanes_list), 3)
    wall_s = max(1e-6, time.time() - started_wall)
    gen = sum(l["generated_tokens"] for l in lanes_list)
    decode_s = sum(l["decode_ms_total"] for l in lanes_list) / 1000.0
    prompt_read = sum(max(0, (l["prompt_tokens"] or 0) - (l["reused_tokens"] or 0)) for l in lanes_list)
    prompt_s = sum(l["prompt_ms_total"] for l in lanes_list) / 1000.0
    overlap = slot_overlap(samples, opts["sample_interval"] * 3)
    mtp = batch_mtp_evidence(lanes_list, opts["arm"], opts.get("confirm_batch_mtp", ""))
    cancelled = next((l for l in lanes_list if l["lane"] == cancel_lane), None) if cancel_lane else None
    survivors = [l for l in lanes_list if l["lane"] != cancel_lane]
    return {
        "schema": 1,
        "run": {"arm": opts["arm"], "context_class": opts["context_class"], "repeat": opts["repeat"],
                "campaign_id": campaign_id, "started_at": round(started_wall, 3),
                "campaign_wall_s": round(wall_s, 3), "campaign_deadline_s": opts["campaign_deadline"],
                "url": url, "model": opts["model"], "lanes": len(specs),
                "batch_mtp_requested": ARMS[opts["arm"]]["batch_mtp"], "declared_tools": opts["declared"],
                "max_tokens": opts["max_tokens"], "max_steps": opts["max_steps"], "temperature": 0.0,
                "tool_char_cap": [l["tool_char_cap"] for l in lanes_list],
                "sampler_errors": sampler.errors, "sampler_rows_dropped": sampler.dropped},
        "guard": opts["guard"],
        "batch_mtp": mtp,
        "lanes": lanes_list,
        "cancellation": ({"lane": cancel_lane, "after_s": cancel_after,
                          "cancelled_lane_stopped": bool(cancelled) and
                          (cancelled.get("stopped") or "").startswith("cancelled"),
                          "survivor_ok": bool(survivors) and all(l["task_complete"] for l in survivors)}
                         if cancel_lane else None),
        "server_evidence": {
            "queue_delay_source": queue_evidence["correlation"],
            "queue_delay": queue_evidence,
            "max_queued": overlap["max_queued"], "max_running": overlap["max_running"],
            "max_nonidle_slots": overlap["max_nonidle_slots"], "slot_overlap_s": overlap["slot_overlap_s"],
            "single_slot_s": overlap["single_slot_s"], "slot_state_source": overlap["slot_state_source"],
            "inflight_overlap_s": overlap["inflight_overlap_s"], "slot_samples": overlap["slot_samples"],
            "hardware_peak": hardware_peak(samples, status_after),
            "status_after": status_after if isinstance(status_after, dict) else {"error": status_after},
            "metrics_after": {k: metrics_after.get(k) for k in
                              ("engine", "live", "totals", "hardware", "hardware_static", "time", "requests_kept")}
            if isinstance(metrics_after, dict) else {},
            "samples": samples},
        "decision_metric": {
            "tasks_completed": completed, "tasks_possible": len(specs), "partial_credit": partial,
            "useful_work_per_minute": round(completed / (wall_s / 60.0), 3),
            "partial_credit_per_minute": round(partial / (wall_s / 60.0), 3),
            "per_lane_decode_tok_s": median([l["decode_tok_s_server"] for l in lanes_list]),
            "aggregate_decode_tok_s": round(gen / wall_s, 1) if gen else None,
            "sum_of_lane_decode_s": round(decode_s, 3),
            "per_slot_decode_tok_s": round(gen / decode_s, 1) if decode_s > 0 and gen else None,
            "aggregate_prompt_tok_s": round(prompt_read / prompt_s, 1) if prompt_s > 0 and prompt_read else None,
            "note": "tasks_completed counts whole tasks only and partial_credit is reported beside it, never "
                    "added to it. per_lane_decode_tok_s is the median of each lane's own decoded tokens over its "
                    "own decode seconds summed over its steps; aggregate_decode_tok_s is the campaign's tokens "
                    "over the campaign's wall seconds; per_slot_decode_tok_s is those tokens over the sum of the "
                    "lanes' decode seconds."},
        "not_measured": [
            "the server's slot-transition latency: /metrics carries no timestamp per transition.  What is recorded "
            "instead is each lane's own client TTFT, the server's per-request queue_s when it can be tied to a "
            "lane, and the seconds two slots were non-idle in a row",
            "a per-lane admission wait: the server keeps no per-lane admission clock, so admission_s is null and "
            "queue delay is per request or unavailable",
            "the instant a request reached the engine: the server records its own clock, which is not the "
            "harness's clock, so the two are never subtracted across machines"]}


def plan_text(arm: str, context_class: str, specs: list, opts: dict, cancel_lane: str, cancel_after: float) -> str:
    lines = [
        f"arm               {arm} (the server must report concurrency.serving={ARMS[arm]['serving']}, "
        f"--batch-mtp {'on' if ARMS[arm]['batch_mtp'] else 'off'})",
        f"context class     {context_class}",
        f"lanes             {len(specs)} started at one instant"
        + (f" (asked --lanes {opts['lanes']}: only {len(specs)} session name(s) given)"
           if len(specs) < opts["lanes"] else ""),
        "tools             " + ", ".join(DECLARED) + "  (anything else ends the lane)",
        f"bounds            max_tokens {opts['max_tokens']}, max_steps {opts['max_steps']}, "
        f"tool result {opts['max_tool_chars']} chars, socket timeout {opts['timeout']} s, campaign deadline "
        f"{opts['campaign_deadline']:.0f} s, sample every {opts['sample_interval']} s",
        "cancellation      " + (f"lane {cancel_lane} is cut off at {cancel_after} s; the other lane must finish"
                                if cancel_lane else "off"),
        f"repeat            {opts['repeat']}   order tag {opts['tag'] or '-'}",
        "",
        "lanes:",
    ]
    for s in specs:
        est = int(s["filler_words"] * TOKENS_PER_FILLER_WORD)
        lines.append(f"  lane {s['lane']}  session {s['session']:<14} filler {s['filler_words']} words "
                     f"(~{est} prompt tokens; the server's own count is what the evidence records)")
        lines.append(f"        task: {s['task'][:110]}...")
    lines += ["", "nothing is sent and nothing is written in a dry run"]
    return red("\n".join(lines))


def render_markdown(rep: dict) -> str:
    red_ = _RED
    ev, d, g = rep["server_evidence"], rep["decision_metric"], rep["guard"]
    h = ev["hardware_peak"]
    lines = [f"# Pi C1/C2 run: arm `{rep['run']['arm']}`, context class `{rep['run']['context_class']}`",
             "",
             f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rep['run']['started_at']))}.  "
             f"Campaign wall {rep['run']['campaign_wall_s']} s.  Model `{rep['run']['model']}`.  "
             f"Server reported `concurrency.serving={g['serving_observed']}` before any traffic "
             f"(and `{(rep['server_evidence'].get('status_after') or {}).get('concurrency', {}).get('serving')}` "
             f"after it).  Engine `{g.get('engine_version')}`, window {g.get('max_context')} tokens.",
             "",
             f"**Decision: {d['tasks_completed']}/{d['tasks_possible']} tasks completed, "
             f"{d['useful_work_per_minute']} completed tasks per wall-clock minute.**  Partial credit "
             f"{d['partial_credit']} ({d['partial_credit_per_minute']}/min) is reported and is not counted as a "
             f"completed task.",
             "",
             "| quantity | value | where it comes from |",
             "|---|---|---|",
             f"| tasks completed | {d['tasks_completed']}/{d['tasks_possible']} | all six fixed checks passed |",
             f"| partial credit | {d['partial_credit']} | checks passed, summed; never counted as done |",
             f"| per-lane decode | {d['per_lane_decode_tok_s']} tok/s | each lane's own tokens over its own "
             f"decode seconds, summed over its steps (server `timings`) |",
             f"| aggregate decode | {d['aggregate_decode_tok_s']} tok/s | the campaign's tokens over the "
             f"campaign's wall seconds |",
             f"| per-slot decode | {d['per_slot_decode_tok_s']} tok/s | those tokens over the sum of the lanes' "
             f"decode seconds ({d['sum_of_lane_decode_s']} s) |",
             f"| aggregate prompt | {d['aggregate_prompt_tok_s']} tok/s | tokens read over prompt seconds "
             f"(server `timings`) |",
             f"| max queued | {ev['max_queued']} | /metrics live.queued |",
             f"| max running | {ev['max_running']} | /metrics live.running (requests, not slots) |",
             f"| max non-idle slots | {ev['max_nonidle_slots']} | /metrics live.slots[].state "
             f"({ev['slot_state_source']}) |",
             f"| two-slot overlap | {ev['slot_overlap_s']} s | /metrics live.slots[].state, two samples in a row |",
             f"| in-flight overlap | {ev['inflight_overlap_s']} s | /metrics live.running, two samples in a row - "
             f"requests, not slots, so not slot evidence |",
             f"| VRAM peak | {h['vram_used_mib']} / {h['vram_total_mib']} MiB | /metrics hardware (bytes, "
             f"converted) |",
             f"| RAM peak | {h['ram_used_gib']} / {h['ram_total_gib']} GiB | /metrics hardware (bytes, "
             f"converted) |",
             f"| GPU power peak | {h['gpu_power_w']} W | /metrics hardware |",
             f"| batch MTP | {'verified' if rep['batch_mtp']['verified'] else 'NOT verified'} | "
             f"{red_(rep['batch_mtp']['why'])} |",
             f"| queue delay | {ev['queue_delay_source']} | {red_(ev['queue_delay']['correlation'])} |",
             ""]
    if rep["batch_mtp"]["verified"] is False:
        lines += [f"> **The arm label was not confirmed:** {red_(rep['batch_mtp']['why'])}", ""]
    if g["mismatch"]:
        lines += ["> **Guard:** " + red_("; ".join(g["mismatch"]))]
        if g.get("serving_observed") != g.get("serving_expected"):
            lines += [f"> The server was not the arm's: asked for {g['serving_expected']} serving slots, it "
                      f"reported {g['serving_observed']} before any traffic."]
        lines += [""]
    lines += ["## Lanes",
              "",
              "| lane | session | TTFT s | queue s | model TTFT s | queue source | tok/s (lane, server) "
              "| prompt tok | gen tok | drafts | tool calls | reads | complete | checks | partial | stopped |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in rep["lanes"]:
        lines.append(f"| {l['lane']} | {l['session']} | {l['ttft_s']} | {l['queue_delay_s']} | "
                     f"{l['model_ttft_s']} | {red_(str(l['queue_delay_source']))} | "
                     f"{l['decode_tok_s_server']} | {l['prompt_tokens']} | {l['generated_tokens']} | "
                     f"{l['drafts_offered']} | {l['tool_calls']} | {len(l['reads'])} | "
                     f"{'yes' if l['task_complete'] else 'no'} | {l['checks_passed']}/{l['checks_total']} | "
                     f"{l['partial_credit']} | {l['stopped']} |")
    lines += ["", "## Fixed checks", "",
              "| lane | " + " | ".join(k.replace("_", " ") for k in CHECKS_ORDER) +
              " |",
              "|---" * (len(CHECKS_ORDER) + 1) + "|"]
    for l in rep["lanes"]:
        cells = ["yes" if l["checks"].get(k) else "no" for k in CHECKS_ORDER]
        lines.append(f"| {l['lane']} | " + " | ".join(cells) + " |")
    if rep["cancellation"]:
        c = rep["cancellation"]
        lines += ["", "## Cancellation", "",
                  f"Lane `{c['lane']}` was cut off at {c['after_s']} s.  It stopped: "
                  f"{'yes' if c['cancelled_lane_stopped'] else 'no'}.  The other lane finished its task: "
                  f"{'yes' if c['survivor_ok'] else 'no'}.", ""]
    errs = [(l["lane"], e) for l in rep["lanes"] for e in l["errors"]]
    if errs:
        lines += ["## Errors", ""] + [f"- lane {l}: {red_(json.dumps(e))}" for l, e in errs] + [""]
    bad = [(l["lane"], t) for l in rep["lanes"] for t in l["invalid_arguments"]]
    trunc = [(l["lane"], t) for l in rep["lanes"] for t in l["truncated_tool_calls"]]
    if bad or trunc:
        lines += ["## Tool calls the harness could not use", ""]
        lines += [f"- lane {l}: {red_(json.dumps(t))} (arguments were not an object)" for l, t in bad]
        lines += [f"- lane {l}: {red_(json.dumps(t))} (the stream ended inside the call, issue #211)"
                  for l, t in trunc]
        lines += [""]
    lines += ["## What is not measured", ""] + [f"- {red_(x)}" for x in rep["not_measured"]] + [""]
    return "\n".join(lines)


def build_specs(names: list, context_class: str, lanes: int) -> list:
    """The lanes of this campaign: one session each, distinct sessions, in lane A, B order."""
    out = []
    for i, name in enumerate(names[:lanes]):
        spec = dict(TASKS[name])
        spec.update({"lane": chr(ord("A") + i), "session": name,
                     "filler_words": CONTEXT_CLASSES[context_class], "context_class": context_class})
        out.append(spec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="sessions: " + ", ".join(TASKS) + "\narms: " + ", ".join(ARMS) +
                                       "\ncontext classes: " + ", ".join(CONTEXT_CLASSES))
    ap.add_argument("--url", default="http://127.0.0.1:8181")
    ap.add_argument("--api-key", default="", help="the server's API key (prefer --api-key-env, so it does not "
                                                  "appear in a command line)")
    ap.add_argument("--api-key-env", default="STRATA_API_KEY",
                    help="environment variable read for the key when --api-key is empty (default STRATA_API_KEY)")
    ap.add_argument("--allow-remote", action="store_true",
                    help="allow a non-loopback --url over http://, which carries the key in clear text")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--expect-serving", type=int, default=0,
                    help="refuse to measure unless the server reports this concurrency.serving (default: what "
                         "the arm claims)")
    ap.add_argument("--model", default="", help="refuse to measure unless the server has this model")
    ap.add_argument("--sessions", default=",".join(list(TASKS)[:2]),
                    help="one per lane, in lane order, distinct (this harness knows: " + ", ".join(TASKS) +
                         ")")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--context-class", default="short", choices=sorted(CONTEXT_CLASSES))
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--max-tool-result-chars", type=int, default=MAX_TOOL_RESULT_CHARS)
    ap.add_argument("--timeout", type=float, default=600.0, help="a socket read may go quiet this long")
    ap.add_argument("--campaign-deadline", type=float, default=1800.0,
                    help="the whole campaign's wall-clock bound: a lane that passes it is written as incomplete")
    ap.add_argument("--sample-interval", type=float, default=0.2)
    ap.add_argument("--cancel-lane", default="", help="the lane letter to cancel mid-request (the fixture)")
    ap.add_argument("--cancel-after", type=float, default=20.0)
    ap.add_argument("--confirm-batch-mtp", default="",
                    help="state how --batch-mtp was checked, for an engine that reports no drafts; recorded in "
                         "the run file")
    ap.add_argument("--allow-context-overflow", action="store_true",
                    help="measure a context class the server's window cannot hold (the run will show errors)")
    ap.add_argument("--output-dir", default="bench/results")
    ap.add_argument("--tag", default="", help="a word for the file name: which server, which order, which try")
    ap.add_argument("--repeat", type=int, default=1, help="which controlled repetition this is (the gate counts "
                                                          "distinct repeats, not files)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and send nothing")
    a = ap.parse_args()

    def die(msg: str) -> int:
        print(red(f"[pi_c1c2] refusing: {msg}"), file=sys.stderr)
        return 2

    api_key = a.api_key or os.environ.get(a.api_key_env or "", "")
    if api_key:
        use_key_for_redaction(api_key)
    host = (urllib.parse.urlsplit(a.url).hostname or "").lower()
    if not a.url.lower().startswith("https://") and host not in ("127.0.0.1", "localhost", "::1"):
        return die(f"{a.url} is not loopback and is plain http: the API key would cross the network in clear "
                   "text.  Use https, or --allow-remote if you meant it")

    names = [x.strip() for x in a.sessions.split(",") if x.strip()]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        return die(f"unknown session(s) {unknown}; this harness knows {sorted(TASKS)}")
    if len(set(names)) != len(names):
        return die(f"--sessions repeats a name: two lanes on one session share one conversation, which is not "
                   "two independent Pi sessions")
    if a.lanes < 1:
        return die("--lanes must be at least 1")
    specs = build_specs(names, a.context_class, a.lanes)
    if len(specs) < 2 and not a.dry_run:
        return die("a C1/C2 campaign needs two lanes and two distinct sessions: give --sessions two names "
                   "(the C1 arm runs both lanes too - one runs while the other waits, and that is the baseline)")
    lane_letters = [s["lane"] for s in specs]
    if a.cancel_lane and a.cancel_lane not in lane_letters:
        return die(f"--cancel-lane {a.cancel_lane!r} is not one of this campaign's lanes {lane_letters} "
                   f"(--lanes {a.lanes}, {len(specs)} session name(s))")
    if a.cancel_lane and len(specs) < 2:
        return die("the cancellation fixture needs a second lane to survive")
    if a.cancel_after <= 0:
        return die("--cancel-after must be positive")
    if a.max_steps < 2:
        return die("--max-steps must be at least 2: a lane has to read before it submits")
    if a.sample_interval < 0.05:
        return die("--sample-interval below 0.05 s would load the server more than the lanes it measures")
    if a.campaign_deadline < 30:
        return die("--campaign-deadline below 30 s cannot hold a campaign of these tasks")

    expect = a.expect_serving or ARMS[a.arm]["serving"]
    if expect != ARMS[a.arm]["serving"]:
        return die(f"--expect-serving {expect} contradicts --arm {a.arm}, which is "
                   f"{ARMS[a.arm]['serving']}-serving: the two arms differ only in what the server was started "
                   "with, and the guard checks that")

    if a.dry_run:
        opts = {"lanes": a.lanes, "max_tokens": a.max_tokens, "max_steps": a.max_steps,
                "max_tool_chars": a.max_tool_result_chars, "timeout": a.timeout,
                "campaign_deadline": a.campaign_deadline, "sample_interval": a.sample_interval,
                "repeat": a.repeat, "tag": a.tag}
        print(plan_text(a.arm, a.context_class, specs, opts, a.cancel_lane, a.cancel_after))
        return 0

    guard_opts = {"arm": a.arm, "repeat": a.repeat, "tag": a.tag, "url": a.url, "model": a.model or "(whatever "
                  "the server has)", "declared": DECLARED, "max_tokens": a.max_tokens, "max_steps": a.max_steps,
                  "max_tool_chars": a.max_tool_result_chars, "timeout": a.timeout,
                  "campaign_deadline": a.campaign_deadline, "sample_interval": a.sample_interval,
                  "api_key": api_key, "context_class": a.context_class,
                  "confirm_batch_mtp": a.confirm_batch_mtp}
    st = get_json(a.url + "/v1/status", api_key, min(30.0, a.timeout))
    mt = get_json(a.url + "/metrics", api_key, min(30.0, a.timeout))
    st = st if isinstance(st, dict) and not st.get("error") else None
    mt = mt if isinstance(mt, dict) and not mt.get("error") else None
    g = guard(st, mt, a.arm, expect, a.model)
    guard_opts["guard"] = g
    if g["mismatch"]:
        for m in g["mismatch"]:
            print(red(f"[pi_c1c2] guard: {m}"), file=sys.stderr)
        return 2

    # keep the conversation inside the window the server reports: the class sizes the first prompt, and the
    # reads come out of what is left
    max_ctx = g.get("max_context") or 0
    prompt_est = int(specs[0]["filler_words"] * TOKENS_PER_FILLER_WORD)
    if max_ctx and prompt_est >= 0.9 * max_ctx and not a.allow_context_overflow:
        return die(f"context class {a.context_class} is about {prompt_est} prompt tokens and this server's window "
                   f"is {max_ctx}: the request would be refused by the server, not by the model.  Measure a "
                   "smaller class, or pass --allow-context-overflow to record that anyway")
    for s in specs:
        s["prompt_tokens_est"] = int(s["filler_words"] * TOKENS_PER_FILLER_WORD)
    guard_opts["max_tool_chars"] = tool_char_cap(prompt_est, max_ctx, a.max_steps, a.max_tool_result_chars)
    if guard_opts["max_tool_chars"] < a.max_tool_result_chars:
        print(red(f"[pi_c1c2] tool results capped at {guard_opts['max_tool_chars']} chars so {a.max_steps} reads "
                  f"cannot pass the server's {max_ctx}-token window (asked "
                  f"{a.max_tool_result_chars}; class {a.context_class} is ~{prompt_est} tokens)"))

    campaign_id = f"{int(time.time())}-{os.getpid()}"
    print(red(f"[pi_c1c2] arm {a.arm} {a.context_class}: {len(specs)} lanes on {a.url}, model "
              f"{g['model']!r}, serving {g['serving_observed']}, window {max_ctx}, repeat {a.repeat}"))
    report = run_campaign(a.url, api_key, specs, guard_opts, cancel_lane=a.cancel_lane,
                          cancel_after=a.cancel_after, campaign_id=campaign_id)
    report = redact_obj(report, _RED)          # the whole report is redacted before anything is written
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(report["run"]["started_at"]))
    stem = f"run-{a.arm}-{a.context_class}-{stamp}-r{a.repeat}" + (f"-{a.tag}" if a.tag else "")
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    d, ev = report["decision_metric"], report["server_evidence"]
    print(red(f"[pi_c1c2] arm {a.arm} {a.context_class}: wall {report['run']['campaign_wall_s']} s, tasks "
              f"{d['tasks_completed']}/{len(specs)} completed (partial {d['partial_credit']}), per-lane decode "
              f"{d['per_lane_decode_tok_s']} tok/s, aggregate {d['aggregate_decode_tok_s']} tok/s, peak non-idle "
              f"slots {ev['max_nonidle_slots']}, slot overlap {ev['slot_overlap_s']} s, queue delay "
              f"{ev['queue_delay_source']}"))
    for lane in report["lanes"]:
        print(red(f"[pi_c1c2]   lane {lane['lane']} {lane['session']}: TTFT {lane['ttft_s']} s "
                  f"(queue {lane['queue_delay_s']} s, model {lane['model_ttft_s']} s), "
                  f"{lane['decode_tok_s_server']} tok/s over {lane['decode_ms_total']} ms of decode, "
                  f"complete {'yes' if lane['task_complete'] else 'no'} "
                  f"({lane['checks_passed']}/{lane['checks_total']}, partial {lane['partial_credit']}), "
                  f"stopped {lane['stopped']}"))
    print(red(f"[pi_c1c2] batch MTP: {'verified' if report['batch_mtp']['verified'] else 'NOT verified'} - "
              f"{report['batch_mtp']['why']}"))
    print(red(f"[pi_c1c2] wrote {out_dir / stem}.json and .md"))
    if not report["batch_mtp"]["verified"]:
        return 4
    if any(l["errors"] for l in report["lanes"]) or any(l["stopped"] == "harness_error" for l in report["lanes"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
