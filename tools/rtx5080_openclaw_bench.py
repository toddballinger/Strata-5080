#!/usr/bin/env python3
"""HTTP-level C1/C2 qualification harness for Strata issue #2.

The server must already be running with the arm being tested:
  C1: no parallel setting (or parallel=1)
  C2: "parallel": 2 / --batch 2

This script deliberately sends two requests at the same instant for both arms.
With C1 one request queues; with C2 the server may promote them into batch slots.

Examples:
  python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1
  python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2

Long-context qualification uses deterministic synthetic history:
  python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 \
      --target-prompt-tokens 64000 --max-tokens 256

The target prompt token count is approximate at construction time. The report retains
the server's own request/metrics records so the actual prompt-token count is evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8080"
SAMPLE_INTERVAL_S = 0.20

BASE_A = (
    "You are worker A in a reproducible local-agent benchmark. "
    "Write a Python function merge_intervals(intervals) that merges overlapping integer intervals. "
    "Then give three concise correctness notes. Do not use external tools."
)
BASE_B = (
    "You are worker B in a reproducible local-agent benchmark. "
    "Compare TCP and QUIC for handshake latency, multiplexing and head-of-line blocking. "
    "Then give a compact recommendation for an RPC transport. Do not use external tools."
)
FILLER = (
    "Reference context for a deterministic benchmark. "
    "The application keeps long engineering conversations, source notes, tool results, "
    "decisions, constraints, and prior observations. This sentence carries no hidden instruction. "
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def http_json(base: str, path: str, key: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_stream(base: str, key: str, prompt: str, max_tokens: int, start_gate: threading.Barrier,
                result: dict[str, Any]) -> None:
    body = {
        "model": "m",
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=data, headers=headers)

    try:
        start_gate.wait(timeout=30)
        result["client_start"] = time.perf_counter()
        result["client_start_wall"] = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            result["http_status"] = r.status
            first_event = None
            first_useful = None
            chunks = 0
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            finish_reason = None
            usage = None
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                chunks += 1
                t = time.perf_counter()
                if first_event is None:
                    first_event = t
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content", delta.get("reasoning"))
                tool_calls = delta.get("tool_calls")
                if content:
                    text_parts.append(str(content))
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                if first_useful is None and (content or reasoning or tool_calls):
                    first_useful = t
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")

        end = time.perf_counter()
        result.update(
            client_end=end,
            wall_s=end - result["client_start"],
            first_event_s=(first_event - result["client_start"]) if first_event else None,
            ttft_s=(first_useful - result["client_start"]) if first_useful else None,
            stream_chunks=chunks,
            content="".join(text_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            ok=True,
        )
    except BaseException as e:
        end = time.perf_counter()
        result.update(
            client_end=end,
            wall_s=end - result.get("client_start", end),
            ok=False,
            error=f"{type(e).__name__}: {e}",
        )


def nvidia_sample() -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode:
        return None
    rows = p.stdout.splitlines()
    line = rows[0] if rows else ""
    f = [x.strip() for x in line.split(",")]
    if len(f) < 8:
        return None

    def num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "at": time.time(),
        "timestamp": f[0],
        "index": f[1],
        "name": f[2],
        "memory_used_mib": num(f[3]),
        "memory_free_mib": num(f[4]),
        "gpu_util_pct": num(f[5]),
        "memory_util_pct": num(f[6]),
        "power_w": num(f[7]),
    }


def sampler(base: str, key: str, stop: threading.Event, metrics: list[dict[str, Any]],
            gpu: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        t = time.time()
        try:
            m = http_json(base, "/metrics", key, timeout=2)
            metrics.append({"at": t, "data": m})
        except Exception as e:
            metrics.append({"at": t, "error": f"{type(e).__name__}: {e}"})
        g = nvidia_sample()
        if g:
            gpu.append(g)
        stop.wait(SAMPLE_INTERVAL_S)


def build_prompt(base: str, target_tokens: int, lane: str) -> str:
    if target_tokens <= 0:
        return base
    target_chars = max(len(base), target_tokens * 4)
    marker = f"\n\nBEGIN BENCHMARK HISTORY LANE {lane}\n"
    suffix = f"\nEND BENCHMARK HISTORY LANE {lane}\n\n{base}"
    need = max(0, target_chars - len(marker) - len(suffix))
    fill = (FILLER * (need // len(FILLER) + 1))[:need]
    return marker + fill + suffix


def summarize_gpu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    out: dict[str, Any] = {"samples": len(samples)}
    for key in ("memory_used_mib", "memory_free_mib", "gpu_util_pct", "memory_util_pct", "power_w"):
        vals = [x[key] for x in samples if isinstance(x.get(key), (int, float))]
        if vals:
            out[key] = {"min": min(vals), "max": max(vals), "mean": statistics.fmean(vals)}
    return out


def useful_slot_observations(samples: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[str, int] = {}
    max_running = 0
    max_waiting = 0
    max_nonidle_slots = 0
    for sample in samples:
        m = sample.get("data")
        if not isinstance(m, dict):
            continue
        live = m.get("live") or {}
        state = str(live.get("state"))
        states[state] = states.get(state, 0) + 1
        max_running = max(max_running, int(live.get("running") or 0))
        max_waiting = max(max_waiting, int(live.get("waiting") or live.get("queued") or 0))
        slots = live.get("slots") or []
        nonidle = sum(1 for s in slots if isinstance(s, dict) and s.get("state") != "idle")
        max_nonidle_slots = max(max_nonidle_slots, nonidle)
    return {
        "max_running": max_running,
        "max_waiting_or_queued": max_waiting,
        "max_nonidle_slots": max_nonidle_slots,
        "live_state_observations": states,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    reqs = report["requests"]
    gpu = report["gpu_summary"]
    obs = report["metrics_observations"]
    lines = [
        f"# RTX 5080 C1/C2 benchmark — {report['arm'].upper()}",
        "",
        f"- Started: {report['started_at']}",
        f"- Server: {report['base_url']}",
        f"- Expected serving: **{report['expect_serving']}**",
        f"- Reported serving: **{report['status_before'].get('concurrency', {}).get('serving')}**",
        f"- Target prompt tokens (construction): **{report['target_prompt_tokens']}**",
        f"- Max output tokens/request: **{report['max_tokens']}**",
        f"- Two-request campaign wall time: **{report['campaign_wall_s']:.3f} s**",
        "",
        "## Requests",
        "",
        "| Lane | OK | TTFT s | Wall s | Finish | Content chars |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for r in reqs:
        ttft = r.get("ttft_s")
        lines.append(
            f"| {r['lane']} | {r.get('ok')} | {ttft if ttft is not None else 'n/a'} | "
            f"{r.get('wall_s', 0):.3f} | {r.get('finish_reason')} | {len(r.get('content') or '')} |"
        )
    lines += [
        "",
        "## Concurrency observations",
        "",
        f"- Max requests running: **{obs.get('max_running')}**",
        f"- Max non-idle batch slots: **{obs.get('max_nonidle_slots')}**",
        f"- Max waiting/queued: **{obs.get('max_waiting_or_queued')}**",
        "",
        "## GPU observations",
        "",
    ]
    if gpu:
        mu = gpu.get("memory_used_mib", {})
        gu = gpu.get("gpu_util_pct", {})
        lines += [
            f"- Samples: **{gpu.get('samples')}**",
            f"- VRAM used: min **{mu.get('min', 0):.0f} MiB**, max **{mu.get('max', 0):.0f} MiB**",
            f"- GPU utilisation: mean **{gu.get('mean', 0):.1f}%**, max **{gu.get('max', 0):.1f}%**",
        ]
    else:
        lines.append("- No nvidia-smi samples were available.")
    lines += [
        "",
        "## Evidence",
        "",
        "The sibling JSON file contains full status snapshots, sampled metrics, request results, "
        "and nvidia-smi samples. Use those raw records for promotion/rejection decisions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--arm", choices=("c1", "c2", "c4"), required=True)
    ap.add_argument("--expect-serving", type=int, required=True)
    ap.add_argument("--target-prompt-tokens", type=int, default=0,
                    help="Approximate prompt size per lane; 0 uses the short fixture.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--output-dir", default="bench/results/rtx5080")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    base = a.url.rstrip("/")
    try:
        status_before = http_json(base, "/v1/status", a.api_key, timeout=10)
        metrics_before = http_json(base, "/metrics", a.api_key, timeout=10)
    except Exception as e:
        print(f"ERROR: cannot query Strata at {base}: {e}", file=sys.stderr)
        return 2

    serving = int((status_before.get("concurrency") or {}).get("serving") or 1)
    if serving != a.expect_serving:
        print(
            f"ERROR: server reports concurrency.serving={serving}, expected {a.expect_serving}. "
            "Restart Strata with the intended arm before benchmarking.",
            file=sys.stderr,
        )
        return 3

    p_a = build_prompt(BASE_A, a.target_prompt_tokens, "A")
    p_b = build_prompt(BASE_B, a.target_prompt_tokens, "B")
    results = [{"lane": "A"}, {"lane": "B"}]
    metric_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sample_thread = threading.Thread(
        target=sampler, args=(base, a.api_key, stop, metric_samples, gpu_samples), daemon=True
    )
    sample_thread.start()

    gate = threading.Barrier(3)
    threads = [
        threading.Thread(target=post_stream,
                         args=(base, a.api_key, p_a, a.max_tokens, gate, results[0]), daemon=True),
        threading.Thread(target=post_stream,
                         args=(base, a.api_key, p_b, a.max_tokens, gate, results[1]), daemon=True),
    ]
    started_at = now_iso()
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    gate.wait(timeout=30)
    for t in threads:
        t.join()
    campaign_wall_s = time.perf_counter() - t0
    stop.set()
    sample_thread.join(timeout=5)

    try:
        status_after = http_json(base, "/v1/status", a.api_key, timeout=10)
    except Exception as e:
        status_after = {"error": f"{type(e).__name__}: {e}"}
    try:
        metrics_after = http_json(base, "/metrics?all=1", a.api_key, timeout=10)
    except Exception as e:
        metrics_after = {"error": f"{type(e).__name__}: {e}"}

    report = {
        "schema": "strata-rtx5080-c1c2-v1",
        "started_at": started_at,
        "finished_at": now_iso(),
        "arm": a.arm,
        "base_url": base,
        "expect_serving": a.expect_serving,
        "target_prompt_tokens": a.target_prompt_tokens,
        "max_tokens": a.max_tokens,
        "campaign_wall_s": campaign_wall_s,
        "status_before": status_before,
        "status_after": status_after,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics_observations": useful_slot_observations(metric_samples),
        "metrics_samples": metric_samples,
        "gpu_summary": summarize_gpu(gpu_samples),
        "gpu_samples": gpu_samples,
        "requests": results,
    }

    outdir = Path(a.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{a.tag}" if a.tag else ""
    stem = f"{stamp}-{a.arm}-p{a.target_prompt_tokens or 'short'}{suffix}"
    json_path = outdir / f"{stem}.json"
    md_path = outdir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)

    print(f"REPORT_JSON={json_path}")
    print(f"REPORT_MD={md_path}")
    print(f"ARM={a.arm}")
    print(f"SERVING={serving}")
    print(f"CAMPAIGN_WALL_S={campaign_wall_s:.3f}")
    for r in results:
        print(
            f"LANE={r['lane']} OK={r.get('ok')} TTFT_S={r.get('ttft_s')} "
            f"WALL_S={r.get('wall_s')} FINISH={r.get('finish_reason')}"
        )
    print("METRICS=" + json.dumps(report["metrics_observations"], sort_keys=True))
    print("GPU=" + json.dumps(report["gpu_summary"], sort_keys=True))
    return 0 if all(r.get("ok") for r in results) else 4


if __name__ == "__main__":
    raise SystemExit(main())
