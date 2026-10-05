#!/usr/bin/env python3
"""Compare RTX 5080 C1/C2 JSON evidence emitted by rtx5080_openclaw_bench.py."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def med(values):
    values = [float(x) for x in values if isinstance(x, (int, float))]
    return statistics.median(values) if values else None


def rng(values):
    values = [float(x) for x in values if isinstance(x, (int, float))]
    return (min(values), max(values)) if values else (None, None)


def load(paths: list[str]) -> list[dict[str, Any]]:
    reports = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for p in matches:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            if r.get("schema") != "strata-rtx5080-c1c2-v1":
                print(f"skip {p}: unexpected schema {r.get('schema')!r}", file=sys.stderr)
                continue
            r["_path"] = p
            reports.append(r)
    return reports


def arm_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [r.get("campaign_wall_s") for r in items]
    ttfts = [
        q.get("ttft_s")
        for r in items
        for q in r.get("requests", [])
        if q.get("ok") and isinstance(q.get("ttft_s"), (int, float))
    ]
    vrams = []
    failed = 0
    serving_mismatch = 0
    max_slots = 0
    for r in items:
        if not all(q.get("ok") for q in r.get("requests", [])):
            failed += 1
        reported = (r.get("status_before", {}).get("concurrency", {}) or {}).get("serving")
        if reported != r.get("expect_serving"):
            serving_mismatch += 1
        max_slots = max(max_slots, int(r.get("metrics_observations", {}).get("max_nonidle_slots") or 0))
        v = r.get("gpu_summary", {}).get("memory_used_mib", {}).get("max")
        if isinstance(v, (int, float)):
            vrams.append(v)
    lo, hi = rng(walls)
    return {
        "n": len(items),
        "campaign_median_s": med(walls),
        "campaign_min_s": lo,
        "campaign_max_s": hi,
        "ttft_median_s": med(ttfts),
        "vram_peak_median_mib": med(vrams),
        "failed_runs": failed,
        "serving_mismatch_runs": serving_mismatch,
        "max_nonidle_slots_seen": max_slots,
    }


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", help="JSON files or shell-style glob patterns")
    ap.add_argument("--output", help="optional Markdown output path")
    a = ap.parse_args()

    reports = load(a.reports)
    if not reports:
        print("ERROR: no compatible reports", file=sys.stderr)
        return 2

    groups: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in reports:
        groups[int(r.get("target_prompt_tokens") or 0)][str(r.get("arm"))].append(r)

    lines = [
        "# RTX 5080 C1/C2 comparison",
        "",
        "| Prompt target | Arm | Runs | Campaign median s | Range s | TTFT median s | Peak VRAM median MiB | Failures | Max slots seen |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    exit_code = 0
    comparisons = []
    for target in sorted(groups):
        stats = {arm: arm_stats(items) for arm, items in groups[target].items()}
        for arm in ("c1", "c2", "c4"):
            if arm not in stats:
                continue
            s = stats[arm]
            lines.append(
                f"| {target or 'short'} | {arm.upper()} | {s['n']} | "
                f"{fmt(s['campaign_median_s'], 3)} | "
                f"{fmt(s['campaign_min_s'], 3)}–{fmt(s['campaign_max_s'], 3)} | "
                f"{fmt(s['ttft_median_s'], 3)} | {fmt(s['vram_peak_median_mib'], 0)} | "
                f"{s['failed_runs']} | {s['max_nonidle_slots_seen']} |"
            )
            if s["failed_runs"] or s["serving_mismatch_runs"]:
                exit_code = 1

        c1, c2 = stats.get("c1"), stats.get("c2")
        if c1 and c2 and c1["campaign_median_s"] and c2["campaign_median_s"]:
            speedup = c1["campaign_median_s"] / c2["campaign_median_s"]
            reduction = 100.0 * (1.0 - c2["campaign_median_s"] / c1["campaign_median_s"])
            comparisons.append((target, speedup, reduction, c1, c2))

    lines += ["", "## C2 vs C1", ""]
    if not comparisons:
        lines.append("No context target has both C1 and C2 evidence yet.")
    for target, speedup, reduction, c1, c2 in comparisons:
        lines += [
            f"### {target or 'short'}",
            "",
            f"- Two-worker campaign speedup: **{speedup:.3f}x**",
            f"- Wall-time reduction: **{reduction:.1f}%**",
            f"- C1 median TTFT: **{fmt(c1['ttft_median_s'], 3)} s**",
            f"- C2 median TTFT: **{fmt(c2['ttft_median_s'], 3)} s**",
            f"- C2 max non-idle slots observed: **{c2['max_nonidle_slots_seen']}**",
            "",
        ]

    lines += [
        "## Interpretation gate",
        "",
        "This script does not emit PROMOTE/REJECT automatically. Issue #2 also requires correctness, "
        "memory headroom, long-context behaviour and agent/tool reliability. A faster median is evidence, "
        "not authorization for an engine change.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
        print(f"\nWROTE={a.output}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
