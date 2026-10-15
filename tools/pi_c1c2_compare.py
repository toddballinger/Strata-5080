"""Compare Pi C1/C2 lane runs written by tools/pi_c1c2_bench.py and recommend PROMOTE, DEFER or REJECT.

    python tools/pi_c1c2_compare.py bench/results/rtx5080-pi-c1c2/*.json
    python tools/pi_c1c2_compare.py 'bench/results/rtx5080-pi-c1c2/*.json' --min-repeats 3 --json compare.json

It reads the run files and does arithmetic: medians and ranges per arm and context class, the C2-to-C1 ratios for the
decision metric (useful completed work per wall-clock minute) and for the quantities that explain it - queue delay,
TTFT, per-lane decode, aggregate decode, peak VRAM, expert-cache hit rate, the seconds two slots were non-idle at
once.  Then it applies the Issue #2 gate: at least --min-repeats repetitions per arm, no correctness violation, no
fall in completed tasks, a C2 run that really held two slots, and a cancellation fixture whose surviving lane
finished.  Anything missing is DEFER with the reason named, not a guess.

The output is a **recommendation, not a decision**: the decision is made by hand in the issue, with the run files
next to it.  A C1-vs-C2 ratio alone does not separate batching from losing MTP drafts (batch slots decode without
drafts unless `--batch-mtp` is on), so when a `c2mtp` group is present it is compared against `c2` as well.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_load import pct                             # noqa: E402  the same percentile every other suite uses

SCHEMA = "strata-pi-c1c2-v1"
BASELINE = "c1"
# per-run fields the comparison reads, all written by pi_c1c2_bench.py
SERIES = (("useful_work_per_minute", ("decision_metric", "useful_work_per_minute")),
          ("campaign_wall_s", ("run", "campaign_wall_s")),
          ("tasks_completed", ("decision_metric", "tasks_completed")),
          ("useful_work", ("decision_metric", "useful_work")),
          ("aggregate_decode_tok_s", ("decision_metric", "aggregate_decode_tok_s")),
          ("median_lane_decode_tok_s", ("decision_metric", "median_per_lane_decode_tok_s")),
          ("last_lane_first_useful_s", ("decision_metric", "last_lane_first_useful_s")),
          ("max_queue_delay_s", ("lanes", "queue_delay_s")),
          ("max_ttft_s", ("lanes", "ttft_s")),
          ("max_model_ttft_s", ("lanes", "model_ttft_s")),
          ("vram_peak_mib", ("server_evidence", "hardware_peak", "vram_used_mib", "peak")),
          ("ram_peak_gib", ("server_evidence", "hardware_peak", "ram_used_gib", "peak")),
          ("slot_overlap_s", ("server_evidence", "slot_overlap_s")),
          ("max_nonidle_slots", ("server_evidence", "max_nonidle_slots")),
          ("queued_seconds", ("server_evidence", "queued_seconds")))


def dig(obj: dict, path: tuple):
    """Walk a fixed path of dict keys; None when the run file does not carry it."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def lane_field(run: dict, field: str):
    vals = [l.get(field) for l in run.get("lanes") or [] if isinstance(l.get(field), (int, float))]
    return vals


def run_value(run: dict, key: str):
    """One number per run: from the run's own blocks, or the worst lane's for the per-lane latency fields."""
    spec = dict(SERIES)[key]
    if spec[0] == "lanes":
        vals = lane_field(run, spec[1])
        if not vals:
            return None
        return max(vals) if key.startswith("max_") else statistics.median(vals)
    return dig(run, spec)


def spread(vals: list) -> dict | None:
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return None
    return {"median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3), "n": len(v)}


def correctness(run: dict) -> dict:
    out = {"undeclared_tools": [], "bad_arguments": 0, "errors": [], "incomplete": 0, "lanes": 0, "guard_ok": True}
    arm_expect = {"c1": 1, "c2": 2, "c2mtp": 2}.get(run["run"]["arm"])
    if run["guard"].get("serving_observed") != arm_expect:
        out["guard_ok"] = False
    for l in run.get("lanes") or []:
        out["lanes"] += 1
        out["undeclared_tools"] += l.get("undeclared_tools") or []
        out["bad_arguments"] += sum(1 for c in l.get("calls") or [] if c.get("bad_arguments"))
        out["errors"] += l.get("errors") or []
        if not l.get("task_complete"):
            out["incomplete"] += 1
    return out


def group(runs: list) -> dict:
    g = {"arm": runs[0]["run"]["arm"], "context_class": runs[0]["run"]["context_class"], "runs": len(runs),
         "repeats": sorted({r["run"].get("repeat") for r in runs}),
         "queue_delay_sources": sorted({l.get("queue_delay_source") for r in runs for l in r["lanes"]
                                        if l.get("queue_delay_source")}),
         "correctness": {"undeclared_tools": [], "bad_arguments": 0, "errors": [], "incomplete": 0, "lanes": 0,
                         "guard_ok": True},
         "cancellation": [r.get("cancellation") for r in runs if r.get("cancellation")]}
    for field, label in (("ttft_s", "ttft"), ("model_ttft_s", "model_ttft"), ("queue_delay_s", "queue_delay")):
        vals = [v for r in runs for v in lane_field(r, field)]
        g[label + "_p50_p90"] = None if not vals else {"p50": round(pct(vals, .5), 3),
                                                       "p90": round(pct(vals, .9), 3), "n": len(vals)}
    for key, _ in SERIES:
        g[key] = spread([run_value(r, key) for r in runs])
    for r in runs:
        c, agg = correctness(r), g["correctness"]
        agg["undeclared_tools"] += c["undeclared_tools"]
        agg["bad_arguments"] += c["bad_arguments"]
        agg["errors"] += c["errors"]
        agg["incomplete"] += c["incomplete"]
        agg["lanes"] += c["lanes"]
        agg["guard_ok"] = agg["guard_ok"] and c["guard_ok"]
    return g


def ratio(a: dict | None, b: dict | None, higher_is_better=True) -> float | None:
    if not a or not b or not a["median"]:
        return None
    return round(a["median"] / b["median"], 3)


def gate(groups: dict, min_repeats: int, promote_margin: float, min_lane_retention: float) -> tuple:
    """-> (recommendation, [reasons]).  Every reason names the number it came from."""
    reasons, promote = [], True
    base = groups.get((BASELINE, None)) or next((g for g in groups.values() if g["arm"] == BASELINE), None)
    c2s = [g for g in groups.values() if g["arm"] != BASELINE]
    if not c2s:
        return "DEFER", ["no C2 arm in these files: nothing to promote"]
    if not base:
        return "DEFER", ["no C1 baseline in these files: the C2 numbers have nothing to be compared against"]
    for g in [base] + c2s:
        label = f"{g['arm']}/{g['context_class']}"
        if g["runs"] < min_repeats:
            promote = False
            reasons.append(f"DEFER: {label} has {g['runs']} repetition(s), the gate wants {min_repeats}")
        c = g["correctness"]
        if c["undeclared_tools"]:
            promote = False
            reasons.append(f"REJECT: {label} called undeclared tools {sorted(set(c['undeclared_tools']))}")
        if c["bad_arguments"]:
            promote = False
            reasons.append(f"REJECT: {label} emitted {c['bad_arguments']} tool call(s) whose arguments were not JSON")
        if c["errors"]:
            promote = False
            reasons.append(f"REJECT: {label} had request errors ({c['errors'][0][:120]})")
        if not c["guard_ok"]:
            promote = False
            reasons.append(f"REJECT: {label} was measured with a serving count that did not match its arm")
        if g["arm"] != BASELINE:
            if (g["max_nonidle_slots"] or {}).get("median", 0) < 2:
                promote = False
                reasons.append(f"DEFER: {label} never held two slots non-idle at once "
                               f"(peak {int((g['max_nonidle_slots'] or {}).get('median', 0))} slot): the lanes did "
                               f"not run together")
            if not g["cancellation"]:
                promote = False
                reasons.append(f"DEFER: {label} has no cancellation fixture run")
            elif not all(x.get("survivor_ok") or x.get("survivor_task_complete") for x in g["cancellation"]):
                promote = False
                reasons.append(f"REJECT: {label} lost the surviving lane when the other was cancelled")
    reject = any(r.startswith("REJECT") for r in reasons)
    if reject:
        return "REJECT", reasons
    for g in c2s:
        label = f"{g['arm']}/{g['context_class']}"
        base_g = groups.get((BASELINE, g["context_class"])) or base
        gain = ratio(g["useful_work_per_minute"], base_g["useful_work_per_minute"])
        if gain is None:
            promote = False
            reasons.append(f"DEFER: {label} or its baseline has no useful-work numbers")
            continue
        if (g["tasks_completed"] or {}).get("median", 0) < (base_g["tasks_completed"] or {}).get("median", 0):
            return "REJECT", reasons + [f"REJECT: {label} completes fewer tasks "
                                        f"({g['tasks_completed']['median']}) than C1 "
                                        f"({base_g['tasks_completed']['median']})"]
        retention = ratio(g["median_lane_decode_tok_s"], base_g["median_lane_decode_tok_s"])
        if retention is not None and retention < min_lane_retention:
            return "REJECT", reasons + [f"REJECT: {label} leaves each lane at {retention}x of its C1 decode rate "
                                        f"(below {min_lane_retention}x): both workers got slower, which is contention, "
                                        f"not concurrency"]
        if gain >= 1 + promote_margin:
            reasons.append(f"{label}: {gain}x the useful work per wall-clock minute of C1 - clears the "
                           f"{promote_margin:.0%} margin")
        elif gain >= 1.0:
            promote = False
            reasons.append(f"DEFER: {label} gains {gain}x per wall-clock minute, inside the "
                           f"{promote_margin:.0%} margin: no reason to change the default")
        else:
            promote = False
            reasons.append(f"DEFER: {label} does {gain}x the useful work per wall-clock minute of C1 - slower, and "
                           f"the limiter has to be named before this can be called a rejection of concurrency")
    return ("PROMOTE" if promote else "DEFER"), reasons


def table(groups: dict) -> str:
    L = ["| arm / context | runs | useful work /min | wall s | tasks | TTFT p50 / p90 s | model TTFT worst s "
         "| queue delay p50 / p90 s | per-lane tok/s | aggregate tok/s | VRAM peak MiB | slot overlap s "
         "| incomplete lanes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups.values():
        def cell(key):
            s = g.get(key)
            if not s:
                return "-"
            return f"{s['median']}" + (f" ({s['min']}-{s['max']})" if s["n"] > 1 else "")

        def pcell(key):
            s = g.get(key)
            return "-" if not s else f"{s['p50']} / {s['p90']}"
        L.append(f"| {g['arm']} / {g['context_class']} | {g['runs']} | {cell('useful_work_per_minute')} "
                 f"| {cell('campaign_wall_s')} | {cell('tasks_completed')} | {pcell('ttft_p50_p90')} "
                 f"| {cell('max_model_ttft_s')} | {pcell('queue_delay_p50_p90')} "
                 f"| {cell('median_lane_decode_tok_s')} "
                 f"| {cell('aggregate_decode_tok_s')} | {cell('vram_peak_mib')} | {cell('slot_overlap_s')} "
                 f"| {g['correctness']['incomplete']}/{g['correctness']['lanes']} |")
    return "\n".join(L)


def ratios_table(groups: dict) -> str:
    L = ["| context | C2 arm | useful work/min | wall time | per-lane tok/s | aggregate tok/s | TTFT worst lane "
         "| VRAM peak |", "|---|---|---|---|---|---|---|---|"]
    for g in groups.values():
        if g["arm"] == BASELINE:
            continue
        base_g = groups.get((BASELINE, g["context_class"]))
        if not base_g:
            L.append(f"| {g['context_class']} | {g['arm']} | no C1 baseline for this context class | | | | | |")
            continue
        cells = [ratio(g[k], base_g[k]) for k in ("useful_work_per_minute", "campaign_wall_s",
                                                  "median_lane_decode_tok_s", "aggregate_decode_tok_s",
                                                  "max_ttft_s", "vram_peak_mib")]
        L.append(f"| {g['context_class']} | {g['arm']} | " + " | ".join("-" if c is None else f"x{c}" for c in cells)
                 + " |")
    return "\n".join(L)


def mtp_note(groups: dict) -> str:
    c2 = next((g for g in groups.values() if g["arm"] == "c2"), None)
    c2m = next((g for g in groups.values() if g["arm"] == "c2mtp"), None)
    if not c2m:
        return ("No `c2mtp` group: batch slots decode without MTP drafts unless `--batch-mtp` is on, so a plain "
                "C1-vs-C2 ratio mixes batching with losing speculation. Add a c2mtp arm before reading one.")
    if not c2:
        return "Only a `c2mtp` group is present; compare it with a `c2` arm to separate batching from MTP drafts."
    per_lane = ratio(c2m["median_lane_decode_tok_s"], c2["median_lane_decode_tok_s"])
    work = ratio(c2m["useful_work_per_minute"], c2["useful_work_per_minute"])
    return (f"c2mtp against c2: {per_lane}x per-lane decode, {work}x useful work per minute. That difference is the "
            f"MTP drafts, not the batching; the C1-vs-C2 number is the batching plus the drafts.")


def load(patterns: list) -> list:
    runs, skipped = [], []
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True) or ([p] if Path(p).is_file() else [])
    for f in sorted(set(files)):
        try:
            obj = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            skipped.append(f"{f}: {e}")
            continue
        if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
            skipped.append(f"{f}: not a {SCHEMA} run file")
            continue
        obj["_file"] = f
        runs.append(obj)
    if skipped:
        for s in skipped:
            print(f"[pi_c1c2_compare] skipped {s}", file=sys.stderr)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("patterns", nargs="+", help="run files or globs written by tools/pi_c1c2_bench.py")
    ap.add_argument("--min-repeats", type=int, default=3, help="the gate's controlled repetitions per arm")
    ap.add_argument("--promote-margin", type=float, default=0.05,
                    help="how much more useful work per wall-clock minute C2 needs to be worth promoting")
    ap.add_argument("--min-lane-retention", type=float, default=0.5,
                    help="below this share of C1's per-lane decode rate each lane is being starved: reject")
    ap.add_argument("--json", default="", help="also write the comparison as JSON here")
    ap.add_argument("--strict", action="store_true", help="exit non-zero unless the recommendation is PROMOTE")
    a = ap.parse_args()
    runs = load(a.patterns)
    if not runs:
        print("[pi_c1c2_compare] no run files matched", file=sys.stderr)
        return 2
    runs.sort(key=lambda r: (r["run"]["context_class"], r["run"]["arm"], r["run"]["started_at"]))
    groups = {}
    for r in runs:
        key = (r["run"]["arm"], r["run"]["context_class"])
        groups.setdefault(key, []).append(r)
    grouped = {k: group(v) for k, v in groups.items()}
    rec, reasons = gate(grouped, a.min_repeats, a.promote_margin, a.min_lane_retention)
    print(f"# Pi C1/C2 comparison - {len(runs)} run file(s), {len(grouped)} arm/context group(s)\n")
    print(table(grouped))
    print("\n## Ratio to the C1 baseline of the same context class\n")
    print(ratios_table(grouped))
    print("\n## Batch-MTP\n")
    print(mtp_note(grouped))
    print(f"\n## Recommendation: {rec} (a recommendation, not the decision - the issue decides)\n")
    for r in reasons or ["every gate checked out and C2 clears the margin on every context class present"]:
        print(f"- {r}")
    print("\nNot covered by these files: before/after READY and C1/C2 peak VRAM (needs a restart per arm), "
          "C1<->C2 transition latency (parallel is fixed at engine start), lane/row association under batch-MTP, "
          "and displacement attribution at an equal memory budget.")
    if a.json:
        Path(a.json).write_text(json.dumps({"schema": SCHEMA, "recommendation": rec, "reasons": reasons,
                                            "groups": [grouped[k] for k in grouped],
                                            "files": [r["_file"] for r in runs]}, indent=1, default=str) + "\n",
                                encoding="utf-8")
        print(f"[pi_c1c2_compare] wrote {a.json}")
    return 1 if (a.strict and rec != "PROMOTE") else 0


if __name__ == "__main__":
    sys.exit(main())
