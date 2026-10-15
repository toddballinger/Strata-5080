#!/usr/bin/env python3
"""Pi C1/C2 comparator (Issue #2): read the run files of a campaign and say what they are worth.

Reads the JSON files tools/pi_c1c2_bench.py writes (globs work), groups them by arm and context class, and
answers the two questions the plan asks:

  is C2 worth it?       the decision metric is COMPLETED tasks per wall-clock minute - a task counts only when
                        every fixed check passed, so partial credit cannot carry an arm over the line.  Partial
                        credit is printed beside it, never inside it.
  is it safe?           did each lane keep its own conversation, did any lane call a tool it was not given, did
                        the server really hold two slots at once, did a cancelled lane leave the other one
                        working, did any request error

The gate is deliberately blunt, and it fails closed:

  REJECT   a lane called an undeclared tool, a request errored, a lane's tool arguments arrived as something
           other than an object, the guard was contradicted, the cancellation fixture lost its survivor, or C1
           completed tasks that C2 did not
  DEFER    not enough controlled repetitions, no C1 baseline FOR THIS CONTEXT CLASS, the arms ran different
           models, the arm label was not confirmed by the server's own numbers, the slot evidence is missing, a
           stream ended inside a tool call, the gain is inside the observed spread, or neither arm completed a
           task - a PROMOTE needs at least one completed task on the arm being promoted, and a comparison of
           nothing with nothing is not a result
  PROMOTE  every correctness check passed, the C2 arm completed more tasks per minute than the C1 arm of the
           same context class by more than --promote-margin, the two arms' per-run ranges do not overlap, and
           each lane still decodes at a usable share of its C1 rate

A run taken with --cancel-lane is a fixture, not a measurement: one lane is cut off on purpose, so its numbers
are structurally lower.  Fixture runs are checked for the cancellation gate and are kept OUT of the medians the
ratio is taken from, so a mandated fixture cannot drag the arm it belongs to down.

Nothing here measures anything: it reads files.  --strict makes a non-PROMOTE recommendation a non-zero exit,
for a script that runs the campaign.
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

BASELINE = "c1"
# (key, where in the run file, better direction).  `useful_work_per_minute` is completed tasks per minute;
# `partial_credit_per_minute` is reported and is never part of the gate.
SERIES = [("tasks_completed", ("decision_metric", "tasks_completed"), True),
          ("useful_work_per_minute", ("decision_metric", "useful_work_per_minute"), True),
          ("partial_credit", ("decision_metric", "partial_credit"), True),
          ("wall_s", ("run", "campaign_wall_s"), False),
          ("ttft_median_s", ("lanes", "ttft_s"), False),
          ("ttft_p90_s", ("lanes", "ttft_s"), False),
          ("model_ttft_median_s", ("lanes", "model_ttft_s"), False),
          ("max_model_ttft_s", ("lanes", "model_ttft_s"), False),
          ("max_queue_delay_s", ("lanes", "queue_delay_s"), False),
          ("decode_tok_s", ("lanes", "decode_tok_s_server"), True),
          ("aggregate_decode_tok_s", ("decision_metric", "aggregate_decode_tok_s"), True),
          ("per_lane_decode_tok_s", ("decision_metric", "per_lane_decode_tok_s"), True),
          ("prompt_tok_s", ("decision_metric", "aggregate_prompt_tok_s"), True),
          ("max_nonidle_slots", ("server_evidence", "max_nonidle_slots"), True),
          ("slot_overlap_s", ("server_evidence", "slot_overlap_s"), True),
          ("vram_peak_mib", ("server_evidence", "hardware_peak", "vram_used_mib"), False)]
LOWER_IS_BETTER = {k for k, _, better in SERIES if not better}
QUEUE_SOURCES_GOOD = ("api_requests:nonce",)
SLOT_STATE_GOOD = "live.slots"


def run_value(run: dict, field: str):
    """One number out of one run file: a lane field takes the worst lane's value (a decision must not be made
    on a mean that hides one bad lane), anything else is read by its path."""
    spec = next((s for s in SERIES if s[0] == field), None)
    path = spec[1] if spec else (field,)
    if path[0] == "lanes":
        vals = [l.get(path[1]) for l in run.get("lanes") or [] if isinstance(l.get(path[1]), (int, float))]
        if not vals:
            return None
        return max(vals) if field.startswith("max_") else (pct(vals, 0.9) if field.endswith("p90_s")
                                                           else statistics.median(vals))
    node = run
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def correctness(run: dict) -> dict:
    """What went wrong in one run, as counts and names.  Empty `problems` means the run is usable as evidence."""
    out = {"lanes": 0, "undeclared_tools": [], "invalid_arguments": 0, "truncated_tool_calls": 0, "errors": 0,
           "guard_ok": True, "serving_ok": True, "batch_mtp_ok": True, "queue_sources": [], "problems": []}
    arm = (run.get("run") or {}).get("arm")
    out["queue_sources"] = sorted({str(l.get("queue_delay_source")) for l in run.get("lanes") or []
                                   if l.get("queue_delay_source")})
    expect = {"c1": 1, "c2": 2, "c2mtp": 2}.get(arm)
    g = run.get("guard") or {}
    if g.get("mismatch"):
        out["problems"].append("guard: " + "; ".join(str(m) for m in g["mismatch"]))
    if expect is not None and g.get("serving_observed") != expect:
        out["serving_ok"] = False
        out["problems"].append(f"guard: the server reported concurrency.serving={g.get('serving_observed')}, "
                               f"not the {expect} the arm {arm!r} claims")
    mtp = run.get("batch_mtp") or {}
    if mtp and mtp.get("verified") is False:
        out["batch_mtp_ok"] = False        # the gate defers on this: the run does not prove the arm it is labelled
    for l in run.get("lanes") or []:
        out["lanes"] += 1
        out["undeclared_tools"] += l.get("undeclared_tools") or []
        out["invalid_arguments"] += len(l.get("invalid_arguments") or [])
        out["truncated_tool_calls"] += len(l.get("truncated_tool_calls") or [])
        out["errors"] += len([e for e in l.get("errors") or [] if "error" in e])
        if l.get("stopped") == "harness_error":
            out["problems"].append(f"lane {l.get('lane')} stopped inside the harness: "
                                   f"{json.dumps(l.get('errors'))[:160]}")
    out["undeclared_tools"] = sorted(set(out["undeclared_tools"]))
    if out["undeclared_tools"]:
        out["problems"].append(f"undeclared tools called: {out['undeclared_tools']}")
    if out["invalid_arguments"]:
        out["problems"].append(f"{out['invalid_arguments']} tool call(s) whose arguments were not an object")
    if out["truncated_tool_calls"]:
        out["problems"].append(f"{out['truncated_tool_calls']} tool call(s) ended mid-arguments (issue #211)")
    if out["errors"]:
        out["problems"].append(f"{out['errors']} request error(s)")
    return out


def is_fixture(run: dict) -> bool:
    """A run taken with --cancel-lane: one lane is cut off on purpose, so its numbers are not measurements."""
    return bool(run.get("cancellation"))


def group(runs: list) -> dict:
    """One arm and one context class.  Medians come from the measured runs only; the cancellation fixtures of
    the same arm are counted and kept apart."""
    measured = [r for r in runs if not is_fixture(r)]
    fixtures = [r for r in runs if is_fixture(r)]
    src = measured or runs                       # a group of nothing but fixtures still gets a row, marked
    g = {"arm": (runs[0].get("run") or {}).get("arm"), "context_class": (runs[0].get("run") or {}).get("context_class"),
         "runs": len(measured), "fixture_runs": len(fixtures),
         "repeats": sorted({(r.get("run") or {}).get("repeat") for r in measured}),
         "models": sorted({str((r.get("run") or {}).get("model")) for r in runs
                           if (r.get("run") or {}).get("model")}),
         "correctness": [c for r in runs for c in correctness(r)["problems"]],
         "queue_delay_sources": sorted({s for r in runs for l in r.get("lanes") or []
                                        for s in [str(l.get("queue_delay_source"))] if s}),
         "slot_state_sources": sorted({str((r.get("server_evidence") or {}).get("slot_state_source"))
                                       for r in runs}),
         "batch_mtp_verified": [bool((r.get("batch_mtp") or {}).get("verified")) for r in runs],
         "cancellation": [r.get("cancellation") for r in fixtures],
         # what each run on its own has to stand up: a median cannot cover for a run that shows nothing
         "per_run": [{"file": str(r.get("_file") or ""), "model": str((r.get("run") or {}).get("model") or ""),
                      "max_nonidle_slots": run_value(r, "max_nonidle_slots"),
                      "slot_state_source": str((r.get("server_evidence") or {}).get("slot_state_source") or ""),
                      "queue_sources": sorted({str(l.get("queue_delay_source")) for l in r.get("lanes") or []
                                               if l.get("queue_delay_source")})} for r in runs]}
    for key, _, _ in SERIES:
        vals = [run_value(r, key) for r in src]
        vals = [v for v in vals if isinstance(v, (int, float))]
        g[key] = {"median": round(statistics.median(vals), 3) if vals else None,
                  "min": round(min(vals), 3) if vals else None,
                  "max": round(max(vals), 3) if vals else None, "n": len(vals)}
    return g


def spread(g: dict, key: str) -> str:
    d = g.get(key) or {}
    if d.get("median") is None:
        return "-"
    return f"{d['median']} (min {d['min']}, max {d['max']}, n {d['n']})"


def ratio(c2: dict, c1: dict, key: str) -> str:
    a, b = (c2.get(key) or {}).get("median"), (c1.get(key) or {}).get("median")
    if a is None or b is None or b == 0:
        return "-"
    x = round(a / b, 3)
    if key in LOWER_IS_BETTER:
        if x < 1:
            return f"x{x} (lower is better)"
        return "x1.0" if x == 1 else f"x{x} WORSE"
    return f"x{x}"


def ratios_table(groups: dict) -> list:
    keys = [k for k, _, _ in SERIES]
    lines = ["| arm / class | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for cls in sorted({g["context_class"] for g in groups.values()}):
        base = groups.get((BASELINE, cls))
        for (arm, gcls), g in sorted(groups.items()):
            if gcls != cls:
                continue
            cells = []
            for k in keys:
                if arm == BASELINE:
                    cells.append(spread(g, k))
                else:
                    other = base
                    cells.append("no C1 baseline for this context class"
                                 if not other else f"{spread(g, k)}  ({ratio(g, other, k)})")
            lines.append(f"| `{arm}/{cls}` | " + " | ".join(cells) + " |")
    return lines


def series_table(groups: dict) -> list:
    lines = ["| arm / class | runs | repeats | models | queue delay source | slot state source |",
             "|---|---|---|---|---|---|"]
    for (arm, cls), g in sorted(groups.items()):
        lines.append(f"| `{arm}/{cls}` | {g['runs']} (+{g['fixture_runs']} fixture) | "
                     f"{', '.join(str(r) for r in g['repeats']) or '-'} | {', '.join(g['models']) or '-'} | "
                     f"{', '.join(g['queue_delay_sources']) or '-'} | "
                     f"{', '.join(g['slot_state_sources']) or '-'} |")
    return lines


def drafter_table(groups: dict) -> list:
    """What the drafts are worth on their own: `c2mtp` (two slots, one drafter each) against `c2` (two slots, no
    drafter) of the same context class.  A plain C1-to-C2 ratio mixes adding a lane with losing the drafts C1 has,
    and this is the half that separates them."""
    keys = ("tasks_completed", "useful_work_per_minute", "per_lane_decode_tok_s", "model_ttft_median_s",
            "max_model_ttft_s")
    lines = ["| context class | tasks_completed | useful_work_per_minute | per-lane decode tok/s | "
             "model TTFT s | worst model TTFT s |", "|---" * (len(keys) + 1) + "|"]
    rows = 0
    for cls in sorted({g["context_class"] for g in groups.values()}):
        mtp, plain = groups.get(("c2mtp", cls)), groups.get(("c2", cls))
        if not mtp or not plain or not mtp["runs"] or not plain["runs"]:
            continue
        rows += 1
        lines.append(f"| `{cls}` | " + " | ".join(ratio(mtp, plain, k) for k in keys) + " |")
    if not rows:
        return []
    return ["## What the drafts are worth (same class, `c2mtp` against `c2`)", "",
            "A C1-to-C2 difference is two effects at once: a lane added, and the MTP drafts C1 has been lost. These "
            "are the same two slots with a drafter and without one, which is the second effect on its own.", ""] \
        + lines + [""]


def gate(groups: dict, min_repeats: int, promote_margin: float, min_lane_retention: float):
    """The recommendation and every reason it is not better than that.  Fails closed: anything unproven is DEFER."""
    reject, defer = [], []
    c2s = [g for (arm, _), g in groups.items() if arm != BASELINE]
    c1s = [g for (arm, _), g in groups.items() if arm == BASELINE]
    if not c2s:
        return "DEFER", ["no C2 run files: the gate compares a C2 arm against the C1 arm of the same class"]
    if not c1s:
        return "DEFER", ["no C1 run files: there is nothing to compare the C2 arm against"]

    for g in list(groups.values()):          # every group, C1 included, has to stand up on its own
        tag = f"{g['arm']}/{g['context_class']}"
        for p in g["correctness"]:
            reject.append(f"{tag}: {p}")
        distinct = {r for r in g["repeats"] if r is not None}
        if len(distinct) < min_repeats:
            defer.append(f"{tag}: {len(distinct)} distinct --repeat value(s) ({sorted(distinct) or 'none'}), "
                         f"the gate wants {min_repeats} - files are not repetitions")
        if g["runs"] == 0:
            defer.append(f"{tag}: every run of this group was a cancellation fixture, so nothing was measured")

    # the cancellation fixture: the survivor has to finish, and the cancelled lane has to have stopped
    for g in c2s + c1s:
        for c in g["cancellation"]:
            if not c.get("survivor_ok"):
                reject.append(f"{g['arm']}/{g['context_class']}: the cancellation fixture lost its survivor - "
                              "the other lane did not finish its task")
            if c.get("cancelled_lane_stopped") is False:
                defer.append(f"{g['arm']}/{g['context_class']}: the cancelled lane was not recorded as stopped, "
                             "so the fixture did not cancel anything")

    # a C2 arm cannot be promoted on a different model, or against a different context class
    for g in c2s:
        base = groups.get((BASELINE, g["context_class"]))
        tag, btag = f"{g['arm']}/{g['context_class']}", f"{BASELINE}/{g['context_class']}"
        if g["runs"] == 0:
            defer.append(f"{tag}: every measured run of this arm was set aside, so there is nothing to compare")
            continue
        if not base or base["runs"] == 0:
            defer.append(f"{tag}: no C1 baseline for this context class - a {g['context_class']} C2 run is never "
                         f"compared against a {BASELINE} run of another class")
            continue
        shared = None
        for r in g["per_run"] + base["per_run"]:
            if not r["model"]:
                continue
            shared = {r["model"]} if shared is None else shared & {r["model"]}
        if not shared:
            defer.append(f"{tag}: the runs do not all name one model "
                         f"({g['models']} against {base['models']}), so the arms are not comparable")
        if any(v is False for v in g["batch_mtp_verified"]):
            defer.append(f"{tag}: at least one run whose arm label the server's own numbers did not confirm")

        done_c2, done_c1 = g["tasks_completed"]["median"], base["tasks_completed"]["median"]
        if done_c2 is None or done_c1 is None:
            defer.append(f"{tag}: no completed-task numbers to compare")
            continue
        if done_c2 == 0 and done_c1 == 0:
            defer.append(f"{tag}: neither arm completed a task ({spread(g, 'tasks_completed')} against "
                         f"{spread(base, 'tasks_completed')}), so there is no useful work to promote - partial "
                         f"credit ({spread(g, 'partial_credit')}) is reported and is not counted")
            continue
        if done_c2 < done_c1:
            reject.append(f"{tag}: C1 completed {spread(base, 'tasks_completed')} tasks where "
                          f"{g['arm']} completed {spread(g, 'tasks_completed')}")
        # the decision metric itself: completed tasks per wall-clock minute, C2 over C1
        gain = (g["useful_work_per_minute"]["median"] / base["useful_work_per_minute"]["median"]) \
            if g["useful_work_per_minute"]["median"] and base["useful_work_per_minute"]["median"] else None
        if gain is None:
            defer.append(f"{tag}: no useful-work-per-minute on one of the arms")
            continue
        if gain < 1 + promote_margin:
            defer.append(f"{tag}: {round(gain, 3)}x the completed tasks per wall-clock minute of "
                         f"{btag}, under the {round(1 + promote_margin, 3)}x the gate wants")
            continue
        # the gain has to be outside the noise the runs themselves show, not just above a fixed margin
        if g["useful_work_per_minute"]["min"] is None or base["useful_work_per_minute"]["max"] is None or \
                g["useful_work_per_minute"]["min"] <= base["useful_work_per_minute"]["max"]:
            defer.append(f"{tag}: the arms' per-run ranges overlap "
                         f"({tag} {spread(g, 'useful_work_per_minute')} against "
                         f"{btag} {spread(base, 'useful_work_per_minute')}), so the gain is not separated from "
                         "the spread of the runs")
            continue
        c2_lane, c1_lane = g["per_lane_decode_tok_s"]["median"], base["per_lane_decode_tok_s"]["median"]
        if c2_lane is None or c1_lane is None or c1_lane == 0:
            defer.append(f"{tag}: no per-lane decode rate to check lane retention against")
            continue
        if c2_lane / c1_lane < min_lane_retention:
            reject.append(f"{tag}: each lane decodes at {round(c2_lane / c1_lane, 3)}x of the C1 lane rate "
                          f"({c2_lane} against {c1_lane} tok/s), under {min_lane_retention}: the lanes are "
                          "starving each other")
            continue
        no_slot = [r for r in g["per_run"] if (r["max_nonidle_slots"] or 0) < 2
                   or r["slot_state_source"] != SLOT_STATE_GOOD]
        if no_slot:
            defer.append(f"{tag}: {len(no_slot)} of {len(g['per_run'])} run(s) never showed two non-idle slots "
                         f"from live.slots[].state (sources {g['slot_state_sources']}), so they do not show the "
                         f"concurrency they are being credited for: "
                         f"{[r['file'] or r['model'] for r in no_slot][:4]}")
            continue
        no_queue = [r for r in g["per_run"] if not any(s.startswith(QUEUE_SOURCES_GOOD)
                                                       for s in r["queue_sources"])]
        if no_queue:
            defer.append(f"{tag}: {len(no_queue)} of {len(g['per_run'])} run(s) tie no lane's queue delay to that "
                         f"lane ({g['queue_delay_sources']}), so the split between waiting and decoding is not "
                         f"evidence yet - run the server with --api-monitor: "
                         f"{[r['file'] or r['model'] for r in no_queue][:4]}")
            continue
    if reject:
        return "REJECT", reject + defer
    if defer:
        return "DEFER", defer
    return "PROMOTE", ["every correctness check passed, and the C2 arm of each context class completed more "
                       "tasks per wall-clock minute than the C1 arm of the same class, by more than the margin "
                       "and outside the spread of the runs, with each lane decoding within the retention limit"]


def render_markdown(cmp: dict) -> str:
    lines = [f"# Pi C1/C2 comparison: **{cmp['recommendation']}**", "",
             f"Read {cmp['files']} run file(s), {len(cmp['groups'])} arm/class group(s).  Gate: "
             f"{cmp['min_repeats']} distinct repetitions per arm, promote margin {cmp['promote_margin']}, "
             f"lane retention {cmp['min_lane_retention']}.", "",
             "## Why", ""]
    lines += [f"- **{r}**" if cmp["recommendation"] == "REJECT" else f"- {r}" for r in cmp["reasons"]]
    lines += ["", "## Groups", ""] + series_table(cmp["groups"])
    lines += ["", "## The numbers (median, with the range the runs showed)", ""] + ratios_table(cmp["groups"])
    draft = drafter_table(cmp["groups"])
    if draft:
        lines += [""] + draft
    lines += ["", "## Reading it", "",
              "`tasks_completed` counts whole tasks only; `partial_credit` is printed beside it and is never "
              "part of the decision.  `queue delay source` says whether a lane's queue seconds are the server's "
              "own per-request number tied to that lane (`api_requests:nonce`) or unavailable.  `slot state "
              "source` says whether the two-slot evidence came from `live.slots[].state`.", ""]
    return "\n".join(lines)


def load(patterns: list) -> tuple:
    runs, skipped = [], []
    for pat in patterns:
        for f in sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]:
            try:
                obj = json.loads(Path(f).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                skipped.append((f, f"{type(e).__name__}: {e}"))
                continue
            # a schema-correct but partial file is skipped, not fatal: one bad file must not stop a comparison
            bad = []
            if not isinstance(obj, dict):
                bad = ["not an object"]
            else:
                bad = [k for k in ("run", "guard", "decision_metric", "server_evidence")
                       if not isinstance(obj.get(k), dict)]
                if not isinstance(obj.get("lanes"), list):
                    bad.append("lanes")
                bad += [f"run.{k}" for k in ("arm", "context_class", "started_at")
                        if not obj["run"].get(k)] if isinstance(obj.get("run"), dict) else []
            if bad:
                skipped.append((f, f"missing or malformed: {', '.join(bad)}"))
                continue
            obj["_file"] = f
            runs.append(obj)
    return runs, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("patterns", nargs="+", help="run files or globs written by tools/pi_c1c2_bench.py")
    ap.add_argument("--min-repeats", type=int, default=3, help="the gate's controlled repetitions per arm")
    ap.add_argument("--promote-margin", type=float, default=0.05,
                    help="how much more completed work per wall-clock minute C2 needs to be worth promoting")
    ap.add_argument("--min-lane-retention", type=float, default=0.5,
                    help="below this share of C1's per-lane decode rate each lane is being starved: reject")
    ap.add_argument("--json", default="", help="also write the comparison as JSON here")
    ap.add_argument("--markdown", default="", help="also write the comparison as a markdown report here")
    ap.add_argument("--strict", action="store_true", help="exit non-zero unless the recommendation is PROMOTE")
    a = ap.parse_args()
    runs, skipped = load(a.patterns)
    for f, why in skipped:
        print(f"[pi_c1c2_compare] skipped {f}: {why}", file=sys.stderr)
    if not runs:
        print("[pi_c1c2_compare] no usable run files", file=sys.stderr)
        return 2
    groups = {}
    for r in runs:
        groups.setdefault((r["run"]["arm"], r["run"]["context_class"]), []).append(r)
    grouped = {k: group(v) for k, v in groups.items()}
    rec, reasons = gate(grouped, a.min_repeats, a.promote_margin, a.min_lane_retention)
    cmp = {"recommendation": rec, "reasons": reasons, "files": len(runs), "skipped": len(skipped),
           "groups": grouped, "min_repeats": a.min_repeats, "promote_margin": a.promote_margin,
           "min_lane_retention": a.min_lane_retention}
    print(f"[pi_c1c2_compare] {rec}")
    for r in reasons:
        print(f"  - {r}")
    print()
    print("\n".join(series_table(grouped)))
    print()
    print("\n".join(ratios_table(grouped)))
    draft = drafter_table(grouped)
    if draft:
        print()
        print("\n".join(draft))
    if a.json:
        # the groups are keyed by (arm, context class); a JSON object needs one string key for both
        out = dict(cmp, groups={f"{arm}/{cls}": g for (arm, cls), g in grouped.items()})
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[pi_c1c2_compare] wrote {a.json}")
    if a.markdown:
        Path(a.markdown).write_text(render_markdown(cmp), encoding="utf-8")
        print(f"[pi_c1c2_compare] wrote {a.markdown}")
    return 0 if (rec == "PROMOTE" or not a.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
