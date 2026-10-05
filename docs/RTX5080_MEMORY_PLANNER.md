# RTX 5080 C1↔C2 memory planner — architecture map

Issue: #3  
Status: research / instrumentation design only  
Implementation gate: blocked on #2 hardware evidence

## Purpose

Map the smallest Strata-native path for changing the single-request versus two-request memory operating point on a 16 GB RTX 5080.

This document is deliberately implementation-neutral until #2 identifies whether C2 is useful and what actually limits it.

## Existing mechanisms

### 1. The server already owns C1 → C2 promotion

`StrataEngine.generate_batched()` in `serve/server.py` is the central policy boundary.

Current behavior:

1. A request alone uses `GEN`, i.e. the ordinary solo/MTP path.
2. When another request arrives, the solo request is stopped.
3. Its prompt plus generated continuation is reused and admitted into a batch slot with `BGEN`.
4. A second request is admitted into another slot.
5. When only one request remains, `_may_go_solo()` may stop that slot and restore the survivor to the solo/MTP path.
6. `SOLO_AGAIN_MAX` bounds repeated slot↔solo transitions.

This means the server already has exactly the lifecycle boundaries a memory planner would need.

### 2. Slot occupancy is explicit

Relevant server state:

- `slot_busy`
- `slot_live`
- `slot_held`
- `slot_used`
- `waiting`
- `slots_view()`

`/metrics` exposes live slot state, running requests and waiting requests.

A planner therefore does not need a second concurrency state machine.

### 3. Runtime expert-cache resizing already exists

`StrataEngine.vram(reserve_mib)` sends:

```text
VRAM <reserve_mib>
```

to the engine.

The existing contract says the command can shrink the expert cache until the requested free-VRAM reserve is available, or grow it back toward its startup reserve.

The engine returns figures including:

- `expert_slots`
- `expert_cache_mib`
- `vram_free_mib`

The server stores those values in engine status.

Important current constraint: this control is documented as a **between-requests** operation. It must not simply be injected while active decode windows own mutable GPU state.

### 4. Batch slots and expert residency compete for the same VRAM

`docs/BATCHING.md` explicitly states that each slot's session consumes VRAM that would otherwise hold experts.

Therefore #3's main causal hypothesis is testable without new allocator code:

```text
slot 2 admitted
   ↓
slot/session VRAM rises
   ↓
expert slots/cache shrink or cannot grow
   ↓
more routed experts leave VRAM
   ↓
CPU / PCIe expert work rises
   ↓
per-request decode falls
```

#2 must establish whether this actually dominates on the RTX 5080.

## Candidate insertion points

### Boundary A — before second slot admission

Location: server batching/admission policy immediately before transitioning from one active request to two slots.

Potential future action:

```text
predict C2 memory requirement
if elastic policy enabled:
    request a larger free-VRAM reserve
    verify returned expert-cache state
admit slot 2
```

This is the preferred place for a resize because the transition is explicit.

### Boundary B — after C2 collapses back to C1

Location: after the second active slot ends and before/while the survivor returns to the solo path.

Potential future action:

```text
release slot-2-owned state
restore startup/default VRAM reserve
allow expert cache to grow
resume/continue solo MTP
```

The resize must occur only at a state-safe boundary supported by the engine protocol.

### Boundary C — admission decision without resizing

A planner may not need to resize anything.

If #2 shows that C2 is slower than serial service for a particular workload, the smallest useful planner may simply choose:

```text
C2 predicted useful  -> admit
C2 predicted harmful -> leave request queued
```

This is preferable to memory-system changes when the evidence supports it.

## Required attribution telemetry

#2's HTTP harness already captures the outer campaign. #3 needs the following fields correlated to transition time:

| Signal | Why |
| --- | --- |
| active/running requests | establish C1/C2 interval |
| non-idle slots | prove actual slot admission |
| `expert_slots` | direct residency capacity |
| `expert_cache_mib` | cache size |
| `vram_free_mib` | allocator headroom |
| GPU VRAM used/free | independent external check |
| decode hit/lookups | expert-cache effectiveness |
| offloaded / PCIe share | routed expert spill |
| per-request decode | user-visible consequence |
| transition latency | cost of C1→C2 / C2→C1 |
| pinned host/KV memory where available | distinguish slot/KV pressure |

The important analysis is temporal, not just before/after averages.

## Decision tree after #2

```text
Does C2 materially improve two-worker wall time?
│
├─ no
│   └─ do not build a memory planner unless a clearly reversible
│      memory operating point is shown to be the cause
│
└─ yes
    │
    ├─ expert residency collapses when slot 2 appears
    │   └─ evaluate elastic expert-cache policy
    │
    ├─ slot/KV state itself exhausts headroom
    │   └─ investigate slot/KV placement/residency
    │
    ├─ admission/prompt serialization dominates
    │   └─ scheduler/admission work, not memory planner
    │
    └─ no material bottleneck
        └─ close/deprioritize #3
```

## Planner design constraints

Any future implementation must satisfy all of these:

1. **Feature off means no behavior change.**
2. Never silently reduce configured max context.
3. Never resize through an unsafe active-engine boundary.
4. Never release or overwrite a live slot's conversation state.
5. Cancellation of one lane cannot alter the survivor's state.
6. C2→C1 must preserve the existing slot→solo MTP restoration semantics.
7. Vision/multimodal state must remain valid.
8. Failure to reach the desired reserve must be bounded and explicit: queue/defer/fallback, not OOM.
9. A generic memory-cost model is preferred over a hard-coded "RTX 5080" branch if the measured variables are sufficient.

## Candidate implementation order

Only after #2:

1. **No-code/static tuning baseline** using existing settings.
2. **Admission-only planner** using observed/predicted C2 benefit.
3. **Existing elastic resize composition** at proven-safe boundaries.
4. **KV/slot placement changes** only if attribution requires them.
5. **New allocator architecture** only as a last resort.

## What is explicitly not authorized yet

- automatic `VRAM` commands during active decode;
- changing `generate_batched()` policy;
- changing slot session allocation;
- changing KV residency;
- C4 optimization;
- RTX-5080-specific heuristics.

Those decisions require #2 target-hardware evidence.

## Exit criteria for the research phase

This architecture phase is complete when:

- the C1↔C2 lifecycle boundaries are identified;
- existing VRAM controls are mapped;
- attribution telemetry is defined;
- implementation choices are ordered from smallest to largest;
- no behavioral change has been made prematurely.

At that point #3 remains open but blocked on #2 evidence.
