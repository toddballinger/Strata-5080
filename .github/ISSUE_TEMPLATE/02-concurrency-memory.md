---
name: RTX 5080 concurrency memory planner
about: Optimize the C1/C2 memory operating point after qualification
title: "[P1][MEMORY] RTX 5080 concurrency memory planner and elastic C1↔C2 operating point"
---

## Gate
Blocked until the C1/C2 baseline demonstrates useful concurrency and identifies memory/expert-residency pressure.

## Questions
- exact slot-2 VRAM cost by context
- expert residency displaced
- dynamic expert-cache resize feasibility
- host-backed slot/KV options
- admission policy
- safe C2 -> C1 recovery

## Preferred mechanism order
1. planner-only policy
2. reuse existing elastic expert-cache machinery
3. slot/KV placement tuning
4. qualified prefill/reserve operating points
5. new allocation architecture only if required

## Hard requirements
No silent context reduction, stale state, cross-lane leakage or OOM-by-design.
