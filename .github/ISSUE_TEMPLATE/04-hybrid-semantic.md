---
name: Hybrid semantic execution R&D
about: Research known-block and host-bound execution with warm MTP resume
title: "[P2][R&D] Hybrid semantic execution: known blocks, host-bound values and warm MTP resume"
---

## Goal
Evaluate whether Strata can advance deterministic output without one sampled decode round per known token.

## Candidate classes
- free text/code -> MTP
- finite semantic choice -> target-authoritative decision
- fixed syntax/key -> KNOWN_BLOCK
- host-supplied exact value -> HOST_BOUND_VALUE
- structured runtime event -> STRUCTURED_EVENT

## Stages
1. feasibility/instrumentation
2. one-token exact commit
3. N-token known block
4. tool/JSON planner
5. concurrent qualification

## Hard invariants
Target authority for uncertain choices; known token != speculative acceptance; exactly-once commit; no full-prefix replay; preserve session/QSA/GDN/MTP/multimodal state; no cross-slot contamination.
