---
name: OpenClaw agent compatibility
about: Qualify and harden Strata for agent/tool workloads
title: "[P1][AGENT] OpenClaw tool/API compatibility qualification and hardening"
---

## Goal
Run the corpus in `docs/OPENCLAW_COMPATIBILITY.md` against untouched main before changing implementation.

## Required classification
For every fixture record PASS / FAIL / UNSUPPORTED plus finish reason, streamed events, parsed tool calls, reused prefix tokens and TTFT.

## Safety
Do not invent undeclared tools, arguments or permissions. Retain at least one independently derived oracle/corpus.
