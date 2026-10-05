# Strata-5080 R&D control board

This fork is an RTX 5080 / OpenClaw qualification and optimization fork of upstream Strata.

## Operating rules

1. Keep `main` close to upstream Strata. Experimental work lives on bounded research branches.
2. Idea -> issue -> benchmark/evidence -> draft PR -> validated PR -> merge or reject.
3. One engineering question per PR.
4. Benchmark before implementation when the expected value depends on workload shape or memory trade-offs.
5. Prefer upstreamable, generic changes over permanent fork-only divergence.
6. Do not mechanically port NInfer CUDA kernels or memory mechanisms into Strata; the architectures differ.
7. Preserve correctness and agent/tool semantics before optimizing throughput.
8. For the RTX 5080 project, optimize **completed OpenClaw work per wall-clock time**, not single-stream tok/s alone.

## Reference platform

Primary target:
- NVIDIA RTX 5080 16 GB
- 192 GB system RAM
- Qwen3.8-Flash-Next
- 131,072-token maximum context target
- OpenClaw local-worker workloads
- current fork baseline: upstream-derived `main`

## Priority queue

### P1-A — C1/C2 concurrency qualification

Question:

> Does `parallel=2` materially improve useful OpenClaw work completed per minute on a single RTX 5080 16 GB while retaining a 128K-class operating profile?

First action: benchmark untouched upstream behaviour. No engine rewrite is authorized before the baseline exists.

Required arms:
- C1 / `parallel=1`
- C2 / `parallel=2`
- C4 exploratory only if memory/startup is safe

Required workloads:
- short independent engineering/tool prompts
- 8K-32K retained conversations
- ~64K
- ~118K where feasible
- C1 -> C2 -> C1 transition
- cancellation of one lane while the other survives

Primary success metric:
- completed two-worker work per wall-clock minute

Secondary metrics:
- aggregate and per-request tok/s
- TTFT
- prompt/prefill rate
- expert-cache capacity/residency/hit rate
- VRAM and pinned host RAM
- MTP acceptance on solo path
- transition latency
- tool-call correctness

### P1-B — Concurrency memory planner

Blocked on P1-A evidence.

If C2 is useful but loses too much expert residency, evaluate the smallest change that improves the memory operating point.

Preferred order:
1. planner-only policy using existing mechanisms
2. reuse/extend existing elastic expert-cache resize at request boundaries
3. slot/KV placement tuning
4. qualified prefill/reserve operating points
5. only then consider new allocation architecture

Candidate dynamic policy:

```text
C1:
  maximize useful expert residency
  keep solo MTP fast

second request arrives:
  rebalance expert cache if necessary
  allocate second slot
  enter batch path

second request finishes:
  release slot state
  recover expert-cache capacity
  return survivor to solo/MTP
```

Hard requirements:
- no silent context reduction
- no stale/cross-conversation state
- deterministic bounded failure rather than OOM
- cancellation preserves surviving lane
- no behaviour change when concurrency is disabled

### P1-C — OpenClaw compatibility qualification

Build a reusable agent/API compatibility corpus before hardening code.

Cover:
- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- benign query strings
- streaming/non-streaming
- multiple/streamed tool calls
- malformed/truncated later tool calls
- duplicate parameters
- undeclared-tool rejection
- quoted reasoning markers
- mid-conversation system/developer reminders
- empty assistant turns
- thinking-stripped history
- disconnect/cancel
- C1 -> C2 -> C1
- follow-up from retained batch/conversation state

At least one fixture/oracle must be independently derived from the production parser.

Compatibility normalization must be measured for prefix-reuse side effects.

### P2-A — Hybrid semantic execution

Research whether Strata can skip autoregressive rounds for output whose exact token sequence is already known.

Candidate classes:

```text
free text / arbitrary code  -> normal MTP
finite semantic choice      -> target-authoritative decision
known syntax / fixed key    -> KNOWN_BLOCK
host-supplied exact value   -> HOST_BOUND_VALUE
structured runtime event    -> STRUCTURED_EVENT
```

Core principle:

> Do not spend one autoregressive decode round on output that is already known.

Stages:
1. feasibility/instrumentation
2. one-token exact commit
3. N-token known block (4/8/32/128)
4. tool/JSON planner
5. concurrent qualification

Hard invariants:
- target model remains authority for uncertain semantic choices
- known token != speculative acceptance
- exactly-once per-sequence commit
- no full-prefix replay for a local transition
- preserve QSA/GDN/MTP/session/multimodal state
- no cross-slot state contamination
- model-visible tokens and wire serialization remain distinct concepts

Promote to P1 only if deterministic/tool-structural spans are shown to consume material agent wall time and a bounded state-update path is viable.

## Work explicitly not copied from NInfer

Do not port these directly without independent Strata evidence:
- dense-model Q4/Q5 projection kernels
- GQA/RoPE tuning
- NInfer KV implementation
- NInfer prefix-cache architecture
- NInfer n-gram controller

Strata has different MoE, expert-residency, CPU/PCIe and speculative-decoding economics.

## Cross-runtime benchmark goal

The long-term comparison is:

```text
OpenClaw workload
      |
  +---+---+
  |       |
NInfer   Strata
  |       |
Qwen3.8  Qwen3.8-Flash-Next
  |       |
same agent benchmark suite
```

The decision metric is which architecture completes more useful agent work per hour on a 16 GB RTX 5080, not which has the largest isolated token/s number.
