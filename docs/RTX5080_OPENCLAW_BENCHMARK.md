# RTX 5080 / OpenClaw concurrency benchmark contract

This benchmark qualifies Strata's current C1/C2 serving behaviour before any engine optimization.

## Hardware identity

Record:
- GPU model and VRAM
- driver
- CUDA/runtime
- CPU
- RAM
- OS/kernel
- PCIe link
- storage
- Strata commit SHA
- model family/quant
- config and engine args

## Test matrix

### Concurrency

- C1: `parallel=1`
- C2: `parallel=2`
- C4: exploratory only after C2 is safe

### Context classes

- short: <4K
- medium: 8K-32K
- long: ~64K
- very long: ~118K
- max-class: configured 131,072

### Agent shapes

1. two independent short coding/tool jobs
2. long parent + short worker
3. two medium retained conversations
4. long prompt admitted while another slot decodes
5. cancellation of one slot
6. C1 -> C2 -> C1
7. follow-up turn after slot/conversation reuse

## Per-request metrics

Capture:
- prompt tokens
- reused tokens
- newly computed prompt tokens
- TTFT
- prompt/prefill tok/s
- decode tok/s
- generated tokens
- wall time
- tool-call result
- finish reason
- slot id/state
- MTP draft accepted/proposed when on solo path

## Server/system metrics

Capture:
- aggregate tok/s
- time until second request begins useful output
- full campaign wall time
- `concurrency.serving`
- expert-cache capacity
- expert-cache hit rate
- VRAM before READY
- VRAM after READY
- C1 VRAM peak
- C2 VRAM peak
- pinned host memory
- CPU utilization
- PCIe traffic where available
- graph/capture fallback
- retries/stalls/crashes

## Correctness

The concurrent result is invalid if any of these fail:
- undeclared tool execution
- malformed tool leakage
- cross-conversation contamination
- missing/duplicated tokens caused by slot transition
- cancellation damages surviving lane
- follow-up conversation resumes the wrong state

## Decision metric

Primary:

```text
completed useful two-worker work / wall-clock minute
```

Do not reject C2 merely because its aggregate tok/s is below C1.

For example, a serial 90 tok/s server can still be less useful to two agents than two concurrent ~38 tok/s lanes if queue latency dominates task completion.

## Evidence record

Every benchmark report should include:
- BASE SHA
- exact config
- fixture identity/hash where possible
- repetitions
- median and range
- failures
- decision: PROMOTE / DEFER / REJECT


## Issue #2 HTTP harness

`tools/rtx5080_openclaw_bench.py` exercises the same HTTP service path used by OpenAI-compatible agent clients.

It deliberately launches **two requests simultaneously in both C1 and C2**:

- C1 shows the real queueing cost when one worker must wait.
- C2 shows the real batch-slot admission/decoding behaviour.
- The script refuses to run if `GET /v1/status` does not report the expected `concurrency.serving` value.

The harness polls `/metrics` every 200 ms during the campaign and samples the first NVIDIA GPU with `nvidia-smi`. It writes:

- a machine-readable JSON evidence record containing before/after status, sampled metrics, request timing and GPU telemetry;
- a compact Markdown summary beside it.

### First-pass short-context arms

Start Strata in the intended configuration before each arm.

```bash
python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1 --tag short-r1
python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 --tag short-r1
```

Repeat at least three times per arm before interpreting small differences.

### Context sweep

The synthetic-history fixture is deterministic. `--target-prompt-tokens` is an approximate construction target; the server's own prompt-token accounting in the JSON evidence is authoritative.

```bash
python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1 --target-prompt-tokens 8192  --tag p8k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 --target-prompt-tokens 8192  --tag p8k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1 --target-prompt-tokens 32000 --tag p32k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 --target-prompt-tokens 32000 --tag p32k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1 --target-prompt-tokens 64000 --tag p64k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 --target-prompt-tokens 64000 --tag p64k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c1 --expect-serving 1 --target-prompt-tokens 118000 --tag p118k-r1
python3 tools/rtx5080_openclaw_bench.py --arm c2 --expect-serving 2 --target-prompt-tokens 118000 --tag p118k-r1
```

The C4 arm remains intentionally gated until the C2 result is safe and useful.

### Existing correctness tests remain authoritative

The HTTP harness supplements rather than replaces the existing engine/server tests:

- `tools/batch_test.py` — solo vs slot token exactness and aggregate batch rate;
- `tools/batch_interleave_test.py` — prompt interleave, yield, retained conversation state and return to solo MTP;
- `tools/early_close_test.py` — cancellation/client disconnect isolation.

A performance result should not be promoted if these correctness paths fail on the candidate configuration.
