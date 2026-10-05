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
