# Hybrid semantic execution — Strata feasibility map

Issue: #5  
Status: Phase A research  
Branch: `research/issue-5-hybrid-semantic`

## Question

Can Strata advance a known token or known token block without paying one ordinary sampled autoregressive round per token, while leaving the model/session state exactly as if those tokens had been generated normally?

This document maps the existing machinery before any behavior change is attempted.

## Important distinction

A known token is not the same thing as a speculative token.

```text
speculative token
    = candidate proposed before target authority accepts it

known token
    = exact token sequence already determined by application/runtime semantics
      and therefore does not require sampling to decide WHAT token is next
```

Even when the token identity is already known, Strata still has to advance every model state component that depends on that token.

## Existing evidence that known text can participate in an in-flight continuation

### Reasoning-budget wrap-up

`Service.run()` in `serve/server.py` already has a narrow known-text insertion path.

When `reasoning_budget_tokens` is reached:

1. generation is stopped at a clean token boundary;
2. the server tokenizes the fixed `REASONING_WRAP_UP` string;
3. those known tokens are fed through the output parser to the client;
4. the next engine pass receives:

```text
old prompt + model-generated segment + known wrap-up
```

5. the engine continues from the prefix it already holds rather than rereading the whole conversation.

This proves a useful but limited fact:

> Strata can splice deterministic server-owned text into the model-visible sequence and continue without a full-prefix replay.

It does **not** prove that the native engine advances state for N known tokens more cheaply than ordinary prompt/verify work.

### Batch-slot → solo MTP restoration

The batching path already moves a surviving conversation from a slot back to the solo path and resumes MTP from restored session state.

That is relevant to the required post-known-block invariant:

```text
known transition
    ↓
session state remains coherent
    ↓
MTP resumes warm
```

Again, this is evidence about restoration/continuation semantics, not an existing known-block executor.

## State that must advance for a known token

Inspection of `src/core/session.cpp` shows that a live sequence is more than KV cache.

At minimum the state equivalence oracle must account for:

### Residual / hyper-connection state

`SessionState.block.R`

The residual/hyper-connection streams are sequence state and must match the ordinary-token path.

### GDN recurrence and convolution history

`session_zero()` explicitly resets:

- GDN recurrent state;
- GDN convolution history.

Therefore a known token cannot simply append a token id and skip the layer path.

### QSA cache and indexer state

Each owned QSA layer maintains:

- QSA K/V state;
- indexer state;
- token/position staging;
- RoPE-related position state.

The session code explicitly initializes and zeros QSA cache/indexer state at sequence boundaries.

### PLE history

The session owns:

- `ple_hist`;
- `ple_prev`;
- `ple_token`.

The source comments explicitly state that stale PLE history would make a real contribution to later tokens.

### Position state

`stage_token()` prepares per-token position/indexer staging for QSA layers.

Any known-block mechanism must advance positions exactly as ordinary execution would, including multimodal/M-RoPE positions where applicable.

### Verifier commit state

`Verifier` owns dedicated mapped commit staging (`h_commit_` / `m_commit_`) and commit graph resources.

This is likely the closest native primitive to investigate for a one-token exact transition because the existing verify path already separates:

```text
evaluate candidate window
       ↓
determine accepted token span
       ↓
commit accepted state
```

The first implementation question is therefore not “how do we bypass the model?” but:

> Can the existing commit/state-advance machinery be driven with a caller-authoritative token while still executing every state update the model requires?

## What cannot be skipped

A correct known-token transition may avoid:

- sampling;
- candidate selection;
- speculative proposal work for determining token identity;
- repeated host-side orchestration between individual known tokens, if the engine can batch the state advance.

It may **not** skip state evolution through:

- GDN layers;
- QSA layers;
- PLE;
- residual/hyper-connection state;
- positional state;
- session/checkpoint ownership;
- multimodal position state.

The speedup ceiling is therefore narrower than “N tokens for free”.

## Smallest safe experiment

### Experiment K1 — one exact token

Add an experimental engine-only command/path, disabled by default, that accepts exactly one caller-selected token at the current sequence frontier.

Conceptually:

```text
current valid session S_t
caller supplies exact token x_t
        ↓
run the minimum existing target/state-advance path for x_t
        ↓
produce S_(t+1)
        ↓
ordinary target/MTP decode next token
```

### Oracle

For a fixed prompt and fixed token X compare two runs:

**Control**

```text
target ordinarily produces/accepts X
then continues
```

**K1**

```text
caller supplies the same X
known-token transition advances state
then continues
```

Require:

1. exact next-token logits/choice equivalence where observable;
2. exact subsequent greedy tokens for a long enough suffix;
3. identical session positions;
4. no checkpoint/prefix-cache divergence;
5. MTP resumes with no persistent acceptance regression;
6. no additional full-prefix read;
7. same result after slot→solo restoration;
8. no Vision/M-RoPE regression before claiming general support.

If K1 cannot satisfy this, stop. Do not proceed to blocks.

## Candidate implementation directions, ordered

### A. Reuse existing verify commit machinery

Preferred if the verifier can accept an externally fixed token identity while still executing the normal per-token state transition.

Advantages:
- smallest semantic delta;
- likely to preserve current state ownership;
- natural place to compare ordinary accepted-token commit versus caller-known commit.

Risk:
- verifier graphs may assume logits/candidate/acceptance buffers are populated by the ordinary verify path.

### B. Reuse prompt-continuation machinery for a tiny suffix

The reasoning-budget path already effectively appends deterministic tokens and continues.

A one-token suffix through existing continuation/prompt machinery provides a correctness baseline.

It may not be fast enough to be the final mechanism, but it gives a useful reference:

```text
known 1 token via ordinary continuation
vs
known 1 token via specialized state advance
```

### C. New dedicated state-advance graph

Only if A cannot be made clean.

This would explicitly execute the per-token model/state path without sampling.

It is a larger architectural change and should not be the first experiment.

## N-token block only after K1

If K1 passes, test:

- K4
- K8
- K32
- K128

The key value question is whether the state path can amortize host orchestration and weight movement across a known block.

If N known tokens still require N nearly identical target rounds, complexity is unlikely to pay.

## Interaction with tools / JSON

Potential deterministic spans include:

- `<tool_call>` framing;
- fixed function name after the semantic tool choice is already made;
- fixed schema keys;
- JSON punctuation;
- host-bound exact identifiers/paths/values;
- closing syntax.

Not deterministic by default:

- free-form strings;
- model-selected numeric values;
- enum values before the target has selected the semantic choice;
- arbitrary tool arguments.

The planner must classify semantics first; syntax compression is secondary.

## Interaction with finite semantic decisions

A bounded semantic choice is different again.

Example:

```text
allowed result ∈ {true, false}
```

The target still has to decide which semantic value is authoritative.

Only after that decision is made can the exact serialized tokens be treated as known.

Therefore the future planner should separate:

```text
DECIDE semantic value
SERIALIZE exact known representation
ADVANCE state through known representation
RESUME MTP
```

## Interaction with C2

No known-block implementation should be qualified only on the solo path.

A batch slot owns independent session state. The eventual C2 test must prove:

- per-slot exact state advancement;
- no cross-slot commit contamination;
- no shared commit-buffer race;
- cancellation of one lane is isolated;
- slot→solo continuation remains exact.

This is Phase E, not the first experiment.

## Value gate before implementation

Before K1 becomes more than a research prototype, measure real agent output and answer:

1. What percentage of output tokens are in structurally deterministic spans?
2. How long are those spans?
3. What fraction of end-to-end agent wall time do they consume?
4. Are they concentrated in tool-heavy workloads where OpenClaw actually spends time?
5. Would the theoretical savings exceed the transition/repair overhead?

If the upper bound is small, keep #5 as research and do not complicate the engine.

## Current conclusion

**Feasible enough to justify K1 research, not enough to justify a production implementation.**

Reasons:

- Strata already supports deterministic server text being inserted into an in-flight continuation without full-prefix replay.
- Strata already restores session state across slot→solo transitions and resumes MTP.
- The live sequence state is complex enough that token-id insertion alone is invalid.
- The verifier has explicit commit machinery that is the best first place to investigate.
- Current structured output is prompt+validate, not grammar-constrained decoding, so there is no existing general known-block executor to simply expose.

## Next permitted work

1. instrument deterministic-span frequency on representative agent outputs;
2. map the verifier's accepted-token commit call path in detail;
3. specify K1 protocol/API as experimental and disabled by default;
4. only then implement K1 on a dedicated follow-up PR.
