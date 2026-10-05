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



## Stronger finding: Strata already has bulk teacher-forced state advancement

A deeper trace of `src/program/generate.cpp` and `include/strata/core/verify.hpp` found that the core state primitive #5 needs is **already present for prompt continuation**.

The serve prompt path contains `read_windows(a, b)`, described in source as:

> tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals

Its behavior is:

1. disable verifier head sampling with `set_head_sampling(false)`;
2. pack up to `T` exact caller-supplied tokens into a verifier window;
3. call `Verifier::run(T, win, position, ...)`;
4. discard the verifier's token picks because the input tokens are authoritative;
5. call `Verifier::commit(T)`, committing **every supplied token**;
6. call `mtp.prefill(ver.final_R_all(), nxt, T, position, ...)` so the MTP drafter is advanced through the same known sequence;
7. call `wait_commit()` before anything else reads the session.

The verifier header documents why this is state-correct:

- `run()` appends QSA K/V and indexer keys for the whole candidate window and records the intermediate state needed for commit;
- `commit(n_keep)` advances GDN recurrent/conv state for the accepted prefix;
- it repairs the QSA indexer tail and re-appends accepted keys;
- it restores PLE history to the snapshot corresponding to the committed prefix;
- rejected speculative state remains overwriteable beyond the committed frontier.

This is materially stronger than the earlier reasoning-budget observation.

### Consequence

The #5 problem is no longer:

> Can Strata advance exact known tokens at all?

The source already shows that it can, in bulk, on the prompt/teacher-forced path.

The new engineering question is:

> Can that existing teacher-forced verifier-window path be invoked safely **at the live generation frontier**, without turning the known suffix into an ordinary prompt replay and without breaking the current-token / next-token semantics of decode?

That is a substantially narrower problem.

## Revised K1 / K-N design target

The preferred implementation should reuse the same semantic sequence already proven by `read_windows`:

```text
exact known tokens
      ↓
Verifier::run(T), head sampling OFF
      ↓
Verifier::commit(T)
      ↓
mtp.prefill(final_R, known-next-token sequence)
      ↓
wait_commit at required ownership boundary
      ↓
resume ordinary MTP decode
```

The prototype should **not** create a second state-update implementation unless the existing verifier/prefill path cannot be safely entered at the live frontier.

### Frontier problem that still needs proof

Ordinary decode treats the current token as `window[0]` and the verifier produces the next-token outputs. Prompt teacher forcing already knows both the current token window and its following `nxt` sequence.

For a mid-generation known block, the protocol must define exactly:

- which token is already committed at the frontier;
- which known token is the next token to consume;
- the `pos0` supplied to `Verifier::run`;
- the `win[]` sequence;
- the `nxt[]` sequence passed to `mtp.prefill`;
- which final known token becomes the current token from which ordinary decode resumes.

An off-by-one error here could yield plausible text while corrupting session state, so this mapping is the next required design artifact.

## Revised smallest experiment

### K1-F — one token at the live frontier

Start from a live session at a known decode frontier and choose a token that the control run will consume next.

Compare:

**Control**
```text
ordinary decode/verify consumes token X
ordinary MTP continues
```

**K1-F**
```text
existing teacher-forced verifier path consumes the same X
MTP is prefilled/repaired through X
ordinary MTP continues
```

Require exact subsequent greedy continuation and the state invariants already listed.

### K-N-F — only after K1-F

If K1-F passes, feed a known block using the existing `T`-token teacher-forced windows rather than one host round per token.

That is where a meaningful speedup may exist: the verifier already amortizes dense-weight reads and expert work across a multi-token window.

## Revised feasibility conclusion

**The native state-advance mechanism is already demonstrated inside Strata.**

The remaining uncertainty is integration, not fundamental state evolution:

1. expose/reuse it at the live decode frontier;
2. prove frontier/off-by-one semantics;
3. preserve checkpoint/session ownership;
4. measure transition overhead;
5. prove warm MTP continuation;
6. then decide whether deterministic spans are common enough in OpenClaw to justify exposing the mechanism.

This raises #5's technical feasibility materially, while leaving the value gate unchanged.


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

**Technically more feasible than first assessed: the bulk exact-token state-advance primitive already exists for teacher-forced prompt windows, but live-frontier integration and value still need proof.**

Reasons:

- Strata already supports deterministic server text being inserted into an in-flight continuation without full-prefix replay.
- Strata already restores session state across slot→solo transitions and resumes MTP.
- The live sequence state is complex enough that token-id insertion alone is invalid.
- The prompt teacher-forcing path already drives verifier windows with head sampling disabled, commits every supplied token, and advances MTP with `mtp.prefill`; this is the preferred primitive to reuse.
- Current structured output is prompt+validate, not grammar-constrained decoding, so there is no existing general known-block executor to simply expose.

## Next permitted work

1. specify the exact live-frontier `win[]` / `nxt[]` / position mapping by comparison with `read_windows` and ordinary decode;
2. instrument deterministic-span frequency on representative agent outputs;
3. specify K1-F protocol/API as experimental and disabled by default;
4. only then implement K1-F on a dedicated follow-up PR using the existing teacher-forced verifier + MTP-prefill path.
