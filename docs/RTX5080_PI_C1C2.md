# Running two coding agents at once on one 16 GB card: the C1/C2 measurement

Issue #2 asks one question with numbers: **does one RTX 5080 with 16 GB do more useful coding work per wall-clock
minute when two independent Pi coding-agent requests run together, or when they run one at a time?** Aggregate
tokens per second is not the answer - two lanes can raise the total while each one gets slower and the user waits
longer. The decision metric is **completed useful work per wall-clock minute**, with time-to-first-token, correctness,
VRAM and RAM beside it.

This page is the contract: what the arms are, what each number means, how a task is scored as useful, and what has to
be true before anyone writes PROMOTE. The tool is `tools/pi_c1c2_bench.py`, the arithmetic is
`tools/pi_c1c2_compare.py`, the CPU-only tests are `tools/test_pi_c1c2_bench.py`.

**Nothing on this page is a measurement yet.** Phase 1 built and tested the harness without a GPU; the campaigns are
the next step and need a machine of their own.

## The arms are start-up configuration, not a switch

`"parallel": N` in a config becomes `--batch N` on the engine's command line (`serve/server.py`, `parallel_args`), and
the server reads the slot count back once when the engine starts (`INFO batch_slots=N`), which is what
`GET /v1/status` reports as `concurrency.serving`. There is no runtime resize.

| arm | engine start-up | serving | MTP drafts while decoding |
|---|---|---|---|
| C1 | no `parallel` (or `"parallel": 1`) | 1 | yes, when the config has `--spec`/`--mtp` |
| C2 | `"parallel": 2` | 2 | **no**, unless `--batch-mtp` is on |
| C2+MTP | `"parallel": 2 --batch-mtp` | 2 | yes, one drafter per slot |

Two consequences, both of which the harness enforces:

- **A C1-to-C2 change is a server restart.** "C1 → C2 transition latency" is not measurable inside one running
  server, so it is not measured. What is measured instead, inside one C2 server: how long a lane waits before it gets
  a slot, whether two slots were ever non-idle at the same time, and what the surviving lane does when the other ends
  or is cancelled. Restart cost itself is `tools/ab_engine.py`'s business and belongs with the GPU window.
- **A plain C1-vs-C2 ratio mixes two effects**: adding a lane, and losing the MTP drafts that C1 has. That is why
  `c2mtp` is a third arm. `tools/pi_c1c2_compare.py` prints the `c2mtp`-against-`c2` ratio separately and says which
  part of the difference is the drafts.

Because the guard refuses to run when the server disagrees, every run file records the serving count the server
reported before any traffic was sent.

## What each number means

| number | what it is | where it comes from |
|---|---|---|
| **queue delay** | how long a request waits before the engine starts it | the server's own per-request `queue_s` from `GET /api/requests` when the API monitor is on; otherwise `GET /metrics` `live.queued` integrated over the campaign at the sampling interval. The server does not echo a client request id, so a record is tied to a lane by waiting, not by identity: the longest queue of the campaign goes to the lane with the largest client TTFT. The raw records are kept in the run file so that attribution can be checked by hand |
| **TTFT** | the client's: request sent → first useful streamed token (content, reasoning, or a tool-call argument) | the client's own clock |
| **model TTFT** | TTFT minus the queue delay: what the engine itself took | derived |
| **decode tok/s** | per lane, decode only, as the server counts it | the `timings` of the last streamed chunk (`predicted_per_second`) |
| **client tok/s** | per lane, whole lane, gaps and prefill included | tokens ÷ the client's own wall time |
| **aggregate tok/s** | all lanes together | tokens ÷ summed decode seconds |
| **useful work** | tasks whose checks all passed (below), 1.0 each; partial credit is recorded but never counted as done | the scorer |
| **useful work per minute** | useful work ÷ campaign wall minutes | the decision metric |
| **slot overlap seconds** | seconds with two slots non-idle in two samples in a row | `/metrics` `live.slots[].state` |
| **VRAM / RAM peak** | highest sampled value during the campaign | `/metrics` `hardware`, falling back to `/v1/status` `machine` |

Queue delay is separated from TTFT on purpose. Under C1 the second request waits for the first to finish, so its
client TTFT contains the whole wait; under C2 it should not. Reporting one number would hide the thing being measured.

## The workload: two real coding-agent sessions, started at the same instant

Each lane is one coding-agent session from the set `tools/agent_turns_bench.py` already uses (the same tasks over this
repository's own files, so anyone with this checkout sends the same bytes). A lane runs the loop an agent runs:

1. the task arrives with three tools declared: `read_file`, `grep`, `submit_solution`;
2. the model answers with a tool call; the harness executes `read_file` and `grep` **read-only** against this
   checkout - `read_file` is confined to the repository, nothing is written and no command is run;
3. the result goes back as a `tool` message and the lane continues, up to `--max-steps`;
4. the lane finishes by calling `submit_solution` with the file it would change, one line of that file quoted
   verbatim as the anchor of the edit, and a summary.

Both lanes are released by one barrier, so their first requests are sent at the same instant. Requests are greedy
(`temperature 0`, fixed seed), thinking off.

A task counts as **useful work** only when all six fixed checks pass. No model is asked to judge the answer:

| check | how it is decided |
|---|---|
| `submitted` | the lane called `submit_solution` |
| `declared_tools_only` | every tool the lane called was one of the three declared - an undeclared call ends the lane and fails the run |
| `read_before_submit` | at least one `read_file`/`grep` was executed and returned before submitting |
| `target_path_expected` | the file named is one the task is actually about |
| `anchor_verbatim` | the quoted anchor appears, character for character, in that file |
| `summary_present` | the summary is at least 40 characters |

The prompt size is a class, not a claim: `short`, `32k`, `64k`, `118k` add a deterministic filler block sized to land
near that many tokens, and the server's own `prompt_tokens` in the run file is what the evidence records.

## Running it

Dry-run by default - without `--run` it prints the plan and opens no connection at all, not even `/v1/status`.

```
python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2                      # the plan, nothing sent
python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2 --run                 # one campaign, short prompts
python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2 --context-class 64k --repeat 2 --run
python tools/pi_c1c2_bench.py --arm c2 --expect-serving 2 --cancel-lane B --cancel-after 20 --run
python tools/pi_c1c2_compare.py 'bench/results/rtx5080-pi-c1c2/*.json' --min-repeats 3
```

It never starts, stops or reconfigures a server. Changing arms means restarting on that arm's configuration, which is
`tools/ab_engine.py`'s job and needs a GPU window of its own. `--expect-serving` is not a formality: a run whose
server reports a different number stops before sending anything, so a mislabelled arm cannot produce evidence.

Run files are `bench/results/rtx5080-pi-c1c2/run-<arm>-<class>-<stamp>.json` and `.md`. Home, user and host names, IP
addresses and the API key are replaced by placeholders before anything is written (the same redaction
`tools/strata_bench.py` uses).

## The gate

`tools/pi_c1c2_compare.py` prints medians and ranges per arm and context class, the ratios to the C1 arm of the same
class, and a recommendation. **A recommendation is not the decision** - the decision is written by hand in the issue
with the run files next to it. It DEFERs rather than guesses when something is missing:

- at least three controlled repetitions per arm, alternating order;
- no undeclared tool call, no tool call whose arguments were not JSON, no request error, and the serving count the arm
  claims;
- C2 completes at least as many tasks as C1 - fewer completed tasks is a REJECT whatever the throughput says;
- a C2 run that actually held two slots non-idle at once (a run that never did has not tested concurrency);
- a cancellation run where closing one lane leaves the other finishing;
- each lane still decoding at a usable share of its C1 rate - if both lanes get slower, that is contention, not
  concurrency, whatever the aggregate says;
- and, to promote, a gain in useful work per wall-clock minute outside the noise band (5 % by default).

The correctness and cancellation fixtures reuse what upstream already has rather than duplicating it:
`tools/early_close_test.py` (an early client close, alone and with a background request), `tools/batch_test.py` (slot
tokens equal to solo tokens), `tools/batch_interleave_test.py` (a long read interleaved with decodes, giving way,
continuing from a slot), `tools/batch_groups_cache_test.py` (later turns of two alternating conversations reusing at
least 90 % of the prompt), `tools/serve_load.py --save/--compare` (greedy answers must not depend on who else ran),
`tools/needle_bench.py` (recall at 32K and 128K), and `serve/test_parallel.py` for the server's queueing and slot
bookkeeping on a fake engine without a GPU.

## What these runs cannot show, and what has to happen before a decision

- **Peak VRAM before and after the model is READY, and the C1/C2 peak difference**, need a restart per arm. The
  harness samples VRAM during a campaign; it cannot see the memory of a start-up it did not watch.
- **Expert-cache displacement** needs both arms at the same memory budget. On a card where the experts mostly do not
  fit, `hit_rate` and the `strata batch:` adapt lines are the evidence; a C2 loss is not called displacement until
  they say so.
- **Lane and row association under batch-MTP** - that a slot's MTP proposals and hidden state stay with its own
  sequence - is an engine-side question. `tests/core/batch_rows_test.cpp` checks the window's row layout on the CPU,
  including the contiguous proposal rows `--batch-mtp` uses; an end-to-end C2 check with drafts on is still open, and
  until it exists a C2 slowdown must not be blamed on anything else first.
- **Shared-prefix behaviour**: with `parallel 2`, two requests that share a long document have been seen to re-read
  it from token 0 (`docs/RESEARCH_RUNS.md`). Two long agent sessions are the same shape, so the conversation-cache and
  prefix settings belong in the record of every run.
- **Reproducibility**: the adaptive expert tier can change greedy output between runs. Performance is measured on the
  production configuration; exactness comparisons need the adapt-off configuration. Say which one a number came from.
- **Not tested by this harness at all**: C4 and beyond, multi-GPU, replicas, the AMD path, and anything that needs a
  second machine.
