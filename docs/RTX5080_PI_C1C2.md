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
  `c2mtp` is a third arm. Every non-C1 arm is compared against the C1 arm of its own context class, and the report
  prints a separate `c2mtp`-against-`c2` table - the same two slots with a drafter and without one - which is the
  drafts' share on its own.

Because the guard refuses to run when the server disagrees, every run file records the serving count the server
reported before any traffic was sent.

## What each number means

| number | what it is | where it comes from |
|---|---|---|
| **queue delay** | how long a request waits before the engine starts it | the server's own per-request `queue_s`, read from `GET /api/requests?id=` - which needs the server started with `--api-monitor`, because it is off by default (`serve/server.py`). The server does not echo a client request id, so a record is tied to a lane by **content**: every lane's prompt carries its own nonce (`[session A]`, `[session B]`) and the detail form returns the request text. A record that carries one nonce is that lane's; a record that carries two, or none, is matched to neither. When the monitor is off, or nothing matches, the lane's `queue_delay_s` stays `null` and `queue_delay_source` says why - an estimate is not written into a field named after a measurement. `queue_delay_record` names the record the number came from, so it can be checked by hand |
| **TTFT** | the client's: request sent → first useful streamed token (content, reasoning, or a tool-call argument) | the client's own clock |
| **model TTFT** | TTFT minus that lane's queue delay: what the engine itself took | derived, and `null` whenever the queue delay is `null` |
| **decode tok/s** | per lane, decode only, as the server counts it | every step's `timings`: `predicted_n` summed over the lane's steps over `decode_ms` summed over the same steps. One step's rate is that step's; the last chunk of a four-step lane says nothing about the other three |
| **per-slot decode** | those same tokens over the sum of the lanes' decode seconds | derived - what one slot is worth while it decodes |
| **aggregate decode** | all lanes' tokens over the campaign's wall seconds | derived - the number that rises while each user waits longer, which is why it is not the decision metric |
| **useful work** | tasks whose six checks all passed, 1.0 each; partial credit is recorded beside it and never counted as done | the scorer |
| **useful work per minute** | completed tasks ÷ campaign wall minutes | the decision metric |
| **slot overlap seconds** | seconds with two slots non-idle in two samples in a row, with no gap wider than the sampling interval between them | `/metrics` `live.slots[].state`. `live.running` counts *requests*, not slots, so it is recorded separately as `inflight_overlap_s` and is never the two-slot evidence |
| **VRAM / RAM peak** | highest sampled value during the campaign | `/metrics` `hardware`, which reports **bytes**, converted to the MiB and GiB the field names claim; falling back to `/v1/status` `machine`, which is already in MiB/GiB |

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

`--dry-run` prints the plan and opens no connection at all, not even `/v1/status`; without it the campaign runs.

```
python tools/pi_c1c2_bench.py --arm c2 --dry-run                                # the plan, nothing sent
python tools/pi_c1c2_bench.py --arm c2                                          # one campaign, short prompts
python tools/pi_c1c2_bench.py --arm c2 --context-class 64k --repeat 2           # a bigger class, second repetition
python tools/pi_c1c2_bench.py --arm c2 --cancel-lane B --cancel-after 20        # the cancellation fixture
python tools/pi_c1c2_compare.py 'bench/results/rtx5080-pi-c1c2/*.json' --min-repeats 3
```

It never starts, stops or reconfigures a server. Changing arms means restarting on that arm's configuration, which is
`tools/ab_engine.py`'s job and needs a GPU window of its own. `--expect-serving` is not a formality: a run whose
server reports a different number stops before sending anything, so a mislabelled arm cannot produce evidence. It
defaults to what the arm claims (1 for `c1`, 2 for `c2` and `c2mtp`), and naming a number that contradicts the arm is
refused rather than quietly obeyed.

Nothing about the arm is taken on trust either. `--batch-mtp` is not in `/v1/status`, so the run file records what the
engine said about drafts on the campaign's own requests (`draft_n`, `draft_n_accepted` in `timings`): a `c2mtp` run
whose requests report no drafts at all is written as NOT verified, and a `c2` run that reports drafts is written as
contradicted. `--confirm-batch-mtp` records how the operator checked it by hand when the engine reports nothing.

The harness bounds itself. `--timeout` is the quiet-socket limit of one read, `--campaign-deadline` bounds the whole
campaign, and a lane cut off by `--cancel-lane` is stopped at the socket, not between steps. A lane whose tool call
arrives as something other than an object - the model controls those bytes - costs that lane a check and is recorded;
it does not end the campaign. Reads are confined to this checkout, `grep` is `git grep` with a timeout and an output
cap, and nothing is written.

Run files are `bench/results/rtx5080-pi-c1c2/run-<arm>-<class>-<stamp>-r<repeat>[-tag].json` and `.md`; the repeat is
in the name because the gate counts distinct repeats, not files. Home, user and host names, IP addresses and the API
key are replaced by placeholders before anything is written (the same redaction `tools/strata_bench.py` uses, with the
key itself added to it), and a plain-HTTP `--url` that is not loopback is refused unless `--allow-remote` says the key
may travel in clear text.

## Checking the harness before any GPU time is spent

`python3 tools/test_pi_c1c2_bench.py` (89 tests, no GPU, no downloads) runs the harness against a stub server whose
answers are built from the shapes `serve/server.py` and `serve/telemetry.py` actually send - `/v1/status`
(`concurrency.serving`, `context.max_positions`, `machine` in MiB), `/metrics` (`live.slots[].state`, `hardware` in
bytes), `/api/requests` in its list and `?id=` forms, and streamed `chat.completions` chunks with fragmented tool
calls, `usage`, `timings` and a final `[DONE]`. It covers the things a wrong harness gets wrong quietly: a stream that
ends inside a tool call, a lane that is cancelled and must stop, a submission whose `path` is an object, a queue record
that belongs to the other lane, bytes written under a `_mib` name, and a `PROMOTE` built on partial credit.

## The gate

`tools/pi_c1c2_compare.py` prints medians and ranges per arm and context class, the ratios to the C1 arm of the same
class, and a recommendation. **A recommendation is not the decision** - the decision is written by hand in the issue
with the run files next to it. It fails closed: anything that is not in the files is DEFER, not assumed.

REJECT - something went wrong:

- an undeclared tool call, a request error, a tool call whose arguments were not an object, a stream that ended inside
  a tool call, a lane that stopped inside the harness, or a guard the server contradicted (including a
  `concurrency.serving` that is not the arm's);
- a cancellation fixture whose surviving lane did not finish;
- C1 completing more tasks than C2 - fewer completed tasks is a REJECT whatever the throughput says;
- each lane decoding under `--min-lane-retention` of its C1 rate: that is contention, not concurrency.

DEFER - something is not yet shown:

- fewer than `--min-repeats` **distinct** `--repeat` values in either arm - files are not repetitions;
- no C1 baseline **of the same context class**: a 64k C2 run is never compared against a `short` C1 run;
- the runs of the two arms not naming one and the same model;
- an arm label the server's own numbers did not confirm (`batch_mtp.verified` false);
- any C2 run that never showed two non-idle slots from `live.slots[].state`, or any C2 run whose lane queue delay is
  not tied to that lane by its nonce - a median cannot cover for a run that shows nothing;
- neither arm completing a task: partial credit is printed beside the decision and buys nothing, so a comparison of
  nothing with nothing is not a result;
- a gain inside `--promote-margin` (5 % by default), or a gain whose per-run ranges still overlap - the noise the runs
  themselves show has to be cleared, not just a fixed percentage.

A run taken with `--cancel-lane` is a fixture, not a measurement: one lane is cut off on purpose, so its numbers are
structurally low. Fixtures are checked for the cancellation gate and kept out of the medians the ratio is taken from,
so running the mandated fixture cannot drag its own arm down.

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
