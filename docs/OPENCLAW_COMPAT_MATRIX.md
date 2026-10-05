# OpenClaw compatibility matrix

Issue: #4  
Branch: `research/issue-4-openclaw-compat`

The goal is to qualify Strata as an agent backend without duplicating tests the upstream tree already has.

## Current classification

| Area | Case | Coverage | Current classification |
| --- | --- | --- | --- |
| OpenAI Chat | ordinary request | existing `serve/test_server.py` | COVERED |
| Anthropic Messages | ordinary request | existing `serve/test_server.py` | COVERED |
| OpenAI Responses | Codex-style history/tool items | existing `serve/test_responses.py` | COVERED |
| Routes | Anthropic `/v1/messages?beta=true` | existing `ClientShapes.test_query_string` | COVERED |
| Routes | Chat query strings | existing `ClientShapes.test_query_string` | COVERED |
| History | mid-conversation Anthropic system reminder | existing `ClientShapes` | COVERED |
| History | late OpenAI developer/system messages | existing `ClientShapes` | COVERED |
| History | Responses late developer item | existing `serve/test_responses.py` | COVERED |
| Reasoning | literal/quoted `<think>` / `</think>` in user/history/tool text | existing `LiteralThinkTags` | COVERED |
| Tools | tag-like text inside parameter values | existing `ToolCallTerminators` | COVERED |
| Tools | streamed call cut off before completion | existing `UnfinishedToolCall` | COVERED |
| Tools | a complete declared call | existing + new boundary regression | COVERED |
| Tools | **model emits an undeclared tool name** | new `serve/test_openclaw_compat.py` + parser guard | FIXED ON BRANCH |
| Tools | valid first call followed by undeclared truncated call | new regression | FIXED ON BRANCH |
| Tools | request tool-list malformed | existing request validation | COVERED |
| Responses | cut-off historical function-call arguments | existing Responses parser regression | COVERED |
| Lifecycle | early client close / cancellation | existing `tools/early_close_test.py` | COVERED, GPU/server qualification still required |
| Lifecycle | C1→C2→C1 state transition | existing batching/interleave tests | COVERED, RTX qualification in #2 |
| Lifecycle | cancel one concurrent lane | existing early-close/batch logic | COVERED, RTX qualification in #2 |
| Prefix | late system/developer keeps history prefix in place | implementation explicitly designed for this | COVERED structurally; measure reuse on RTX |
| Empty assistant | OpenClaw-shaped empty turn + later developer reminder | `serve/test_openclaw_compat.py` | ADDED ON BRANCH |
| Finish semantics | OpenClaw-specific output-limit/tool cut-off combinations | partial upstream coverage | TODO |
| Responses route | benign `/v1/responses?beta=true` query string | `serve/test_openclaw_compat.py` | ADDED ON BRANCH |
| Real agent loop | OpenClaw tool loop on target model | hardware/integration | TODO |

## New safety boundary

The current upstream-style parser accepted any syntactically valid model-emitted tool name.

That meant:

```text
caller declares: [read_file, write_file]
model emits:     delete_everything(...)
                         ↓
parser surfaces executable tool_call
                         ↓
client is expected to reject it
```

That is not an acceptable security boundary for this fork.

The #4 branch changes the boundary to:

```text
model tool name ∈ caller-declared tools
    ├─ yes -> existing tool streaming/final-call path
    └─ no  -> no tool_start, no tool_args, no executable tool_call,
              and unfinished undeclared markup is suppressed
```

The client may still apply its own allowlist, but Strata no longer relies on that downstream defense.

## Deliberately unchanged behavior

- Declared tool calls keep their current IDs, streaming and argument parsing.
- A **declared** call cut off mid-output keeps the existing incomplete-call semantics so the client can distinguish it from a runnable completed call.
- Tag-like strings inside declared argument values remain data.
- The change does not authorize undeclared MCP tools.
- Request-side tool schemas are still validated by the existing OpenAI/Anthropic/Responses normalization paths.

## Remaining CPU work

Before calling the CPU corpus complete:

1. run the complete relevant unittest set, including the existing server/Responses suites and the new compatibility suite;
2. retain any newly demonstrated failure as a minimal regression before fixing it.

The existing `UnfinishedToolCall` coverage already pins token-limit cut calls and preservation of a valid complete call when a later streamed call is incomplete, so the fork does not duplicate that case.

## Remaining target work

On the RTX/OpenClaw service:

1. one complete tool loop through each API dialect actually used by OpenClaw;
2. repeated follow-up turns with prefix reuse recorded;
3. two simultaneous independent tool loops;
4. cancel one lane while the other continues;
5. verify no cross-lane tool or conversation state leakage.

Those integration items can share #2's C1/C2 evidence harness.
