# OpenClaw compatibility qualification

This corpus is shared methodology for Strata-5080 agent testing. It is not a license to normalize arbitrary malformed model output.

## API cases

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- streaming and non-streaming
- benign query-string variants

## Tool cases

- one valid tool call
- multiple valid calls
- streamed tool name then arguments
- valid first call + malformed/truncated later call
- duplicate parameters
- undeclared tool
- long tool name
- alternate well-defined Qwen/Claude forms
- output limit during tool generation
- disconnect during tool generation

## Reasoning/history cases

- quoted reasoning-close marker
- retained thinking
- thinking omitted by client on next turn
- empty assistant turn
- later system/developer reminder
- same conversation follow-up
- rewritten suffix
- cancellation/retry

## Concurrency cases

- two independent tool loops
- cancel one lane
- C1 -> C2 -> C1
- follow-up after retained slot
- no state/tool leakage between lanes

## Required observations

For each case record:
- PASS / FAIL / UNSUPPORTED
- wire events
- finish reason
- parsed tool calls
- reused prefix tokens
- TTFT
- concurrent slot behaviour when applicable

## Safety boundary

A recovery or normalization may only preserve a demonstrably valid capability declared by the client. It must not invent tools, arguments or permissions.

At least one regression corpus must be independently derived rather than sharing the production parser's assumptions.
