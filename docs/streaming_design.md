# Streaming Design Specification

## Scope

This document defines the required streaming behavior for `run_task()` and its interaction with prompt rendering, tool-calling semantics, and usage accounting.

## Streaming Lifecycle

The streaming path MUST follow this sequence:

1. Resolve and render prompt via the prompt rendering layer.
2. Tokenize rendered prompt to establish input token baseline.
3. Start generation with a streamer attached.
4. Emit callback chunks for decoded assistant deltas.
5. Emit one final callback chunk with `finish_reason`.

The streaming path MUST preserve existing callback shape and MUST continue returning callback-driven chunks instead of a generator result object.

## Callback Contract

Each stream callback payload MUST include:

- `model`
- `choices` with one `delta` message and `finish_reason`
- `usage`

Intermediate chunks MUST set `finish_reason=null`. The terminal chunk MUST set `finish_reason` to `stop` or `length`.

## Token Accounting Semantics

Prompt token counting MUST be derived from prompt-boundary detection against generated token sequences and MUST stay aligned with non-streaming semantics.

For multimodal requests (image blocks in canonical input), prompt token baseline MUST be derived from the normalized text view of canonical content so streaming and non-streaming usage counters stay parity-compatible within one runtime contract.

Completion token counting MUST increment only for generated assistant tokens after prompt boundary resolution.

Total tokens MUST equal:

- `prompt_tokens + completion_tokens`

Usage values in each emitted chunk MUST reflect current cumulative streaming state and the final chunk MUST contain final usage totals.

## Finish-Reason Determination

- `stop` MUST be emitted when generation ends on EOS.
- `length` MUST be emitted when generation stops without EOS.

The final chunk MUST always contain a terminal `finish_reason`.

## Prompt Rendering Coexistence

Streaming MUST consume rendered prompts exactly as non-streaming does.
For multimodal requests, streaming MUST consume the same HF chat-message mapping as non-streaming.

Prompt rendering is input-only and MUST NOT perform output parsing in either streaming or non-streaming paths. Rendering integration MUST NOT alter streaming chunk shape or usage semantics.

## Tool-Calling Coexistence

When `tools` are provided, rendering MUST continue to pass tool metadata through rendering-controlled formatting. Streaming emission MUST remain content-first and MUST NOT remove or rewrite generated tool-call text in transit.

## Non-Regression Invariants

After compatibility-layer refactor, the following invariants MUST hold:

- Streaming callback payload schema remains unchanged.
- Usage counters remain parity-compatible with prior behavior.
- Final chunk contract remains one explicit terminal chunk.
- Streaming path remains independent from non-streaming response-object assembly.
- Prompt-rendering backend selection MUST NOT break existing streaming control flow.
