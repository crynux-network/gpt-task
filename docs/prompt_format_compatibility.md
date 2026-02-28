# Prompt Format Compatibility Specification

## Scope

This document defines required compatibility behavior for prompt rendering in `run_task()`. The compatibility layer SHALL operate as a pluggable adapter system and SHALL be responsible for model-family-specific **input rendering** decisions only.

## Unified Input Contract

The compatibility layer MUST accept the unified `GPTTaskArgs` contract with:

- `model`
- `messages`
- `tools` (optional)
- `template_args` (optional raw passthrough map)

`messages[].content` MUST support:

- Legacy string content.
- Block-list content with HF-compatible blocks:
  - Text block: `{ "type": "text", "text": "<text>" }`
  - Image block: `{ "type": "image", "base64": "<base64>" }`

## Adapter Contract

Each adapter MUST implement input rendering from unified messages/tools into model-compatible prompt text.

Adapters MUST be registered through a central registry. The registry MUST resolve one adapter per request, based on model-family matching rules.

## Model-Family Resolution Rules

Resolution MUST follow this order:

1. Family adapters for explicit targets.
2. Tokenizer-template adapter.
3. Generic fallback adapter.

Family adapters MUST take precedence over generic adapters when model identifiers match.

## Current Adapter Implementations

### DeepSeek-V3.2 (`deepseek-ai/DeepSeek-V3.2`)

- The DeepSeek adapter MUST use the vendored official encoder module:
  - `src/gpt_task/inference/prompt_adapters/encoding_dsv32.py`
- It MUST call official `encode_messages(...)` for prompt rendering.
- It MUST NOT use generic `tokenizer.apply_chat_template(...)` for this family.
- If `tools` are provided, it MUST inject tool schemas into a system message path compatible with the official encoder.
- It MUST remain focused on input rendering only.
- It MUST fail on invalid `thinking_mode` values.

#### DeepSeek `template_args` behavior

The DeepSeek adapter accepts `template_args` as a compatibility surface and normalizes them to official encoder arguments:

- Supported keys: `thinking`, `enable_thinking`, `thinking_mode`, `context`, `drop_thinking`, `add_default_bos_token`
- Unknown keys: silently ignored
- Precedence:
  - If `thinking_mode` is provided, it is used directly (`thinking` or `chat`)
  - Otherwise `thinking` / `enable_thinking` are mapped to `thinking_mode`
- Defaults:
  - `drop_thinking = (last message role == user)` when not explicitly set
  - `add_default_bos_token = true` when not explicitly set

### Tokenizer-Template Adapter

- This adapter MUST be used only when no explicit family adapter matches and `tokenizer.chat_template` exists.
- It MUST preserve tool rendering through template arguments when `tools` is provided.
- It MUST pass `template_args` through as raw keyword arguments to `tokenizer.apply_chat_template(...)`.
- For block-list content without images, it MUST normalize blocks to text before applying chat templates.
- Unknown keys MAY be ignored by the underlying tokenizer template implementation.

### Generic Fallback Adapter

- This adapter MUST be used only when no explicit family adapter matches and tokenizer-template rendering is unavailable.
- It MUST render prompts via plain message-content concatenation after block-to-text normalization.
- It MUST ignore `tools` and `template_args` with explicit warnings.
- It MUST remain reserved for future non-target model families.

## HF Message Mapping For VLM

When any canonical message contains image blocks, the runtime MUST bypass text-only adapter rendering and MUST map canonical blocks to HF chat-template message structures:

- Text block -> `{ "type": "text", "text": ... }`
- Image block -> `{ "type": "image", "base64": ... }`

## Template Args Semantics

`template_args` is a model-family-specific input extension map:

- For generic template path, keys are passthrough to tokenizer templates.
- For DeepSeek-V3.2 family, keys may be normalized to official encoder options as defined above.
- Unsupported keys may be dropped by adapter/tokenizer logic.
- Callers are responsible for passing model-compatible values.

Unsupported families MUST NOT fail solely due to unsupported `template_args`; they MUST log explicit warnings and continue where possible.

## Output Invariants

- `run_task()` response assembly MUST always use plain assistant text messages:
  - `{"role": "assistant", "content": decoded_text}`
- Compatibility adapters MUST NOT implement output parsing logic.

## Extensibility Requirements

- Adding a new model family MUST require only a new adapter implementation plus registry registration.
- Core `run_task()` streaming and generation control flow MUST remain adapter-agnostic.
- Adapter failures MUST propagate as task execution errors and MUST NOT silently downgrade target family behavior.
