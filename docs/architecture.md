# Architecture Principles

## Project Goals

The project targets maximum generality while keeping scope intentionally narrow.

1. Maximize input-side compatibility across model/tokenizer variations.
2. Do no output-side processing; return raw decoded text directly (important for result cross-validation in the Crynux Network).
3. Delegate downstream-specific post-processing (such as tool call parsing) to downstream components.
4. The system must enforce deterministic execution across heterogeneous multi-GPU environments so outputs remain reproducible and verifiable across nodes.

## Principles

1. Input is the compatibility boundary
   - This project solves input normalization and prompt rendering compatibility.
   - The canonical message contract accepts `content` as either a legacy string or a block list.
   - Image blocks in canonical content use base64 source only.
   - It does not solve output structuring/parsing for downstream consumers.

2. Output stays raw and transport-stable
   - Non-streaming responses always return plain assistant text.
   - Streaming responses emit plain assistant text deltas without output post-processing.

3. Keep core generation flow stable
   - Model loading, generation config, token accounting, and streaming contracts remain independent from rendering backend details.

## Component Split

1. Task contract and public API
   - Defines request/response schema and entrypoint parameters.
   - Files:
     - `src/gpt_task/models/args.py`
     - `src/gpt_task/inference/inference.py`

2. Prompt rendering layer
   - Converts unified task input into model-ready prompt text or HF chat message blocks.
   - Receives `template_args` as input-side extension arguments.
   - Related docs:
     - `docs/prompt_format_compatibility.md`
   - Files:
     - `src/gpt_task/inference/prompt_adapters/interface.py`
     - `src/gpt_task/inference/prompt_adapters/registry.py`
     - `src/gpt_task/inference/prompt_adapters/deepseek_v32.py`
     - `src/gpt_task/inference/prompt_adapters/template.py`
     - `src/gpt_task/inference/prompt_adapters/fallback.py`
     - `src/gpt_task/inference/prompt_adapters/utils.py`

3. Generation and streaming runtime
   - Executes model inference, streaming callbacks, finish reasons, and usage accounting.
   - Uses one model-capability-driven flow: `AutoProcessor.from_pretrained(...)` + `pipeline(task=None, ...)`.
   - Related docs:
     - `docs/streaming_design.md`
     - `docs/model_cache.md`
   - File:
     - `src/gpt_task/inference/inference.py`

4. Output boundary
   - Assembles plain-text assistant messages only.
   - No tool-call formatting/parsing in this layer.
   - Downstream systems own output interpretation.
   - File:
     - `src/gpt_task/inference/inference.py`

## Implementation Path

1. `run_task()` validates/binds input into `GPTTaskArgs`.
2. Prompt-rendering layer resolves either prompt text (LLM path) or HF chat blocks (VLM path).
3. Runtime executes generation with a single auto-dispatch pipeline (streaming or non-streaming).
4. Response assembly returns raw decoded text in assistant content.

## File Navigation Map

- Request/response schema: `src/gpt_task/models/args.py`
- Entrypoint and runtime flow: `src/gpt_task/inference/inference.py`
- Renderer protocol: `src/gpt_task/inference/prompt_adapters/interface.py`
- Renderer registry/routing: `src/gpt_task/inference/prompt_adapters/registry.py`
- Family-specific renderer: `src/gpt_task/inference/prompt_adapters/deepseek_v32.py`
- Generic template renderer: `src/gpt_task/inference/prompt_adapters/template.py`
- Generic fallback renderer: `src/gpt_task/inference/prompt_adapters/fallback.py`
- Shared rendering helpers: `src/gpt_task/inference/prompt_adapters/utils.py`
- Prompt compatibility spec: `docs/prompt_format_compatibility.md`
- Streaming runtime spec: `docs/streaming_design.md`
- Model cache spec: `docs/model_cache.md`

## Scope Boundary

- In scope: input compatibility, prompt rendering, generation execution, raw text output.
- Out of scope: output formatting, tool-call parsing/normalization, downstream business interpretation.
