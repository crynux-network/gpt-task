# Model Compatibility Documentation

## Document Index

- `architecture.md`: The common six-stage execution contract from canonical `GPTTaskArgs` to raw assistant text.
- `compatibility-matrix.md`: Model and execution-path coverage across validation, runtime resolution, artifact loading, input rendering, generation, and raw decoding.
- `tool-history-arguments.md`: The Transformers object contract and verified model-interface exceptions for assistant tool-call history.
- `deepseek-v3.2.md`: Verified DeepSeek-V3.2 family-adapter coverage sample.
- `qwen3.5.md`: Verified Qwen3.5 coverage sample.
- `qwen3.6.md`: Verified Qwen3.6 coverage sample.
- `ernie-4.5-vl.md`: Verified ERNIE 4.5 VL coverage sample.

## Authority

This directory is the authoritative specification for model execution inside gpt-task, beginning with validated canonical `GPTTaskArgs` and ending with raw assistant text.

Questions about canonical task validation, prompt and chat-template behavior, processor input, tools, tool history, thinking template controls, AutoClass selection, remote `auto_map`, tensor-parallel plans, classic or tensor-parallel execution, generation, and raw decoding MUST be answered from this directory.

Model pages are verified coverage samples. They MUST NOT be interpreted or implemented as runtime model-ID allowlists. Runtime behavior MUST be selected from request data, loaded model configuration, processor or tokenizer capability, AutoClass resolution, effective TP plans, and validated dimensions.

Classic and tensor-parallel execution MUST remain backend strategies inside the common six-stage flow. Core flow, backend, and registry modules MUST NOT contain model IDs, local paths, `model_type` values, architecture names, custom model methods, custom processor methods, model-specific message formats, or nonstandard TP-plan semantics. Those values and behaviors MUST exist only in the adapter or hook module that owns them. Adapter matching MUST use loaded configuration identity inside `matches(context)`.

Every implemented stage registry MUST use ordered adapters and a standard Transformers fallback. Stage 1, Stage 5, and Stage 6 hook contracts MUST remain documented even when no nonstandard implementation exists; empty hook packages MUST NOT be created.

OpenAI-compatible request conversion, Relay and Node transport, and raw-output conversion into public API responses are specified by `crynux-bridge/docs/model-compatibility/`. This directory MUST NOT redefine those API transformations.

## Update Requirements

Every document in this directory MUST:

1. State final, testable behavior with `MUST`, `MUST NOT`, or `SHALL`.
2. Separate system-wide contracts from model-specific verified coverage.
3. Identify the evidence boundary for every model-specific claim.
4. Record an unsupported or unverified feature as not verified; it MUST NOT infer support from a related model name.
5. Organize behavior under the common processing stages in `architecture.md`.
6. Preserve model-generated assistant text as raw gpt-task output.
7. Keep classic, tensor-parallel, and reduced-world-size behavior consistent with `../tensor_parallel.md`.

Documentation MUST NOT contain recommendations, alternatives, speculation, future placeholders, or model-ID-based runtime gates.
