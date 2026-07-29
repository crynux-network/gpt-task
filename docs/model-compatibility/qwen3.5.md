# Qwen3.5 Verified Coverage Sample

## Evidence Scope

This page records verified behavior for `Qwen/Qwen3.5-35B-A3B`. It is a coverage sample and MUST NOT be used as a runtime model-ID allowlist or as evidence for every Qwen3.5 checkpoint.

The verified sources are the published model configuration and template, the Transformers 5.14.1 `qwen3_5_moe` configuration and AutoClass mappings, and gpt-task VLM and TP tests.

## Task Validation

The published config declares `Qwen3_5MoeForConditionalGeneration`, `model_type=qwen3_5_moe`, a text sub-config, and a vision sub-config. Runtime MUST resolve multimodal generation from the loaded config and AutoClass mapping, not from the `Qwen3.5` name.

Canonical text and image messages MUST retain their order. Image content MUST use raw-base64 canonical blocks before entering gpt-task.

## Runtime and Artifact Resolution

### Shared Behavior

Both backends MUST use the loaded native configuration, standard artifact fallback, tokenizer or processor templates, and common generation and raw-output contracts.

### Classic Execution

Classic execution MUST load the native image-text model and processor through pipeline auto-dispatch.

### Tensor-Parallel Execution

The native text config provides a TP plan. The effective plan covers attention, linear attention, routed experts, the shared expert, embeddings, and `lm_head`. The vision config has no TP plan, so every TP rank MUST replicate the complete vision tower.

## Input Rendering

Text-only input MUST use the tokenizer chat-template adapter. Image input MUST use `AutoProcessor` chat-template processing with canonical raw-base64 image blocks.

Tools, template-ready assistant tool history, tool results, and compatible `template_args` MUST pass to the selected tokenizer or processor template. gpt-task MUST NOT derive an API response policy from `enable_thinking`.

The published Qwen3.5 tool template iterates assistant-history function arguments as object items. The supplied canonical history MUST therefore contain object-valued `function.arguments`. gpt-task MUST preserve that representation and pass it to the template without OpenAI response conversion.

## Generation and Raw Output

Classic execution MUST use GPU-only `device_map="auto"`. TP execution MUST use the same rendered messages and generation configuration on every rank.

Generated thinking, final text, and native tool-call syntax MUST remain unchanged in raw assistant content. gpt-task MUST NOT parse Qwen XML, create structured API tool calls, or remove thinking.

Model emission of Hermes JSON is not verified and MUST NOT be inferred from parser support in another component.

## Tensor-Parallel Evidence

The verified MoE config has:

- `hidden_size=2048`;
- `num_attention_heads=16`;
- `num_key_value_heads=2`;
- `moe_intermediate_size=512`;
- `shared_expert_intermediate_size=512`;
- `vocab_size=248320`.

World size 2 satisfies the verified sharded dimensions. A full world size greater than 2 MUST fail `num_key_value_heads` divisibility. Under `reduce_gpus`, the runtime MUST select world size 2 when every other sharded dimension is divisible by 2. Under `device_map`, or when no reduced size is valid, the runtime MUST use classic execution.

Quantized requests and requests with fewer than two visible GPUs MUST use classic execution.
