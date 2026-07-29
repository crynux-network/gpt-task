# Qwen3.6 Verified Coverage Sample

## Evidence Scope

This page records verified behavior for `Qwen/Qwen3.6-35B-A3B`. It is a coverage sample and MUST NOT be used as a runtime model-ID allowlist or as evidence for every Qwen3.6 checkpoint.

The verified sources are the published model configuration and template, its Transformers `qwen3_5_moe` architecture mapping, and gpt-task VLM and TP tests.

## Task Validation

The published Qwen3.6 config declares `Qwen3_5MoeForConditionalGeneration`, `model_type=qwen3_5_moe`, a text sub-config, and a vision sub-config. The marketing family name and implementation architecture name differ. Runtime MUST use the loaded config and AutoClass mapping and MUST NOT derive the class from `Qwen3.6`.

Canonical text and image messages MUST retain their order. Image content MUST use raw-base64 canonical blocks before entering gpt-task.

## Runtime and Artifact Resolution

### Shared Behavior

Both backends MUST use the loaded `qwen3_5_moe` configuration, standard artifact fallback, tokenizer or processor templates, and common generation and raw-output contracts. Runtime MUST NOT create a separate Qwen3.6 architecture rule from the repository name.

### Classic Execution

Classic execution MUST load the native image-text model and processor through pipeline auto-dispatch.

### Tensor-Parallel Execution

TP execution MUST use the native AutoClass mappings and effective plans from the loaded configuration.

## Input Rendering

Text-only input MUST use the tokenizer chat-template adapter. Image input MUST use `AutoProcessor` chat-template processing with canonical raw-base64 image blocks.

Tools, template-ready assistant tool history, tool results, and compatible `template_args` MUST pass to the selected tokenizer or processor template. gpt-task MUST preserve the supplied thinking control and MUST NOT force a mode based on the presence of tools.

The published Qwen3.6 tool template consumes object-valued function arguments in assistant history. The supplied canonical history MUST contain object-valued `function.arguments`, and gpt-task MUST pass that representation to the template unchanged.

## Generation and Raw Output

Classic execution MUST use GPU-only `device_map="auto"`. TP execution MUST use the same rendered messages and generation configuration on every rank.

Generated thinking, final text, and native tool-call syntax MUST remain unchanged in raw assistant content. gpt-task MUST NOT parse Qwen XML, create structured API tool calls, or remove thinking.

Hermes JSON emission by this sample is not verified.

## Tensor-Parallel Evidence

The verified config has:

- `hidden_size=2048`;
- `num_attention_heads=16`;
- `num_key_value_heads=2`;
- `moe_intermediate_size=512`;
- `shared_expert_intermediate_size=512`;
- `vocab_size=248320`.

The native effective TP plan covers attention, linear attention, routed experts, the shared expert, embeddings, and vocabulary-sharded `lm_head`. The vision config has no TP plan, so the complete vision tower MUST be replicated on every rank.

World size 2 is the verified TP size. A larger full world MUST fail because `num_key_value_heads=2` is not divisible by that world size. With `reduce_gpus`, a larger visible GPU set MUST reduce to world size 2 when all remaining sharded dimensions are compatible. With `device_map`, or without a valid reduced world, execution MUST fall back to classic.

Quantization and fewer than two visible GPUs MUST force classic execution.
