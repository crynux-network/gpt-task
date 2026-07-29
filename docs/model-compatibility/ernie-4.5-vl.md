# ERNIE 4.5 VL Verified Coverage Sample

## Evidence Scope

This page records verified behavior for `baidu/ERNIE-4.5-VL-28B-A3B-PT` and the native Transformers 5.14.1 `ernie4_5_vl_moe` configuration used by TP-plan validation tests. It is a coverage sample and MUST NOT be used as a runtime model-ID allowlist.

The published remote checkpoint config and the native Transformers config are distinct evidence sources. Runtime MUST evaluate the configuration actually loaded for the task.

## Task Validation

The published checkpoint config declares:

- `model_type=ernie4_5_moe_vl`;
- `architectures=["Ernie4_5_VLMoeForConditionalGeneration"]`;
- remote `AutoConfig`;
- remote `AutoModel`;
- remote `AutoModelForCausalLM`;
- a vision config and multimodal image token fields.

Canonical text and image messages MUST retain their order. Image content MUST use raw-base64 canonical blocks before entering gpt-task.

The published template verifies `system` and `user` handling. Assistant tool-call history, tool results, and tool rendering are not verified for this sample and MUST NOT be claimed.

## Runtime and Artifact Resolution

The runtime MUST honor the repository `auto_map` with `trust_remote_code=True` when the loaded config type is not in the installed static AutoClass mapping. It MUST use the repository-declared `AutoModelForCausalLM` generation class and MUST NOT reject multimodal capability solely because the repository does not declare `AutoModelForImageTextToText`.

### Shared Behavior

The artifact and vision adapters MUST match the published remote `model_type=ernie4_5_moe_vl` inside their own modules. A repository ID, local path, or unrelated object exposing the same custom method MUST NOT select either adapter.

The artifact adapter MUST require callable `model.add_image_preprocess` and MUST register the loaded processor before generation. A matched model without that method or processor MUST fail explicitly.

### Classic Execution

Classic execution MUST use trusted remote code through pipeline auto-dispatch, MUST remain GPU-only, and MUST invoke the shared artifact registry after loading.

### Tensor-Parallel Execution

TP rank loading MUST invoke the same artifact registry after loading each model shard.

Remote AutoClass resolution does not establish TP eligibility. The actual resolved class and loaded config MUST expose an effective TP plan and MUST pass every general dimension validator before TP starts. A published remote config without the required effective text plan MUST use classic execution.

## Input Rendering

Image execution MUST require a compatible processor. The published chat template accepts text and image content and inserts model image placeholders.

The ERNIE vision adapter MUST require callable `processor.process_vision_info`. It MUST place canonical raw base64 in a transient `image_url.url`, render the prompt, call `process_vision_info`, and pass the resolved image data to the processor. A matched processor without the required method MUST fail explicitly and MUST NOT fall back to the standard processor adapter. This transient representation MUST NOT replace canonical task data.

Request tools and compatible `template_args` MUST pass to the processor template. No tool rendering or thinking behavior beyond the published template evidence MAY be inferred.

## Generation and Raw Output

Classic and eligible TP execution MUST preserve the same adapter identity, rendered messages, generation configuration, and raw decoding semantics.

gpt-task MUST return generated assistant text unchanged. Model-specific thinking syntax, Hermes emission, Qwen XML emission, tool-call history, and tool results are not verified for this sample.

## Tensor-Parallel Evidence

The native Transformers 5.14.1 `ernie4_5_vl_moe` config is the verified sharded-vision validation reference. Its vision plan shards attention qkv and projection plus MLP fc1 and fc2. Validation MUST cover vision `hidden_size`, `num_heads`, and `intermediate_size`. Its list-valued text MoE intermediate dimensions MUST each divide the selected world size.

For the native config, six visible GPUs are not a valid full world and `reduce_gpus` MUST select world size 4. This verified result is plan-driven and MUST NOT be generalized to the published remote config without independently resolving its plans.

The native config's successful validation MUST NOT be transferred automatically to the published remote config. The runtime MUST resolve and validate the loaded remote class and its plans independently.

Quantized requests, fewer than two GPUs, missing plans, unknown nonstandard vision plans, or indivisible sharded dimensions MUST use classic execution. `reduce_gpus` MUST select the largest independently valid reduced world of at least two.
