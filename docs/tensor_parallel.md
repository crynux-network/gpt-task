# Tensor Parallel Inference

## Authority and Scope

This document defines the tensor-parallel backend contract subordinate to the immutable six-stage execution and adapter contracts in [`model-compatibility/architecture.md`](model-compatibility/architecture.md). Model compatibility evidence MUST come from [`model-compatibility/`](model-compatibility/AGENTS.md). Model pages are verified samples and MUST NOT become runtime model-ID allowlists.

Runtime selection MUST depend on loaded configuration, AutoClass resolution, effective plans, visible GPUs, quantization, and dimension validation.

## Config and AutoClass Resolution

`run_task_tp` MUST load the complete model config exactly once while resolving a task. Installed Transformers mappings and trusted repository `auto_map` entries MUST both participate in AutoClass resolution.

A static config mapping MUST use its mapped class. When a config type is absent from a static mapping, a repository `auto_map` entry MUST be resolved through the requested AutoClass with `trust_remote_code=True`. The runtime MUST NOT hardcode remote class names.

`AutoModelForImageTextToText` MUST take precedence when a static mapping or remote `auto_map` entry exists. Otherwise TP MUST select `AutoModelForCausalLM`. A trusted remote causal mapping combined with image input MUST also require `AutoProcessor`; text-only input MUST use `AutoTokenizer`. `vision_config` alone MUST NOT select a model loader.

Remote AutoClass resolution does not prove TP compatibility. The resolved class and loaded config MUST satisfy the same plan and dimension requirements as installed classes.

## Eligibility and Fallback

TP requires:

1. At least two visible GPUs.
2. `quantize_bits=None`.
3. A selected generation AutoClass strategy.
4. A text config with a non-empty `base_model_tp_plan`.
5. A complete effective TP plan that passes all applicable validators for the candidate world size.

Image content MUST NOT by itself force classic execution.

`GPT_TP_FALLBACK` has two behaviors:

- `device_map` MUST use classic `run_task` with `device_map="auto"` when the full visible GPU count is incompatible.
- `reduce_gpus` MUST test smaller world sizes in descending order, select the largest compatible size of at least two, and use classic execution when none is compatible.

Every value other than `reduce_gpus`, including `classic`, MUST use `device_map` behavior. A TP model load MUST pass `tp_plan="auto"` and MUST NOT pass `device_map="auto"`.

Before classic fallback starts, the TP rank group MUST shut down. Before TP starts, the worker-level classic cache MUST clear. Classic and TP model copies MUST NOT occupy GPU memory simultaneously.

### Single GPU-Resident Cache

The inference coordinator MUST keep exactly one GPU-resident model owner at a time: the worker-level classic or SD cache, or the TP rank-group shard cache. Same-backend reuse remains allowed. Cross-backend transitions MUST evict the previous owner before the next backend loads:

- Before the coordinator dispatches classic `run_task`, any live TP rank group MUST shut down through `shutdown_tp_executor()`. `run_task` MUST remain independent of TP lifecycle management.
- Before SD inference or SD fine-tuning starts, any live TP rank group MUST shut down.
- Before SD fine-tuning starts, the worker-level cache MUST also clear because that path loads outside the shared cache.
- Consecutive eligible TP tasks MUST NOT shut down the rank group between tasks so compatible shards remain cached.

## Effective TP Plan

The effective plan MUST include:

- the resolved model class `_tp_plan`;
- the text config `base_model_tp_plan`;
- every applicable nested sub-config `base_model_tp_plan`, including vision plans.

The validator MUST infer required dimensions from non-replicated effective-plan entries. Recognized attention, linear-attention, MLP, expert, embedding, and output-head entries MUST map to their corresponding dimensions. Every inferred dimension MUST divide the candidate world size:

- `hidden_size`;
- `num_attention_heads`;
- `num_key_value_heads`;
- `intermediate_size`;
- `moe_intermediate_size`;
- `shared_expert_intermediate_size`.

Each integer in a list-valued inferred dimension MUST divide the candidate world size. `vocab_size` MUST divide the candidate when the effective plan shards an embedding or `lm_head`. A dimension that is present in config but not sharded by the effective plan MUST NOT reject a candidate.

An unknown non-replicated text-plan entry MUST use classic execution. A `replicated` entry and `moe_tp_experts` MUST NOT add a divisibility constraint.

An empty vision plan means the complete vision tower is replicated on every rank. Vision dimensions MUST NOT affect world-size selection in that case.

A non-empty standard vision plan MUST validate `hidden_size`, `num_heads` or `num_attention_heads`, and `intermediate_size`. Nonstandard patch-embedding and adapter entries MUST have a complete model-type-specific validator. An unknown nonstandard plan MUST use classic execution.

Static validation establishes compatibility only for known sharded dimensions. A Transformers DTensor failure during model load or generation MUST propagate as a task execution failure.

## Rank Loading

The selected runtime strategy MUST carry a model loader and a processor requirement independently. `AutoModelForImageTextToText` always requires `AutoProcessor`. `AutoModelForCausalLM` MUST require `AutoProcessor` for image input when a remote causal `auto_map` exists, and MUST otherwise use `AutoTokenizer`. Trusted remote mappings MUST be invoked through their declaring AutoClass.

Every rank MUST load with `tp_plan="auto"`, use its explicit CUDA rank device, set the same seed, and use the same generation configuration. Rank 0 MUST log `model.tp_plan`, create the streamer when streaming, and emit the response. Nonzero ranks MUST NOT create a streamer or emit a response.

After loading, every rank MUST construct the shared model adapter context and resolve the same backend-neutral artifact registry used by classic execution. Adapter matching MUST use loaded configuration identity inside the adapter module. The adapter MUST perform any required upstream model-processor registration before generation. The common rank loader MUST NOT discover or invoke a model-specific registration method directly.

Persistent rank processes MUST cache one model tuple per rank. A runtime-strategy, model, dtype, or quantization key change MUST replace that tuple. A world-size change MUST recreate the rank group.

## Input and Output

Image and text requests MUST use the shared backend-neutral input renderer. It MUST resolve the same vision or text adapter as classic execution from the loaded adapter context and MUST pass the complete task arguments to that adapter.

The standard Transformers processor adapter MUST:

1. Convert canonical messages with `to_hf_chat_messages`.
2. Call `processor.apply_chat_template` with generation prompt insertion, tokenization, dictionary output, and PyTorch tensors.
3. Pass tools and processor-supported `template_args`.

The shared renderer MUST recursively move every adapter-returned tensor, including nested token, attention, and image tensors, to the current-rank CUDA device.

`TPRuntimeStrategy.requires_processor` MUST govern image routing. An image request without that strategy contract or without the required loaded processor MUST fail explicitly and MUST NOT enter text rendering. Text-only requests MUST use the shared text adapter and tokenizer and MUST preserve tools, tool history, `template_args`, and model-specific prompt behavior.

Non-streaming rank 0 output MUST match the canonical gpt-task response shape. Direct streaming MUST emit raw assistant deltas and one terminal finish reason. TP execution MUST NOT parse thinking or tool-call output.

## Determinism

Ranks MUST set deterministic PyTorch behavior and pin NCCL to `Ring`, `Simple`, with NVLS disabled before importing torch. CPU and disk offload MUST NOT occur.

Classic and TP output MUST remain in separate validation pools. Every node in one TP pool MUST use the same GPU model, world size, platform, executor marker, and fallback behavior.

## Verified Coverage

Verified model samples and their exact evidence boundaries are:

- [Qwen3.5](model-compatibility/qwen3.5.md)
- [Qwen3.6](model-compatibility/qwen3.6.md)
- [ERNIE 4.5 VL](model-compatibility/ernie-4.5-vl.md)
- [Compatibility matrix](model-compatibility/compatibility-matrix.md)

These links document coverage only. Runtime MUST NOT compare the requested model ID against them.
