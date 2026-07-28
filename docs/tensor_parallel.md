# Tensor Parallel Inference

## Eligibility and Fallback

`run_task_tp` MUST load the complete model config exactly once while resolving eligibility. A config mapped by `AutoModelForImageTextToText` MUST use the VLM TP path. Otherwise, a config mapped by `AutoModelForCausalLM` MUST use the causal-LM TP path. A config with neither mapping MUST use classic `run_task`; the presence of `vision_config` alone MUST NOT determine the model family.

The text config returned by `get_text_config()` MUST declare a non-empty `base_model_tp_plan`. Tensor parallel execution also requires at least two visible GPUs and `quantize_bits=None`. Image content does not change these eligibility rules.

`GPT_TP_FALLBACK` has two behaviors:

- `device_map` MUST immediately use classic `run_task`, which loads through `device_map="auto"`, when the full visible GPU count is incompatible.
- `reduce_gpus` MUST test smaller world sizes in descending order and select the largest compatible K where K is at least 2. It MUST use classic `run_task` when no compatible K exists.

Every value other than `reduce_gpus`, including `classic`, MUST use `device_map` behavior. A TP model load MUST pass `tp_plan="auto"` and MUST NOT pass `device_map="auto"`.

## Effective TP Plan Validation

The effective plan MUST include the mapped model class `_tp_plan`, the text config `base_model_tp_plan`, and the vision config `base_model_tp_plan`. Every candidate world size MUST divide each present text dimension:

- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`
- `moe_intermediate_size`
- `shared_expert_intermediate_size`

`vocab_size` MUST also be divisible when an effective-plan entry shards an embedding or `lm_head` on the vocabulary dimension.

An empty vision plan means the complete vision tower is replicated on every rank, and vision dimensions MUST NOT affect world-size selection. A non-empty vision plan MUST validate vision `hidden_size`, `num_heads` or `num_attention_heads`, and `intermediate_size`. Nonstandard vision patch-embedding and adapter entries MUST have a complete model-type-specific validator; an unknown nonstandard plan MUST use classic execution.

`ernie4_5_vl_moe` is the verified sharded-vision reference. Its validator covers `hidden_size`, `num_heads`, and `intermediate_size` for the plan entries that shard attention qkv/proj and MLP fc1/fc2. Llama 4 additionally validates its patch-embedding and vision-adapter plan through its model-specific validation boundary.

## Rank Model and Input Handling

The causal-LM rank path MUST load `AutoTokenizer` and `AutoModelForCausalLM`. The VLM rank path MUST load `AutoProcessor` and `AutoModelForImageTextToText`. Both model classes MUST load with `tp_plan="auto"`, and rank 0 MUST log the resulting `model.tp_plan`.

Image requests MUST use the shared `prepare_vlm_inputs` helper. The helper MUST:

1. Convert canonical messages with `to_hf_chat_messages`.
2. Invoke `processor.apply_chat_template` with generation prompt insertion, tokenization, dictionary output, and PyTorch tensors.
3. Move every returned tensor, including token, attention, and image tensors, to the explicit current-rank CUDA device.

Text-only VLM requests MUST use the existing prompt adapter and `processor.tokenizer`. They MUST preserve tools, `template_args`, and model-specific prompt-template behavior. Only rank 0 MAY create a streamer, while every rank MUST call generation with the same generation configuration.

## Transformers 5.14.1 Coverage

Representative VLM families with a verified native text TP plan are Qwen2-VL, Qwen2.5-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.6, Gemma 3, Gemma 3n, Gemma 4, Llama 4, GLM-4V, GLM-4V-MoE, GLM-OCR, Mistral 4, Aria, Ovis2, ERNIE 4.5 VL MoE, DeepSeek-OCR2, InternVL, LLaVA, SmolVLM, PaliGemma, Kimi K2.5, and Cohere2 Vision.

Representative mapped VLM families without a native text TP plan are Qwen3-VL dense, the original Idefics architecture, mLlama/Llama 3.2 Vision, BLIP, BLIP-2, Chameleon, Florence-2, and Fuyu. They MUST use classic `device_map="auto"` execution.

These family lists are versioned examples and MUST NOT be used as runtime allowlists. Runtime eligibility MUST be derived from the loaded Transformers config, the applicable AutoModel mapping, the effective plan, and dimension validation.

Qwen3.6-35B-A3B is represented by the `qwen3_5_moe` architecture. Its effective plan covers attention, MoE experts, the shared expert, linear attention, and the vocabulary-sharded `lm_head`. Its default `num_key_value_heads=2` permits two ranks and rejects larger world sizes. With `reduce_gpus`, a larger visible GPU set MUST reduce to K=2 when all remaining dimensions are compatible. Its vision tower has no plan and is replicated on every rank.

Static plan validation establishes compatibility only for known sharded dimensions. Runtime Transformers DTensor failures remain task execution failures and MUST propagate through the existing error path.
