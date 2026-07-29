# Compatibility Matrix

## Interpretation

This matrix records only gpt-task handling that differs from the standard Transformers execution flow defined in `architecture.md`.

The matrix MUST describe what gpt-task does at a specific stage. It MUST NOT repeat behavior implemented by a model, tokenizer template, processor, or Transformers itself.

Each row is a verified model coverage sample. An empty stage cell means gpt-task applies no model-specific handling at that stage. Rows MUST NOT become runtime model-ID allowlists.

## Model-Specific Handling Matrix

| Model | 1. Validate and determine | 2. Resolve runtime | 3. Load artifacts | 4. Render input | 5. Generate | 6. Decode raw output |
|---|---|---|---|---|---|---|
| [DeepSeek-V3.2](deepseek-v3.2.md) |  |  |  | **Both:** Match the loaded `deepseek_v32` configuration inside the family adapter; inject request tools into system data; map supported thinking controls; invoke the bundled encoder instead of `apply_chat_template`. |  |  |
| [ERNIE-4.5-VL-28B-A3B-PT](ernie-4.5-vl.md) |  | **TP:** Validate `hidden_size`, `num_heads`, and `intermediate_size` from the standard `ernie4_5_vl_moe` vision TP plan. | **Both:** Match the loaded remote configuration inside the artifact adapter and register the processor through the required model hook. | **Both:** Match the loaded remote configuration inside the vision adapter and resolve canonical image input through the required processor hook. |  |  |

## Evidence Boundaries

DeepSeek-V3.2 claims are bounded to the loaded `deepseek_v32` configuration, bundled official encoder integration, and prompt-adapter tests or examples. No model-specific image or TP handling is inferred.

ERNIE vision TP validation is plan-driven. Its artifact and vision adapters match the verified loaded remote `model_type` inside their own modules and fail when required hooks are absent. None of these behaviors is selected from the repository name.
