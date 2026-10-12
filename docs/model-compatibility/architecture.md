# Model Compatibility Architecture

## Scope

gpt-task owns the execution contract from canonical `GPTTaskArgs` to raw assistant text. OpenAI-compatible request conversion, network task transport, and downstream response parsing are outside this boundary.

Quantization TaskArgs, runtime adapters, artifact configurations, platform
restrictions, cache identity, and verification requirements are specified in
[quantization.md](quantization.md).

Every supported model MUST pass through the same six processing stages. A model-specific adapter or runtime strategy MAY specialize a stage, but it MUST preserve the stage input and output contracts.

Classic `device_map="auto"` and tensor-parallel execution are backend strategies inside this flow. Neither backend encloses or replaces the six-stage model-compatibility contract.

Every stage registry MUST evaluate ordered adapters through `matches(context)` and MUST end with a standard Transformers fallback. The context MUST carry loaded configuration identity and the artifacts required by that stage. Core flow and registry modules MUST NOT inspect model IDs, local paths, `model_type`, architecture names, custom method names, model-specific message formats, or model-specific TP-plan formats. An adapter MAY inspect loaded `config.model_type` or architectures inside its own `matches` implementation. All matching values and nonstandard behavior MUST remain private to that adapter module.

## Stage 1: Validate TaskArgs and Establish Determinism

### Purpose

This stage establishes one canonical, reproducible inference request.

### Input and Output

The input is JSON or programmatic arguments. The output is a validated `GPTTaskArgs` instance containing:

- a non-empty model repository ID or local path;
- ordered `system`, `user`, `assistant`, and `tool` messages;
- string, text-block, and raw-base64 image-block content;
- optional tools and tool history;
- optional `tool_choice` with `none`, `auto`, `required`, or one declared named function;
- optional `response_format` with `json_object` or `json_schema`;
- generation configuration and template arguments;
- seed, dtype, and quantization controls.

Canonical image blocks MUST contain valid raw base64 and MUST NOT contain `image_url`, a data URL prefix, or additional fields.

The runtime MUST enable deterministic PyTorch behavior and set the task seed before model execution. Validation failures MUST become task-argument errors rather than model execution errors.

Canonical task objects MUST reject unknown top-level fields. Tool names MUST be unique, and a named choice MUST reference exactly one declared function. Each tool parameter schema and response schema MUST be valid JSON data, MUST be at most 65,536 UTF-8 bytes, MUST have nesting depth at most 32, and MUST contain at most 1,024 declared properties.

### Adapter or Hook Boundary

A validation hook MAY add constraints required by a loaded model interface. It MUST receive and return the canonical request semantics and MUST NOT rewrite, replace, or extend `GPTTaskArgs`.

## Stage 2: Resolve Runtime and Execution Backend

### Purpose

This stage selects a loadable generation interface and an execution backend from runtime capabilities.

### Shared Contract

Runtime-strategy and TP-plan hooks MUST consume loaded configuration and return standard strategy or eligibility results. Stage 2 core functions MAY resolve and invoke those hooks but MUST NOT branch on model identity or nonstandard plan vocabulary.

### Classic Execution

Classic execution MUST use `AutoProcessor.from_pretrained(..., trust_remote_code=True)` and Transformers pipeline auto-dispatch with `device_map="auto"`. It MUST prohibit CPU and disk offload.

### Tensor-Parallel Execution

Tensor-parallel resolution MUST load the complete model config with `trust_remote_code=True`. It MUST resolve static Transformers mappings and trusted repository `auto_map` entries without reproducing a remote class-name allowlist.

`AutoModelForImageTextToText` MUST take precedence when declared. Otherwise the runtime MUST use `AutoModelForCausalLM`. Image input through a remote causal mapping MUST additionally require a compatible processor.

TP eligibility MUST require at least two visible GPUs, a non-empty text TP
plan, and successful validation of every dimension sharded by the effective
plan. Quantization presence MUST NOT make a task ineligible. The effective
plan MUST merge the resolved class `_tp_plan`, text `base_model_tp_plan`,
applicable nested sub-config plans, and updates supplied by the selected
quantizer.

When the full visible world is invalid, `reduce_gpus` MUST select the largest compatible world size of at least two. All other fallback values MUST select classic execution. Remote `auto_map` resolution alone MUST NOT establish TP support.

### Adapter or Hook Boundary

Classic resolution has no dimension-sharding validator. TP-plan validators are TP-only and MUST own model-specific plan semantics. The TP backend MUST assemble the effective plan, pass it to the validator registry, and consume only the returned supported dimensions or unsupported result.

### Output

The stage output is either:

- a classic pipeline and processor; or
- a TP runtime strategy, validated world size, and identical rank-loading contract.

## Stage 3: Load Model Artifacts

### Purpose

This stage creates or reuses the exact processor, tokenizer, model, and cache entry required by the resolved backend.

### Classic Execution

Classic execution MUST load one GPU-only pipeline keyed by model, dtype, and quantization settings.

### Tensor-Parallel Execution

Every TP rank MUST load the same model strategy with `tp_plan="auto"` on its explicit CUDA rank. Runtime strategy, model, dtype, quantization, or world-size changes MUST invalidate incompatible cached artifacts.

Classic and TP model copies MUST NOT occupy GPU memory simultaneously. TP fallback MUST shut down the rank group before classic loading, and TP execution MUST clear the worker classic cache before rank loading. The owning execution coordinator MUST shut down any live TP rank group before dispatching direct classic or other non-TP GPU work on the same visible devices. Classic `run_task` MUST remain independent of TP lifecycle management. The process MUST keep exactly one GPU-resident model owner at a time while allowing same-backend cache reuse.

After each load or cache reuse, the backend MUST read the loaded main model's parameter dtype and expose its normalized PyTorch name as task execution metadata. Classic execution MUST publish this metadata from the selected pipeline model. TP rank 0 MUST return the dtype with the internal rank result, and the parent process MUST publish it before returning the public task response. TP fallback MUST publish the classic model's dtype. The dtype metadata MUST remain separate from the raw assistant response contract.

### Adapter or Hook Boundary

After either backend loads its artifacts, it MUST construct the same adapter context from the loaded model configuration, model, processor, and tokenizer. It MUST resolve and invoke the shared artifact registry. An artifact adapter MAY configure a nonstandard model-processor relationship. The standard adapter MUST leave artifacts unchanged. Backend loaders MUST NOT discover or invoke model-specific methods directly.

## Stage 4: Render Model Input

### Purpose

This stage converts canonical messages into the exact prompt or tensor inputs accepted by the loaded model without changing conversation order or meaning.

Classic and TP execution MUST call the same backend-neutral input-rendering entry point with the loaded adapter context.

### Text Input

Text-only execution MUST resolve prompt handling in this order:

1. registered family adapter;
2. tokenizer chat-template adapter;
3. plain fallback adapter.

A family adapter MUST implement an upstream model interface that cannot be expressed by the generic tokenizer template. Family selection MUST use loaded configuration identity inside the adapter. The generic template adapter MUST pass tools and compatible `template_args` to `apply_chat_template`. The fallback adapter MUST warn when it ignores tools or template arguments.

### Image Input

Image execution MUST use the loaded processor.

Image execution MUST resolve input rendering in this order:

1. registered vision adapter matching the loaded configuration identity;
2. standard Transformers processor adapter.

A registered vision adapter MUST implement an upstream processor interface that cannot be expressed by the standard Transformers processor flow. Its module MUST own custom message conversion and custom processor method invocation. A matched adapter with a missing required method MUST fail explicitly and MUST NOT fall back to the standard adapter.

The standard Transformers processor adapter MUST convert canonical messages to Hugging Face chat messages and call the processor template with `add_generation_prompt=True`, `tokenize=True`, `return_dict=True`, and `return_tensors="pt"`.

Processor-only representations MUST NOT replace canonical task data. Every vision adapter MUST receive the complete task arguments, pass tools and compatible template arguments, and return model-ready inputs. The common stage MUST recursively move every returned tensor to the model or rank device.

### Tools, History, Template Arguments, and Thinking

Tools, assistant tool-call history, tool results, and compatible template arguments MUST reach the selected adapter or processor in canonical order.

The generic tokenizer and processor paths MUST pass canonical `tool_choice` as a chat-template argument when it is present. A family adapter MUST decide whether and how its verified interface consumes that value. Core rendering code MUST NOT convert the choice into model-specific tags.

The generic Transformers tokenizer and processor paths MUST receive object-valued `function.arguments` in assistant tool history. This is the Transformers chat-template contract. A registered family adapter MAY require a different representation when its authoritative upstream interface defines one; the DeepSeek-V3.2 encoder requires a JSON string and expands it with `json.loads`.

gpt-task MUST preserve the representation supplied in `GPTTaskArgs` and MUST NOT infer whether a template expects an object or string by trial rendering. OpenAI-compatible string-to-object adaptation belongs to the upstream API boundary.

gpt-task MUST receive `template_args` as an optional input-extension map. The generic tokenizer and processor paths MUST pass compatible entries as template keyword arguments. A family adapter MUST normalize only the entries supported by its upstream interface. The plain fallback adapter MUST warn when it ignores the map.

Thinking is an input-template capability. gpt-task MUST apply a compatible thinking control when present and MUST NOT infer an OpenAI response policy from that control.

### Classic Execution

Classic execution MUST pass rendered text to the pipeline and MUST pass rendered multimodal tensors to the loaded model generation interface.

### Tensor-Parallel Execution

TP execution MUST use `TPRuntimeStrategy.requires_processor` as the image-routing contract. An image request whose selected strategy does not require a processor, or whose required processor is missing, MUST fail before text rendering. Text requests MUST continue through the shared text adapter registry.

### Adapter or Hook Boundary

Text and vision adapters MUST convert canonical messages into a standard prompt string or model-ready tensor mapping. The common renderer MUST preserve canonical task data and recursively move returned tensors to the selected device.

## Stage 5: Generate

### Purpose

This stage executes model generation with one resolved generation configuration.

Task generation overrides MUST be merged with the loaded model generation defaults. Classic and TP paths MUST preserve the same task-level generation semantics.

Stage 5 MUST resolve constraints through an independent ordered `GenerationAdapterRegistry` with a standard Transformers fallback. A generation adapter MUST match loaded configuration identity or tokenizer capability. Core generation and registry modules MUST NOT select a format from the request model ID.

`response_format=json_object` MUST compile the XGrammar built-in JSON grammar. `response_format=json_schema` MUST compile its JSON Schema. These response constraints MUST use the loaded tokenizer and MUST NOT select a model-family tool adapter.

Tool constraints MUST apply for `required`, a named function, and `auto` with at least one `strict=true` tool. `none`, and `auto` without a strict tool, MUST NOT add a tool constraint. Tools MUST still reach Stage 4 when the choice is `none`. A tool constraint MUST take precedence over `response_format`.

The tool generation registry MUST support `llama`, `kimi`, `deepseek_r1`, `deepseek_v3_1`, `deepseek_v3_2`, `deepseek_v4`, `qwen_3`, `qwen_3_coder`, `qwen_3_5`, `glm_4_7`, `hermes`, `hy_v4`, and `kimi_k3`. XGrammar built-ins MUST own the first ten formats. The `hermes`, `hy_v4`, and `kimi_k3` builders MUST retain the Apache-2.0 behavior from vLLM commit `588a813a60448f6f71dfa0e0ecd52cc0bacc7329` and MUST retain source attribution.

Tokenizer information and compiled grammars MAY be cached. Every generation invocation MUST create a new XGrammar Transformers logits processor so each returned sequence owns independent matcher state. Constraint compilation MUST complete within five seconds. Invalid schema syntax, unsupported model format, tokenizer incompatibility, and compilation timeout MUST become task-argument errors and MUST NOT fall back to unconstrained generation.

Constrained generation MUST set `num_beams=1`. Multiple return sequences MUST use one independent matcher per sequence.

CUDA constrained generation MUST provide a Triton package that matches the installed PyTorch release and platform. On Linux that package MUST be the standard `triton` dependency selected by PyTorch. On Windows that package MUST be `triton-windows`. Unconstrained generation MUST NOT require Triton.

### Classic Execution

Classic execution MUST invoke the pipeline or model with the rendered text or multimodal tensors.

Text pipeline, multimodal `model.generate`, and direct streaming MUST receive the same resolved logits processor list.

### Tensor-Parallel Execution

Every TP rank MUST execute the same inputs and generation configuration. Rank 0 MUST own streaming and response emission; nonzero ranks MUST participate in generation without emitting output.

Every TP rank MUST build the same grammar from the same tokenizer contract and apply the logits processor to its complete vocabulary logits.

### Adapter or Hook Boundary

A generation hook MAY specialize invocation for a verified model interface. It MUST preserve the resolved generation configuration, deterministic backend invariants, rank participation, and streaming ownership.

## Stage 6: Decode Raw Output

### Purpose

This stage converts generated token IDs into the stable gpt-task response contract.

Non-streaming output MUST contain one choice per generated sequence with:

- the original choice index;
- `role="assistant"`;
- decoded raw assistant text in `content`;
- `finish_reason="stop"` or `finish_reason="length"`;
- prompt, completion, and total token usage.

Direct streaming MUST emit decoded raw assistant deltas, cumulative usage, and one terminal finish reason.

Decoding MUST preserve generated thinking, final text, and tool-call syntax. gpt-task MUST NOT strip reasoning, parse output-format protocols, create OpenAI tool calls, or change finish reason to `tool_calls`.

### Classic Execution

Classic decoding MUST normalize pipeline text or generated token IDs into the common raw-output contract.

### Tensor-Parallel Execution

TP rank 0 MUST decode generated token IDs into the same raw-output contract. Nonzero ranks MUST NOT decode or emit output.

### Adapter or Hook Boundary

A decode hook MAY specialize token decoding for a verified model interface. It MUST return unchanged raw assistant text semantics and MUST NOT parse or normalize model-generated protocols.

## Cross-Backend Invariants

Classic and TP execution MUST consume equivalent canonical messages, tools, template arguments, and generation settings.

Stage 1 validation and Stage 4 input contracts are shared. Stages 2 and 3 use backend-specific loading strategies with the same model adapter identity. Stage 5 uses backend-specific invocation through one generation contract. Stage 6 preserves one raw-output contract. Artifact, text, vision, generation, and decode hooks apply to both backends whenever required; TP-plan validators apply only to TP eligibility.

Classic, streaming, multimodal, and TP invocation MUST apply equivalent structured-output constraints. This equivalence does not establish model-specific output quality or numeric equivalence across backend validation pools.

Classic and TP results MUST remain in separate validation pools because their floating-point operation order differs. Nodes in one TP pool MUST use the same GPU model, world size, platform, executor marker, and fallback configuration.

Model pages and the compatibility matrix document verified coverage of these stages. They MUST NOT be used as runtime model-ID allowlists.
