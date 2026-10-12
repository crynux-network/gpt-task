# Quantization

## Scope and authority

This document specifies quantized model execution from canonical `GPTTaskArgs`
through runtime selection and model loading. It is subordinate to the common
six-stage contract in [architecture.md](architecture.md).

The task schema MUST provide a stable backend-neutral container. Current
backend names, keys, values, and combinations MUST NOT be encoded as a closed
TaskArgs enum or an upstream execution allowlist. gpt-task MUST validate a
configuration against its runtime adapter, official configuration class,
installed libraries, hardware, and model metadata.

The configurations in this document define current runtime adapter coverage.
They MUST NOT be interpreted as model-ID allowlists.

## Stage 1: Validate TaskArgs

`quantize_bits` MUST NOT be accepted.

`quantization_config` MUST be absent, `null`, or an object with this stable
shape:

```json
{
  "quant_method": "bitsandbytes",
  "quant_kwargs": {}
}
```

`quant_method` MUST be a non-empty string. `quant_kwargs` MUST be a JSON
object. The TaskArgs schema MUST NOT enumerate backend names,
backend-specific keys, or backend-specific values.

gpt-task MUST select an adapter by `quant_method`. The adapter MUST
construct the corresponding official Transformers configuration explicitly
and MUST NOT forward unchecked keyword arguments to a third-party
constructor.

The complete canonical `quantization_config` MUST participate in execution
identity.

Model files with an embedded `quantization_config` and a task with a non-null
`quantization_config` MUST be rejected. Embedded configuration and task
configuration MUST NOT be merged or overridden.

## Common task-time configurations

### BitsAndBytes 4-bit

The current adapter MUST support this configuration structure:

```json
{
  "quant_method": "bitsandbytes",
  "quant_kwargs": {
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": false
  }
}
```

The adapter MUST support:

- `bnb_4bit_quant_type` values `nf4` and `fp4`;
- `bnb_4bit_compute_dtype` values `float16`, `bfloat16`, and `float32`; and
- both boolean values for `bnb_4bit_use_double_quant`.

When `bnb_4bit_compute_dtype` is absent, the runtime MUST resolve the task
`dtype`. An explicit task dtype MUST be used directly. `dtype="auto"` MUST use
the dtype declared by `AutoConfig`, or the Transformers float32 fallback when
the configuration does not declare one.

All fields accepted by the installed official `BitsAndBytesConfig` MUST remain
representable. The adapter MUST use that class to validate field types, values,
and relationships.

### BitsAndBytes INT8

The current adapter MUST support this configuration structure:

```json
{
  "quant_method": "bitsandbytes",
  "quant_kwargs": {
    "load_in_8bit": true,
    "llm_int8_threshold": 6.0,
    "llm_int8_skip_modules": null
  }
}
```

The adapter MUST pass `llm_int8_threshold` through the official configuration
class. Tests MUST cover the common values `0.0` and `6.0`.

`llm_int8_skip_modules` MUST remain representable. Its effective value MUST
participate in execution and cache identity.

`llm_int8_enable_fp32_cpu_offload` MUST be false. CPU and disk offload are
prohibited. `llm_int8_has_fp16_weight` MUST be false for network inference.

### Fine-grained FP8

The current adapter MUST support this configuration structure:

```json
{
  "quant_method": "fp8",
  "quant_kwargs": {
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "float",
    "dequantize": false
  }
}
```

The adapter MUST pass activation scheme, weight block size, scale format, and
`dequantize` through the official fine-grained FP8 configuration class.
Dynamic activation, a `[128, 128]` weight block, and `scale_fmt` values
`float` and `ue8m0` are required test profiles. A dequantized load MUST be
reported as floating-point execution rather than quantized execution.

## Stage 2: Resolve runtime and execution backend

Quantization and tensor parallelism are independent execution dimensions.
Classic quantized execution MUST use `device_map="auto"`.

Quantization presence MUST NOT prevent a task that satisfies the normal TP
plan and static dimension checks from attempting tensor-parallel execution.
TP loading MUST use the Transformers distributed TP configuration and MUST
NOT pass `device_map`.

The effective TP plan MUST include updates returned by the selected quantizer.
Weight shards and their scale, zero-point, packing, grouped-expert, and other
kernel-required state MUST remain aligned on every rank.

The runtime MUST NOT maintain a verified model, quantization configuration, or
kernel allowlist for TP. Absence of a prior multi-GPU test MUST NOT reject a
statically eligible task. A runtime incompatibility discovered during
quantized TP loading or generation MUST propagate through the existing
execution-error contract.

CPU and disk offload MUST remain disabled for every quantization method.

Quantized classic and quantized TP execution MUST use separate validation
pools. Quantized TP pools MUST also separate GPU model, world size, platform,
effective quantization configuration, weight format, selected kernel, and
dependency contract.

### Windows CUDA

The current Windows CUDA dependency set supports:

- BitsAndBytes NF4 and FP4 4-bit execution;
- BitsAndBytes INT8 execution; and
- fine-grained FP8 execution when the GPU, PyTorch, `kernels`, and
  `triton-windows` runtime pass the required probe.

AWQ and GPTQModel MUST return a clear unsupported-runtime error while the
required GPTQModel runtime is unavailable on the supported Windows build.

### Linux CUDA

The current Linux CUDA dependency set supports:

- BitsAndBytes NF4 and FP4 4-bit execution;
- BitsAndBytes INT8 execution;
- fine-grained FP8 execution;
- AWQ INT4 model execution; and
- GPTQModel INT4 model execution.

Actual execution MUST still satisfy the selected backend's dependency,
kernel, model configuration, and GPU-architecture requirements.

### macOS

The CUDA quantization backends defined by this document MUST return an
unsupported-runtime error on macOS.

Fine-grained FP8 CUDA execution MUST require compute capability 8.9 or newer.
A fallback that dequantizes weights because the GPU or kernel is unsupported
MUST be rejected and MUST NOT be reported as FP8 execution.

## Stage 3: Load model files

### Embedded fine-grained FP8

The runtime MUST recognize fine-grained FP8 from model metadata rather than
from a model ID or filename. Current coverage SHALL include:

- dynamic activation, `[128, 128]` blocks, and float32 scales; and
- dynamic activation, `[128, 128]` blocks, and UE8M0 scales.

### AWQ

AWQ configuration belongs to the model files. TaskArgs MUST NOT override its
bits, group size, zero-point convention, packing format, or kernel.

Current Linux coverage SHALL include:

- 4-bit, group size 128, asymmetric GEMM models;
- group size 64 models;
- symmetric, `zero_point=false`, Marlin-compatible models; and
- GEMM and batch-one GEMV execution paths.

### GPTQModel

GPTQModel configuration belongs to the model files. TaskArgs MUST NOT override
its bits, group size, activation ordering, symmetry, checkpoint format, or
kernel.

Current Linux coverage SHALL include:

- 4-bit, group size 128, `desc_act=false`, symmetric models;
- `desc_act=true` models;
- group size 32 models;
- GPTQ v1 and GPTQ v2 checkpoint formats; and
- asymmetric GPTQ v2 models.

Model files whose matrix dimensions, group size, packing, zero-point
semantics, or checkpoint format do not satisfy an available kernel MUST fail
while loading before generation. The runtime MUST NOT alter the embedded
configuration to make it loadable.

### Unsupported model execution

Compressed-tensors checkpoints that are expanded to floating-point weights
MUST NOT be reported as quantized execution.

LLM GGUF packed execution is outside current CUDA coverage. A Transformers
path that dequantizes GGUF weights MUST NOT be reported as GGUF quantized
execution.

TorchAO, Quanto, HQQ, SINQ, AQLM, VPTQ, generic MXFP4, and generic NVFP4 are
outside current runtime adapter coverage. Their names remain representable by
the stable TaskArgs container. gpt-task MUST return an unsupported-backend
error when no adapter exists.

## Cache and execution metadata

The loaded-model cache identity MUST include:

- the resolved model-file identity;
- the resolved task dtype;
- the complete canonical task-time configuration or complete embedded model
  configuration;
- the resolved compute dtype;
- skipped modules;
- the selected quantization implementation and kernel-relevant configuration;
  and
- classic executor identity.

Cache entries with different effective configurations MUST NOT be shared.

The runtime MUST report the actual execution dtype, quantization method,
normalized effective configuration identity, and executor identity. It MUST
not report a requested method when loading fell back to dequantized weights.

## Example TaskArgs for 24 GB CUDA

The following TaskArgs are coverage samples for common Hub checkpoints that
fit a single 24 GB CUDA device under the current adapters. They MUST NOT be
interpreted or implemented as a runtime model-ID allowlist.

### Task-time BitsAndBytes NF4 on Qwen2.5-14B

```json
{
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "dtype": "bfloat16",
  "seed": 42,
  "generation_config": {"max_new_tokens": 64},
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "quant_kwargs": {
      "load_in_4bit": true,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": "bfloat16",
      "bnb_4bit_use_double_quant": false
    }
  }
}
```

### Task-time BitsAndBytes INT8 on Qwen3-8B

```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "dtype": "bfloat16",
  "seed": 42,
  "generation_config": {"max_new_tokens": 64},
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "quant_kwargs": {
      "load_in_8bit": true,
      "llm_int8_threshold": 6.0
    }
  }
}
```

### Task-time fine-grained FP8 on Qwen2.5-7B

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "dtype": "bfloat16",
  "seed": 42,
  "generation_config": {"max_new_tokens": 64},
  "quantization_config": {
    "quant_method": "fp8",
    "quant_kwargs": {
      "activation_scheme": "dynamic",
      "weight_block_size": [128, 128],
      "scale_fmt": "float",
      "dequantize": false
    }
  }
}
```

### Embedded AWQ INT4 Qwen2.5-32B

Linux CUDA only. TaskArgs MUST NOT set `quantization_config`; the checkpoint
owns the AWQ configuration.

```json
{
  "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "dtype": "auto",
  "seed": 42,
  "generation_config": {"max_new_tokens": 64}
}
```

### Embedded GPTQ INT4 Qwen2.5-32B

Linux CUDA only. TaskArgs MUST NOT set `quantization_config`; the checkpoint
owns the GPTQ configuration.

```json
{
  "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "dtype": "auto",
  "seed": 42,
  "generation_config": {"max_new_tokens": 64}
}
```

Runnable scripts for these samples are under `examples/quantization_*.py`.

## Verification boundary

Common configuration profiles MUST pass a real model load and generation test
on each target platform. Repeated execution of one
canonical task inside one validation pool MUST produce an identical final
score.

Multi-GPU tests MUST cover model loading, weight and quantization-state
sharding, kernel alignment, generation, cache reuse, and repeated final-score
equality for representative BitsAndBytes, fine-grained FP8, AWQ, and GPTQModel
configurations. Test results MUST document compatibility and drive framework
fixes. They MUST NOT become a runtime model or configuration allowlist.

The following observations establish baseline feasibility only:

- BitsAndBytes NF4, BitsAndBytes INT8, and float-scale fine-grained FP8
  produced identical repeated tokens on Windows CUDA;
- AWQ INT4 and GPTQModel INT4 produced identical repeated tokens on Linux
  CUDA; and
- GPTQModel models can fail validly when their matrix dimensions do not
  satisfy the selected Marlin kernel.

These observations MUST NOT be generalized to an untested configuration,
model-file format, GPU architecture, operating system, or dependency version.
