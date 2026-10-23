from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

from gpt_task import models
from gpt_task.cache import ModelCache
from gpt_task.config import Config, get_config

from ..errors import error_context
from ..executed_gpu_count import clear_executed_gpu_count, set_executed_gpu_count
from ..execution_dtype import clear_execution_dtype, set_execution_dtype
from ..execution_quantization import clear_execution_quantization, set_execution_quantization
from ..inference import run_task
from ..model_adapters.input import contains_image_blocks
from ..model_adapters.quantization import resolve_task_quantization
from ..model_adapters.runtime_strategy import (
    has_remote_auto_map,
    is_builtin_mapped,
    resolve_runtime_strategy,
)
from ..model_adapters.tp_plan import validate_effective_tp_plan
from ..utils import load_model_kwargs
from .executor import shutdown_tp_executor, submit_tp_task
from .result import TPTaskResult
from .runtime_strategy import (
    TP_MODEL_LOADER_CAUSAL_LM,
    TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
    TPRuntimeStrategy,
)

_logger = logging.getLogger(__name__)

TP_FALLBACK_DEVICE_MAP = "device_map"
TP_FALLBACK_REDUCE_GPUS = "reduce_gpus"

@dataclass(frozen=True)
class _TPTaskResolution:
    world_size: int
    strategy: TPRuntimeStrategy


def _resolve_tp_fallback() -> str:
    value = os.environ.get("GPT_TP_FALLBACK", TP_FALLBACK_DEVICE_MAP)
    if value == TP_FALLBACK_REDUCE_GPUS:
        return TP_FALLBACK_REDUCE_GPUS
    return TP_FALLBACK_DEVICE_MAP


def _load_model_config(args: models.GPTTaskArgs, config: Config):
    from transformers import AutoConfig

    model_kwargs = load_model_kwargs(config=config)
    if args.revision is not None:
        model_kwargs["revision"] = args.revision
    return AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
        **model_kwargs,
    )


def _mapped_model_class(model_config, model_loader: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.auto_factory import _get_model_class

    if model_loader == TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT:
        mapping = AutoModelForImageTextToText._model_mapping
    else:
        mapping = AutoModelForCausalLM._model_mapping

    config_type = type(model_config)
    if config_type not in mapping:
        return None
    try:
        return _get_model_class(model_config, mapping)
    except (ImportError, RuntimeError):
        return None


def _is_builtin_mapped(model_config, model_loader: str) -> bool:
    return is_builtin_mapped(model_config, model_loader)


def _has_remote_auto_map(model_config, loader: str) -> bool:
    return has_remote_auto_map(model_config, loader)


def _resolve_runtime_strategy(
    model_config,
    *,
    has_image_input: bool = False,
) -> TPRuntimeStrategy:
    return resolve_runtime_strategy(
        model_config,
        has_image_input=has_image_input,
    )


def _has_tp_plan(model_config) -> bool:
    text_config = model_config.get_text_config()
    return bool(getattr(text_config, "base_model_tp_plan", None))


def _iter_config_tp_plans(model_config):
    pending = [("config", model_config)]
    seen = set()
    while pending:
        prefix, current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        plan = getattr(current, "base_model_tp_plan", None)
        if plan:
            yield prefix, plan

        for name in getattr(current, "sub_configs", {}):
            pending.append((f"{prefix}.{name}", getattr(current, name, None)))


def _effective_tp_plan(
    model_config,
    strategy: TPRuntimeStrategy,
    *,
    quantizer_config: Any | None = None,
) -> Dict[str, str]:
    plan: Dict[str, str] = {}
    model_class = _mapped_model_class(model_config, strategy.model_loader)
    if model_class is not None:
        for name, style in (getattr(model_class, "_tp_plan", None) or {}).items():
            plan[f"model.{name}"] = style

    text_config = model_config.get_text_config()
    configs = [
        ("text", getattr(text_config, "base_model_tp_plan", None) or {}),
        *_iter_config_tp_plans(model_config),
    ]
    seen_configs = set()
    for prefix, config_plan in configs:
        plan_id = id(config_plan)
        if plan_id in seen_configs:
            continue
        seen_configs.add(plan_id)
        for name, style in config_plan.items():
            plan[f"{prefix}.{name}"] = style

    plan.update(_quantizer_tp_plan_updates(quantizer_config, base_plan=plan))
    return plan


def _quantizer_tp_plan_updates(
    quantizer_config: Any | None,
    *,
    base_plan: Dict[str, str] | None = None,
) -> Dict[str, str]:
    if quantizer_config is None:
        return {}
    seed = dict(base_plan or {})
    update_tp_plan = getattr(quantizer_config, "update_tp_plan", None)
    if callable(update_tp_plan):
        updated = update_tp_plan(dict(seed))
        if isinstance(updated, dict):
            return {str(k): str(v) for k, v in updated.items()}
    try:
        from transformers.quantizers import AutoHfQuantizer

        quantizer = AutoHfQuantizer.from_config(quantizer_config)
        update_tp_plan = getattr(quantizer, "update_tp_plan", None)
        if callable(update_tp_plan):
            updated = update_tp_plan(dict(seed))
            if isinstance(updated, dict):
                return {str(k): str(v) for k, v in updated.items()}
    except Exception:
        return {}
    return {}


def _dims_divisible_by(
    model_config,
    strategy: TPRuntimeStrategy,
    world_size: int,
    *,
    quantizer_config: Any | None = None,
) -> bool:
    effective_plan = _effective_tp_plan(
        model_config,
        strategy,
        quantizer_config=quantizer_config,
    )
    return validate_effective_tp_plan(
        model_config,
        effective_plan,
        world_size,
    )


def _resolve_tp_task(
    args: models.GPTTaskArgs, config: Config, visible_gpus: int
) -> Optional[_TPTaskResolution]:
    # The fallback decision depends on the task args, the model config, the
    # visible GPU count, and the node-owned GPT_TP_FALLBACK setting. Nodes in
    # the same TP pool MUST use the same GPT_TP_FALLBACK value so every node
    # makes the identical choice and results stay consistent across the pool.
    if visible_gpus < 2:
        return None

    model_config = _load_model_config(args, config)
    quant_resolution = resolve_task_quantization(
        args,
        model_config=model_config,
    )
    quantizer_config = (
        None
        if quant_resolution is None
        else quant_resolution.transformers_config
    )
    if not _has_tp_plan(model_config):
        return None
    strategy = _resolve_runtime_strategy(
        model_config,
        has_image_input=contains_image_blocks(args.messages),
    )

    if _dims_divisible_by(
        model_config,
        strategy,
        visible_gpus,
        quantizer_config=quantizer_config,
    ):
        return _TPTaskResolution(visible_gpus, strategy)

    if _resolve_tp_fallback() != TP_FALLBACK_REDUCE_GPUS:
        return None

    for k in range(visible_gpus - 1, 1, -1):
        if _dims_divisible_by(
            model_config,
            strategy,
            k,
            quantizer_config=quantizer_config,
        ):
            return _TPTaskResolution(k, strategy)
    return None


def _resolve_tp_world_size(
    args: models.GPTTaskArgs, config: Config, visible_gpus: int
) -> Optional[int]:
    resolution = _resolve_tp_task(args, config, visible_gpus)
    return None if resolution is None else resolution.world_size


def run_task_tp(
    args: models.GPTTaskArgs | None = None,
    *,
    version: str | None = None,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    tools: Sequence[Dict[str, Any]] | None = None,
    tool_choice: models.ToolChoice | None = None,
    response_format: models.ResponseFormat | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    template_args: Mapping[str, Any] | None = None,
    stream_callback: Callable[[models.GPTTaskStreamResponse], None] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantization_config: models.QuantizationConfig | Mapping[str, Any] | None = None,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> Union[models.GPTTaskResponse, models.GPTTaskStreamResponse]:
    """Run a GPT task on the tensor parallel executor.

    Tasks that cannot run under tensor parallelism are delegated to the
    classic run_task path in-process. When GPT_TP_FALLBACK=reduce_gpus and
    the full visible GPU count cannot shard the model, the largest K >= 2
    that divides all TP-sharded dimensions is used instead.
    """
    if config is None:
        config = get_config()

    clear_executed_gpu_count()
    clear_execution_dtype()
    clear_execution_quantization()

    with error_context(local_files_only=config.local_files_only):
        if args is None:
            args = models.GPTTaskArgs.model_validate(
                {
                    "version": version,
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "response_format": response_format,
                    "generation_config": generation_config,
                    "template_args": template_args,
                    "seed": seed,
                    "dtype": dtype,
                    "quantization_config": quantization_config,
                }
            )

        import torch

        visible_gpus = torch.cuda.device_count()
        resolution = _resolve_tp_task(args, config, visible_gpus)

    # The two execution paths must never hold models in VRAM at the same
    # time: a classic-fallback task tears down the rank group so its full
    # model load does not compete with the cached shards, and a TP task
    # evicts the worker-level cache before the rank group loads shards.
    if resolution is None:
        shutdown_tp_executor()
        # run_task records visible GPU count as the classic executed count
        # and logs the final device_map execution plan.
        return run_task(
            args,
            stream_callback=stream_callback,
            config=config,
            model_cache=model_cache,
        )

    world_size = resolution.world_size
    set_executed_gpu_count(world_size)
    _logger.info(
        "Task execution plan: mode=%s, gpu_count=%d, visible_gpus=%d, model=%s",
        "tensor_parallel",
        world_size,
        visible_gpus,
        args.model,
    )

    if model_cache is not None:
        model_cache.clear()

    result = submit_tp_task(
        world_size,
        resolution.strategy,
        args,
        config,
        stream_callback,
    )
    if not isinstance(result, TPTaskResult):
        raise RuntimeError("Tensor-parallel executor returned an invalid result.")
    set_execution_dtype(result.execution_dtype)
    set_execution_quantization(result.execution_quantization)
    return result.response
