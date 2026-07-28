from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

from gpt_task import models
from gpt_task.cache import ModelCache
from gpt_task.config import Config, get_config

from ..errors import error_context
from ..inference import run_task
from ..utils import load_model_kwargs
from .executor import shutdown_tp_executor, submit_tp_task
from .model_family import (
    TP_MODEL_FAMILY_CAUSAL_LM,
    TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
)

_logger = logging.getLogger(__name__)

TP_FALLBACK_DEVICE_MAP = "device_map"
TP_FALLBACK_REDUCE_GPUS = "reduce_gpus"

_TEXT_TP_SHARDED_DIM_ATTRS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)
_VISION_TP_SHARDED_DIM_ATTRS = (
    "hidden_size",
    "num_heads",
    "num_attention_heads",
    "intermediate_size",
)
_STANDARD_VISION_PLAN_PARTS = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".attn.qkv",
    ".attn.proj",
    ".mlp.fc1",
    ".mlp.fc2",
)
_VISION_PLAN_VALIDATORS = {
    "ernie4_5_vl_moe": (
        "hidden_size",
        "num_heads",
        "intermediate_size",
    ),
    "llama4": (
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
    ),
}


@dataclass(frozen=True)
class _TPTaskResolution:
    world_size: int
    model_family: str


def _resolve_tp_fallback() -> str:
    value = os.environ.get("GPT_TP_FALLBACK", TP_FALLBACK_DEVICE_MAP)
    if value == TP_FALLBACK_REDUCE_GPUS:
        return TP_FALLBACK_REDUCE_GPUS
    return TP_FALLBACK_DEVICE_MAP


def _load_model_config(args: models.GPTTaskArgs, config: Config):
    from transformers import AutoConfig

    model_kwargs = load_model_kwargs(config=config)
    return AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
        **model_kwargs,
    )


def _mapped_model_class(model_config, model_family: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.auto_factory import _get_model_class

    if model_family == TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT:
        mapping = AutoModelForImageTextToText._model_mapping
    else:
        mapping = AutoModelForCausalLM._model_mapping

    config_type = type(model_config)
    if config_type not in mapping:
        return None
    return _get_model_class(model_config, mapping)


def _resolve_model_family(model_config) -> Optional[str]:
    if _mapped_model_class(
        model_config, TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT
    ) is not None:
        return TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT
    if _mapped_model_class(model_config, TP_MODEL_FAMILY_CAUSAL_LM) is not None:
        return TP_MODEL_FAMILY_CAUSAL_LM
    return None


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


def _effective_tp_plan(model_config, model_family: str) -> Dict[str, str]:
    plan: Dict[str, str] = {}
    model_class = _mapped_model_class(model_config, model_family)
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
    return plan


def _plan_shards_vocabulary(plan: Mapping[str, str]) -> bool:
    for name, style in plan.items():
        if "embed_tokens" in name and style == "embedding_rowwise":
            return True
        if "lm_head" in name and style.startswith("colwise"):
            return True
    return False


def _dims_are_divisible(config, attrs: Sequence[str], world_size: int) -> bool:
    for attr in attrs:
        dim = getattr(config, attr, None)
        if isinstance(dim, int) and dim % world_size != 0:
            return False
        if isinstance(dim, (list, tuple)) and any(
            isinstance(value, int) and value % world_size != 0
            for value in dim
        ):
            return False
    return True


def _vision_plan_dims(model_config) -> Optional[Sequence[str]]:
    vision_config = getattr(model_config, "vision_config", None)
    vision_plan = getattr(vision_config, "base_model_tp_plan", None) or {}
    if not vision_plan:
        return ()

    model_type = getattr(model_config, "model_type", "")
    validator_dims = _VISION_PLAN_VALIDATORS.get(model_type)
    if validator_dims is not None:
        return validator_dims

    if all(
        any(name.endswith(part) for part in _STANDARD_VISION_PLAN_PARTS)
        for name in vision_plan
    ):
        return _VISION_TP_SHARDED_DIM_ATTRS
    return None


def _dims_divisible_by(
    model_config, model_family: str, world_size: int
) -> bool:
    text_config = model_config.get_text_config()
    if not _dims_are_divisible(
        text_config, _TEXT_TP_SHARDED_DIM_ATTRS, world_size
    ):
        return False

    effective_plan = _effective_tp_plan(model_config, model_family)
    if _plan_shards_vocabulary(effective_plan) and not _dims_are_divisible(
        text_config, ("vocab_size",), world_size
    ):
        return False

    vision_dims = _vision_plan_dims(model_config)
    if vision_dims is None:
        return False
    vision_config = getattr(model_config, "vision_config", None)
    if vision_dims and not _dims_are_divisible(
        vision_config, vision_dims, world_size
    ):
        return False
    return True


def _resolve_tp_task(
    args: models.GPTTaskArgs, config: Config, visible_gpus: int
) -> Optional[_TPTaskResolution]:
    # The fallback decision depends on the task args, the model config, the
    # visible GPU count, and the node-owned GPT_TP_FALLBACK setting. Nodes in
    # the same TP pool MUST use the same GPT_TP_FALLBACK value so every node
    # makes the identical choice and results stay consistent across the pool.
    if visible_gpus < 2:
        return None
    if args.quantize_bits is not None:
        return None

    model_config = _load_model_config(args, config)
    model_family = _resolve_model_family(model_config)
    if model_family is None or not _has_tp_plan(model_config):
        return None

    if _dims_divisible_by(model_config, model_family, visible_gpus):
        return _TPTaskResolution(visible_gpus, model_family)

    if _resolve_tp_fallback() != TP_FALLBACK_REDUCE_GPUS:
        return None

    for k in range(visible_gpus - 1, 1, -1):
        if _dims_divisible_by(model_config, model_family, k):
            return _TPTaskResolution(k, model_family)
    return None


def _resolve_tp_world_size(
    args: models.GPTTaskArgs, config: Config, visible_gpus: int
) -> Optional[int]:
    resolution = _resolve_tp_task(args, config, visible_gpus)
    return None if resolution is None else resolution.world_size


def run_task_tp(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    tools: Sequence[Dict[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    template_args: Mapping[str, Any] | None = None,
    stream_callback: Callable[[models.GPTTaskStreamResponse], None] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
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

    with error_context(local_files_only=config.local_files_only):
        if args is None:
            args = models.GPTTaskArgs.model_validate(
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "generation_config": generation_config,
                    "template_args": template_args,
                    "seed": seed,
                    "dtype": dtype,
                    "quantize_bits": quantize_bits,
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
        _logger.info(
            "Task is not eligible for tensor parallelism, "
            "falling back to the classic executor"
        )
        shutdown_tp_executor()
        return run_task(
            args,
            stream_callback=stream_callback,
            config=config,
            model_cache=model_cache,
        )

    world_size = resolution.world_size
    if world_size < visible_gpus:
        _logger.info(
            "TP-sharded dimensions are not divisible by %d visible GPUs; "
            "reducing tensor parallel world size to %d",
            visible_gpus,
            world_size,
        )

    if model_cache is not None:
        model_cache.clear()

    return submit_tp_task(
        world_size,
        resolution.model_family,
        args,
        config,
        stream_callback,
    )
