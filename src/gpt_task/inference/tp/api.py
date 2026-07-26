from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

from gpt_task import models
from gpt_task.cache import ModelCache
from gpt_task.config import Config, get_config

from ..errors import error_context
from ..inference import run_task
from ..prompt_adapters.utils import contains_image_blocks
from ..utils import load_model_kwargs
from .executor import shutdown_tp_executor, submit_tp_task

_logger = logging.getLogger(__name__)

TP_FALLBACK_DEVICE_MAP = "device_map"
TP_FALLBACK_REDUCE_GPUS = "reduce_gpus"


# Weight dimensions sharded by the transformers tp_plan: q/k/v/o projections
# are sharded by head counts, dense MLP projections by intermediate_size, and
# MoE expert projections (including the packed gate_up weight) by the expert
# and shared expert intermediate sizes. Every dimension present in the model
# config must be divisible by the rank count, otherwise the checkpoint shards
# and the allocated parameter shards disagree in shape and the mismatched
# weights are silently reinitialized.
_TP_SHARDED_DIM_ATTRS = (
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)


def _resolve_tp_fallback() -> str:
    value = os.environ.get("GPT_TP_FALLBACK", TP_FALLBACK_DEVICE_MAP)
    if value == TP_FALLBACK_REDUCE_GPUS:
        return TP_FALLBACK_REDUCE_GPUS
    return TP_FALLBACK_DEVICE_MAP


def _load_text_config(args: models.GPTTaskArgs, config: Config):
    from transformers import AutoConfig

    model_kwargs = load_model_kwargs(config=config)
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
        **model_kwargs,
    )
    return model_config.get_text_config()


def _has_tp_plan(text_config) -> bool:
    return getattr(text_config, "base_model_tp_plan", None) is not None


def _dims_divisible_by(text_config, world_size: int) -> bool:
    for attr in _TP_SHARDED_DIM_ATTRS:
        dim = getattr(text_config, attr, None)
        if isinstance(dim, int) and dim % world_size != 0:
            return False
    return True


def _resolve_tp_world_size(
    args: models.GPTTaskArgs, config: Config, visible_gpus: int
) -> Optional[int]:
    # The fallback decision depends on the task args, the model config, the
    # visible GPU count, and the node-owned GPT_TP_FALLBACK setting. Nodes in
    # the same TP pool MUST use the same GPT_TP_FALLBACK value so every node
    # makes the identical choice and results stay consistent across the pool.
    if visible_gpus < 2:
        return None
    if args.quantize_bits is not None:
        return None
    if contains_image_blocks(args.messages):
        return None

    text_config = _load_text_config(args, config)
    if not _has_tp_plan(text_config):
        return None

    if _dims_divisible_by(text_config, visible_gpus):
        return visible_gpus

    if _resolve_tp_fallback() != TP_FALLBACK_REDUCE_GPUS:
        return None

    for k in range(visible_gpus - 1, 1, -1):
        if _dims_divisible_by(text_config, k):
            return k
    return None


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
        world_size = _resolve_tp_world_size(args, config, visible_gpus)

    # The two execution paths must never hold models in VRAM at the same
    # time: a classic-fallback task tears down the rank group so its full
    # model load does not compete with the cached shards, and a TP task
    # evicts the worker-level cache before the rank group loads shards.
    if world_size is None:
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

    if world_size < visible_gpus:
        _logger.info(
            "TP-sharded dimensions are not divisible by %d visible GPUs; "
            "reducing tensor parallel world size to %d",
            visible_gpus,
            world_size,
        )

    if model_cache is not None:
        model_cache.clear()

    return submit_tp_task(world_size, args, config, stream_callback)
