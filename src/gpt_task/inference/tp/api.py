from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Mapping, Sequence, Union

from gpt_task import models
from gpt_task.cache import ModelCache
from gpt_task.config import Config, get_config

from ..errors import error_context
from ..inference import run_task
from ..prompt_adapters.utils import contains_image_blocks
from ..utils import load_model_kwargs
from .executor import shutdown_tp_executor, submit_tp_task

_logger = logging.getLogger(__name__)


def _model_supports_tp_plan(args: models.GPTTaskArgs, config: Config) -> bool:
    from transformers import AutoConfig

    model_kwargs = load_model_kwargs(config=config)
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
        **model_kwargs,
    )
    text_config = model_config.get_text_config()
    return getattr(text_config, "base_model_tp_plan", None) is not None


def _should_fallback_to_classic(
    args: models.GPTTaskArgs, config: Config, world_size: int
) -> bool:
    # The fallback decision depends only on the task args and the model
    # config, so every node in a TP pool makes the identical choice and
    # results stay consistent across the pool.
    if world_size < 2:
        return True
    if args.quantize_bits is not None:
        return True
    if contains_image_blocks(args.messages):
        return True
    if not _model_supports_tp_plan(args, config):
        return True
    return False


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
    classic run_task path in-process.
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

        world_size = torch.cuda.device_count()
        use_classic = _should_fallback_to_classic(args, config, world_size)

    # The two execution paths must never hold models in VRAM at the same
    # time: a classic-fallback task tears down the rank group so its full
    # model load does not compete with the cached shards, and a TP task
    # evicts the worker-level cache before the rank group loads shards.
    if use_classic:
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

    if model_cache is not None:
        model_cache.clear()

    return submit_tp_task(world_size, args, config, stream_callback)
