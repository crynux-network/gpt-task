from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Dict, List, Tuple

from gpt_task import models
from gpt_task.config import Config

from ..execution_dtype import resolve_model_execution_dtype
from ..input_rendering import encode_rendered_task_input, render_task_input
from ..model_adapters import ModelAdapterContext
from ..model_adapters.artifacts import configure_artifacts
from ..model_adapters.input import contains_image_blocks
from .result import TPTaskResult
from .runtime_strategy import TPRuntimeStrategy

_logger = logging.getLogger(__name__)


def _prepare_task_inputs(
    strategy: TPRuntimeStrategy,
    model_config: Any,
    processor: Any,
    tokenizer: Any,
    args: models.GPTTaskArgs,
    device: Any,
):
    has_image_input = contains_image_blocks(args.messages)
    if has_image_input and not strategy.requires_processor:
        raise RuntimeError(
            "TP runtime strategy does not provide the processor required "
            "for image input."
        )
    if has_image_input and processor is None:
        raise RuntimeError("TP image input requires a loaded processor.")

    context = ModelAdapterContext(
        config=model_config,
        processor=processor,
        tokenizer=tokenizer,
    )
    rendered = render_task_input(context, args, device)
    return encode_rendered_task_input(rendered, tokenizer, device)


def _load_rank_artifacts(
    strategy: TPRuntimeStrategy,
    args: models.GPTTaskArgs,
    config: Config,
    torch_dtype: Any,
):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    from .runtime_strategy import (
        TP_MODEL_LOADER_CAUSAL_LM,
        TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
    )
    from ..utils import load_model_kwargs

    model_kwargs = load_model_kwargs(config=config)
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
        **model_kwargs,
    }

    processor = None
    if strategy.requires_processor:
        processor = AutoProcessor.from_pretrained(args.model, **load_kwargs)
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)

    if strategy.model_loader == TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT:
        model_class = AutoModelForImageTextToText
    elif strategy.model_loader == TP_MODEL_LOADER_CAUSAL_LM:
        model_class = AutoModelForCausalLM
    else:
        raise RuntimeError(f"Unsupported TP model loader: {strategy.model_loader}")

    model = model_class.from_pretrained(
        args.model,
        tp_plan="auto",
        dtype=torch_dtype,
        **load_kwargs,
    )
    configure_artifacts(
        ModelAdapterContext(
            config=model.config,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
        )
    )
    return model, tokenizer, processor


def rank_worker_main(
    rank: int,
    world_size: int,
    port: int,
    task_queue: Any,
    result_queue: Any,
):
    """Entry point of one persistent tensor parallel rank process.

    Environment variables must be pinned before torch is imported so the
    process group and NCCL pick them up. NCCL_ALGO/NCCL_PROTO are fixed and
    NVLS is disabled so the floating point reduction order does not depend
    on the machine's interconnect topology, keeping collectives bitwise
    deterministic across nodes with the same GPU model and count.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    import torch
    import torch.distributed as dist

    from ..errors import error_context

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    # One cached model per rank process; replaced when the model key changes.
    model_cache: Dict[str, Tuple[Any, Any, Any]] = {}

    try:
        while True:
            msg = task_queue.get()
            if msg[0] == "stop":
                break
            _, seq, strategy, args, config, stream = msg
            try:
                with error_context(local_files_only=config.local_files_only):
                    resp = _execute_task(
                        rank,
                        seq,
                        strategy,
                        args,
                        config,
                        stream,
                        result_queue,
                        model_cache,
                    )
                if rank == 0:
                    result_queue.put(("result", seq, resp))
            except Exception as e:
                result_queue.put(
                    ("error", seq, type(e).__name__, str(e), traceback.format_exc())
                )
    finally:
        dist.destroy_process_group()


def _execute_task(
    rank: int,
    seq: int,
    strategy: TPRuntimeStrategy,
    args: models.GPTTaskArgs,
    config: Config,
    stream: bool,
    result_queue: Any,
    model_cache: Dict[str, Tuple[Any, Any, Any]],
):
    import torch
    from transformers import set_seed

    from ..inference import TokenStreamer
    from ..key import generate_model_key
    from ..utils import resolve_generation_config, use_deterministic_mode

    if rank == 0:
        _logger.info("TP task starts")
        _logger.info(f"task args: {args}")

    use_deterministic_mode()
    set_seed(args.seed)

    model_key = f"{strategy!r}:{generate_model_key(args)}"
    if model_key not in model_cache:
        model_cache.clear()
        torch.cuda.empty_cache()

        torch_dtype = None
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "float32":
            torch_dtype = torch.float32
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model, tokenizer, processor = _load_rank_artifacts(
            strategy,
            args,
            config,
            torch_dtype,
        )
        model.eval()
        if rank == 0:
            _logger.info("Effective TP plan: %s", getattr(model, "tp_plan", None))
        model_cache[model_key] = (model, tokenizer, processor)

    model, tokenizer, processor = model_cache[model_key]
    execution_dtype = resolve_model_execution_dtype(model)

    resolved_generation_config = resolve_generation_config(
        model.generation_config, args
    )
    if resolved_generation_config.pad_token_id is None:
        resolved_generation_config.pad_token_id = tokenizer.eos_token_id
    if stream:
        resolved_generation_config.use_cache = True

    encoded = _prepare_task_inputs(
        strategy,
        model.config,
        processor,
        tokenizer,
        args,
        torch.device(f"cuda:{rank}"),
    )
    input_tokens: List[int] = encoded["input_ids"][0].tolist()

    streamer = None
    if stream and rank == 0:
        streamer = TokenStreamer(
            tokenizer,
            input_tokens,
            args.model,
            lambda resp: result_queue.put(("stream", seq, resp)),
        )

    with torch.no_grad():
        output = model.generate(
            **encoded,
            generation_config=resolved_generation_config,
            streamer=streamer,
        )

    if rank != 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    if stream:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TPTaskResult(response=None, execution_dtype=execution_dtype)

    prompt_tokens = len(input_tokens)
    sequences: List[List[int]] = [
        [int(t) for t in sequence] for sequence in output.tolist()
    ]

    completion_tokens = 0
    choices: List[models.ResponseChoice] = []
    for i, sequence in enumerate(sequences):
        generated_tokens = sequence[prompt_tokens:]
        if len(generated_tokens) > 0 and generated_tokens[-1] == tokenizer.eos_token_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        completion_tokens += len(generated_tokens)
        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        choices.append(
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
                "index": i,
            }
        )

    usage: models.Usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    resp: models.GPTTaskResponse = {
        "model": args.model,
        "choices": choices,
        "usage": usage,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _logger.info(f"task response: {resp}")
    _logger.info("TP text generation completes")
    return TPTaskResult(response=resp, execution_dtype=execution_dtype)
