from __future__ import annotations

import copy
import os
from collections import UserDict
from typing import Any, Dict

import torch
from pydantic import TypeAdapter
from transformers.utils import ModelOutput

from gpt_task import models
from gpt_task.config import Config, get_config


def load_model_kwargs(config: Config | None = None) -> Dict[str, Any]:
    """
    generate model kwargs from config.
    config may contains:
        - cache_dir
        - proxies
    """
    if config is None:
        config = get_config()

    res = {}
    if config.data_dir is not None:
        res["cache_dir"] = config.data_dir.models.huggingface
    if config.proxy is not None and config.proxy.host != "":
        if "://" in config.proxy.host:
            scheme, host = config.proxy.host.split("://", 2)
        else:
            scheme, host = "", config.proxy.host
        
        proxy_str = ""
        if scheme != "":
            proxy_str += f"{scheme}://"
        if config.proxy.username != "":
            proxy_str += f"{config.proxy.username}:{config.proxy.password}@"
        proxy_str += f"{host}:{config.proxy.port}"

        res["proxies"] = {"http": proxy_str, "https": proxy_str}

    return res


def resolve_generation_config(base_generation_config: Any, args: models.GPTTaskArgs) -> Any:
    """Resolve the effective GenerationConfig for a task by overlaying the
    task's generation args on the model's own generation config."""
    generation_kwargs: Dict[str, Any] = {
        "num_return_sequences": 1,
        "max_new_tokens": 256,
    }
    if args.generation_config is not None:
        customer_config = TypeAdapter(models.GPTGenerationConfig).dump_python(
            args.generation_config,
            exclude_none=True,
            exclude_unset=True,
        )
        for k, v in customer_config.items():
            if v is not None:
                generation_kwargs[k] = v

    resolved_generation_config = copy.deepcopy(base_generation_config)
    for k, v in generation_kwargs.items():
        setattr(resolved_generation_config, k, v)
    # Generation length is controlled exclusively by max_new_tokens, which is
    # always set; max_length (default 20) would conflict and cause a
    # transformers warning.
    resolved_generation_config.max_length = None
    return resolved_generation_config


def use_deterministic_mode():
    r"""
    use deterministic mode
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        # Use deterministic CUDA backends for reproducible worker results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
