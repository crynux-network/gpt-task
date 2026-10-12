from __future__ import annotations

from typing import Any, Literal

import torch

from gpt_task.models import GPTTaskArgs

DtypeName = Literal["float16", "bfloat16", "float32", "auto"]

_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "half": torch.float16,
}


def torch_dtype_from_name(name: str) -> torch.dtype:
    key = name.removeprefix("torch.")
    if key not in _TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype value: {name}")
    return _TORCH_DTYPES[key]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def resolve_task_torch_dtype(
    args: GPTTaskArgs,
    *,
    model_config: Any | None = None,
) -> torch.dtype | None:
    if args.dtype != "auto":
        return torch_dtype_from_name(args.dtype)
    if model_config is None:
        return None
    declared = getattr(model_config, "torch_dtype", None)
    if isinstance(declared, torch.dtype):
        return declared
    if isinstance(declared, str) and declared != "auto":
        return torch_dtype_from_name(declared)
    return torch.float32


def resolve_compute_dtype(
    raw: Any | None,
    args: GPTTaskArgs,
    *,
    model_config: Any | None = None,
) -> torch.dtype:
    if raw is None:
        resolved = resolve_task_torch_dtype(args, model_config=model_config)
        return resolved if resolved is not None else torch.float32
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return torch_dtype_from_name(raw)
    raise ValueError(f"Unsupported compute dtype value: {raw!r}")
