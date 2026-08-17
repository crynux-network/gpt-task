from __future__ import annotations

from typing import Any

import torch

_execution_dtype: str | None = None


def resolve_model_execution_dtype(model: Any) -> str:
    dtype = getattr(model, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError("Loaded model does not expose a valid parameter dtype.")
    return str(dtype).removeprefix("torch.")


def clear_execution_dtype() -> None:
    global _execution_dtype
    _execution_dtype = None


def set_execution_dtype(dtype: str) -> None:
    global _execution_dtype
    if not dtype:
        raise ValueError("execution dtype must not be empty")
    _execution_dtype = dtype


def get_execution_dtype() -> str | None:
    return _execution_dtype
