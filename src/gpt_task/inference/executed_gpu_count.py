"""Track how many GPUs the current GPT task actually used.

Under tensor parallelism this is the final TP world size (after reduce_gpus
when applicable). Under classic execution it is the visible CUDA device
count. Worker error reports read this value after a failed task.
"""

from __future__ import annotations

_executed_gpu_count: int | None = None


def clear_executed_gpu_count() -> None:
    global _executed_gpu_count
    _executed_gpu_count = None


def set_executed_gpu_count(count: int) -> None:
    global _executed_gpu_count
    if count < 0:
        raise ValueError(f"executed GPU count must be >= 0, got {count}")
    _executed_gpu_count = count


def get_executed_gpu_count() -> int | None:
    return _executed_gpu_count
