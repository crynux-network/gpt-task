from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelAdapterContext:
    config: Any
    model: Any = None
    processor: Any = None
    tokenizer: Any = None
