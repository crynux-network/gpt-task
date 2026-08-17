from __future__ import annotations

from dataclasses import dataclass

from gpt_task import models


@dataclass(frozen=True)
class TPTaskResult:
    response: models.GPTTaskResponse | None
    execution_dtype: str
