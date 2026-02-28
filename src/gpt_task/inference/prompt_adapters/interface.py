from __future__ import annotations

from typing import Any, Protocol

from gpt_task import models


class PromptCompatibilityAdapter(Protocol):
    def matches(self, model_id: str) -> bool:
        ...

    def render_input(self, args: models.GPTTaskArgs, tokenizer: Any) -> str:
        ...
