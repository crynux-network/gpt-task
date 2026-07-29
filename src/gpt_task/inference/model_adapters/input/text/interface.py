from __future__ import annotations

from typing import Protocol

from gpt_task import models

from ...context import ModelAdapterContext


class TextInputAdapter(Protocol):
    def matches(self, context: ModelAdapterContext) -> bool:
        ...

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> str:
        ...
