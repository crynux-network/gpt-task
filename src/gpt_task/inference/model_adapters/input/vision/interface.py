from __future__ import annotations

from typing import Any, Dict, Protocol

from gpt_task import models

from ...context import ModelAdapterContext


class VisionInputAdapter(Protocol):
    def matches(self, context: ModelAdapterContext) -> bool:
        ...

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> Dict[str, Any]:
        ...
