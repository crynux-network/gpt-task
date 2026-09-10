from __future__ import annotations

from typing import Any, Protocol

from ..context import ModelAdapterContext


class GenerationAdapter(Protocol):
    def matches(self, context: ModelAdapterContext) -> bool:
        ...

    def structural_tag_model(self, context: ModelAdapterContext) -> str | None:
        ...

    def build_structural_tag(
        self,
        context: ModelAdapterContext,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        reasoning: bool,
    ) -> Any | None:
        ...
