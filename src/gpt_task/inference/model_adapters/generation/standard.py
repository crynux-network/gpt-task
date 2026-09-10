from __future__ import annotations

from typing import Any

from ..context import ModelAdapterContext


class StandardGenerationAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return True

    def structural_tag_model(self, context: ModelAdapterContext) -> str | None:
        return None

    def build_structural_tag(
        self,
        context: ModelAdapterContext,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        reasoning: bool,
    ) -> None:
        return None
