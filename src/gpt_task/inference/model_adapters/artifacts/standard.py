from __future__ import annotations

from ..context import ModelAdapterContext


class StandardArtifactAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return True

    def configure(self, context: ModelAdapterContext) -> None:
        return None
