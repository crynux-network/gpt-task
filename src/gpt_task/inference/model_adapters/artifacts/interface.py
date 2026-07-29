from __future__ import annotations

from typing import Protocol

from ..context import ModelAdapterContext


class ArtifactAdapter(Protocol):
    def matches(self, context: ModelAdapterContext) -> bool:
        ...

    def configure(self, context: ModelAdapterContext) -> None:
        ...
