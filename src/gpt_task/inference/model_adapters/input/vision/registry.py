from __future__ import annotations

from typing import Iterable

from ...context import ModelAdapterContext
from .ernie import ErnieVisionInputAdapter
from .interface import VisionInputAdapter
from .standard import StandardVisionInputAdapter


class VisionInputAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[VisionInputAdapter] | None = None,
    ) -> None:
        self._adapters = list(
            adapters
            if adapters is not None
            else (
                ErnieVisionInputAdapter(),
                StandardVisionInputAdapter(),
            )
        )

    def resolve(self, context: ModelAdapterContext) -> VisionInputAdapter:
        for adapter in self._adapters:
            if adapter.matches(context):
                return adapter
        raise RuntimeError("No compatible vision input adapter found.")


_REGISTRY = VisionInputAdapterRegistry()


def resolve_vision_input_adapter(
    context: ModelAdapterContext,
) -> VisionInputAdapter:
    return _REGISTRY.resolve(context)
