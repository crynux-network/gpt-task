from __future__ import annotations

from typing import Iterable

from ..context import ModelAdapterContext
from .family import family_generation_adapters
from .interface import GenerationAdapter
from .standard import StandardGenerationAdapter


class GenerationAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[GenerationAdapter] | None = None,
    ) -> None:
        self._adapters = list(
            adapters
            if adapters is not None
            else (
                *family_generation_adapters(),
                StandardGenerationAdapter(),
            )
        )

    def resolve(self, context: ModelAdapterContext) -> GenerationAdapter:
        for adapter in self._adapters:
            if adapter.matches(context):
                return adapter
        raise RuntimeError("No compatible generation adapter found.")


_REGISTRY = GenerationAdapterRegistry()


def resolve_generation_adapter(
    context: ModelAdapterContext,
) -> GenerationAdapter:
    return _REGISTRY.resolve(context)
