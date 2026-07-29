from __future__ import annotations

from typing import Iterable

from ..context import ModelAdapterContext
from .ernie import ErnieArtifactAdapter
from .interface import ArtifactAdapter
from .standard import StandardArtifactAdapter


class ArtifactAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[ArtifactAdapter] | None = None,
    ) -> None:
        self._adapters = list(
            adapters
            if adapters is not None
            else (
                ErnieArtifactAdapter(),
                StandardArtifactAdapter(),
            )
        )

    def resolve(self, context: ModelAdapterContext) -> ArtifactAdapter:
        for adapter in self._adapters:
            if adapter.matches(context):
                return adapter
        raise RuntimeError("No compatible artifact adapter found.")


_REGISTRY = ArtifactAdapterRegistry()


def resolve_artifact_adapter(
    context: ModelAdapterContext,
) -> ArtifactAdapter:
    return _REGISTRY.resolve(context)


def configure_artifacts(context: ModelAdapterContext) -> None:
    resolve_artifact_adapter(context).configure(context)
