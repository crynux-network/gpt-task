from __future__ import annotations

from typing import Iterable

from ...context import ModelAdapterContext
from .deepseek_v32 import DeepSeekV32TextInputAdapter
from .fallback import FallbackTextInputAdapter
from .interface import TextInputAdapter
from .template import TemplateTextInputAdapter


class TextInputAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[TextInputAdapter] | None = None,
    ) -> None:
        self._adapters = list(
            adapters
            if adapters is not None
            else (
                DeepSeekV32TextInputAdapter(),
                TemplateTextInputAdapter(),
                FallbackTextInputAdapter(),
            )
        )

    def resolve(self, context: ModelAdapterContext) -> TextInputAdapter:
        for adapter in self._adapters:
            if adapter.matches(context):
                return adapter
        raise RuntimeError("No compatible text input adapter found.")


_REGISTRY = TextInputAdapterRegistry()


def resolve_text_input_adapter(
    context: ModelAdapterContext,
) -> TextInputAdapter:
    return _REGISTRY.resolve(context)
