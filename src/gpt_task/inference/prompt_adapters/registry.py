from __future__ import annotations

from typing import Any, List

from .deepseek_v32 import DeepSeekV32PromptAdapter
from .fallback import FallbackPromptAdapter
from .interface import PromptCompatibilityAdapter
from .template import TemplatePromptAdapter


class PromptAdapterRegistry:
    def __init__(self) -> None:
        self._family_adapters: List[PromptCompatibilityAdapter] = [
            DeepSeekV32PromptAdapter(),
        ]
        self._template_adapter = TemplatePromptAdapter()
        self._fallback_adapter = FallbackPromptAdapter()

    def resolve(self, model_id: str, tokenizer: Any) -> PromptCompatibilityAdapter:
        for adapter in self._family_adapters:
            if adapter.matches(model_id):
                return adapter

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            return self._template_adapter
        return self._fallback_adapter


_REGISTRY = PromptAdapterRegistry()


def resolve_adapter(model_id: str, tokenizer: Any) -> PromptCompatibilityAdapter:
    return _REGISTRY.resolve(model_id, tokenizer)
