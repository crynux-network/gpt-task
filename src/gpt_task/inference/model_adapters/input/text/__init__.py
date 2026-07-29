from .interface import TextInputAdapter
from .registry import TextInputAdapterRegistry, resolve_text_input_adapter

__all__ = [
    "TextInputAdapter",
    "TextInputAdapterRegistry",
    "resolve_text_input_adapter",
]
