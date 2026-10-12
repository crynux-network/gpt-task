from __future__ import annotations

from typing import Any

_execution_quantization: dict[str, Any] | None = None


def clear_execution_quantization() -> None:
    global _execution_quantization
    _execution_quantization = None


def set_execution_quantization(identity: dict[str, Any] | None) -> None:
    global _execution_quantization
    _execution_quantization = identity


def get_execution_quantization() -> dict[str, Any] | None:
    return _execution_quantization


def resolve_loaded_quantization_identity(
    model: Any,
    *,
    requested: dict[str, Any] | None,
    reports_as_quantized: bool,
) -> dict[str, Any] | None:
    if not reports_as_quantized:
        return {
            "quant_method": "float",
            "quant_kwargs": {},
            "source": "dequantized",
        }

    def _method_name(raw: Any) -> str:
        from enum import Enum

        if isinstance(raw, Enum):
            return str(raw.value)
        text = str(raw)
        if text.startswith("QuantizationMethod."):
            return text.rsplit(".", 1)[-1].lower().replace("_", "")
        return text

    hf_quantizer = getattr(model, "hf_quantizer", None)
    if hf_quantizer is not None:
        quantizer_config = getattr(hf_quantizer, "quantization_config", None)
        method = getattr(quantizer_config, "quant_method", None)
        if method is None and requested is not None:
            method = requested.get("quant_method")
        if method is not None:
            return {
                "quant_method": _method_name(method),
                "quant_kwargs": (
                    requested.get("quant_kwargs", {})
                    if requested is not None
                    else {}
                ),
                "source": "runtime",
            }

    embedded = getattr(getattr(model, "config", None), "quantization_config", None)
    if embedded is None:
        return requested
    method = getattr(embedded, "quant_method", None)
    if method is None and isinstance(embedded, dict):
        method = embedded.get("quant_method")
    if method is None:
        return requested
    return {
        "quant_method": _method_name(method),
        "quant_kwargs": (
            requested.get("quant_kwargs", {}) if requested is not None else {}
        ),
        "source": "embedded",
    }
