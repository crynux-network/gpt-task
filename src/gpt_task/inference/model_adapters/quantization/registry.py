from __future__ import annotations

from typing import Any, Iterable

from gpt_task.models import GPTTaskArgs, QuantizationConfig

from .bitsandbytes import BitsAndBytesQuantizationAdapter
from .errors import (
    QuantizationConfigConflict,
    UnsupportedQuantizationBackend,
)
from .fp8 import FineGrainedFP8QuantizationAdapter
from .interface import QuantizationAdapter, QuantizationResolution


class QuantizationAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[QuantizationAdapter] | None = None,
    ) -> None:
        self._adapters = list(
            adapters
            if adapters is not None
            else (
                BitsAndBytesQuantizationAdapter(),
                FineGrainedFP8QuantizationAdapter(),
            )
        )

    def resolve(
        self,
        quantization_config: QuantizationConfig,
        args: GPTTaskArgs,
        *,
        model_config: Any | None = None,
    ) -> QuantizationResolution:
        for adapter in self._adapters:
            if adapter.matches(quantization_config.quant_method):
                return adapter.resolve(
                    quantization_config,
                    args,
                    model_config=model_config,
                )
        raise UnsupportedQuantizationBackend(quantization_config.quant_method)


_REGISTRY = QuantizationAdapterRegistry()


def resolve_quantization_adapter(
    quantization_config: QuantizationConfig,
    args: GPTTaskArgs,
    *,
    model_config: Any | None = None,
) -> QuantizationResolution:
    return _REGISTRY.resolve(
        quantization_config,
        args,
        model_config=model_config,
    )


def model_has_embedded_quantization(model_config: Any) -> bool:
    if model_config is None:
        return False
    embedded = getattr(model_config, "quantization_config", None)
    if embedded is None:
        return False
    if isinstance(embedded, dict):
        return bool(embedded)
    return True


def resolve_task_quantization(
    args: GPTTaskArgs,
    *,
    model_config: Any | None = None,
) -> QuantizationResolution | None:
    task_config = args.quantization_config
    has_embedded = model_has_embedded_quantization(model_config)
    if task_config is not None and has_embedded:
        raise QuantizationConfigConflict()
    if task_config is None:
        if not has_embedded:
            return None
        return _resolution_from_embedded(model_config)
    return resolve_quantization_adapter(
        task_config,
        args,
        model_config=model_config,
    )


def _resolution_from_embedded(model_config: Any) -> QuantizationResolution:
    from .config_utils import config_to_effective_dict

    embedded = model_config.quantization_config
    if hasattr(embedded, "quant_method"):
        method = str(embedded.quant_method)
        effective = config_to_effective_dict(embedded)
    elif isinstance(embedded, dict):
        method = str(embedded.get("quant_method") or "embedded")
        effective = dict(embedded)
    else:
        method = type(embedded).__name__
        effective = config_to_effective_dict(embedded)

    # compressed-tensors and similar loaders that restore floating weights
    # are reported as non-quantized by the post-load identity helpers.
    reports_as_quantized = method not in {
        "compressed-tensors",
        "compressed_tensors",
        "gguf",
    }
    return QuantizationResolution(
        quant_method=method,
        transformers_config=None,
        effective_config={
            "quant_method": method,
            "quant_kwargs": effective,
            "source": "embedded",
        },
        reports_as_quantized=reports_as_quantized,
    )
