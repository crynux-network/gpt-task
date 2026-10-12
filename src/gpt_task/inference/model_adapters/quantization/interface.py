from __future__ import annotations

from typing import Any, Protocol

from gpt_task.models import GPTTaskArgs, QuantizationConfig


class QuantizationResolution:
    def __init__(
        self,
        *,
        quant_method: str,
        transformers_config: Any,
        effective_config: dict[str, Any],
        reports_as_quantized: bool,
    ) -> None:
        self.quant_method = quant_method
        self.transformers_config = transformers_config
        self.effective_config = effective_config
        self.reports_as_quantized = reports_as_quantized


class QuantizationAdapter(Protocol):
    def matches(self, quant_method: str) -> bool: ...

    def resolve(
        self,
        quantization_config: QuantizationConfig,
        args: GPTTaskArgs,
        *,
        model_config: Any | None = None,
    ) -> QuantizationResolution: ...
