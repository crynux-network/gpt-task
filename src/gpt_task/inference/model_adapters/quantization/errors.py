from __future__ import annotations


class UnsupportedQuantizationBackend(ValueError):
    def __init__(self, quant_method: str) -> None:
        super().__init__(
            f"Unsupported quantization backend: {quant_method}"
        )
        self.quant_method = quant_method


class QuantizationRuntimeUnsupported(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class QuantizationConfigConflict(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "Task quantization_config conflicts with the model embedded "
            "quantization_config"
        )
