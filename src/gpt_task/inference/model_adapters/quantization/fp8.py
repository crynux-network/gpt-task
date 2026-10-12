from __future__ import annotations

import sys
from typing import Any

from gpt_task.models import GPTTaskArgs, QuantizationConfig

from .config_utils import build_official_config, config_to_effective_dict
from .errors import QuantizationRuntimeUnsupported
from .interface import QuantizationResolution


class FineGrainedFP8QuantizationAdapter:
    def matches(self, quant_method: str) -> bool:
        return quant_method == "fp8"

    def resolve(
        self,
        quantization_config: QuantizationConfig,
        args: GPTTaskArgs,
        *,
        model_config: Any | None = None,
    ) -> QuantizationResolution:
        del args, model_config
        if sys.platform == "darwin":
            raise QuantizationRuntimeUnsupported(
                "Fine-grained FP8 CUDA quantization is unsupported on macOS"
            )
        try:
            from transformers import FineGrainedFP8Config
        except ImportError as exc:
            raise QuantizationRuntimeUnsupported(
                "FineGrainedFP8Config is unavailable"
            ) from exc

        kwargs = dict(quantization_config.quant_kwargs)

        def normalize(key: str, value: Any) -> Any:
            if key == "weight_block_size" and isinstance(value, list):
                return tuple(value)
            return value

        transformers_config = build_official_config(
            FineGrainedFP8Config,
            kwargs,
            normalize=normalize,
        )
        dequantize = bool(getattr(transformers_config, "dequantize", False))
        effective = config_to_effective_dict(transformers_config)
        if "weight_block_size" in effective and isinstance(
            effective["weight_block_size"], tuple
        ):
            effective["weight_block_size"] = list(effective["weight_block_size"])
        return QuantizationResolution(
            quant_method="fp8",
            transformers_config=transformers_config,
            effective_config={
                "quant_method": "fp8",
                "quant_kwargs": effective,
            },
            reports_as_quantized=not dequantize,
        )
