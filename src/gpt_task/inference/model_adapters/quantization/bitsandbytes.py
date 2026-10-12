from __future__ import annotations

import sys
from typing import Any

from gpt_task.models import GPTTaskArgs, QuantizationConfig

from .config_utils import build_official_config, config_to_effective_dict
from .dtype import dtype_name, resolve_compute_dtype
from .errors import QuantizationRuntimeUnsupported
from .interface import QuantizationResolution


class BitsAndBytesQuantizationAdapter:
    def matches(self, quant_method: str) -> bool:
        return quant_method == "bitsandbytes"

    def resolve(
        self,
        quantization_config: QuantizationConfig,
        args: GPTTaskArgs,
        *,
        model_config: Any | None = None,
    ) -> QuantizationResolution:
        if sys.platform == "darwin":
            raise QuantizationRuntimeUnsupported(
                "BitsAndBytes CUDA quantization is unsupported on macOS"
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise QuantizationRuntimeUnsupported(
                "BitsAndBytesConfig is unavailable"
            ) from exc
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise QuantizationRuntimeUnsupported(
                "bitsandbytes is not installed"
            ) from exc

        kwargs = dict(quantization_config.quant_kwargs)
        compute_raw = kwargs.get("bnb_4bit_compute_dtype")
        if "bnb_4bit_compute_dtype" in kwargs or kwargs.get("load_in_4bit"):
            compute_dtype = resolve_compute_dtype(
                compute_raw,
                args,
                model_config=model_config,
            )
            kwargs["bnb_4bit_compute_dtype"] = compute_dtype

        def normalize(key: str, value: Any) -> Any:
            if key == "bnb_4bit_compute_dtype":
                return resolve_compute_dtype(
                    value, args, model_config=model_config
                )
            return value

        transformers_config = build_official_config(
            BitsAndBytesConfig,
            kwargs,
            overrides={
                "llm_int8_enable_fp32_cpu_offload": False,
                "llm_int8_has_fp16_weight": False,
            },
            normalize=normalize,
        )
        if getattr(transformers_config, "llm_int8_enable_fp32_cpu_offload", False):
            raise QuantizationRuntimeUnsupported(
                "BitsAndBytes CPU offload is prohibited"
            )

        effective = config_to_effective_dict(transformers_config)
        if "bnb_4bit_compute_dtype" in effective:
            effective["bnb_4bit_compute_dtype"] = dtype_name(
                resolve_compute_dtype(
                    effective["bnb_4bit_compute_dtype"],
                    args,
                    model_config=model_config,
                )
            )
        return QuantizationResolution(
            quant_method="bitsandbytes",
            transformers_config=transformers_config,
            effective_config={
                "quant_method": "bitsandbytes",
                "quant_kwargs": effective,
            },
            reports_as_quantized=True,
        )
