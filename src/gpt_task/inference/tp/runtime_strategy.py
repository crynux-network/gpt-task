from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TP_MODEL_LOADER_CAUSAL_LM = "AutoModelForCausalLM"
TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT = "AutoModelForImageTextToText"


@dataclass(frozen=True)
class TPRuntimeStrategy:
    model_loader: Literal[
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
    ]
    requires_processor: bool
