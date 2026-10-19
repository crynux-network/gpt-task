from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


MODEL_LOADER_CAUSAL_LM = "AutoModelForCausalLM"
MODEL_LOADER_IMAGE_TEXT_TO_TEXT = "AutoModelForImageTextToText"

# Backward-compatible aliases used by the TP backend.
TP_MODEL_LOADER_CAUSAL_LM = MODEL_LOADER_CAUSAL_LM
TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT = MODEL_LOADER_IMAGE_TEXT_TO_TEXT

PipelineTask = Literal["text-generation", "image-text-to-text"]


@dataclass(frozen=True)
class RuntimeStrategy:
    model_loader: Literal[
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
    ]
    requires_processor: bool


# Backward-compatible alias used by the TP backend.
TPRuntimeStrategy = RuntimeStrategy


def is_builtin_mapped(model_config, model_loader: str) -> bool:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    mapping = (
        AutoModelForImageTextToText._model_mapping
        if model_loader == MODEL_LOADER_IMAGE_TEXT_TO_TEXT
        else AutoModelForCausalLM._model_mapping
    )
    return type(model_config) in mapping


def has_remote_auto_map(model_config, loader: str) -> bool:
    auto_map = getattr(model_config, "auto_map", None) or {}
    return isinstance(auto_map, Mapping) and loader in auto_map


def resolve_runtime_strategy(
    model_config,
    *,
    has_image_input: bool = False,
) -> RuntimeStrategy:
    if is_builtin_mapped(
        model_config, MODEL_LOADER_IMAGE_TEXT_TO_TEXT
    ) or has_remote_auto_map(model_config, MODEL_LOADER_IMAGE_TEXT_TO_TEXT):
        return RuntimeStrategy(MODEL_LOADER_IMAGE_TEXT_TO_TEXT, True)

    remote_causal_lm = has_remote_auto_map(model_config, MODEL_LOADER_CAUSAL_LM)
    return RuntimeStrategy(
        MODEL_LOADER_CAUSAL_LM,
        has_image_input and remote_causal_lm,
    )


def resolve_pipeline_task(model_config) -> PipelineTask:
    strategy = resolve_runtime_strategy(model_config)
    if strategy.model_loader == MODEL_LOADER_IMAGE_TEXT_TO_TEXT:
        return "image-text-to-text"
    return "text-generation"
