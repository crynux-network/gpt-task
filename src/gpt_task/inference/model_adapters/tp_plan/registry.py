from __future__ import annotations

from typing import Iterable, Mapping

from .interface import (
    TPPlanValidationContext,
    TPPlanValidationResult,
    TPPlanValidator,
)
from .llama4 import Llama4VisionTPPlanValidator
from .standard import StandardTPPlanValidator


class TPPlanValidatorRegistry:
    def __init__(
        self,
        validators: Iterable[TPPlanValidator] | None = None,
    ) -> None:
        self._validators = list(
            validators
            if validators is not None
            else (
                Llama4VisionTPPlanValidator(),
                StandardTPPlanValidator(),
            )
        )

    def resolve(
        self,
        context: TPPlanValidationContext,
    ) -> TPPlanValidator:
        for validator in self._validators:
            if validator.matches(context):
                return validator
        raise RuntimeError("No TP plan validator found.")

    def validate(
        self,
        context: TPPlanValidationContext,
    ) -> TPPlanValidationResult:
        result = self.resolve(context).validate(context)
        if not result.supported:
            return result
        if not _dimensions_are_divisible(
            context.config,
            result.dimensions,
            context.candidate_world_size,
        ):
            return TPPlanValidationResult(False)
        return result


_REGISTRY = TPPlanValidatorRegistry()


def resolve_tp_plan_validator(
    context: TPPlanValidationContext,
) -> TPPlanValidator:
    return _REGISTRY.resolve(context)


def validate_effective_tp_plan(
    model_config,
    effective_plan: Mapping[str, str],
    candidate_world_size: int,
) -> bool:
    text_config = model_config.get_text_config()
    text_plan = {
        name: style
        for name, style in effective_plan.items()
        if ".vision_config." not in name
        and not name.startswith("config.vision_config.")
    }
    text_context = TPPlanValidationContext(
        model_config=model_config,
        config=text_config,
        plan=text_plan,
        plan_scope="text",
        candidate_world_size=candidate_world_size,
    )
    if not _REGISTRY.validate(text_context).supported:
        return False

    vision_config = getattr(model_config, "vision_config", None)
    vision_plan = (
        getattr(vision_config, "base_model_tp_plan", None) or {}
        if vision_config is not None
        else {}
    )
    vision_context = TPPlanValidationContext(
        model_config=model_config,
        config=vision_config,
        plan=vision_plan,
        plan_scope="vision",
        candidate_world_size=candidate_world_size,
    )
    return _REGISTRY.validate(vision_context).supported


def _dimensions_are_divisible(
    config,
    dimensions: Iterable[str],
    world_size: int,
) -> bool:
    for dimension in dimensions:
        value = getattr(config, dimension, None)
        if isinstance(value, int) and value % world_size != 0:
            return False
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, int) and item % world_size != 0
            for item in value
        ):
            return False
    return True
