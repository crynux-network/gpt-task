from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Tuple


@dataclass(frozen=True)
class TPPlanValidationContext:
    model_config: Any
    config: Any
    plan: Mapping[str, str]
    plan_scope: Literal["text", "vision"]
    candidate_world_size: int


@dataclass(frozen=True)
class TPPlanValidationResult:
    supported: bool
    dimensions: Tuple[str, ...] = ()


class TPPlanValidator(Protocol):
    def matches(self, context: TPPlanValidationContext) -> bool:
        ...

    def validate(
        self,
        context: TPPlanValidationContext,
    ) -> TPPlanValidationResult:
        ...
