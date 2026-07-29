from __future__ import annotations

from .interface import TPPlanValidationContext, TPPlanValidationResult

_MODEL_TYPE = "llama4"


class Llama4VisionTPPlanValidator:
    def matches(self, context: TPPlanValidationContext) -> bool:
        return (
            context.plan_scope == "vision"
            and any(
                name.lower().endswith("patch_embedding.linear")
                for name in context.plan
            )
            and getattr(context.model_config, "model_type", None)
            == _MODEL_TYPE
        )

    def validate(
        self,
        context: TPPlanValidationContext,
    ) -> TPPlanValidationResult:
        for name, style in context.plan.items():
            if style.startswith("replicated"):
                continue
            normalized = name.lower()
            if not (
                normalized.endswith("patch_embedding.linear")
                or "vision_adapter.mlp.fc1" in normalized
                or "vision_adapter.mlp.fc2" in normalized
                or any(
                    attention_name in normalized
                    for attention_name in (
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.o_proj",
                    )
                )
            ):
                return TPPlanValidationResult(False)
        return TPPlanValidationResult(
            True,
            (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
            ),
        )
