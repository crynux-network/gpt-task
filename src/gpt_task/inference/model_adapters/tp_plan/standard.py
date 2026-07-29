from __future__ import annotations

from typing import Optional, Tuple

from .interface import TPPlanValidationContext, TPPlanValidationResult


class StandardTPPlanValidator:
    def matches(self, context: TPPlanValidationContext) -> bool:
        return True

    def validate(
        self,
        context: TPPlanValidationContext,
    ) -> TPPlanValidationResult:
        dimensions = infer_plan_dimensions(context)
        if dimensions is None:
            return TPPlanValidationResult(False)
        return TPPlanValidationResult(True, dimensions)


def infer_plan_dimensions(
    context: TPPlanValidationContext,
) -> Optional[Tuple[str, ...]]:
    dimensions = set()
    for name, style in context.plan.items():
        if style.startswith("replicated") or style == "moe_tp_experts":
            continue

        normalized = name.lower()
        if "embed_tokens" in normalized or "lm_head" in normalized:
            dimensions.add("vocab_size")
        elif any(
            part in normalized
            for part in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                ".attn.qkv",
                ".attn.proj",
                "linear_attn.in_proj",
                "linear_attn.out_proj",
            )
        ):
            dimensions.add("hidden_size")
            if context.plan_scope == "vision":
                if getattr(context.config, "num_heads", None) is not None:
                    dimensions.add("num_heads")
                elif (
                    getattr(context.config, "num_attention_heads", None)
                    is not None
                ):
                    dimensions.add("num_attention_heads")
            elif "k_proj" in normalized or "v_proj" in normalized:
                if (
                    getattr(context.config, "num_key_value_heads", None)
                    is not None
                ):
                    dimensions.add("num_key_value_heads")
            elif (
                getattr(context.config, "num_attention_heads", None)
                is not None
            ):
                dimensions.add("num_attention_heads")
        elif any(
            part in normalized
            for part in (
                ".mlp.",
                ".experts.",
                ".shared_expert.",
                ".shared_experts.",
            )
        ):
            dimension = _dimension_for_mlp_entry(
                context.config, normalized
            )
            if getattr(context.config, dimension, None) is not None:
                dimensions.add(dimension)
        else:
            return None
    return tuple(sorted(dimensions))


def _dimension_for_mlp_entry(config, name: str) -> str:
    if "shared_expert" in name and getattr(
        config, "shared_expert_intermediate_size", None
    ) is not None:
        return "shared_expert_intermediate_size"
    if (
        "expert" in name
        or getattr(config, "moe_intermediate_size", None) is not None
    ):
        return "moe_intermediate_size"
    return "intermediate_size"
