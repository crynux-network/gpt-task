from __future__ import annotations

from typing import Any

import xgrammar as xgr

from ..context import ModelAdapterContext
from .vllm_tags import build_vllm_structural_tag

_SUPPORTED_STYLES = {
    "llama",
    "kimi",
    "deepseek_r1",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "deepseek_v4",
    "qwen_3",
    "qwen_3_coder",
    "qwen_3_5",
    "glm_4_7",
    "hermes",
    "hy_v4",
    "kimi_k3",
}

_MODEL_TYPE_STYLES = {
    "llama": "llama",
    "llama4": "llama",
    "kimi_k2": "kimi",
    "kimi_k25": "kimi",
    "kimi_k3": "kimi_k3",
    "deepseek_v32": "deepseek_v3_2",
    "deepseek_v4": "deepseek_v4",
    "qwen3": "qwen_3",
    "qwen3_moe": "qwen_3",
    "qwen3_next": "qwen_3_coder",
    "qwen3_5": "qwen_3_5",
    "qwen3_5_moe": "qwen_3_5",
    "glm4_moe": "glm_4_7",
    "glm4_moe_lite": "glm_4_7",
    "hunyuan_v1_dense": "hy_v4",
    "hunyuan_v1_moe": "hy_v4",
}
_VLLM_STRUCTURAL_TAG_STYLES = {"hermes", "hy_v4", "kimi_k3"}


class FamilyGenerationAdapter:
    def __init__(self, style: str) -> None:
        self._style = style

    def matches(self, context: ModelAdapterContext) -> bool:
        return _resolve_style(context) == self._style

    def structural_tag_model(self, context: ModelAdapterContext) -> str | None:
        return self._style if self.matches(context) else None

    def build_structural_tag(
        self,
        context: ModelAdapterContext,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        reasoning: bool,
    ) -> Any | None:
        if not self.matches(context):
            return None
        if self._style in _VLLM_STRUCTURAL_TAG_STYLES:
            return build_vllm_structural_tag(
                self._style,
                tools,
                tool_choice,
                reasoning,
                context.tokenizer,
            )
        return xgr.get_model_structural_tag(
            model=self._style,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
        )


def family_generation_adapters() -> tuple[FamilyGenerationAdapter, ...]:
    return tuple(
        FamilyGenerationAdapter(style)
        for style in (
            "llama",
            "kimi",
            "deepseek_r1",
            "deepseek_v3_1",
            "deepseek_v3_2",
            "deepseek_v4",
            "qwen_3",
            "qwen_3_coder",
            "qwen_3_5",
            "glm_4_7",
            "hermes",
            "hy_v4",
            "kimi_k3",
        )
    )


def _resolve_style(context: ModelAdapterContext) -> str | None:
    for source in (context.config, context.tokenizer):
        explicit = _read_string(source, "structural_tag_model")
        if explicit in _SUPPORTED_STYLES:
            return explicit
        explicit = _read_string(source, "tool_call_format")
        if explicit in _SUPPORTED_STYLES:
            return explicit

    template = _read_string(context.tokenizer, "chat_template")
    for marker, template_style in (
        ("<|open|>tools<|sep|>", "kimi_k3"),
        ("<tool_calls", "hy_v4"),
        ("<｜DSML｜function_calls>", "deepseek_v3_2"),
        ("<｜DSML｜tool_calls>", "deepseek_v4"),
        ("<|tool_calls_section_begin|>", "kimi"),
        ("<|python_tag|>", "llama"),
    ):
        if marker in template and (
            template_style != "hy_v4" or "<arg_key" in template
        ):
            return template_style
    if "<function=" in template and "<parameter=" in template:
        return "qwen_3_5"
    if "<｜tool▁calls▁begin｜>" in template:
        return "deepseek_r1" if "```json" in template else "deepseek_v3_1"

    model_type = _read_string(context.config, "model_type")
    style = _MODEL_TYPE_STYLES.get(model_type)
    if style == "qwen_3" and "<function=" in template:
        return "qwen_3_5"
    if style == "llama" and "<tool_call>" in template:
        return "hermes"
    if style is not None:
        return style
    if "<tool_call>" in template:
        return "hermes"
    return None


def _read_string(value: Any, name: str) -> str:
    resolved = getattr(value, name, "")
    return resolved if isinstance(resolved, str) else ""
