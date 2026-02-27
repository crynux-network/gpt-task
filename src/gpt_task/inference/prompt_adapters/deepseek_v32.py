from __future__ import annotations

from typing import Any, Dict, List

from gpt_task import models

from .encoding_dsv32 import encode_messages as deepseek_encode_messages
from .utils import copy_messages, copy_tools


class DeepSeekV32PromptAdapter:
    def matches(self, model_id: str) -> bool:
        normalized = model_id.lower()
        return (
            "deepseek-v3.2" in normalized
            or "deepseek-v3_2" in normalized
            or "deepseek_v3.2" in normalized
            or "deepseek_v3_2" in normalized
        )

    def render_input(self, args: models.GPTTaskArgs, tokenizer: Any) -> str:
        del tokenizer  # DeepSeek-V3.2 uses official encoder script, not HF Jinja template.
        chats = copy_messages(args.messages)
        tools = copy_tools(args.tools)
        if tools:
            chats = _inject_tools_system_message(chats, tools)

        encode_kwargs = _build_encode_kwargs(args.template_args, chats)
        try:
            return deepseek_encode_messages(chats, **encode_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "DeepSeek-V3.2 official encoder received unsupported template args. "
                "Use thinking/enable_thinking/thinking_mode/context/drop_thinking/"
                "add_default_bos_token."
            ) from exc


def _build_encode_kwargs(
    template_args: Dict[str, Any] | None,
    chats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(template_args or {})
    allowed_keys = {
        "thinking",
        "enable_thinking",
        "thinking_mode",
        "context",
        "drop_thinking",
        "add_default_bos_token",
    }
    merged = {k: v for k, v in merged.items() if k in allowed_keys}

    thinking_mode = merged.get("thinking_mode")
    if thinking_mode is None:
        thinking_enabled = _normalize_thinking_flag(
            merged.get("thinking", merged.get("enable_thinking", False))
        )
        thinking_mode = "thinking" if thinking_enabled else "chat"
    elif thinking_mode not in {"thinking", "chat"}:
        raise RuntimeError("DeepSeek-V3.2 thinking_mode must be 'thinking' or 'chat'.")

    return {
        "thinking_mode": thinking_mode,
        "context": merged.get("context"),
        "drop_thinking": merged.get("drop_thinking", chats[-1].get("role") == "user"),
        "add_default_bos_token": merged.get("add_default_bos_token", True),
    }


def _normalize_thinking_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "enabled", "thinking"}
    return bool(value)


def _inject_tools_system_message(
    chats: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = [dict(m) for m in chats]
    if merged and merged[0].get("role") == "system":
        existing_tools = merged[0].get("tools")
        if isinstance(existing_tools, list):
            merged[0]["tools"] = [*existing_tools, *tools]
        else:
            merged[0]["tools"] = tools
        return merged
    return [{"role": "system", "tools": tools}, *merged]
