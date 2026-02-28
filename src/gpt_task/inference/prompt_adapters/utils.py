from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpt_task import models


def copy_messages(messages: List[models.Message]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for message in messages:
        normalized = dict(**message)
        content = message.get("content")
        if isinstance(content, list):
            normalized["content"] = content_to_text(content)
        copied.append(normalized)
    return copied


def copy_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    return [dict(**tool) for tool in tools]


def apply_chat_template(
    tokenizer: Any,
    chats: List[Dict[str, Any]],
    template_args: Dict[str, Any],
    optional_args: Optional[Dict[str, Any]] = None,
) -> str:
    merged_args = dict(template_args)
    if optional_args:
        for key, value in optional_args.items():
            merged_args[key] = value

    try:
        return tokenizer.apply_chat_template(chats, **merged_args)
    except TypeError:
        if not optional_args:
            raise

    retry_args = dict(template_args)
    for key, value in optional_args.items():
        retry_args[key] = value
        try:
            return tokenizer.apply_chat_template(chats, **retry_args)
        except TypeError:
            retry_args.pop(key, None)
            continue

    return tokenizer.apply_chat_template(chats, **template_args)


def content_to_text(content: models.MessageContent | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    segments: List[str] = []
    for block in content:
        if block.get("type") == "text":
            segments.append(str(block.get("text", "")))
        elif block.get("type") == "image":
            segments.append("<image>")
    return "".join(segments)


def contains_image_blocks(messages: List[models.Message]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image":
                return True
    return False


def to_hf_chat_messages(messages: List[models.Message]) -> List[Dict[str, Any]]:
    hf_messages: List[Dict[str, Any]] = []
    for message in messages:
        mapped: Dict[str, Any] = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            mapped["content"] = _to_hf_content_blocks(content)
        hf_messages.append(mapped)
    return hf_messages


def _to_hf_content_blocks(
    blocks: List[models.MessageContentBlock],
) -> List[Dict[str, Any]]:
    hf_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            hf_blocks.append({"type": "text", "text": block.get("text", "")})
            continue
        if block_type == "image":
            hf_blocks.append(_normalize_hf_image_block(block))
            continue
        raise RuntimeError(f"Unsupported content block type: {block_type}")
    return hf_blocks


def _normalize_hf_image_block(block: models.ImageContentBlock) -> Dict[str, Any]:
    return {"type": "image", "base64": block["base64"]}
