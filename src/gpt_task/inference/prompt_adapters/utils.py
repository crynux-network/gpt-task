from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpt_task import models

def copy_messages(messages: List[models.Message]) -> List[Dict[str, Any]]:
    return [dict(**m) for m in messages]


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
