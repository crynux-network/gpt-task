from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from gpt_task import models

from .model_adapters import ModelAdapterContext
from .model_adapters.input import (
    contains_image_blocks,
    to_hf_chat_messages,
)
from .model_adapters.input.text import resolve_text_input_adapter
from .model_adapters.input.vision import resolve_vision_input_adapter


@dataclass(frozen=True)
class RenderedTaskInput:
    generation_input: Union[str, List[Dict[str, Any]]]
    encoded: Dict[str, Any] | None


def render_task_input(
    context: ModelAdapterContext,
    args: models.GPTTaskArgs,
    device: Any,
) -> RenderedTaskInput:
    if contains_image_blocks(args.messages):
        if context.processor is None:
            raise RuntimeError("Image input requires a loaded processor.")
        adapter = resolve_vision_input_adapter(context)
        encoded = adapter.render_input(context, args)
        return RenderedTaskInput(
            generation_input=to_hf_chat_messages(args.messages),
            encoded=_move_tensors_to_device(encoded, device),
        )

    adapter = resolve_text_input_adapter(context)
    return RenderedTaskInput(
        generation_input=adapter.render_input(context, args),
        encoded=None,
    )


def encode_rendered_task_input(
    rendered: RenderedTaskInput,
    tokenizer: Any,
    device: Any,
) -> Dict[str, Any]:
    if rendered.encoded is not None:
        return rendered.encoded
    encoded = tokenizer(
        rendered.generation_input,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return _move_tensors_to_device(encoded, device)


def _move_tensors_to_device(value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_tensors_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value
