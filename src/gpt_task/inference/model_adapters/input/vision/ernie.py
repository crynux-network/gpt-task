from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Mapping, Sequence

from gpt_task import models

from ...context import ModelAdapterContext
from ..utils import apply_chat_template, copy_tools, to_hf_chat_messages

_MODEL_TYPE = "ernie4_5_moe_vl"


class ErnieVisionInputAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return getattr(context.config, "model_type", None) == _MODEL_TYPE

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> Dict[str, Any]:
        processor = context.processor
        process_vision_info = getattr(processor, "process_vision_info", None)
        if not callable(process_vision_info):
            raise RuntimeError(
                "ERNIE vision compatibility requires "
                "processor.process_vision_info."
            )

        messages = _to_image_url_messages(
            to_hf_chat_messages(list(args.messages))
        )
        template_args = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        tools = copy_tools(args.tools)
        if tools is not None:
            template_args["tools"] = tools

        prompt = apply_chat_template(
            tokenizer=processor,
            chats=copy.deepcopy(messages),
            template_args=template_args,
            optional_args=dict(args.template_args or {}),
        )
        vision_info = process_vision_info(messages)
        return _invoke_processor(processor, prompt, vision_info)


def _to_image_url_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    transformed: list[Dict[str, Any]] = []
    for message in messages:
        mapped = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            mapped["content"] = [
                (
                    {
                        "type": "image_url",
                        "image_url": {"url": block["base64"]},
                    }
                    if block.get("type") == "image"
                    else dict(block)
                )
                for block in content
            ]
        transformed.append(mapped)
    return transformed


def _supported_kwargs(
    callable_obj: Any,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _invoke_processor(
    processor: Any,
    prompt: str,
    vision_info: Any,
) -> Dict[str, Any]:
    if isinstance(vision_info, Mapping):
        vision_kwargs = dict(vision_info)
    elif isinstance(vision_info, tuple):
        vision_kwargs = {}
        if len(vision_info) > 0:
            vision_kwargs["images"] = vision_info[0]
        if len(vision_info) > 1:
            vision_kwargs["videos"] = vision_info[1]
    else:
        vision_kwargs = {"images": vision_info}

    call_kwargs = {
        "text": prompt,
        **vision_kwargs,
        "padding": True,
        "return_tensors": "pt",
    }
    supported = _supported_kwargs(processor, call_kwargs)
    if "text" in supported:
        return processor(**supported)
    return processor(prompt, **supported)
