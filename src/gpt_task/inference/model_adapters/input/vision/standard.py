from __future__ import annotations

from typing import Any, Dict

from gpt_task import models

from ...context import ModelAdapterContext
from ..utils import apply_chat_template, copy_tools, to_hf_chat_messages


class StandardVisionInputAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return True

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> Dict[str, Any]:
        template_args = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        tools = copy_tools(args.tools)
        if tools is not None:
            template_args["tools"] = tools

        return apply_chat_template(
            tokenizer=context.processor,
            chats=to_hf_chat_messages(list(args.messages)),
            template_args=template_args,
            optional_args=dict(args.template_args or {}),
        )
