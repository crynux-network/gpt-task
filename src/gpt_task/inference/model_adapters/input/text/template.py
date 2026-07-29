from __future__ import annotations

from gpt_task import models

from ...context import ModelAdapterContext
from ..utils import apply_chat_template, copy_messages, copy_tools


class TemplateTextInputAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return (
            callable(getattr(context.tokenizer, "apply_chat_template", None))
            and getattr(context.tokenizer, "chat_template", None) is not None
        )

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> str:
        chats = copy_messages(args.messages)
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        tools = copy_tools(args.tools)
        if tools is not None:
            template_args["tools"] = tools

        return apply_chat_template(
            tokenizer=context.tokenizer,
            chats=chats,
            template_args=template_args,
            optional_args=args.template_args,
        )
