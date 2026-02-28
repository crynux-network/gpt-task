from __future__ import annotations

from typing import Any

from gpt_task import models

from .utils import apply_chat_template, copy_messages, copy_tools


class TemplatePromptAdapter:
    def matches(self, model_id: str) -> bool:
        return True

    def render_input(self, args: models.GPTTaskArgs, tokenizer: Any) -> str:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Template adapter requires tokenizer.apply_chat_template")

        chats = copy_messages(args.messages)
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        tools = copy_tools(args.tools)
        if tools is not None:
            template_args["tools"] = tools

        return apply_chat_template(
            tokenizer=tokenizer,
            chats=chats,
            template_args=template_args,
            optional_args=args.template_args,
        )
