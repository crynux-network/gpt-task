from __future__ import annotations

import logging
from typing import Any

from gpt_task import models

from .utils import content_to_text

_logger = logging.getLogger(__name__)


class FallbackPromptAdapter:
    def matches(self, model_id: str) -> bool:
        return True

    def render_input(self, args: models.GPTTaskArgs, tokenizer: Any) -> str:
        if args.tools is not None:
            _logger.warning(
                "Tools are ignored for model %s because no prompt adapter is registered and "
                "tokenizer chat template is unavailable.",
                args.model,
            )
        if args.template_args:
            _logger.warning(
                "Ignoring template_args for unsupported model family %s because "
                "tokenizer chat template is unavailable.",
                args.model,
            )

        return "\n".join(content_to_text(message.get("content")) for message in args.messages)
