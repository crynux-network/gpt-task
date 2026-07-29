from __future__ import annotations

import logging

from gpt_task import models

from ...context import ModelAdapterContext
from ..utils import content_to_text

_logger = logging.getLogger(__name__)


class FallbackTextInputAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return True

    def render_input(
        self,
        context: ModelAdapterContext,
        args: models.GPTTaskArgs,
    ) -> str:
        if args.tools is not None:
            _logger.warning(
                "Tools are ignored for model %s because no text input adapter "
                "is registered and the tokenizer chat template is unavailable.",
                args.model,
            )
        if args.template_args:
            _logger.warning(
                "Ignoring template_args for model %s because the tokenizer "
                "chat template is unavailable.",
                args.model,
            )
        return "\n".join(
            content_to_text(message.get("content")) for message in args.messages
        )
