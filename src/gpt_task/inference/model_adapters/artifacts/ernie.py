from __future__ import annotations

from ..context import ModelAdapterContext

_MODEL_TYPE = "ernie4_5_moe_vl"


class ErnieArtifactAdapter:
    def matches(self, context: ModelAdapterContext) -> bool:
        return getattr(context.config, "model_type", None) == _MODEL_TYPE

    def configure(self, context: ModelAdapterContext) -> None:
        add_image_preprocess = getattr(
            context.model, "add_image_preprocess", None
        )
        if not callable(add_image_preprocess):
            raise RuntimeError(
                "ERNIE artifact compatibility requires "
                "model.add_image_preprocess."
            )
        if context.processor is None:
            raise RuntimeError(
                "ERNIE artifact compatibility requires a loaded processor."
            )
        add_image_preprocess(context.processor)
