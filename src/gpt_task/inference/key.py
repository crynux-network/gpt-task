import hashlib
import json
from typing import Any, Dict

from gpt_task import models


def generate_model_key(
    args: models.GPTTaskArgs,
    *,
    executor_mode: str = "device_map",
    model_revision: str | None = None,
    effective_quantization: dict[str, Any] | None = None,
) -> str:
    revision = model_revision if model_revision is not None else args.revision
    model_args: Dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "executor_mode": executor_mode,
    }
    if revision is not None:
        model_args["model_revision"] = revision
    if args.quantization_config is not None:
        model_args["quantization_config"] = (
            args.quantization_config.model_dump(mode="json")
        )
    elif effective_quantization is not None:
        model_args["quantization_config"] = effective_quantization

    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    key = hashlib.md5(model_args_str.encode("utf-8")).hexdigest()
    return key
