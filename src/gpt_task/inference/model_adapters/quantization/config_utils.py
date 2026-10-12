from __future__ import annotations

import inspect
from typing import Any, Callable


def accepted_init_params(config_cls: type) -> set[str]:
    signature = inspect.signature(config_cls.__init__)
    return {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def build_official_config(
    config_cls: type,
    quant_kwargs: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
    normalize: Callable[[str, Any], Any] | None = None,
) -> Any:
    accepted = accepted_init_params(config_cls)
    unknown = sorted(set(quant_kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"{config_cls.__name__} does not accept fields: {', '.join(unknown)}"
        )

    values: dict[str, Any] = {}
    for key, value in quant_kwargs.items():
        values[key] = normalize(key, value) if normalize is not None else value
    if overrides:
        values.update(overrides)
    return config_cls(**values)


def config_to_effective_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        data = config.to_dict()
        if isinstance(data, dict):
            return _normalize_jsonable(data)
    return _normalize_jsonable(dict(vars(config)))


def _normalize_jsonable(value: Any) -> Any:
    from enum import Enum

    if isinstance(value, dict):
        return {
            str(k): _normalize_jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return _normalize_jsonable(value.value)
    if hasattr(value, "removeprefix") and str(value).startswith("torch."):
        return str(value).removeprefix("torch.")
    try:
        import torch

        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
    except Exception:
        pass
    return value
