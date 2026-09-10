from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable

import xgrammar as xgr

from gpt_task import models

from .errors import TaskArgsInvalid
from .model_adapters import ModelAdapterContext
from .model_adapters.generation import resolve_generation_adapter

_COMPILATION_TIMEOUT_SECONDS = 5.0
_cache_lock = threading.Lock()
_tokenizer_info_cache: dict[tuple[int, int], Any] = {}
_compiled_grammar_cache: dict[tuple[int, int, str], Any] = {}


def build_generation_logits_processors(
    context: ModelAdapterContext,
    args: models.GPTTaskArgs,
) -> list[Any]:
    specification = _resolve_constraint_specification(context, args)
    if specification is None:
        return []

    tokenizer = context.tokenizer
    if tokenizer is None:
        raise TaskArgsInvalid("Structured generation requires a tokenizer.")
    vocab_size = _resolve_vocab_size(context.config, tokenizer)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise TaskArgsInvalid("Structured generation requires a known vocabulary size.")

    tokenizer_key = (id(tokenizer), vocab_size)
    serialized_specification = json.dumps(
        specification,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    grammar_key = (*tokenizer_key, serialized_specification)

    with _cache_lock:
        compiled = _compiled_grammar_cache.get(grammar_key)
    if compiled is None:
        tokenizer_info = _get_tokenizer_info(tokenizer, tokenizer_key, vocab_size)
        compiled = _compile_with_timeout(
            lambda: _compile_specification(
                tokenizer_info,
                tokenizer,
                specification,
                context,
            )
        )
        with _cache_lock:
            _compiled_grammar_cache[grammar_key] = compiled

    return [xgr.contrib.hf.LogitsProcessor(compiled)]


def _resolve_vocab_size(config: Any, tokenizer: Any) -> int | None:
    for candidate in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
        getattr(config, "language_config", None),
        tokenizer,
    ):
        vocab_size = getattr(candidate, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
    return None


def constrain_generation_config(generation_config: Any, constrained: bool) -> Any:
    if constrained:
        generation_config.num_beams = 1
    return generation_config


def _get_tokenizer_info(
    tokenizer: Any,
    key: tuple[int, int],
    vocab_size: int,
) -> Any:
    with _cache_lock:
        tokenizer_info = _tokenizer_info_cache.get(key)
    if tokenizer_info is not None:
        return tokenizer_info
    try:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size,
        )
    except Exception as exc:
        raise TaskArgsInvalid(
            "Tokenizer is incompatible with structured generation."
        ) from exc
    with _cache_lock:
        _tokenizer_info_cache[key] = tokenizer_info
    return tokenizer_info


def _resolve_constraint_specification(
    context: ModelAdapterContext,
    args: models.GPTTaskArgs,
) -> dict[str, Any] | None:
    tools = args.tools or []
    choice = args.tool_choice or ("auto" if tools else "none")
    strict_auto = choice == "auto" and any(
        tool["function"].get("strict") is True for tool in tools
    )
    tool_constraint = bool(tools) and (
        isinstance(choice, dict)
        or choice == "required"
        or strict_auto
    )
    if tool_constraint:
        adapter = resolve_generation_adapter(context)
        style = adapter.structural_tag_model(context)
        if style is None:
            raise TaskArgsInvalid(
                "The loaded model does not declare a supported tool-call format."
            )
        return {
            "kind": "tool",
            "style": style,
            "tools": tools,
            "tool_choice": choice,
            "reasoning": _reasoning_enabled(args.template_args),
        }

    if args.response_format is None:
        return None
    if args.response_format["type"] == "json_object":
        return {"kind": "json_object"}
    return {
        "kind": "json_schema",
        "schema": args.response_format["json_schema"]["schema"],
    }


def _reasoning_enabled(template_args: dict[str, Any] | None) -> bool:
    values = template_args or {}
    if values.get("thinking_mode") == "chat":
        return False
    for key in ("thinking", "enable_thinking"):
        if key in values:
            return bool(values[key])
    return True


def _compile_specification(
    tokenizer_info: Any,
    tokenizer: Any,
    specification: dict[str, Any],
    context: ModelAdapterContext,
) -> Any:
    compiler = xgr.GrammarCompiler(tokenizer_info)
    kind = specification["kind"]
    if kind == "json_object":
        return compiler.compile_json_schema('{"type":"object"}')
    if kind == "json_schema":
        schema = json.dumps(
            specification["schema"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return compiler.compile_json_schema(schema)

    adapter = resolve_generation_adapter(context)
    structural_tag = adapter.build_structural_tag(
        context,
        specification["tools"],
        specification["tool_choice"],
        specification["reasoning"],
    )
    if structural_tag is None:
        raise TaskArgsInvalid(
            "The loaded model does not declare a supported tool-call format."
        )
    return compiler.compile_structural_tag(structural_tag)


def _compile_with_timeout(compile_fn: Callable[[], Any]) -> Any:
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def compile_target() -> None:
        try:
            result.put((True, compile_fn()))
        except BaseException as exc:
            result.put((False, exc))

    thread = threading.Thread(target=compile_target, daemon=True)
    thread.start()
    try:
        succeeded, value = result.get(timeout=_COMPILATION_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TaskArgsInvalid(
            "Structured generation grammar compilation timed out."
        ) from exc
    if not succeeded:
        raise TaskArgsInvalid(
            "Structured generation grammar is invalid or incompatible."
        ) from value
    return value
