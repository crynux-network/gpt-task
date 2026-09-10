# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM structural_tag_registry.py at
# 588a813a60448f6f71dfa0e0ecd52cc0bacc7329.

from __future__ import annotations

import re
from typing import Any

from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)


def build_vllm_structural_tag(
    style: str,
    tools: list[dict[str, Any]],
    tool_choice: Any,
    reasoning: bool,
    tokenizer: Any,
) -> StructuralTag:
    function_tools, builtin_tools, simplified_choice = normalize_tool_choice(
        tools, tool_choice
    )
    if builtin_tools:
        raise ValueError(f"{style} does not support builtin tools.")
    if style == "hermes":
        return _hermes_structural_tag(function_tools, simplified_choice)
    if style == "hy_v4":
        return _hy_v4_structural_tag(
            function_tools,
            simplified_choice,
            reasoning,
            _hy_v4_token_suffix(tokenizer),
        )
    if style == "kimi_k3":
        return _kimi_k3_structural_tag(function_tools, simplified_choice)
    raise ValueError(f"Unsupported vLLM structural tag style: {style}")


def _function_parameters(tool: Any) -> dict[str, Any] | bool:
    function = tool.function
    if function.strict is False:
        return True
    return function.parameters if function.parameters is not None else True


def _hermes_tool_tags(tools: list[Any]) -> list[TagFormat]:
    formats = (
        ('<tool_call>\n{"name": "', "}\n</tool_call>"),
        ('<tool_call>{"name": "', "}</tool_call>"),
    )
    return [
        TagFormat(
            begin=begin + tool.function.name + '", "arguments": ',
            content=JSONSchemaFormat(
                json_schema=_function_parameters(tool),
            ),
            end=end,
        )
        for tool in tools
        for begin, end in formats
    ]


def _hermes_structural_tag(
    tools: list[Any],
    tool_choice: str,
) -> StructuralTag:
    tags = _hermes_tool_tags(tools)
    if tool_choice == "auto":
        output_format: Any = (
            TriggeredTagsFormat(triggers=["<tool_call>"], tags=tags)
            if tags
            else AnyTextFormat()
        )
    else:
        output_format = TagsWithSeparatorFormat(
            tags=tags,
            separator="",
            at_least_one=True,
            stop_after_first=tool_choice == "forced",
        )
    return StructuralTag(format=output_format)


def _hy_v4_argument_keys(
    parameters: dict[str, Any],
) -> tuple[list[str], list[str]]:
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    required_set = set(required)
    required_keys = [key for key in properties if key in required_set]
    required_keys += [key for key in required if key not in properties]
    optional_keys = [key for key in properties if key not in required_set]
    return required_keys, optional_keys


def _hy_v4_structural_tag(
    tools: list[Any],
    tool_choice: str,
    reasoning: bool,
    token_suffix: str,
) -> StructuralTag:
    think_end = f"</think{token_suffix}>"
    tool_calls_begin = f"<tool_calls{token_suffix}>"
    tool_calls_end = f"</tool_calls{token_suffix}>"
    tool_call_begin = f"<tool_call{token_suffix}>"
    tool_call_end = f"</tool_call{token_suffix}>"
    arg_key_begin = f"<arg_key{token_suffix}>"
    arg_key_end = f"</arg_key{token_suffix}>"
    arg_value_begin = f"<arg_value{token_suffix}>"
    arg_value_end = f"</arg_value{token_suffix}>"

    def argument_pair(key: str) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=arg_key_begin),
                ConstStringFormat(value=key),
                ConstStringFormat(value=arg_key_end),
                ConstStringFormat(value=arg_value_begin),
                AnyTextFormat(),
                ConstStringFormat(value=arg_value_end),
            ]
        )

    def tool_tag(tool: Any) -> TagFormat:
        parameters = (
            tool.function.parameters
            if isinstance(tool.function.parameters, dict)
            else {}
        )
        required, optional = _hy_v4_argument_keys(parameters)
        elements: list[Any] = [argument_pair(key) for key in required]
        if optional:
            pairs = [argument_pair(key) for key in optional]
            elements.append(
                StarFormat(
                    content=pairs[0]
                    if len(pairs) == 1
                    else OrFormat(elements=pairs)
                )
            )
        content: Any = (
            SequenceFormat(elements=elements) if elements else AnyTextFormat()
        )
        return TagFormat(
            begin=tool_call_begin + tool.function.name,
            content=content,
            end=tool_call_end,
        )

    tags = [tool_tag(tool) for tool in tools]
    if tool_choice == "auto":
        block = TagFormat(
            begin=tool_calls_begin,
            content=TagsWithSeparatorFormat(
                tags=tags,
                separator="",
                at_least_one=True,
            ),
            end=tool_calls_end,
        )
        output_format: Any = TriggeredTagsFormat(
            triggers=[tool_calls_begin],
            tags=[block],
        )
    else:
        output_format = SequenceFormat(
            elements=[
                ConstStringFormat(value=tool_calls_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=tool_choice == "forced",
                ),
                ConstStringFormat(value=tool_calls_end),
            ]
        )
    if reasoning:
        output_format = SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end=think_end),
                output_format,
            ]
        )
    return StructuralTag(format=output_format)


def _hy_v4_token_suffix(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return ""
    pattern = re.compile(r"^<tool_calls(?P<suffix>:[^>]+)>$")
    for token in get_vocab():
        match = pattern.match(token)
        if match is not None:
            return match.group("suffix")
    return ""


_K3_OPEN = "<|open|>"
_K3_CLOSE = "<|close|>"
_K3_SEP = "<|sep|>"
_K3_RESPONSE_OPEN = f"{_K3_OPEN}response{_K3_SEP}"
_K3_RESPONSE_CLOSE = f"{_K3_CLOSE}response{_K3_SEP}"
_K3_TOOLS_OPEN = f"{_K3_OPEN}tools{_K3_SEP}"
_K3_TOOLS_CLOSE = f"{_K3_CLOSE}tools{_K3_SEP}"
_K3_CALL_CLOSE = f"{_K3_CLOSE}call{_K3_SEP}"
_K3_ARG_CLOSE = f"{_K3_CLOSE}argument{_K3_SEP}"
_K3_MESSAGE_CLOSE = f"{_K3_CLOSE}message{_K3_SEP}"
_K3_END_OF_MSG = "<|end_of_msg|>"
_K3_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}
_K3_STRING_ATOM = r"(?:[^<]|<[^|])"


def _k3_escape_attr(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if not isinstance(max_len, int) or max_len < 0 or max_len > 4096:
        return None
    if not isinstance(min_len, int) or min_len < 0 or min_len > max_len:
        min_len = 0
    return _K3_STRING_ATOM + f"{{{min_len},{max_len}}}"


def _k3_argument_tag(
    key: str,
    schema: dict[str, Any],
    root_defs: dict[str, Any] | None = None,
) -> TagFormat | None:
    prop = schema if isinstance(schema, dict) else {}
    json_type = prop.get("type")
    xtml_type = (
        _K3_JSON_TO_XTML_TYPE.get(json_type)
        if isinstance(json_type, str)
        else None
    )
    if xtml_type is None:
        return None
    begin = (
        f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" '
        f'type="{xtml_type}"{_K3_SEP}'
    )
    if xtml_type == "string":
        enum_values = prop.get("enum")
        if enum_values is None and isinstance(prop.get("const"), str):
            enum_values = [prop["const"]]
        if (
            isinstance(enum_values, list)
            and enum_values
            and len(enum_values) <= 256
            and all(isinstance(value, str) for value in enum_values)
            and not any("<|" in value for value in enum_values)
        ):
            branches = [ConstStringFormat(value=value) for value in enum_values]
            content: Any = (
                branches[0]
                if len(branches) == 1
                else OrFormat(elements=branches)
            )
        elif (bounded := _k3_bounded_string_regex(prop)) is not None:
            content = RegexFormat(pattern=bounded)
        else:
            content = AnyTextFormat(excludes=[_K3_CLOSE])
    else:
        embedded = prop
        if root_defs:
            embedded = dict(prop)
            for defs_key, defs_value in root_defs.items():
                embedded.setdefault(defs_key, defs_value)
        content = JSONSchemaFormat(json_schema=embedded)
    return TagFormat(begin=begin, content=content, end=_K3_ARG_CLOSE)


def _k3_permissive_argument_tag() -> TagFormat:
    return TagFormat(
        begin=_K3_OPEN + "argument ",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^<]*" + _K3_SEP.replace("|", r"\|")),
                AnyTextFormat(excludes=[_K3_CLOSE]),
            ]
        ),
        end=_K3_ARG_CLOSE,
    )


def _k3_arguments_block(parameters: dict[str, Any] | bool) -> Any:
    if not isinstance(parameters, dict):
        return StarFormat(content=_k3_permissive_argument_tag())
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not properties:
        return StarFormat(content=_k3_permissive_argument_tag())
    root_defs = {
        key: parameters[key]
        for key in ("$defs", "definitions")
        if isinstance(parameters.get(key), dict)
    }
    tags = [
        _k3_argument_tag(key, prop, root_defs) or _k3_permissive_argument_tag()
        for key, prop in properties.items()
    ]
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=tags)
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _k3_call_tag(tool: Any) -> TagFormat:
    function = tool.function
    begin = f'{_K3_OPEN}call tool="{_k3_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[0-9]+"),
                ConstStringFormat(value=f'"{_K3_SEP}'),
                _k3_arguments_block(_function_parameters(tool)),
            ]
        ),
        end=_K3_CALL_CLOSE,
    )


def _k3_response_prefix() -> list[Any]:
    return [
        OptionalFormat(content=ConstStringFormat(value=_K3_RESPONSE_OPEN)),
        TagFormat(
            begin="",
            content=AnyTextFormat(
                excludes=[_K3_OPEN, _K3_CLOSE, _K3_END_OF_MSG]
            ),
            end=_K3_RESPONSE_CLOSE,
        ),
    ]


def _k3_tools_channel(tools: list[Any]) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[_k3_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


def _kimi_k3_structural_tag(
    tools: list[Any],
    tool_choice: str,
) -> StructuralTag:
    trailer = OptionalFormat(content=ConstStringFormat(value=_K3_MESSAGE_CLOSE))
    if not tools:
        return StructuralTag(
            format=SequenceFormat(elements=[*_k3_response_prefix(), trailer])
        )
    if tool_choice == "auto":
        tools_part: Any = OptionalFormat(content=_k3_tools_channel(tools))
    elif tool_choice == "forced":
        tools_part = _k3_tools_channel(tools[:1])
    else:
        tools_part = _k3_tools_channel(tools)
    return StructuralTag(
        format=SequenceFormat(
            elements=[*_k3_response_prefix(), tools_part, trailer]
        )
    )
