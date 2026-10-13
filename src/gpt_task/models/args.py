import base64
import binascii
import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import NotRequired, Required, TypedDict

from .utils import NonEmptyString


class TextContentBlock(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: Annotated[str, Field(min_length=1)]


class ImageContentBlock(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["image"]
    base64: Annotated[str, Field(min_length=1)]


MessageContentBlock = Union[TextContentBlock, ImageContentBlock]
MessageContent = Union[str, List[MessageContentBlock]]


class Message(TypedDict, total=False):
    # Required fields
    role: Literal["system", "user", "assistant", "tool"]

    # Optional fields
    content: Optional[MessageContent]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


class FunctionDefinition(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")

    name: Required[Annotated[str, Field(min_length=1)]]
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]
    strict: NotRequired[bool]


class FunctionTool(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionDefinition


class NamedToolChoiceFunction(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]


class NamedToolChoice(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: NamedToolChoiceFunction


ToolChoice = Union[Literal["none", "auto", "required"], NamedToolChoice]


class JSONObjectResponseFormat(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class JSONSchemaDefinition(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")

    name: Required[Annotated[str, Field(min_length=1)]]
    description: NotRequired[str]
    schema: Required[Dict[str, Any]]
    strict: NotRequired[bool]


class JSONSchemaResponseFormat(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: JSONSchemaDefinition


ResponseFormat = Union[JSONObjectResponseFormat, JSONSchemaResponseFormat]


class GPTGenerationConfig(TypedDict, total=False):
    max_new_tokens: int
    stop_strings: List[str]

    do_sample: bool
    num_beams: int

    temperature: float
    typical_p: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float

    num_return_sequences: int

class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class StreamChoice(TypedDict):
    index: int
    delta: Message
    finish_reason: Optional[Literal["stop", "length"]]

class GPTTaskStreamResponse(TypedDict):
    model: NonEmptyString
    choices: List[StreamChoice]
    usage: Usage

class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quant_method: NonEmptyString
    quant_kwargs: Dict[str, Any]

    @model_validator(mode="after")
    def _reject_dequantize(self) -> "QuantizationConfig":
        if "dequantize" in self.quant_kwargs:
            raise ValueError(
                "quantization_config.quant_kwargs must not contain dequantize"
            )
        return self


class GPTTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: NonEmptyString
    model: NonEmptyString
    revision: Optional[NonEmptyString] = None
    messages: List[Message]
    tools: Optional[List[FunctionTool]] = None
    tool_choice: Optional[ToolChoice] = None
    response_format: Optional[ResponseFormat] = None
    generation_config: Optional[GPTGenerationConfig] = None
    template_args: Optional[Dict[str, Any]] = None

    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    quantization_config: Optional[QuantizationConfig] = None

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, messages: List[Message]) -> List[Message]:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    _validate_content_block(block)
        return messages

    @model_validator(mode="after")
    def _validate_structured_output(self) -> "GPTTaskArgs":
        tools = self.tools or []
        names = [tool["function"]["name"] for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool function names must be unique.")

        if self.tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' needs at least one tool.")

        if isinstance(self.tool_choice, dict):
            selected_name = self.tool_choice["function"]["name"]
            if selected_name not in names:
                raise ValueError(
                    "Named tool_choice must reference one declared tool function."
                )

        schemas: List[Dict[str, Any]] = []
        for tool in tools:
            parameters = tool["function"].get("parameters")
            if parameters is not None:
                schemas.append(parameters)
        if (
            self.response_format is not None
            and self.response_format["type"] == "json_schema"
        ):
            schemas.append(self.response_format["json_schema"]["schema"])
        for schema in schemas:
            _validate_json_schema_limits(schema)
        return self


class ResponseChoice(TypedDict):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class GPTTaskResponse(TypedDict):
    model: NonEmptyString
    choices: List[ResponseChoice]
    usage: Usage


def _validate_content_block(block: MessageContentBlock) -> None:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError("Text content block must include string field 'text'.")
        return
    if block_type == "image":
        _validate_image_block(block)
        return
    raise ValueError("Unsupported content block type.")


def _validate_image_block(block: ImageContentBlock) -> None:
    base64_value = block.get("base64")
    if not isinstance(base64_value, str) or not base64_value.strip():
        raise ValueError("Image content block must include non-empty string field 'base64'.")
    _validate_base64_data(base64_value)

    extra_keys = set(block.keys()) - {"type", "base64"}
    if extra_keys:
        raise ValueError("Image content block supports only fields 'type' and 'base64'.")


def _validate_base64_data(data: str) -> None:
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Image content block field 'base64' must be valid base64.") from exc


_MAX_JSON_SCHEMA_BYTES = 64 * 1024
_MAX_JSON_SCHEMA_DEPTH = 32
_MAX_JSON_SCHEMA_PROPERTIES = 1024


def _validate_json_schema_limits(schema: Dict[str, Any]) -> None:
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON schema must contain JSON-compatible values.") from exc
    if len(encoded) > _MAX_JSON_SCHEMA_BYTES:
        raise ValueError("JSON schema exceeds the 65536-byte limit.")

    property_count = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal property_count
        if depth > _MAX_JSON_SCHEMA_DEPTH:
            raise ValueError("JSON schema exceeds the maximum nesting depth of 32.")
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                property_count += len(properties)
                if property_count > _MAX_JSON_SCHEMA_PROPERTIES:
                    raise ValueError("JSON schema exceeds the 1024-property limit.")
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    visit(schema, 1)
