import base64
import binascii
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from pydantic import field_validator
from typing_extensions import TypedDict

from .utils import NonEmptyString


class TextContentBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageContentBlock(TypedDict, total=False):
    type: Literal["image"]
    base64: str


MessageContentBlock = Union[TextContentBlock, ImageContentBlock]
MessageContent = Union[str, List[MessageContentBlock]]


class Message(TypedDict, total=False):
    # Required fields
    role: Literal["system", "user", "assistant", "tool"]

    # Optional fields
    content: Optional[MessageContent]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]

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

class GPTTaskArgs(BaseModel):
    model: NonEmptyString
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    generation_config: Optional[GPTGenerationConfig] = None
    template_args: Optional[Dict[str, Any]] = None

    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    quantize_bits: Optional[Literal[4, 8]] = None

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, messages: List[Message]) -> List[Message]:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    _validate_content_block(block)
        return messages


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
