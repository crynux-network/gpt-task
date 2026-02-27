from typing import Any, Dict, List, Literal, Optional, TypedDict, Generator

from pydantic import BaseModel
from typing_extensions import TypedDict

from .utils import NonEmptyString

class Message(TypedDict, total=False):
    # Required fields
    role: Literal["system", "user", "assistant", "tool"]

    # Optional fields
    content: Optional[str]
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


class ResponseChoice(TypedDict):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class GPTTaskResponse(TypedDict):
    model: NonEmptyString
    choices: List[ResponseChoice]
    usage: Usage
