from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union, Callable

import torch
from accelerate.utils import get_max_memory
from transformers import AutoProcessor, pipeline, set_seed
from transformers.generation.streamers import BaseStreamer

from gpt_task import models
from gpt_task.config import Config, get_config
from gpt_task.cache import ModelCache

from .errors import error_context
from .utils import (load_model_kwargs, resolve_generation_config,
                    use_deterministic_mode)
from .key import generate_model_key
from .prompt_adapters import resolve_adapter
from .prompt_adapters.utils import contains_image_blocks, content_to_text, to_hf_chat_messages

_logger = logging.getLogger(__name__)


class TokenStreamer(BaseStreamer):
    """Streamer that yields tokens as they are generated."""

    def __init__(self, tokenizer, input_tokens: List[int], model_name: str, stream_callback=None):
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens  # Store the actual input tokens
        self.tokens = []
        self.is_eos = False
        self.completion_tokens = 0
        self.is_done = False
        self.text_queue = []
        self.found_prompt_end = False  # Flag to track if we've found the end of the prompt
        self.first_token = True  # Flag to track if this is the first token being returned
        self.prompt_tokens = len(input_tokens)  # Initialize with input length, will be updated when prompt end is found
        self.model_name = model_name
        self.stream_callback = stream_callback  # Callback to send stream responses

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        token_list = value.tolist()

        for token in token_list:
            self.tokens.append(token)

            # Always check for prompt end first
            if not self.found_prompt_end:
                if len(self.tokens) >= len(self.input_tokens):
                    # Try to find the end of the input sequence
                    prompt_end = _find_prompt_tokens(self.input_tokens, self.tokens)
                    if prompt_end > 0:
                        self.found_prompt_end = True
                        self.prompt_tokens = prompt_end  # Update prompt tokens count to match non-streaming mode
                        self.tokens = self.tokens[prompt_end:]  # Keep only the new tokens
                        _logger.debug(f"Found prompt end at position {prompt_end}")

                        # Check if we've already collected any completion tokens
                        for completion_token in self.tokens:
                            self.completion_tokens += 1
                            # Process each completion token (decode, etc.)
                            new_text = self.tokenizer.decode(
                                [completion_token],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True
                            )
                            if new_text and self.stream_callback:
                                if self.first_token:
                                    new_text = new_text.lstrip()
                                    self.first_token = False
                                self._send_token(new_text)

                            # Check for EOS in completion tokens
                            if completion_token == self.tokenizer.eos_token_id:
                                self.is_eos = True
                                break
                        continue

            # For tokens after prompt identification
            if self.found_prompt_end:
                # Check for EOS after we've found the prompt
                if token == self.tokenizer.eos_token_id:
                    self.is_eos = True
                    break

                self.completion_tokens += 1

                # Decode the new token
                new_text = self.tokenizer.decode(
                    [token],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                if new_text and self.stream_callback:
                    if self.first_token:
                        new_text = new_text.lstrip()
                        self.first_token = False
                    self._send_token(new_text)

    def _send_token(self, text):
        """Send a token through the callback."""
        if self.stream_callback:
            response = {
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None
                }],
                "usage": self.get_usage()
            }
            self.stream_callback(response)

    def end(self):
        """Called when generation is complete."""
        self.is_done = True
        # Send final chunk with finish_reason
        if self.stream_callback:
            finish_reason = self.get_finish_reason()
            _logger.debug(f"Sending final chunk with finish_reason={finish_reason}")
            response = {
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": finish_reason
                }],
                "usage": self.get_usage()
            }
            _logger.info(f"task response: {response}")
            self.stream_callback(response)

    def get_finish_reason(self) -> Literal["stop", "length"]:
        return "stop" if self.is_eos else "length"

    def get_usage(self) -> models.Usage:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }


def _find_prompt_tokens(input_tokens: List[int], output_tokens: List[int]) -> int:

    _logger.debug(f"Finding prompt tokens: input_tokens={input_tokens}")
    _logger.debug(f"Finding prompt tokens: output_tokens={output_tokens}")

    try:
        start = output_tokens.index(input_tokens[0])
        _logger.debug(f"Finding prompt tokens: start={start}")
        end = output_tokens.index(input_tokens[-1], start + len(input_tokens) - 1)
        _logger.debug(f"Finding prompt tokens: end={end}")
        return end + 1
    except ValueError:
        _logger.debug(f"Finding prompt tokens: ValueError")
        return 0


def _resolve_pipeline_tokenizer(pipe: Any) -> Any:
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    processor = getattr(pipe, "processor", None)
    if processor is not None:
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            return processor_tokenizer

    raise RuntimeError("Failed to resolve tokenizer from pipeline.")


def _build_prompt_token_baseline(
    tokenizer: Any,
    inputs: Union[str, List[Dict[str, Any]]],
    messages: Sequence[models.Message],
) -> List[int]:
    if isinstance(inputs, str):
        return tokenizer.encode(inputs, add_special_tokens=False)

    text_prompt = "\n".join(content_to_text(message.get("content")) for message in messages)
    return tokenizer.encode(text_prompt, add_special_tokens=False)


def _extract_generated_text(generated: Any) -> str:
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    if isinstance(generated, str):
        return generated
    if isinstance(generated, list):
        if len(generated) == 0:
            return ""
        last = generated[-1]
        if isinstance(last, dict):
            return _content_to_text(last.get("content"))
    if isinstance(generated, dict):
        return _content_to_text(generated.get("content"))
    return ""


def _to_token_id_list(token_ids: Any) -> List[int] | None:
    if token_ids is None:
        return None
    if torch.is_tensor(token_ids):
        if token_ids.ndim == 0:
            return [int(token_ids.item())]
        if token_ids.ndim == 1:
            return [int(x) for x in token_ids.tolist()]
        return [int(x) for x in token_ids[0].tolist()]
    if isinstance(token_ids, list):
        if len(token_ids) == 0:
            return []
        if isinstance(token_ids[0], list):
            return [int(x) for x in token_ids[0]]
        return [int(x) for x in token_ids]
    return None


def _is_vlm_pipeline(pipe: Any) -> bool:
    return getattr(pipe, "task", None) == "image-text-to-text"


def _invoke_pipeline(
    pipe: Any,
    inputs: Union[str, List[Dict[str, Any]]],
    generation_config: Any,
    *,
    return_tensors: bool = False,
    streamer: BaseStreamer | None = None,
) -> Any:
    call_kwargs: Dict[str, Any] = {}
    if return_tensors:
        call_kwargs["return_tensors"] = True

    # Both TextGenerationPipeline and ImageTextToTextPipeline fall back to
    # self.generation_config when no generation_config is passed through the
    # call. Setting it directly avoids the inconsistent parameter routing
    # between the two pipeline types (flat kwargs vs generate_kwargs dict).
    
    # Save and restore: the pipe object may be cached and reused across
    # calls (via model_cache), so we must not leave a modified
    # generation_config on it after we return.
    saved_generation_config = pipe.generation_config
    pipe.generation_config = generation_config

    try:
        if streamer is not None:
            # streamer is a runtime arg for model.generate(), not part of
            # GenerationConfig. TextGenerationPipeline accepts it as a flat
            # kwarg; ImageTextToTextPipeline requires it inside a
            # generate_kwargs dict — an upstream API inconsistency.
            if _is_vlm_pipeline(pipe):
                call_kwargs["generate_kwargs"] = {"streamer": streamer}
            else:
                call_kwargs["streamer"] = streamer
        return pipe(inputs, **call_kwargs)
    finally:
        pipe.generation_config = saved_generation_config


def _resolve_prompt_input_tokens(
    pipe: Any,
    tokenizer: Any,
    inputs: Union[str, List[Dict[str, Any]]],
    messages: Sequence[models.Message],
) -> List[int]:
    baseline = _build_prompt_token_baseline(tokenizer, inputs, messages)

    processor = getattr(pipe, "processor", None)
    if isinstance(inputs, list) and processor is not None and hasattr(processor, "apply_chat_template"):
        try:
            chat_inputs = processor.apply_chat_template(
                inputs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            resolved_chat = _to_token_id_list(chat_inputs.get("input_ids"))
            if resolved_chat is not None and len(resolved_chat) > 0:
                return resolved_chat
        except Exception:
            pass

    preprocess = getattr(pipe, "preprocess", None)
    if preprocess is None:
        return baseline

    try:
        model_inputs = preprocess(inputs)
    except Exception:
        return baseline

    if not isinstance(model_inputs, dict):
        return baseline

    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        input_ids = model_inputs.get("decoder_input_ids")

    resolved = _to_token_id_list(input_ids)
    if resolved is None and torch.is_tensor(input_ids) and input_ids.ndim == 2:
        resolved = _to_token_id_list(input_ids[0])

    if resolved is not None and len(resolved) > 0:
        return resolved
    return baseline


def run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    tools: Sequence[Dict[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    template_args: Mapping[str, Any] | None = None,
    stream_callback: Callable[[models.GPTTaskStreamResponse], None] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> Union[models.GPTTaskResponse, models.GPTTaskStreamResponse]:
    if config is None:
        config = get_config()

    with error_context(local_files_only=config.local_files_only):
        return _run_task(
            args,
            model=model,
            messages=messages,
            tools=tools,
            generation_config=generation_config,
            template_args=template_args,
            stream_callback=stream_callback,
            seed=seed,
            dtype=dtype,
            quantize_bits=quantize_bits,
            config=config,
            model_cache=model_cache,
        )


def _run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    tools: Sequence[Dict[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    template_args: Mapping[str, Any] | None = None,
    stream_callback: Callable[[models.GPTTaskStreamResponse], None] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> Union[models.GPTTaskResponse, models.GPTTaskStreamResponse]:
    if config is None:
        config = get_config()

    if args is None:
        args = models.GPTTaskArgs.model_validate(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "generation_config": generation_config,
                "template_args": template_args,
                "seed": seed,
                "dtype": dtype,
                "quantize_bits": quantize_bits,
            }
        )

    _logger.info("Task starts")
    _logger.info(f"task args: {args}")

    use_deterministic_mode()

    set_seed(args.seed)

    model_key = generate_model_key(args)

    def model_loader():
        _logger.info("Start loading pipeline")

        torch_dtype = None
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "float32":
            torch_dtype = torch.float32
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model_kwargs = load_model_kwargs(config=config)
        local_files_only = config.local_files_only
        _logger.debug(
            f"model kwargs: {model_kwargs}, local_files_only: {local_files_only}"
        )

        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=local_files_only,
            **model_kwargs,
        )

        if args.quantize_bits == 4:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True
            )
        elif args.quantize_bits == 8:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # CPU/disk offload is prohibited: CPU kernels differ numerically from
        # GPU kernels, so offloaded layers would produce divergent results.
        # With a zero CPU budget the device_map="auto" planner can only place
        # overflow on disk, and without an offload_folder that load fails --
        # a model that does not fit the visible GPUs never starts executing.
        max_memory = get_max_memory()
        max_memory["cpu"] = 0

        try:
            # local_files_only must be a top-level argument: transformers 5.x
            # merges hub kwargs and model_kwargs when loading the config and
            # weights, so putting it inside model_kwargs raises a duplicate
            # keyword argument error.
            pipe = pipeline(
                task=None,
                model=args.model,
                processor=processor,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch_dtype,
                local_files_only=local_files_only,
                model_kwargs=dict(
                    max_memory=max_memory,
                    **model_kwargs,
                ),
            )
        except ValueError as e:
            if "offload" in str(e):
                raise torch.cuda.OutOfMemoryError(
                    "Model does not fit in the visible GPU memory"
                ) from e
            raise

        # transformers 5.x forwards the top-level local_files_only into the
        # pipeline constructor, where it ends up in the cached call parameters
        # and later fails model.generate() as an unknown argument. Drop it
        # from the cached parameter dicts.
        for params in (
            pipe._preprocess_params,
            pipe._forward_params,
            pipe._postprocess_params,
        ):
            params.pop("local_files_only", None)

        _logger.info("Loading pipeline completes")
        return pipe

    if model_cache is not None:
        pipe = model_cache.load(model_key, model_loader)
    else:
        pipe = model_loader()

    tokenizer = _resolve_pipeline_tokenizer(pipe)
    assert tokenizer is not None

    _logger.info("Start text generation")

    resolved_generation_config = resolve_generation_config(
        pipe.model.generation_config, args
    )

    has_image_input = contains_image_blocks(args.messages)
    if has_image_input:
        inputs: Union[str, List[Dict[str, Any]]] = to_hf_chat_messages(args.messages)
    else:
        adapter = resolve_adapter(args.model, tokenizer)
        inputs = adapter.render_input(args, tokenizer)

    _logger.debug(f"Generation config: {resolved_generation_config}")
    _logger.debug(f"Input text: {inputs}")

    input_tokens = _resolve_prompt_input_tokens(pipe, tokenizer, inputs, args.messages)

    if stream_callback is not None:
        streamer = TokenStreamer(tokenizer, input_tokens, args.model, stream_callback)
        resolved_generation_config.pad_token_id = tokenizer.eos_token_id
        resolved_generation_config.use_cache = True

        _invoke_pipeline(
            pipe,
            inputs,
            resolved_generation_config,
            streamer=streamer,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _logger.info("Text generation completes")

        # Return None since we're using callbacks instead of returning a generator
        return None

    output = _invoke_pipeline(
        pipe,
        inputs,
        resolved_generation_config,
        return_tensors=True,
    )
    assert output is not None
    assert isinstance(output, list)

    _logger.debug(f"Raw output: {output}")

    generations: List[Dict[str, Any]] = []
    for single in output:
        assert isinstance(single, dict)
        token_ids = _to_token_id_list(single.get("generated_token_ids"))
        generated_text = _extract_generated_text(single.get("generated_text"))
        generations.append(
            {
                "token_ids": token_ids,
                "generated_text": generated_text,
            }
        )

    assert len(generations) > 0

    del output

    prompt_tokens = len(input_tokens)
    first_generation_tokens = generations[0].get("token_ids")
    if isinstance(first_generation_tokens, list):
        detected_prompt_tokens = _find_prompt_tokens(input_tokens, first_generation_tokens)
        if detected_prompt_tokens > 0:
            prompt_tokens = detected_prompt_tokens

    del input_tokens

    completion_tokens = 0
    output_texts: List[str] = []
    finish_reasons = []

    for generation in generations:
        token_ids = generation.get("token_ids")
        generated_text = generation.get("generated_text", "")
        if isinstance(token_ids, list):
            # when the last token is eos token, finish reason is stop, otherwise is length
            if token_ids[-1] == tokenizer.eos_token_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            finish_reasons.append(finish_reason)

            completion_tokens += max(len(token_ids) - prompt_tokens, 0)

            text = tokenizer.decode(
                token_ids[prompt_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            output_texts.append(text)
        else:
            finish_reasons.append("length")
            completion_tokens += len(
                tokenizer.encode(generated_text, add_special_tokens=False)
            )
            output_texts.append(generated_text.strip())

    usage: models.Usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    choices: List[models.ResponseChoice] = []
    for i, (reason, text) in enumerate(zip(finish_reasons, output_texts)):
        choices.append(
            {
                "finish_reason": reason,
                "message": {"role": "assistant", "content": text},
                "index": i,
            }
        )

    resp: models.GPTTaskResponse = {
        "model": args.model,
        "choices": choices,
        "usage": usage,
    }

    del generations

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _logger.info(f"task response: {resp}")
    _logger.info("Text generation completes")
    return resp
