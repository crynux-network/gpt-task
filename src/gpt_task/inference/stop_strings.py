from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)


def normalize_stop_strings(
    stop_strings: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if stop_strings is None:
        return ()
    if isinstance(stop_strings, str):
        return (stop_strings,) if stop_strings else ()
    if not isinstance(stop_strings, (list, tuple)):
        return ()
    return tuple(
        value for value in stop_strings if isinstance(value, str) and value
    )


def extract_and_clear_stop_strings(generation_config: Any) -> tuple[str, ...]:
    stop_strings = normalize_stop_strings(
        getattr(generation_config, "stop_strings", None)
    )
    if hasattr(generation_config, "stop_strings"):
        generation_config.stop_strings = None
    return stop_strings


def truncate_at_stop_strings(
    text: str,
    stop_strings: Sequence[str] | str | None,
) -> tuple[str, bool]:
    normalized = normalize_stop_strings(stop_strings)
    if not text or not normalized:
        return text, False

    earliest = -1
    for stop_string in normalized:
        pos = text.find(stop_string)
        if pos != -1 and (earliest == -1 or pos < earliest):
            earliest = pos
    if earliest == -1:
        return text, False
    return text[:earliest], True


def compute_stream_safe_prefix_len(
    text: str,
    stop_strings: Sequence[str] | str | None,
) -> int:
    if not text:
        return 0

    replacement_holdback = 0
    while (
        replacement_holdback < len(text)
        and text[-(replacement_holdback + 1)] == "\ufffd"
    ):
        replacement_holdback += 1

    clean_len = len(text) - replacement_holdback
    if clean_len <= 0:
        return 0

    normalized = normalize_stop_strings(stop_strings)
    if not normalized:
        return clean_len

    clean_text = text[:clean_len]
    prefix_holdback = 0
    for stop_string in normalized:
        max_prefix = min(len(clean_text), len(stop_string) - 1)
        for prefix_len in range(max_prefix, 0, -1):
            if stop_string.startswith(clean_text[-prefix_len:]):
                prefix_holdback = max(prefix_holdback, prefix_len)
                break

    return max(0, clean_len - prefix_holdback)


def decode_and_truncate(
    tokenizer: Any,
    generated_tokens: Sequence[int],
    stop_strings: Sequence[str],
) -> tuple[str, bool]:
    raw_text = tokenizer.decode(
        list(generated_tokens),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    truncated, matched = truncate_at_stop_strings(raw_text, stop_strings)
    return truncated.strip(), matched


def finish_reason_for_generation(
    generated_tokens: Sequence[int],
    eos_token_ids: frozenset[int],
    matched_stop: bool,
) -> Literal["stop", "length"]:
    if matched_stop:
        return "stop"
    if len(generated_tokens) > 0 and generated_tokens[-1] in eos_token_ids:
        return "stop"
    return "length"


def build_stop_string_criteria(
    tokenizer: Any,
    stop_strings: Sequence[str],
    initial_prompt_len: int | None = None,
) -> StoppingCriteriaList | None:
    normalized = normalize_stop_strings(stop_strings)
    if not normalized:
        return None
    return StoppingCriteriaList(
        [
            _GeneratedStopStringCriteria(
                tokenizer=tokenizer,
                stop_strings=normalized,
                initial_prompt_len=initial_prompt_len,
            )
        ]
    )


class _GeneratedStopStringCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: Any,
        stop_strings: Sequence[str],
        initial_prompt_len: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.stop_strings = normalize_stop_strings(stop_strings)
        self._prompt_len = initial_prompt_len
        self._prompt_len_resolved = False
        self._max_stop_bytes = max(
            len(stop_string.encode("utf-8")) for stop_string in self.stop_strings
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: tuple[torch.FloatTensor] | None,
        new_token_length: int = 1,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        if not self._prompt_len_resolved:
            inferred_max_prompt_len = max(
                0, int(input_ids.shape[1]) - max(1, int(new_token_length))
            )
            if (
                self._prompt_len is None
                or self._prompt_len > inferred_max_prompt_len
            ):
                self._prompt_len = inferred_max_prompt_len
            self._prompt_len_resolved = True

        done = torch.zeros(
            (input_ids.shape[0],),
            dtype=torch.bool,
            device=input_ids.device,
        )
        prompt_len = self._prompt_len
        for row_idx, row in enumerate(input_ids.tolist()):
            generated = row[prompt_len:]
            if not generated:
                continue
            if self._generated_contains_stop(generated):
                done[row_idx] = True
        return done

    def _generated_contains_stop(self, generated_tokens: list[int]) -> bool:
        window = max(self._max_stop_bytes + 8, 32)
        start = max(0, len(generated_tokens) - window)
        last_token_bytes: int | None = None
        while True:
            tail_tokens = generated_tokens[start:]
            decoded = self.tokenizer.decode(
                tail_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            _, matched = truncate_at_stop_strings(decoded, self.stop_strings)
            if matched:
                if start == 0:
                    return True
                full_decoded = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                _, full_matched = truncate_at_stop_strings(
                    full_decoded, self.stop_strings
                )
                return full_matched
            if start == 0:
                return False
            if last_token_bytes is None:
                last_decoded = self.tokenizer.decode(
                    [generated_tokens[-1]],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                last_token_bytes = len(last_decoded.encode("utf-8"))
            prefix_bytes = len(decoded.encode("utf-8")) - last_token_bytes
            if prefix_bytes >= self._max_stop_bytes + 4:
                return False
            start = max(0, start - window)
