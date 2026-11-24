import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from gpt_task.config import Config
from gpt_task.inference import run_task
from gpt_task.inference.inference import TokenStreamer
from gpt_task.inference.input_rendering import RenderedTaskInput
from gpt_task.inference.stop_strings import (
    build_stop_string_criteria,
    compute_stream_safe_prefix_len,
    decode_and_truncate,
    extract_and_clear_stop_strings,
    finish_reason_for_generation,
    normalize_stop_strings,
    truncate_at_stop_strings,
)
from gpt_task.models import GPTTaskArgs


class NormalizeStopStringsTests(unittest.TestCase):
    def test_filters_empty_values(self):
        self.assertEqual(normalize_stop_strings(None), ())
        self.assertEqual(normalize_stop_strings(""), ())
        self.assertEqual(normalize_stop_strings([]), ())
        self.assertEqual(normalize_stop_strings([""]), ())
        self.assertEqual(normalize_stop_strings(["", "END", ""]), ("END",))
        self.assertEqual(normalize_stop_strings("END"), ("END",))


class TruncateAtStopStringsTests(unittest.TestCase):
    def test_truncates_at_earliest_match(self):
        text, matched = truncate_at_stop_strings(
            "hello Observation: next",
            ["Observation:", "Obs"],
        )
        self.assertTrue(matched)
        self.assertEqual(text, "hello ")

    def test_overlapping_stops_use_earliest_start(self):
        text, matched = truncate_at_stop_strings("abc", ["bc", "abc"])
        self.assertTrue(matched)
        self.assertEqual(text, "")

    def test_no_match_keeps_text(self):
        text, matched = truncate_at_stop_strings("hello", ["STOP"])
        self.assertFalse(matched)
        self.assertEqual(text, "hello")

    def test_newline_stop_before_strip(self):
        text, matched = truncate_at_stop_strings("hello\n\n", ["\n\n"])
        self.assertTrue(matched)
        self.assertEqual(text, "hello")
        self.assertEqual(text.strip(), "hello")

    def test_ignores_empty_stop_strings_on_direct_call(self):
        text, matched = truncate_at_stop_strings("hello", [""])
        self.assertFalse(matched)
        self.assertEqual(text, "hello")


class StreamSafePrefixTests(unittest.TestCase):
    def test_holds_back_proper_prefix(self):
        self.assertEqual(
            compute_stream_safe_prefix_len("Thought: Observ", ["Observation:"]),
            len("Thought: "),
        )

    def test_false_alarm_releases_prefix(self):
        self.assertEqual(
            compute_stream_safe_prefix_len("Thought: Observed", ["Observation:"]),
            len("Thought: Observed"),
        )

    def test_holds_back_replacement_char(self):
        self.assertEqual(
            compute_stream_safe_prefix_len("hello\ufffd", ["STOP"]),
            len("hello"),
        )

    def test_holds_back_stop_prefix_before_replacement_char(self):
        self.assertEqual(
            compute_stream_safe_prefix_len("ok END\ufffd", ["END\u2014"]),
            len("ok "),
        )


class GeneratedStopStringCriteriaTests(unittest.TestCase):
    def test_ignores_prompt_suffix_match(self):
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            {1: "PROMPT\n", 2: "\n", 3: "ok"}.get(token, "?") for token in ids
        )
        criteria = build_stop_string_criteria(
            tokenizer,
            ["\n\n"],
            initial_prompt_len=1,
        )
        self.assertIsNotNone(criteria)

        first = criteria(torch.tensor([[1, 2]]), None)
        self.assertFalse(bool(first[0].item()))

        second = criteria(torch.tensor([[1, 2, 3]]), None)
        self.assertFalse(bool(second[0].item()))
        tokenizer.decode.assert_called()

    def test_stops_when_generated_contains_stop(self):
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            {1: "PROMPT", 2: "hello", 3: "STOP", 4: "x"}.get(token, "?")
            for token in ids
        )
        criteria = build_stop_string_criteria(
            tokenizer,
            ["STOP"],
            initial_prompt_len=1,
        )
        done = criteria(torch.tensor([[1, 2, 3]]), None)
        self.assertTrue(bool(done[0].item()))

    def test_infers_prompt_len_on_first_call(self):
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "STOP"
        criteria = build_stop_string_criteria(tokenizer, ["STOP"])
        done = criteria(torch.tensor([[9, 9, 7]]), None, new_token_length=1)
        self.assertTrue(bool(done[0].item()))
        self.assertEqual(criteria[0]._prompt_len, 2)

    def test_clamps_initial_prompt_len_when_decoder_starts_empty(self):
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            {5: "ST", 6: "OP"}.get(token, "") for token in ids
        )
        criteria = build_stop_string_criteria(
            tokenizer,
            ["STOP"],
            initial_prompt_len=128,
        )
        first = criteria(torch.tensor([[5]]), None, new_token_length=1)
        self.assertFalse(bool(first[0].item()))
        self.assertEqual(criteria[0]._prompt_len, 0)

        second = criteria(torch.tensor([[5, 6]]), None, new_token_length=1)
        self.assertTrue(bool(second[0].item()))

    def test_expands_window_when_final_token_is_long(self):
        tokenizer = MagicMock()
        mapping = {1: "ST", 2: "OP_trailing_text"}
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            mapping.get(token, "") for token in ids
        )
        criteria = build_stop_string_criteria(
            tokenizer,
            ["STOP"],
            initial_prompt_len=0,
        )
        generated = [1] + [0] * 31 + [2]
        done = criteria(torch.tensor([generated]), None)
        self.assertTrue(bool(done[0].item()))


class DecodeAndFinishReasonTests(unittest.TestCase):
    def test_decode_and_truncate_strips_after_cut(self):
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "hello   STOP"
        text, matched = decode_and_truncate(tokenizer, [1, 2], ["STOP"])
        self.assertTrue(matched)
        self.assertEqual(text, "hello")

    def test_finish_reason_priority(self):
        self.assertEqual(
            finish_reason_for_generation([1, 2], frozenset({9}), True),
            "stop",
        )
        self.assertEqual(
            finish_reason_for_generation([1, 9], frozenset({9}), False),
            "stop",
        )
        self.assertEqual(
            finish_reason_for_generation([1, 2], frozenset({9}), False),
            "length",
        )


class ExtractAndClearTests(unittest.TestCase):
    def test_clears_generation_config_stop_strings(self):
        cfg = SimpleNamespace(stop_strings=["END", ""])
        self.assertEqual(extract_and_clear_stop_strings(cfg), ("END",))
        self.assertIsNone(cfg.stop_strings)


class TokenStreamerStopStringsTests(unittest.TestCase):
    def test_buffers_prefix_then_stops_without_emitting_stop(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)

        def _decode(ids, **kwargs):
            mapping = {1: "a", 2: "b", 3: "Observ", 4: "ation:", 5: "x"}
            if isinstance(ids, list) and len(ids) == 1:
                return mapping.get(ids[0], "")
            return "".join(mapping.get(token, "") for token in ids)

        tokenizer.decode.side_effect = _decode
        streamer = TokenStreamer(
            tokenizer,
            [1, 2],
            "test/model",
            9,
            responses.append,
            stop_strings=["Observation:"],
        )
        streamer.put(torch.tensor([[1, 2]]))
        streamer.put(torch.tensor([3]))
        streamer.put(torch.tensor([4]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "stop")

    def test_false_alarm_flushes_held_prefix(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)

        def _decode(ids, **kwargs):
            mapping = {1: "a", 2: "b", 3: "Observ", 4: "ed"}
            if isinstance(ids, list) and len(ids) == 1:
                return mapping.get(ids[0], "")
            return "".join(mapping.get(token, "") for token in ids)

        tokenizer.decode.side_effect = _decode
        streamer = TokenStreamer(
            tokenizer,
            [1, 2],
            "test/model",
            9,
            responses.append,
            stop_strings=["Observation:"],
        )
        streamer.put(torch.tensor([[1, 2]]))
        streamer.put(torch.tensor([3]))
        streamer.put(torch.tensor([4]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "Observed")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "length")

    def test_leading_whitespace_on_first_token_does_not_duplicate_chars(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)
        mapping = {1: "p", 2: " Hello", 3: " world"}
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            mapping.get(token, "") for token in ids
        )
        streamer = TokenStreamer(
            tokenizer,
            [1],
            "test/model",
            9,
            responses.append,
            stop_strings=["STOP"],
        )
        streamer.put(torch.tensor([[1]]))
        streamer.put(torch.tensor([2]))
        streamer.put(torch.tensor([3]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "Hello world")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "length")

    def test_single_token_leading_double_newline_marks_stop(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)
        mapping = {1: "p", 2: "\n\n"}
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            mapping.get(token, "") for token in ids
        )
        streamer = TokenStreamer(
            tokenizer,
            [1],
            "test/model",
            9,
            responses.append,
            stop_strings=["\n\n"],
        )
        streamer.put(torch.tensor([[1]]))
        streamer.put(torch.tensor([2]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "stop")

    def test_leading_newline_stop_prefix_is_held_back(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)
        mapping = {1: "p", 2: "\nObserv", 3: "ation:"}
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            mapping.get(token, "") for token in ids
        )
        streamer = TokenStreamer(
            tokenizer,
            [1],
            "test/model",
            9,
            responses.append,
            stop_strings=["\nObservation:"],
        )
        streamer.put(torch.tensor([[1]]))
        streamer.put(torch.tensor([2]))
        streamer.put(torch.tensor([3]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "stop")

    def test_multibyte_replacement_char_after_stop_prefix_does_not_leak(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)

        def _decode(ids, **kwargs):
            if ids == [2]:
                return "END\ufffd"
            if ids == [2, 3]:
                return "END\u2014"
            return ""

        tokenizer.decode.side_effect = _decode
        streamer = TokenStreamer(
            tokenizer,
            [1],
            "test/model",
            9,
            responses.append,
            stop_strings=["END\u2014"],
        )
        streamer.put(torch.tensor([[1]]))
        streamer.put(torch.tensor([2]))
        streamer.put(torch.tensor([3]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "stop")

    def test_empty_string_list_uses_unbuffered_path(self):
        responses = []
        tokenizer = MagicMock(eos_token_id=9)
        mapping = {1: "p", 2: "hello"}
        tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            mapping.get(token, "") for token in ids
        )
        streamer = TokenStreamer(
            tokenizer,
            [1],
            "test/model",
            9,
            responses.append,
            stop_strings=[""],
        )
        self.assertEqual(streamer.stop_strings, ())
        streamer.put(torch.tensor([[1]]))
        streamer.put(torch.tensor([2]))
        streamer.end()

        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in responses
            if chunk["choices"][0]["finish_reason"] is None
        ]
        self.assertEqual("".join(contents), "hello")
        self.assertEqual(responses[-1]["choices"][0]["finish_reason"], "length")


def _args_with_stop(stop_strings):
    return GPTTaskArgs(
        version="3.7.0",
        model="test/model",
        messages=[{"role": "user", "content": "hello"}],
        dtype="float32",
        generation_config={"max_new_tokens": 16, "stop_strings": stop_strings},
    )


class ClassicPipelineStopStringsTests(unittest.TestCase):
    def test_nonstream_truncates_and_marks_stop(self):
        args = _args_with_stop(["STOP"])
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 3
        tokenizer.decode.return_value = "hello STOP extra"
        processor = MagicMock(tokenizer=tokenizer)
        model = MagicMock()
        model.dtype = torch.float32
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(model_type="test")
        model.generation_config = MagicMock()
        pipe = MagicMock(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            generation_config=MagicMock(),
        )
        pipe._preprocess_params = {}
        pipe._forward_params = {}
        pipe._postprocess_params = {}
        pipe.return_value = [{"generated_token_ids": [1, 2, 5, 6]}]
        generation_config = SimpleNamespace(
            pad_token_id=None,
            eos_token_id=3,
            stop_strings=["STOP"],
        )

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(),
            ),
            patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch("transformers.pipeline", return_value=pipe),
            patch(
                "gpt_task.inference.inference.get_max_memory",
                return_value={"cpu": 1},
            ),
            patch("gpt_task.inference.inference.configure_artifacts"),
            patch(
                "gpt_task.inference.inference.render_task_input",
                return_value=RenderedTaskInput("prompt", None),
            ),
            patch(
                "gpt_task.inference.inference._resolve_prompt_input_tokens",
                return_value=[1, 2],
            ),
            patch(
                "gpt_task.inference.inference.resolve_generation_config",
                return_value=generation_config,
            ),
            patch(
                "gpt_task.inference.inference.resolve_pipeline_task",
                return_value="text-generation",
            ),
            patch("gpt_task.inference.inference.use_deterministic_mode"),
            patch("torch.cuda.device_count", return_value=1),
        ):
            result = run_task(args, config=Config(local_files_only=True))

        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["choices"][0]["message"]["content"], "hello")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertIsNone(generation_config.stop_strings)
        _, call_kwargs = pipe.call_args
        self.assertIsNotNone(call_kwargs.get("stopping_criteria"))


if __name__ == "__main__":
    unittest.main()
