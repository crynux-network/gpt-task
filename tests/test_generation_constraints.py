import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from gpt_task.inference import generation_constraints
from gpt_task.inference.inference import _invoke_pipeline
from gpt_task.inference.generation_constraints import (
    _compile_specification,
    _resolve_constraint_specification,
    build_generation_logits_processors,
    constrain_generation_config,
)
from gpt_task.inference.model_adapters import ModelAdapterContext
from gpt_task.inference.model_adapters.generation import resolve_generation_adapter
from gpt_task.inference.model_adapters.input.utils import apply_chat_template
from gpt_task.models import GPTTaskArgs


def _tool(strict=False):
    return {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {"type": "object"},
            "strict": strict,
        },
    }


class CanonicalStructuredOutputTests(unittest.TestCase):
    def test_unknown_top_level_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            GPTTaskArgs.model_validate(
                {
                    "version": "3.7.0",
                    "model": "m",
                    "messages": [],
                    "structured_outputs": {},
                }
            )

    def test_named_choice_requires_one_unique_declared_function(self):
        with self.assertRaises(ValidationError):
            GPTTaskArgs.model_validate(
                {
                    "version": "3.7.0",
                    "model": "m",
                    "messages": [],
                    "tools": [_tool()],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "other"},
                    },
                }
            )

    def test_tool_constraint_overrides_response_schema(self):
        args = GPTTaskArgs(
            version="3.7.0",
            model="m",
            messages=[],
            tools=[_tool()],
            tool_choice="required",
            response_format={"type": "json_object"},
        )
        context = ModelAdapterContext(
            config=SimpleNamespace(
                structural_tag_model="qwen_3",
                vocab_size=10,
            ),
            tokenizer=SimpleNamespace(),
        )
        specification = _resolve_constraint_specification(context, args)
        self.assertEqual(specification["kind"], "tool")
        self.assertEqual(specification["style"], "qwen_3")

    def test_auto_only_constrains_strict_tools(self):
        context = ModelAdapterContext(
            config=SimpleNamespace(
                structural_tag_model="hermes",
                vocab_size=10,
            ),
            tokenizer=SimpleNamespace(),
        )
        loose = GPTTaskArgs(
            version="3.7.0",
            model="m",
            messages=[],
            tools=[_tool(False)],
            tool_choice="auto",
        )
        strict = GPTTaskArgs(
            version="3.7.0",
            model="m",
            messages=[],
            tools=[_tool(True)],
            tool_choice="auto",
        )
        self.assertIsNone(_resolve_constraint_specification(context, loose))
        self.assertEqual(
            _resolve_constraint_specification(context, strict)["kind"],
            "tool",
        )

    def test_template_args_cannot_override_canonical_tool_choice(self):
        class Tokenizer:
            def apply_chat_template(self, chats, **kwargs):
                return kwargs

        rendered = apply_chat_template(
            Tokenizer(),
            [],
            {"tool_choice": "required"},
            {"tool_choice": "none", "enable_thinking": False},
        )
        self.assertEqual(rendered["tool_choice"], "required")
        self.assertFalse(rendered["enable_thinking"])


class GenerationAdapterTests(unittest.TestCase):
    def test_registry_resolves_all_supported_styles_from_loaded_capability(self):
        styles = {
            "llama",
            "kimi",
            "deepseek_r1",
            "deepseek_v3_1",
            "deepseek_v3_2",
            "deepseek_v4",
            "qwen_3",
            "qwen_3_coder",
            "qwen_3_5",
            "glm_4_7",
            "hermes",
            "hy_v4",
            "kimi_k3",
        }
        for style in styles:
            with self.subTest(style=style):
                context = ModelAdapterContext(
                    config=SimpleNamespace(structural_tag_model=style),
                    tokenizer=SimpleNamespace(),
                )
                adapter = resolve_generation_adapter(context)
                self.assertEqual(adapter.structural_tag_model(context), style)

    def test_each_generation_gets_a_new_logits_processor(self):
        args = GPTTaskArgs(
            version="3.7.0",
            model="m",
            messages=[],
            response_format={"type": "json_object"},
        )
        tokenizer = SimpleNamespace(vocab_size=10)
        context = ModelAdapterContext(
            config=SimpleNamespace(vocab_size=10),
            tokenizer=tokenizer,
        )
        generation_constraints._compiled_grammar_cache.clear()
        processors = []

        class Processor:
            def __init__(self, compiled):
                self.compiled = compiled
                processors.append(self)

        with (
            patch.object(
                generation_constraints,
                "_get_tokenizer_info",
                return_value="tokenizer-info",
            ),
            patch.object(
                generation_constraints,
                "_compile_with_timeout",
                side_effect=lambda fn: "compiled",
            ),
            patch.object(
                generation_constraints.xgr.contrib.hf,
                "LogitsProcessor",
                Processor,
            ),
        ):
            first = build_generation_logits_processors(context, args)
            second = build_generation_logits_processors(context, args)

        self.assertIsNot(first[0], second[0])
        self.assertEqual(len(processors), 2)

    def test_hermes_template_takes_priority_over_llama_model_type(self):
        context = ModelAdapterContext(
            config=SimpleNamespace(model_type="llama"),
            tokenizer=SimpleNamespace(chat_template="<tool_call>"),
        )
        adapter = resolve_generation_adapter(context)
        self.assertEqual(adapter.structural_tag_model(context), "hermes")

    def test_qwen_coder_model_type_is_not_overridden_by_generic_tool_marker(self):
        context = ModelAdapterContext(
            config=SimpleNamespace(model_type="qwen3_next"),
            tokenizer=SimpleNamespace(chat_template="<tool_call>"),
        )
        adapter = resolve_generation_adapter(context)
        self.assertEqual(adapter.structural_tag_model(context), "qwen_3_coder")

    def test_json_object_uses_object_only_schema(self):
        compiler = MagicMock()
        with patch.object(
            generation_constraints.xgr,
            "GrammarCompiler",
            return_value=compiler,
        ):
            _compile_specification(
                "tokenizer-info",
                SimpleNamespace(),
                {"kind": "json_object"},
                ModelAdapterContext(config=SimpleNamespace()),
            )
        compiler.compile_json_schema.assert_called_once_with('{"type":"object"}')
        compiler.compile_builtin_json_grammar.assert_not_called()

    def test_text_pipeline_receives_logits_processor(self):
        pipe = MagicMock()
        pipe.task = "text-generation"
        pipe.generation_config = "saved"
        processors = [object()]
        _invoke_pipeline(
            pipe,
            "prompt",
            "constrained",
            logits_processor=processors,
        )
        pipe.assert_called_once_with("prompt", logits_processor=processors)
        self.assertEqual(pipe.generation_config, "saved")

    def test_vlm_pipeline_receives_logits_processor_in_generate_kwargs(self):
        pipe = MagicMock()
        pipe.task = "image-text-to-text"
        pipe.generation_config = "saved"
        processors = [object()]
        _invoke_pipeline(
            pipe,
            [{"role": "user", "content": "prompt"}],
            "constrained",
            logits_processor=processors,
        )
        pipe.assert_called_once_with(
            [{"role": "user", "content": "prompt"}],
            generate_kwargs={"logits_processor": processors},
        )
        self.assertEqual(pipe.generation_config, "saved")

    def test_constrained_generation_uses_one_beam(self):
        config = SimpleNamespace(num_beams=8)
        constrain_generation_config(config, True)
        self.assertEqual(config.num_beams, 1)

    def test_xgrammar_compiles_all_supported_structural_tags(self):
        import xgrammar as xgr

        tokenizer_info = xgr.TokenizerInfo(
            [bytes([token]) for token in range(256)],
            vocab_size=256,
            stop_token_ids=[0],
        )
        tokenizer = SimpleNamespace(get_vocab=lambda: {})
        tools = [_tool(True)]
        styles = {
            "llama",
            "kimi",
            "deepseek_r1",
            "deepseek_v3_1",
            "deepseek_v3_2",
            "deepseek_v4",
            "qwen_3",
            "qwen_3_coder",
            "qwen_3_5",
            "glm_4_7",
            "hermes",
            "hy_v4",
            "kimi_k3",
        }
        for style in styles:
            with self.subTest(style=style):
                compiled = _compile_specification(
                    tokenizer_info,
                    tokenizer,
                    {
                        "kind": "tool",
                        "style": style,
                        "tools": tools,
                        "tool_choice": "required",
                        "reasoning": False,
                    },
                    ModelAdapterContext(
                        config=SimpleNamespace(structural_tag_model=style),
                        tokenizer=tokenizer,
                    ),
                )
                self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
