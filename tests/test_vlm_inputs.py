import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from gpt_task.inference.inference import _run_task, prepare_vlm_inputs
from gpt_task.inference.input_rendering import RenderedTaskInput
from gpt_task.inference.model_adapters import ModelAdapterContext
from gpt_task.inference.model_adapters.artifacts import (
    configure_artifacts,
    resolve_artifact_adapter,
)
from gpt_task.inference.model_adapters.artifacts.ernie import (
    ErnieArtifactAdapter,
)
from gpt_task.inference.model_adapters.artifacts.standard import (
    StandardArtifactAdapter,
)
from gpt_task.inference.model_adapters.input.vision import (
    resolve_vision_input_adapter,
)
from gpt_task.inference.model_adapters.input.vision.ernie import (
    ErnieVisionInputAdapter,
)
from gpt_task.inference.model_adapters.input.vision.standard import (
    StandardVisionInputAdapter,
)
from gpt_task.inference.model_adapters.input.text import (
    resolve_text_input_adapter,
)
from gpt_task.inference.model_adapters.input.text.deepseek_v32 import (
    DeepSeekV32TextInputAdapter,
)
from gpt_task.inference.tp import api
from gpt_task.inference.tp.runtime_strategy import TPRuntimeStrategy
from gpt_task.inference.tp.rank_worker import (
    _load_rank_artifacts,
    _prepare_task_inputs,
)
from gpt_task.config import Config
from gpt_task.models import GPTTaskArgs


class _Processor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values": torch.tensor([1.0]),
            "metadata": "unchanged",
        }


class VLMInputTests(unittest.TestCase):
    def test_text_adapter_uses_config_identity_not_repository_id(self):
        tokenizer = SimpleNamespace(chat_template=None)
        repository_only = ModelAdapterContext(
            config=SimpleNamespace(model_type="other"),
            tokenizer=tokenizer,
        )
        loaded_identity = ModelAdapterContext(
            config=SimpleNamespace(model_type="deepseek_v32"),
            tokenizer=tokenizer,
        )

        self.assertNotIsInstance(
            resolve_text_input_adapter(repository_only),
            DeepSeekV32TextInputAdapter,
        )
        self.assertIsInstance(
            resolve_text_input_adapter(loaded_identity),
            DeepSeekV32TextInputAdapter,
        )

    def test_prepares_hf_image_blocks_and_moves_every_tensor(self):
        processor = _Processor()
        args = GPTTaskArgs(
            model="test/model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "describe",
                        },
                        {"type": "image", "base64": "aQ=="},
                    ],
                }
            ],
            tools=[{"type": "function", "function": {"name": "inspect"}}],
            template_args={"enable_thinking": False},
        )

        inputs = prepare_vlm_inputs(
            processor,
            args,
            torch.device("cpu"),
            SimpleNamespace(model_type="other"),
        )

        self.assertEqual(
            processor.messages[0]["content"][1],
            {"type": "image", "base64": "aQ=="},
        )
        self.assertEqual(
            processor.kwargs,
            {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "tools": args.tools,
                "enable_thinking": False,
            },
        )
        for key in ("input_ids", "attention_mask", "pixel_values"):
            self.assertEqual(inputs[key].device.type, "cpu")
        self.assertEqual(inputs["metadata"], "unchanged")

    def test_resolves_vision_adapter_from_loaded_config_identity(self):
        standard_context = ModelAdapterContext(
            config=SimpleNamespace(model_type="other"),
            processor=_Processor(),
        )
        self.assertIsInstance(
            resolve_vision_input_adapter(standard_context),
            StandardVisionInputAdapter,
        )

        custom_processor = Mock()
        custom_processor.process_vision_info = Mock()
        self.assertIsInstance(
            resolve_vision_input_adapter(
                ModelAdapterContext(
                    config=SimpleNamespace(model_type="other"),
                    processor=custom_processor,
                )
            ),
            StandardVisionInputAdapter,
        )
        self.assertIsInstance(
            resolve_vision_input_adapter(
                ModelAdapterContext(
                    config=SimpleNamespace(
                        model_type="ernie4_5_moe_vl"
                    ),
                    processor=custom_processor,
                )
            ),
            ErnieVisionInputAdapter,
        )

    def test_ernie_missing_required_vision_method_fails(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "base64": "aQ=="}],
                }
            ],
        )

        with self.assertRaisesRegex(
            RuntimeError, "processor.process_vision_info"
        ):
            prepare_vlm_inputs(
                _Processor(),
                args,
                torch.device("cpu"),
                SimpleNamespace(model_type="ernie4_5_moe_vl"),
            )

    def test_image_input_rejects_strategy_without_processor_contract(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "base64": "aQ=="}],
                }
            ],
        )

        with self.assertRaisesRegex(
            RuntimeError, "does not provide the processor"
        ):
            _prepare_task_inputs(
                TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_CAUSAL_LM,
                    False,
                ),
                SimpleNamespace(model_type="other"),
                None,
                Mock(),
                args,
                torch.device("cpu"),
            )

    def test_text_only_vlm_uses_existing_prompt_adapter(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            template_args={"enable_thinking": False},
        )
        tokenizer = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )

        with (
            patch(
                "gpt_task.inference.tp.rank_worker.render_task_input",
                return_value=RenderedTaskInput("adapted prompt", None),
            ) as render,
        ):
            encoded = _prepare_task_inputs(
                TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_CAUSAL_LM,
                    False,
                ),
                SimpleNamespace(model_type="other"),
                Mock(),
                tokenizer,
                args,
                torch.device("cpu"),
            )

        render.assert_called_once()
        tokenizer.assert_called_once_with(
            "adapted prompt",
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.assertEqual(encoded["input_ids"].tolist(), [[1, 2]])

    def test_image_vlm_uses_shared_multimodal_helper(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "base64": "aQ=="},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        )
        prepared = {"input_ids": torch.tensor([[1]])}
        processor = Mock()
        device = torch.device("cpu")

        with (
            patch(
                "gpt_task.inference.tp.rank_worker.render_task_input",
                return_value=RenderedTaskInput([], prepared),
            ) as render,
        ):
            result = _prepare_task_inputs(
                TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
                    True,
                ),
                SimpleNamespace(model_type="other"),
                processor,
                Mock(),
                args,
                device,
            )

        self.assertIs(result, prepared)
        render.assert_called_once()

    def test_classic_image_vlm_uses_shared_multimodal_helper(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "base64": "aQ=="},
                    ],
                }
            ],
        )
        processor = Mock()
        tokenizer = Mock()
        tokenizer.eos_token_id = 3
        tokenizer.decode.return_value = "answer"
        model = Mock()
        model.dtype = torch.float16
        model.device = torch.device("cpu")
        model.generation_config = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        pipe = Mock(processor=processor, tokenizer=tokenizer, model=model)
        model_cache = Mock()
        model_cache.load.return_value = pipe
        prepared = {"input_ids": torch.tensor([[1, 2]])}
        generation_config = Mock(pad_token_id=None)

        with (
            patch(
                "gpt_task.inference.inference.render_task_input",
                return_value=RenderedTaskInput([], prepared),
            ) as render,
            patch(
                "gpt_task.inference.inference.resolve_generation_config",
                return_value=generation_config,
            ),
            patch("gpt_task.inference.inference.use_deterministic_mode"),
        ):
            result = _run_task(
                args,
                config=Config(),
                model_cache=model_cache,
            )

        render.assert_called_once()
        model.generate.assert_called_once()
        self.assertEqual(result["choices"][0]["message"]["content"], "answer")

    def test_custom_vision_processor_uses_image_urls_and_supported_template_args(self):
        class CustomProcessor:
            def apply_chat_template(
                self,
                messages,
                *,
                add_generation_prompt,
                tokenize,
                tools,
                enable_thinking=True,
            ):
                self.messages = messages
                self.template_image_url = messages[0]["content"][0]["image_url"]["url"]
                self.template_call = {
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": tokenize,
                    "tools": tools,
                    "enable_thinking": enable_thinking,
                }
                messages[0]["content"].clear()
                return "rendered prompt"

            def process_vision_info(self, messages):
                self.vision_messages = messages
                return ["decoded image"], None

            def __call__(self, text, images, padding, return_tensors):
                self.processor_call = {
                    "text": text,
                    "images": images,
                    "padding": padding,
                    "return_tensors": return_tensors,
                }
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "nested": {"pixel_values": torch.tensor([1.0])},
                }

        processor = CustomProcessor()
        args = GPTTaskArgs(
            model="test/custom-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "base64": "aQ=="},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            tools=[{"type": "function", "function": {"name": "inspect"}}],
            template_args={
                "enable_thinking": False,
                "unsupported_argument": "ignored",
            },
        )
        inputs = prepare_vlm_inputs(
            processor,
            args,
            torch.device("cpu"),
            SimpleNamespace(model_type="ernie4_5_moe_vl"),
        )

        image_block = processor.vision_messages[0]["content"][0]
        self.assertEqual(image_block["type"], "image_url")
        self.assertEqual(
            image_block["image_url"]["url"],
            "aQ==",
        )
        self.assertEqual(processor.template_image_url, "aQ==")
        self.assertEqual(
            processor.template_call,
            {
                "add_generation_prompt": True,
                "tokenize": False,
                "tools": args.tools,
                "enable_thinking": False,
            },
        )
        self.assertEqual(
            processor.processor_call,
            {
                "text": "rendered prompt",
                "images": ["decoded image"],
                "padding": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(inputs["input_ids"].device.type, "cpu")
        self.assertEqual(inputs["nested"]["pixel_values"].device.type, "cpu")

    def test_custom_vision_adapter_passes_mapping_results(self):
        class MappingProcessor:
            def apply_chat_template(self, messages, **kwargs):
                return "rendered prompt"

            def process_vision_info(self, messages):
                return {
                    "images": ["decoded image"],
                    "videos": ["decoded video"],
                }

            def __call__(
                self,
                text,
                images,
                videos,
                padding,
                return_tensors,
            ):
                self.processor_call = {
                    "text": text,
                    "images": images,
                    "videos": videos,
                    "padding": padding,
                    "return_tensors": return_tensors,
                }
                return {"input_ids": torch.tensor([[1, 2]])}

        processor = MappingProcessor()
        args = GPTTaskArgs(
            model="test/custom-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "base64": "aQ=="},
                    ],
                }
            ],
        )

        inputs = prepare_vlm_inputs(
            processor,
            args,
            torch.device("cpu"),
            SimpleNamespace(model_type="ernie4_5_moe_vl"),
        )

        self.assertEqual(
            processor.processor_call,
            {
                "text": "rendered prompt",
                "images": ["decoded image"],
                "videos": ["decoded video"],
                "padding": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(inputs["input_ids"].device.type, "cpu")


class ArtifactAdapterTests(unittest.TestCase):
    def test_resolves_and_configures_ernie_from_loaded_config(self):
        model = Mock()
        model.add_image_preprocess = Mock()
        processor = Mock()
        context = ModelAdapterContext(
            config=SimpleNamespace(model_type="ernie4_5_moe_vl"),
            model=model,
            processor=processor,
        )

        adapter = resolve_artifact_adapter(context)

        self.assertIsInstance(adapter, ErnieArtifactAdapter)
        adapter.configure(context)
        model.add_image_preprocess.assert_called_once_with(processor)

    def test_named_method_does_not_select_ernie_for_unrelated_config(self):
        class StandardModel:
            def add_image_preprocess(self, processor):
                raise AssertionError("must not be called")

        model = StandardModel()
        processor = Mock()
        context = ModelAdapterContext(
            config=SimpleNamespace(model_type="other"),
            model=model,
            processor=processor,
        )

        adapter = resolve_artifact_adapter(context)

        self.assertIsInstance(adapter, StandardArtifactAdapter)
        adapter.configure(context)

    def test_ernie_missing_required_artifact_method_fails(self):
        context = ModelAdapterContext(
            config=SimpleNamespace(model_type="ernie4_5_moe_vl"),
            model=object(),
            processor=Mock(),
        )

        with self.assertRaisesRegex(
            RuntimeError, "model.add_image_preprocess"
        ):
            configure_artifacts(context)


class RankLoaderTests(unittest.TestCase):
    def test_remote_causal_vlm_delegates_artifact_configuration(self):
        processor = Mock()
        tokenizer = Mock()
        processor.tokenizer = tokenizer
        model = Mock()
        model.config = SimpleNamespace(model_type="other")
        load_processor = Mock(return_value=processor)
        load_tokenizer = Mock()
        load_causal = Mock(return_value=model)
        load_native = Mock()
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoProcessor = Mock(from_pretrained=load_processor)
        fake_transformers.AutoTokenizer = Mock(from_pretrained=load_tokenizer)
        fake_transformers.AutoModelForCausalLM = Mock(
            from_pretrained=load_causal
        )
        fake_transformers.AutoModelForImageTextToText = Mock(
            from_pretrained=load_native
        )
        with (
            patch.dict(sys.modules, {"transformers": fake_transformers}),
            patch(
                "gpt_task.inference.tp.rank_worker.configure_artifacts",
            ) as configure,
        ):
            loaded = _load_rank_artifacts(
                TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_CAUSAL_LM,
                    True,
                ),
                GPTTaskArgs(
                    model="remote/model",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                Config(local_files_only=True),
                torch.float16,
            )

        self.assertEqual(loaded, (model, tokenizer, processor))
        load_processor.assert_called_once()
        load_tokenizer.assert_not_called()
        load_causal.assert_called_once()
        load_native.assert_not_called()
        configure.assert_called_once()
        context = configure.call_args.args[0]
        self.assertIs(context.model, model)
        self.assertIs(context.processor, processor)


if __name__ == "__main__":
    unittest.main()
