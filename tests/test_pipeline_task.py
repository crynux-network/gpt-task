import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
)

from gpt_task.config import Config
from gpt_task.inference import run_task
from gpt_task.inference.input_rendering import RenderedTaskInput
from gpt_task.inference.model_adapters.runtime_strategy import (
    MODEL_LOADER_CAUSAL_LM,
    MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
    resolve_pipeline_task,
    resolve_runtime_strategy,
)
from gpt_task.models import GPTTaskArgs


def _args() -> GPTTaskArgs:
    return GPTTaskArgs(
        version="3.7.0",
        model="test/model",
        messages=[{"role": "user", "content": "hello"}],
        dtype="float32",
    )


class PipelineTaskResolutionTests(unittest.TestCase):
    def test_image_text_mapping_selects_image_text_to_text(self):
        self.assertEqual(
            resolve_pipeline_task(Qwen3_5MoeConfig()),
            "image-text-to-text",
        )
        self.assertEqual(
            resolve_runtime_strategy(Qwen3_5MoeConfig()).model_loader,
            MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
        )

    def test_text_config_and_vision_only_select_text_generation(self):
        self.assertEqual(
            resolve_pipeline_task(Qwen3_5MoeConfig().get_text_config()),
            "text-generation",
        )
        self.assertEqual(
            resolve_pipeline_task(SimpleNamespace(vision_config=object())),
            "text-generation",
        )
        self.assertEqual(
            resolve_runtime_strategy(
                SimpleNamespace(vision_config=object())
            ).model_loader,
            MODEL_LOADER_CAUSAL_LM,
        )

    def test_remote_auto_map_selects_image_text_to_text(self):
        model_config = SimpleNamespace(
            auto_map={
                "AutoModelForImageTextToText": "remote.NativeVLM",
                "AutoModelForCausalLM": "remote.CausalVLM",
            }
        )
        self.assertEqual(
            resolve_pipeline_task(model_config),
            "image-text-to-text",
        )

    def test_remote_causal_only_selects_text_generation(self):
        model_config = SimpleNamespace(
            auto_map={"AutoModelForCausalLM": "remote.CausalVLM"}
        )
        self.assertEqual(
            resolve_pipeline_task(model_config),
            "text-generation",
        )


class ClassicPipelineTaskWiringTests(unittest.TestCase):
    def test_classic_passes_resolved_task_and_skips_hub_get_task(self):
        args = _args()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 3
        tokenizer.decode.return_value = "answer"
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
        pipe.return_value = [{"generated_token_ids": [1, 2, 3]}]
        generation_config = SimpleNamespace(pad_token_id=None)
        model_config = SimpleNamespace(
            auto_map={"AutoModelForCausalLM": "remote.CausalLM"}
        )

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=model_config,
            ) as load_config,
            patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch("transformers.pipeline", return_value=pipe) as load_pipeline,
            patch(
                "transformers.pipelines.get_task",
                side_effect=AssertionError("Hub get_task must not be called"),
            ),
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
            patch("gpt_task.inference.inference.use_deterministic_mode"),
            patch("torch.cuda.device_count", return_value=1),
        ):
            run_task(args, config=Config(local_files_only=True))

        load_config.assert_called_once()
        load_pipeline.assert_called_once()
        _, kwargs = load_pipeline.call_args
        self.assertEqual(kwargs["task"], "text-generation")

    def test_classic_passes_image_text_to_text_for_vlm_mapping(self):
        args = _args()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 3
        tokenizer.decode.return_value = "answer"
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
            task="image-text-to-text",
        )
        pipe._preprocess_params = {}
        pipe._forward_params = {}
        pipe._postprocess_params = {}
        pipe.return_value = [{"generated_token_ids": [1, 2, 3]}]
        generation_config = SimpleNamespace(pad_token_id=None)

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=Qwen3_5MoeConfig(),
            ),
            patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=processor,
            ),
            patch("transformers.pipeline", return_value=pipe) as load_pipeline,
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
            patch("gpt_task.inference.inference.use_deterministic_mode"),
            patch("torch.cuda.device_count", return_value=1),
        ):
            run_task(args, config=Config(local_files_only=True))

        _, kwargs = load_pipeline.call_args
        self.assertEqual(kwargs["task"], "image-text-to-text")


if __name__ == "__main__":
    unittest.main()
