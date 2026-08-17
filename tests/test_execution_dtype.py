import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from gpt_task.cache import MemoryModelCache
from gpt_task.config import Config
from gpt_task.inference import get_execution_dtype, run_task
from gpt_task.inference.input_rendering import RenderedTaskInput
from gpt_task.inference.key import generate_model_key
from gpt_task.inference.tp.rank_worker import _execute_task
from gpt_task.inference.tp.result import TPTaskResult
from gpt_task.inference.tp.runtime_strategy import (
    TP_MODEL_LOADER_CAUSAL_LM,
    TPRuntimeStrategy,
)
from gpt_task.models import GPTTaskArgs


def _args() -> GPTTaskArgs:
    return GPTTaskArgs(
        model="test/model",
        messages=[{"role": "user", "content": "hello"}],
        dtype="float32",
    )


class ClassicExecutionDtypeTests(unittest.TestCase):
    def test_reports_loaded_model_dtype_on_load_and_cache_hit(self):
        args = _args()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 3
        tokenizer.decode.return_value = "answer"
        processor = MagicMock(tokenizer=tokenizer)
        model = MagicMock()
        model.dtype = torch.bfloat16
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
        model_cache = MemoryModelCache()

        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
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
            first = run_task(args, config=Config(), model_cache=model_cache)
            self.assertEqual(get_execution_dtype(), "bfloat16")

            model.dtype = torch.float16
            second = run_task(args, config=Config(), model_cache=model_cache)

        self.assertEqual(get_execution_dtype(), "float16")
        self.assertEqual(first, second)
        self.assertNotIn("execution_dtype", second)
        load_pipeline.assert_called_once()


class TensorParallelExecutionDtypeTests(unittest.TestCase):
    def test_rank_result_reports_loaded_model_dtype_on_cache_hit(self):
        args = _args()
        strategy = TPRuntimeStrategy(TP_MODEL_LOADER_CAUSAL_LM, False)
        model_key = f"{strategy!r}:{generate_model_key(args)}"
        tokenizer = MagicMock(eos_token_id=3)
        tokenizer.decode.return_value = "answer"
        model = MagicMock()
        model.dtype = torch.bfloat16
        model.config = SimpleNamespace()
        model.generation_config = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        model_cache = {model_key: (model, tokenizer, None)}
        generation_config = SimpleNamespace(pad_token_id=None)

        with (
            patch("transformers.set_seed"),
            patch("gpt_task.inference.utils.use_deterministic_mode"),
            patch(
                "gpt_task.inference.utils.resolve_generation_config",
                return_value=generation_config,
            ),
            patch(
                "gpt_task.inference.tp.rank_worker._prepare_task_inputs",
                return_value={"input_ids": torch.tensor([[1, 2]])},
            ),
            patch(
                "gpt_task.inference.tp.rank_worker._load_rank_artifacts"
            ) as load_artifacts,
        ):
            first = _execute_task(
                0,
                1,
                strategy,
                args,
                Config(),
                False,
                MagicMock(),
                model_cache,
            )
            model.dtype = torch.float16
            second = _execute_task(
                0,
                2,
                strategy,
                args,
                Config(),
                False,
                MagicMock(),
                model_cache,
            )

        self.assertIsInstance(first, TPTaskResult)
        self.assertEqual(first.execution_dtype, "bfloat16")
        self.assertEqual(second.execution_dtype, "float16")
        self.assertNotIn("execution_dtype", second.response)
        load_artifacts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
