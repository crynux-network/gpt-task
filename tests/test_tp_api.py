import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeConfig,
)
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from gpt_task.config import Config
from gpt_task.inference.tp import api
from gpt_task.models import GPTTaskArgs


def _args(*, image: bool = False) -> GPTTaskArgs:
    content = (
        [
            {"type": "text", "text": "describe"},
            {"type": "image", "base64": "aQ=="},
        ]
        if image
        else "hello"
    )
    return GPTTaskArgs(model="test/model", messages=[{"role": "user", "content": content}])


class TPModelResolutionTests(unittest.TestCase):
    def test_resolves_image_text_model_before_causal_mapping(self):
        model_config = Qwen3_5MoeConfig()

        self.assertEqual(
            api._resolve_model_family(model_config),
            api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
        )
        self.assertEqual(
            api._resolve_model_family(model_config.get_text_config()),
            api.TP_MODEL_FAMILY_CAUSAL_LM,
        )
        self.assertIsNone(
            api._resolve_model_family(SimpleNamespace(vision_config=object()))
        )

    def test_image_request_uses_tp_when_mapping_and_plan_are_compatible(self):
        with patch.object(
            api, "_load_model_config", return_value=Qwen3_5MoeConfig()
        ) as load:
            resolution = api._resolve_tp_task(
                _args(image=True),
                Config(local_files_only=True),
                2,
            )

        self.assertEqual(
            resolution,
            api._TPTaskResolution(
                2,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
            ),
        )
        load.assert_called_once()

    def test_quantized_and_single_gpu_tasks_do_not_load_config(self):
        quantized = _args()
        quantized.quantize_bits = 4

        with patch.object(api, "_load_model_config") as load:
            self.assertIsNone(
                api._resolve_tp_task(
                    quantized,
                    Config(local_files_only=True),
                    2,
                )
            )
            self.assertIsNone(
                api._resolve_tp_task(
                    _args(),
                    Config(local_files_only=True),
                    1,
                )
            )
        load.assert_not_called()

    def test_reduce_gpus_uses_largest_compatible_world_size(self):
        with (
            patch.object(
                api, "_load_model_config", return_value=Qwen3_5MoeConfig()
            ),
            patch.dict(os.environ, {"GPT_TP_FALLBACK": "reduce_gpus"}),
        ):
            resolution = api._resolve_tp_task(
                _args(image=True),
                Config(local_files_only=True),
                4,
            )

        self.assertEqual(resolution.world_size, 2)

    def test_unknown_fallback_value_uses_device_map_behavior(self):
        with (
            patch.object(
                api, "_load_model_config", return_value=Qwen3_5MoeConfig()
            ),
            patch.dict(os.environ, {"GPT_TP_FALLBACK": "classic"}),
        ):
            self.assertIsNone(
                api._resolve_tp_task(
                    _args(image=True),
                    Config(local_files_only=True),
                    4,
                )
            )

    def test_missing_text_plan_and_missing_mapping_fall_back(self):
        for model_config in (
            Qwen3VLConfig(),
            SimpleNamespace(get_text_config=lambda: SimpleNamespace()),
        ):
            with self.subTest(config=type(model_config).__name__), patch.object(
                api,
                "_load_model_config",
                return_value=model_config,
            ):
                self.assertIsNone(
                    api._resolve_tp_task(
                        _args(image=True),
                        Config(local_files_only=True),
                        2,
                    )
                )


class TPPlanValidationTests(unittest.TestCase):
    def test_each_text_sharded_dimension_must_be_divisible(self):
        for attr in api._TEXT_TP_SHARDED_DIM_ATTRS:
            with self.subTest(attr=attr):
                model_config = Qwen3_5MoeConfig()
                setattr(model_config.get_text_config(), attr, 3)
                self.assertFalse(
                    api._dims_divisible_by(
                        model_config,
                        api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                        2,
                    )
                )

    def test_vocab_is_checked_when_model_class_shards_lm_head(self):
        model_config = Qwen3_5MoeConfig()
        model_config.get_text_config().vocab_size = 248321

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                2,
            )
        )

    def test_list_valued_moe_dimensions_are_checked(self):
        model_config = Ernie4_5_VLMoeConfig()
        model_config.get_text_config().moe_intermediate_size = [1536, 511]

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                2,
            )
        )

    def test_replicated_vision_dimensions_are_not_checked(self):
        model_config = Qwen3_5MoeConfig()
        model_config.vision_config.hidden_size = 1279

        self.assertTrue(
            api._dims_divisible_by(
                model_config,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                2,
            )
        )

    def test_sharded_vision_dimensions_are_checked(self):
        model_config = Ernie4_5_VLMoeConfig()
        model_config.vision_config.num_heads = 15

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                2,
            )
        )

    def test_unknown_nonstandard_vision_plan_is_not_accepted(self):
        model_config = SimpleNamespace(
            model_type="new_vlm",
            vision_config=SimpleNamespace(
                base_model_tp_plan={"patch_adapter.proj": "colwise"},
            ),
        )

        self.assertIsNone(api._vision_plan_dims(model_config))

    def test_llama4_nonstandard_vision_plan_uses_specific_dimensions(self):
        model_config = Llama4Config()
        model_config.vision_config.hidden_size = 767

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                2,
            )
        )

    def test_empty_text_plan_is_not_tp_eligible(self):
        model_config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(base_model_tp_plan={})
        )

        self.assertFalse(api._has_tp_plan(model_config))


if __name__ == "__main__":
    unittest.main()
