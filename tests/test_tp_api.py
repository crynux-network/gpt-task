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
from gpt_task.inference.model_adapters.tp_plan import (
    TPPlanValidationContext,
    resolve_tp_plan_validator,
)
from gpt_task.inference.model_adapters.tp_plan.llama4 import (
    Llama4VisionTPPlanValidator,
)
from gpt_task.inference.model_adapters.tp_plan.standard import (
    StandardTPPlanValidator,
    infer_plan_dimensions,
)
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


def _image_strategy() -> api.TPRuntimeStrategy:
    return api.TPRuntimeStrategy(
        api.TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
        True,
    )


class TPModelResolutionTests(unittest.TestCase):
    def test_resolves_image_text_strategy_before_causal_mapping(self):
        model_config = Qwen3_5MoeConfig()

        self.assertEqual(
            api._resolve_runtime_strategy(model_config),
            _image_strategy(),
        )
        self.assertEqual(
            api._resolve_runtime_strategy(model_config.get_text_config()),
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_CAUSAL_LM,
                False,
            ),
        )
        self.assertEqual(
            api._resolve_runtime_strategy(SimpleNamespace(vision_config=object())),
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_CAUSAL_LM,
                False,
            ),
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
                api.TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
                    True,
                ),
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

    def test_ernie_reduce_gpus_resolves_six_visible_gpus_to_four(self):
        with (
            patch.object(
                api, "_load_model_config", return_value=Ernie4_5_VLMoeConfig()
            ),
            patch.dict(os.environ, {"GPT_TP_FALLBACK": "reduce_gpus"}),
        ):
            resolution = api._resolve_tp_task(
                _args(image=True),
                Config(local_files_only=True),
                6,
            )

        self.assertEqual(resolution.world_size, 4)

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

    def test_missing_text_plan_falls_back(self):
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

    def test_remote_auto_map_selects_native_then_remote_causal_strategy(self):
        native = SimpleNamespace(
            auto_map={
                "AutoModelForImageTextToText": "remote.NativeVLM",
                "AutoModelForCausalLM": "remote.CausalVLM",
            }
        )
        causal = SimpleNamespace(
            auto_map={"AutoModelForCausalLM": "remote.CausalVLM"}
        )

        self.assertEqual(
            api._resolve_runtime_strategy(native, has_image_input=True),
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
                True,
            ),
        )
        self.assertEqual(
            api._resolve_runtime_strategy(causal, has_image_input=True),
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_CAUSAL_LM,
                True,
            ),
        )
        self.assertEqual(
            api._resolve_runtime_strategy(causal, has_image_input=False),
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_CAUSAL_LM,
                False,
            ),
        )

    def test_unmapped_config_with_compatible_plan_is_not_rejected(self):
        text_config = SimpleNamespace(
            base_model_tp_plan={"layers.*.self_attn.q_proj": "colwise"},
            hidden_size=16,
            num_attention_heads=4,
        )
        model_config = SimpleNamespace(
            get_text_config=lambda: text_config,
            sub_configs={},
        )
        with patch.object(api, "_load_model_config", return_value=model_config):
            resolution = api._resolve_tp_task(
                _args(),
                Config(local_files_only=True),
                2,
            )

        self.assertEqual(
            resolution.strategy.model_loader,
            api.TP_MODEL_LOADER_CAUSAL_LM,
        )


class TPPlanValidationTests(unittest.TestCase):
    def test_generic_dimension_check_delegates_to_plan_registry(self):
        model_config = Qwen3_5MoeConfig()
        strategy = _image_strategy()

        with patch.object(
            api,
            "validate_effective_tp_plan",
            return_value=True,
        ) as validate:
            result = api._dims_divisible_by(model_config, strategy, 2)

        self.assertTrue(result)
        validate.assert_called_once_with(
            model_config,
            api._effective_tp_plan(model_config, strategy),
            2,
        )

    def test_each_plan_inferred_text_dimension_must_be_divisible(self):
        model_config = Qwen3_5MoeConfig()
        effective_plan = api._effective_tp_plan(
            model_config,
            api.TPRuntimeStrategy(
                api.TP_MODEL_LOADER_IMAGE_TEXT_TO_TEXT,
                True,
            ),
        )
        inferred_dims = infer_plan_dimensions(
            TPPlanValidationContext(
                model_config=model_config,
                config=model_config.get_text_config(),
                plan={
                name: style
                for name, style in effective_plan.items()
                if ".vision_config." not in name
                },
                plan_scope="text",
                candidate_world_size=2,
            )
        )

        for attr in inferred_dims:
            with self.subTest(attr=attr):
                model_config = Qwen3_5MoeConfig()
                setattr(model_config.get_text_config(), attr, 3)
                self.assertFalse(
                    api._dims_divisible_by(
                        model_config,
                        _image_strategy(),
                        2,
                    )
                )

    def test_unsharded_dimension_does_not_restrict_world_size(self):
        model_config = Qwen3_5MoeConfig()
        model_config.get_text_config().intermediate_size = 3

        self.assertTrue(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
                2,
            )
        )

    def test_vocab_is_checked_when_plan_shards_lm_head(self):
        text_config = SimpleNamespace(
            base_model_tp_plan={"lm_head": "colwise"},
            vocab_size=17,
        )
        model_config = SimpleNamespace(
            get_text_config=lambda: text_config,
            sub_configs={},
        )

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
                2,
            )
        )

    def test_list_valued_moe_dimensions_are_checked(self):
        model_config = Ernie4_5_VLMoeConfig()
        model_config.get_text_config().moe_intermediate_size = [1536, 511]

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
                2,
            )
        )

    def test_replicated_vision_dimensions_are_not_checked(self):
        model_config = Qwen3_5MoeConfig()
        model_config.vision_config.hidden_size = 1279

        self.assertTrue(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
                2,
            )
        )

    def test_sharded_vision_dimensions_are_checked(self):
        model_config = Ernie4_5_VLMoeConfig()
        model_config.vision_config.num_heads = 15

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
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

        context = TPPlanValidationContext(
            model_config=model_config,
            config=model_config.vision_config,
            plan=model_config.vision_config.base_model_tp_plan,
            plan_scope="vision",
            candidate_world_size=2,
        )
        result = resolve_tp_plan_validator(context).validate(context)
        self.assertFalse(result.supported)

    def test_unknown_text_plan_entry_is_not_accepted(self):
        text_config = SimpleNamespace(
            base_model_tp_plan={"layers.*.mystery_proj": "colwise"},
            hidden_size=16,
        )
        model_config = SimpleNamespace(
            get_text_config=lambda: text_config,
            sub_configs={},
        )

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                api.TPRuntimeStrategy(
                    api.TP_MODEL_LOADER_CAUSAL_LM,
                    False,
                ),
                2,
            )
        )

    def test_llama4_nonstandard_vision_plan_uses_specific_dimensions(self):
        model_config = Llama4Config()
        model_config.vision_config.hidden_size = 767

        self.assertFalse(
            api._dims_divisible_by(
                model_config,
                _image_strategy(),
                2,
            )
        )

    def test_registry_selects_standard_and_llama4_validators(self):
        standard_context = TPPlanValidationContext(
            model_config=SimpleNamespace(model_type="other"),
            config=SimpleNamespace(hidden_size=16),
            plan={"layers.*.self_attn.q_proj": "colwise"},
            plan_scope="text",
            candidate_world_size=2,
        )
        self.assertIsInstance(
            resolve_tp_plan_validator(standard_context),
            StandardTPPlanValidator,
        )

        llama = Llama4Config()
        llama_context = TPPlanValidationContext(
            model_config=llama,
            config=llama.vision_config,
            plan=llama.vision_config.base_model_tp_plan,
            plan_scope="vision",
            candidate_world_size=2,
        )
        self.assertIsInstance(
            resolve_tp_plan_validator(llama_context),
            Llama4VisionTPPlanValidator,
        )

    def test_empty_text_plan_is_not_tp_eligible(self):
        model_config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(base_model_tp_plan={})
        )

        self.assertFalse(api._has_tp_plan(model_config))


if __name__ == "__main__":
    unittest.main()
