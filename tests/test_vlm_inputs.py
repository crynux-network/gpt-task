import unittest
from unittest.mock import Mock, patch

import torch

from gpt_task.inference.inference import prepare_vlm_inputs
from gpt_task.inference.tp import api
from gpt_task.inference.tp.rank_worker import _prepare_task_inputs
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
    def test_prepares_hf_image_blocks_and_moves_every_tensor(self):
        for source in ("aQ==", "data:image/png;base64,aQ==", "https://example/image.png"):
            with self.subTest(source=source):
                processor = _Processor()
                inputs = prepare_vlm_inputs(
                    processor,
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image", "base64": source},
                            ],
                        }
                    ],
                    torch.device("cpu"),
                )

                self.assertEqual(
                    processor.messages[0]["content"][1],
                    {"type": "image", "base64": source},
                )
                self.assertEqual(
                    processor.kwargs,
                    {
                        "add_generation_prompt": True,
                        "tokenize": True,
                        "return_dict": True,
                        "return_tensors": "pt",
                    },
                )
                for key in ("input_ids", "attention_mask", "pixel_values"):
                    self.assertEqual(inputs[key].device.type, "cpu")
                self.assertEqual(inputs["metadata"], "unchanged")

    def test_text_only_vlm_uses_existing_prompt_adapter(self):
        args = GPTTaskArgs(
            model="test/model",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            template_args={"enable_thinking": False},
        )
        adapter = Mock()
        adapter.render_input.return_value = "adapted prompt"
        tokenizer = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )

        with (
            patch(
                "gpt_task.inference.prompt_adapters.resolve_adapter",
                return_value=adapter,
            ) as resolve_adapter,
            patch(
                "gpt_task.inference.inference.prepare_vlm_inputs"
            ) as prepare,
        ):
            encoded = _prepare_task_inputs(
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                Mock(),
                tokenizer,
                args,
                torch.device("cpu"),
            )

        resolve_adapter.assert_called_once_with(args.model, tokenizer)
        adapter.render_input.assert_called_once_with(args, tokenizer)
        tokenizer.assert_called_once_with(
            "adapted prompt",
            return_tensors="pt",
            add_special_tokens=False,
        )
        prepare.assert_not_called()
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
                "gpt_task.inference.inference.prepare_vlm_inputs",
                return_value=prepared,
            ) as prepare,
            patch(
                "gpt_task.inference.prompt_adapters.resolve_adapter"
            ) as resolve_adapter,
        ):
            result = _prepare_task_inputs(
                api.TP_MODEL_FAMILY_IMAGE_TEXT_TO_TEXT,
                processor,
                Mock(),
                args,
                device,
            )

        self.assertIs(result, prepared)
        prepare.assert_called_once_with(processor, args.messages, device)
        resolve_adapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
