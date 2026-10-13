import unittest

from pydantic import ValidationError

from gpt_task.inference.model_adapters.quantization import (
    UnsupportedQuantizationBackend,
    resolve_task_quantization,
)
from gpt_task.inference.model_adapters.quantization.bitsandbytes import (
    BitsAndBytesQuantizationAdapter,
)
from gpt_task.models import GPTTaskArgs, QuantizationConfig


def _args(**kwargs) -> GPTTaskArgs:
    payload = {
        "version": "3.7.0",
        "model": "org/model",
        "messages": [{"role": "user", "content": "hi"}],
        "dtype": "bfloat16",
    }
    payload.update(kwargs)
    return GPTTaskArgs.model_validate(payload)


class QuantizationArgsTests(unittest.TestCase):
    def test_rejects_quantize_bits(self):
        with self.assertRaises(ValidationError):
            _args(quantize_bits=4)

    def test_accepts_stable_quantization_config(self):
        args = _args(
            quantization_config={
                "quant_method": "bitsandbytes",
                "quant_kwargs": {"load_in_4bit": True},
            }
        )
        self.assertEqual(args.quantization_config.quant_method, "bitsandbytes")
        self.assertEqual(
            args.quantization_config.quant_kwargs["load_in_4bit"],
            True,
        )

    def test_rejects_unknown_quantization_fields(self):
        with self.assertRaises(ValidationError):
            _args(
                quantization_config={
                    "quant_method": "bitsandbytes",
                    "quant_kwargs": {},
                    "extra": 1,
                }
            )

    def test_rejects_dequantize_true(self):
        with self.assertRaises(ValidationError):
            _args(
                quantization_config={
                    "quant_method": "fp8",
                    "quant_kwargs": {"dequantize": True},
                }
            )

    def test_rejects_dequantize_false(self):
        with self.assertRaises(ValidationError):
            _args(
                quantization_config={
                    "quant_method": "fp8",
                    "quant_kwargs": {"dequantize": False},
                }
            )

    def test_revision_defaults_to_none(self):
        args = _args()
        self.assertIsNone(args.revision)

    def test_accepts_explicit_revision(self):
        args = _args(revision="abc123def456")
        self.assertEqual(args.revision, "abc123def456")

    def test_rejects_empty_revision(self):
        with self.assertRaises(ValidationError):
            _args(revision="")

    def test_rejects_format_fields(self):
        with self.assertRaises(ValidationError):
            _args(format="gguf")
        with self.assertRaises(ValidationError):
            _args(format_kwargs={"filename": "model.gguf"})

    def test_unsupported_backend(self):
        args = _args(
            quantization_config={
                "quant_method": "torchao",
                "quant_kwargs": {},
            }
        )
        with self.assertRaises(UnsupportedQuantizationBackend):
            resolve_task_quantization(args)

    def test_bitsandbytes_builds_official_config(self):
        args = _args(
            quantization_config={
                "quant_method": "bitsandbytes",
                "quant_kwargs": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                },
            }
        )
        resolution = BitsAndBytesQuantizationAdapter().resolve(
            args.quantization_config,
            args,
        )
        self.assertTrue(resolution.transformers_config.load_in_4bit)
        self.assertEqual(
            resolution.effective_config["quant_kwargs"]["bnb_4bit_quant_type"],
            "nf4",
        )
        self.assertEqual(
            resolution.effective_config["quant_kwargs"]["bnb_4bit_compute_dtype"],
            "bfloat16",
        )

    def test_embedded_and_task_conflict(self):
        args = _args(
            quantization_config={
                "quant_method": "bitsandbytes",
                "quant_kwargs": {"load_in_4bit": True},
            }
        )
        model_config = type(
            "Cfg",
            (),
            {"quantization_config": {"quant_method": "fp8"}},
        )()
        with self.assertRaises(Exception):
            resolve_task_quantization(args, model_config=model_config)


if __name__ == "__main__":
    unittest.main()
