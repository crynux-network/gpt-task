import unittest

from pydantic import ValidationError

from gpt_task.models import GPTTaskArgs


class TaskArgsVersionTests(unittest.TestCase):
    def test_rejects_missing_version(self):
        with self.assertRaises(ValidationError):
            GPTTaskArgs.model_validate(
                {
                    "model": "org/model",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )

    def test_accepts_explicit_version(self):
        args = GPTTaskArgs.model_validate(
            {
                "version": "3.7.0",
                "model": "org/model",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        self.assertEqual(args.version, "3.7.0")

    def test_rejects_empty_version(self):
        with self.assertRaises(ValidationError):
            GPTTaskArgs.model_validate(
                {
                    "version": "",
                    "model": "org/model",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
