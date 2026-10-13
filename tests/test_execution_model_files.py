import unittest
from dataclasses import dataclass

from gpt_task.inference import ExecutionModelFileError, apply_execution_model_files
from gpt_task.inference.key import generate_model_key
from gpt_task.models import GPTTaskArgs

SHA = "a" * 40
OTHER_SHA = "b" * 40


@dataclass(frozen=True)
class Entry:
    repository: str
    revision: str
    format: str = ""


def _args(**kwargs) -> GPTTaskArgs:
    payload = {
        "version": "3.7.0",
        "model": "org/model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(kwargs)
    return GPTTaskArgs.model_validate(payload)


class ExecutionModelFilesTests(unittest.TestCase):
    def test_applies_allocated_revision(self):
        args = _args()

        applied = apply_execution_model_files(args, [Entry("org/model", SHA)])

        self.assertEqual(applied.revision, SHA)
        self.assertIsNone(args.revision)

    def test_accepts_matching_task_revision(self):
        applied = apply_execution_model_files(
            _args(revision=SHA), [Entry("org/model", SHA)]
        )

        self.assertEqual(applied.revision, SHA)

    def test_rejects_conflicting_task_revision(self):
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(
                _args(revision=OTHER_SHA), [Entry("org/model", SHA)]
            )

    def test_branch_task_revision_uses_allocated_sha(self):
        applied = apply_execution_model_files(
            _args(revision="main"), [Entry("org/model", SHA)]
        )

        self.assertEqual(applied.revision, SHA)

    def test_rejects_missing_allocation(self):
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(_args(), [])
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(_args(), [Entry("org/other", SHA)])

    def test_rejects_empty_revision(self):
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(_args(), [Entry("org/model", "")])

    def test_rejects_conflicting_allocations(self):
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(
                _args(), [Entry("org/model", SHA), Entry("org/model", OTHER_SHA)]
            )

    def test_rejects_unsupported_format(self):
        with self.assertRaises(ExecutionModelFileError):
            apply_execution_model_files(
                _args(), [Entry("org/model", SHA, format="gguf")]
            )

    def test_model_key_depends_on_allocated_revision(self):
        first = apply_execution_model_files(_args(), [Entry("org/model", SHA)])
        second = apply_execution_model_files(_args(), [Entry("org/model", OTHER_SHA)])

        self.assertNotEqual(
            generate_model_key(first, executor_mode="device_map"),
            generate_model_key(second, executor_mode="device_map"),
        )


if __name__ == "__main__":
    unittest.main()
