import unittest

from gpt_task.inference.executed_gpu_count import (
    clear_executed_gpu_count,
    get_executed_gpu_count,
    set_executed_gpu_count,
)


class ExecutedGPUCountTests(unittest.TestCase):
    def tearDown(self):
        clear_executed_gpu_count()

    def test_default_is_none(self):
        clear_executed_gpu_count()
        self.assertIsNone(get_executed_gpu_count())

    def test_set_and_get(self):
        set_executed_gpu_count(2)
        self.assertEqual(get_executed_gpu_count(), 2)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            set_executed_gpu_count(-1)


if __name__ == "__main__":
    unittest.main()
