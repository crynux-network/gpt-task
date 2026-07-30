import unittest
from unittest.mock import MagicMock, patch

from gpt_task.config import Config
from gpt_task.inference.tp import api, executor, shutdown_tp_executor
from gpt_task.models import GPTTaskArgs


def _args() -> GPTTaskArgs:
    return GPTTaskArgs(
        model="test/model",
        messages=[{"role": "user", "content": "hello"}],
    )


class TPExecutorLifecycleTests(unittest.TestCase):
    def tearDown(self):
        with executor._executor_lock:
            executor._executor = None

    def test_shutdown_tp_executor_is_idempotent(self):
        mock_exec = MagicMock()
        with executor._executor_lock:
            executor._executor = mock_exec

        shutdown_tp_executor()
        shutdown_tp_executor()

        mock_exec.shutdown.assert_called_once()
        with executor._executor_lock:
            self.assertIsNone(executor._executor)

    def test_run_task_tp_fallback_shuts_down_before_classic(self):
        order = []

        def record_shutdown():
            order.append("shutdown")

        def record_classic(*args, **kwargs):
            order.append("classic")
            return {"ok": True}

        with (
            patch.object(api, "_resolve_tp_task", return_value=None),
            patch.object(api, "shutdown_tp_executor", side_effect=record_shutdown),
            patch.object(api, "run_task", side_effect=record_classic) as run_task,
            patch.object(api, "submit_tp_task") as submit,
            patch("torch.cuda.device_count", return_value=2),
        ):
            result = api.run_task_tp(_args(), config=Config(local_files_only=True))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(order, ["shutdown", "classic"])
        submit.assert_not_called()
        run_task.assert_called_once()

    def test_run_task_tp_eligible_clears_worker_cache_without_shutdown(self):
        model_cache = MagicMock()
        resolution = api._TPTaskResolution(
            2,
            api.TPRuntimeStrategy(api.TP_MODEL_LOADER_CAUSAL_LM, False),
        )

        with (
            patch.object(api, "_resolve_tp_task", return_value=resolution),
            patch.object(api, "shutdown_tp_executor") as shutdown,
            patch.object(api, "submit_tp_task", return_value={"ok": True}) as submit,
            patch("torch.cuda.device_count", return_value=2),
        ):
            result = api.run_task_tp(
                _args(),
                model_cache=model_cache,
                config=Config(local_files_only=True),
            )

        self.assertEqual(result, {"ok": True})
        model_cache.clear.assert_called_once()
        shutdown.assert_not_called()
        submit.assert_called_once()

if __name__ == "__main__":
    unittest.main()
