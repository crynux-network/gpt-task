import unittest
from unittest.mock import MagicMock, patch

from gpt_task.config import Config
from gpt_task.inference import get_execution_dtype
from gpt_task.inference.execution_dtype import set_execution_dtype
from gpt_task.inference.tp import api, executor, shutdown_tp_executor
from gpt_task.inference.tp.result import TPTaskResult
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
            set_execution_dtype("bfloat16")
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
        self.assertEqual(get_execution_dtype(), "bfloat16")
        self.assertEqual(order, ["shutdown", "classic"])
        submit.assert_not_called()
        run_task.assert_called_once()

    def test_run_task_logs_device_map_plan(self):
        from gpt_task.inference import inference

        with (
            patch.object(inference, "_run_task", return_value={"ok": True}),
            patch("torch.cuda.device_count", return_value=3),
            self.assertLogs(inference._logger, level="INFO") as logs,
        ):
            result = inference.run_task(
                _args(),
                config=Config(local_files_only=True),
            )

        self.assertEqual(result, {"ok": True})
        self.assertTrue(
            any(
                "Task execution plan: mode=device_map, gpu_count=3, "
                "visible_gpus=3, model=test/model" in message
                for message in logs.output
            )
        )

    def test_run_task_tp_eligible_logs_tensor_parallel_plan(self):
        model_cache = MagicMock()
        resolution = api._TPTaskResolution(
            2,
            api.TPRuntimeStrategy(api.TP_MODEL_LOADER_CAUSAL_LM, False),
        )

        with (
            patch.object(api, "_resolve_tp_task", return_value=resolution),
            patch.object(api, "shutdown_tp_executor") as shutdown,
            patch.object(
                api,
                "submit_tp_task",
                return_value=TPTaskResult(
                    response={"ok": True},
                    execution_dtype="float16",
                ),
            ) as submit,
            patch("torch.cuda.device_count", return_value=2),
            self.assertLogs(api._logger, level="INFO") as logs,
        ):
            result = api.run_task_tp(
                _args(),
                model_cache=model_cache,
                config=Config(local_files_only=True),
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(get_execution_dtype(), "float16")
        model_cache.clear.assert_called_once()
        shutdown.assert_not_called()
        submit.assert_called_once()
        self.assertTrue(
            any(
                "Task execution plan: mode=tensor_parallel, gpu_count=2, "
                "visible_gpus=2, model=test/model" in message
                for message in logs.output
            )
        )

    def test_run_task_tp_reduce_gpus_logs_reduced_world_size(self):
        resolution = api._TPTaskResolution(
            2,
            api.TPRuntimeStrategy(api.TP_MODEL_LOADER_CAUSAL_LM, False),
        )

        with (
            patch.object(api, "_resolve_tp_task", return_value=resolution),
            patch.object(
                api,
                "submit_tp_task",
                return_value=TPTaskResult(
                    response={"ok": True},
                    execution_dtype="float32",
                ),
            ),
            patch("torch.cuda.device_count", return_value=4),
            self.assertLogs(api._logger, level="INFO") as logs,
        ):
            result = api.run_task_tp(_args(), config=Config(local_files_only=True))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(get_execution_dtype(), "float32")
        self.assertTrue(
            any(
                "Task execution plan: mode=tensor_parallel, gpu_count=2, "
                "visible_gpus=4, model=test/model" in message
                for message in logs.output
            )
        )

    def test_run_task_tp_eligible_clears_worker_cache_without_shutdown(self):
        model_cache = MagicMock()
        resolution = api._TPTaskResolution(
            2,
            api.TPRuntimeStrategy(api.TP_MODEL_LOADER_CAUSAL_LM, False),
        )

        with (
            patch.object(api, "_resolve_tp_task", return_value=resolution),
            patch.object(api, "shutdown_tp_executor") as shutdown,
            patch.object(
                api,
                "submit_tp_task",
                return_value=TPTaskResult(
                    response={"ok": True},
                    execution_dtype="float16",
                ),
            ) as submit,
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
