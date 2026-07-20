from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_lib
import socket
import threading
from typing import Callable, List, Optional

from gpt_task import models
from gpt_task.config import Config

from ..errors import (ModelDownloadError, ModelInvalid, ModelNotDownloaded,
                      TaskArgsInvalid, TaskExecutionError)
from .rank_worker import rank_worker_main

_logger = logging.getLogger(__name__)

_RESULT_POLL_INTERVAL = 1.0

# Errors raised deterministically before any collective operation. The rank
# group stays alive after them; every other error tears the group down
# because ranks may be left in an inconsistent collective state.
_PRE_EXECUTION_ERROR_TYPES = {
    cls.__name__: cls
    for cls in (
        TaskArgsInvalid,
        ModelInvalid,
        ModelDownloadError,
        ModelNotDownloaded,
    )
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rebuild_error(name: str, message: str) -> Exception:
    if name == "OutOfMemoryError":
        import torch

        return torch.cuda.OutOfMemoryError(message)
    error_type = _PRE_EXECUTION_ERROR_TYPES.get(name)
    if error_type is not None:
        return error_type()
    return TaskExecutionError(message)


class TPExecutor:
    """Persistent tensor parallel rank group.

    Spawns one rank process per visible GPU and keeps them alive across
    tasks so per-rank model shards stay cached. Task args go in and results
    come back over multiprocessing queues.
    """

    def __init__(self, world_size: int) -> None:
        assert world_size >= 2
        self._world_size = world_size
        self._seq = 0

        ctx = mp.get_context("spawn")
        self._task_queues = [ctx.Queue() for _ in range(world_size)]
        self._result_queue = ctx.Queue()

        port = _find_free_port()
        self._processes: List[mp.process.BaseProcess] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=rank_worker_main,
                args=(
                    rank,
                    world_size,
                    port,
                    self._task_queues[rank],
                    self._result_queue,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        _logger.info(
            "Spawned %d tensor parallel rank processes on port %d",
            world_size,
            port,
        )

    def all_ranks_alive(self) -> bool:
        return all(p.is_alive() for p in self._processes)

    def shutdown(self) -> None:
        for q in self._task_queues:
            try:
                q.put_nowait(("stop",))
            except queue_lib.Full:
                pass
        for p in self._processes:
            p.join(timeout=5)
        for p in self._processes:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for q in [*self._task_queues, self._result_queue]:
            q.close()
        _logger.info("Tensor parallel rank processes stopped")

    def submit(
        self,
        args: models.GPTTaskArgs,
        config: Config,
        stream_callback: Optional[Callable] = None,
    ):
        """Run one task on the rank group. Returns the task response, or
        None in stream mode. Raises the task error reconstructed from the
        failing rank; the caller owns group teardown on failure."""
        self._seq += 1
        seq = self._seq

        payload = ("task", seq, args, config, stream_callback is not None)
        for q in self._task_queues:
            q.put(payload)

        while True:
            try:
                msg = self._result_queue.get(timeout=_RESULT_POLL_INTERVAL)
            except queue_lib.Empty:
                if not self.all_ranks_alive():
                    raise TaskExecutionError(
                        "a tensor parallel rank process died"
                    )
                continue

            kind, msg_seq = msg[0], msg[1]
            if msg_seq != seq:
                continue

            if kind == "stream":
                if stream_callback is not None:
                    stream_callback(msg[2])
            elif kind == "result":
                return msg[2]
            elif kind == "error":
                _, _, error_name, error_message, error_traceback = msg
                _logger.error(
                    "TP rank task error %s: %s\n%s",
                    error_name,
                    error_message,
                    error_traceback,
                )
                raise _rebuild_error(error_name, error_message)


_executor_lock = threading.Lock()
_executor: Optional[TPExecutor] = None


def shutdown_tp_executor() -> None:
    """Tear down the persistent rank group if it exists, releasing the VRAM
    held by the cached model shards. The group is respawned lazily by the
    next TP task."""
    global _executor

    with _executor_lock:
        if _executor is not None:
            _executor.shutdown()
            _executor = None


def submit_tp_task(
    world_size: int,
    args: models.GPTTaskArgs,
    config: Config,
    stream_callback: Optional[Callable] = None,
):
    """Run one task on the lazily-spawned persistent executor, respawning
    the rank group if it died or was torn down after a previous failure."""
    global _executor

    with _executor_lock:
        if _executor is not None and not _executor.all_ranks_alive():
            _executor.shutdown()
            _executor = None
        if _executor is None:
            _executor = TPExecutor(world_size)
        executor = _executor

    try:
        return executor.submit(args, config, stream_callback)
    except Exception as e:
        if type(e).__name__ not in _PRE_EXECUTION_ERROR_TYPES:
            with _executor_lock:
                if _executor is executor:
                    executor.shutdown()
                    _executor = None
        raise
