"""Fully-async rollout with explicit checkpoint and weight-update barriers."""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, abort, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
    "quiesce_rollout",
    "shutdown_rollout",
]

logger = logging.getLogger("slime.rollout.fully_async")

FULLY_ASYNC_SUPPORTS_EVALUATION = False

FULLY_ASYNC_QUIESCE_TIMEOUT_SECONDS = 60.0
FULLY_ASYNC_ABORT_TIMEOUT_SECONDS = 15.0
FULLY_ASYNC_TASK_DRAIN_TIMEOUT_SECONDS = 15.0
WORKER_POLL_INTERVAL_SECONDS = 0.05

GeneratedGroup = list[Sample] | list[list[Sample]]


@dataclass
class _CompletedGroup:
    gid: int
    result: GeneratedGroup
    original: list[Sample]


@dataclass
class _ActiveGroup:
    gid: int
    original: list[Sample]
    generated: list[Sample] | None = None


class WorkerPhase(Enum):
    NEW = auto()
    RUNNING = auto()
    QUIESCING = auto()
    QUIESCED = auto()
    FAILED = auto()
    STOPPING = auto()
    STOPPED = auto()


_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer, *, worker_factory=None) -> AsyncRolloutWorker:
    global _global_worker
    worker_factory = worker_factory or AsyncRolloutWorker
    unsupported_options = {
        "dynamic_sampling_filter_path": getattr(args, "dynamic_sampling_filter_path", None),
        "rollout_sample_filter_path": getattr(args, "rollout_sample_filter_path", None),
        "rollout_all_samples_process_path": getattr(args, "rollout_all_samples_process_path", None),
        "partial_rollout": getattr(args, "partial_rollout", False),
        "buffer_filter_path": getattr(args, "buffer_filter_path", None),
    }
    enabled_unsupported = [name for name, value in unsupported_options.items() if value]
    if enabled_unsupported:
        raise ValueError("fully-async rollout does not implement these stock rollout semantics: " + ", ".join(enabled_unsupported))
    if getattr(args, "eval_interval", None) is not None:
        raise ValueError("fully-async rollout does not support evaluation")
    if not callable(getattr(data_buffer, "prepend_samples", None)):
        raise TypeError("fully-async rollout requires a data source with prepend_samples()")
    with _worker_lock:
        if _global_worker is None:
            concurrency = args.sglang_server_concurrency * get_rollout_num_engines(args)
            if concurrency <= 0:
                raise ValueError("fully-async rollout requires at least one rollout engine")
            logger.info("starting fully-async rollout worker")
            _global_worker = worker_factory(
                args,
                data_buffer,
                concurrency=concurrency,
            )
            _global_worker.start()
        worker = _global_worker
    if type(worker) is not worker_factory:
        raise RuntimeError("fully-async rollout cannot switch a live worker factory")
    if worker.data_buffer is not data_buffer:
        raise RuntimeError("fully-async rollout worker cannot switch data sources")
    worker.ensure_healthy()
    return worker


def quiesce_rollout() -> None:
    with _worker_lock:
        worker = _global_worker
    if worker is not None:
        worker.quiesce()


def shutdown_rollout() -> None:
    global _global_worker
    with _worker_lock:
        worker = _global_worker
    if worker is None:
        return
    try:
        worker.shutdown()
    finally:
        if not worker.is_alive() and not worker.has_pending_resources():
            with _worker_lock:
                if _global_worker is worker:
                    _global_worker = None


def _shutdown_rollout_at_exit() -> None:
    try:
        shutdown_rollout()
    except BaseException:
        logger.exception("fully-async worker failed during interpreter shutdown")


atexit.register(_shutdown_rollout_at_exit)


class AsyncRolloutWorker:
    """Generate bounded prefetch on one persistent asyncio event loop."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        if concurrency <= 0:
            raise ValueError("fully-async concurrency must be greater than zero")
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.prefetch_limit = max(concurrency, int(args.rollout_batch_size))
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.state: GenerateState | None = None

        self._pause_requested = threading.Event()
        self._pause_requested.set()
        self._paused = threading.Event()
        self._shutdown_requested = threading.Event()
        self._stopped = threading.Event()
        self._wake_requested = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._phase_lock = threading.Lock()
        self._phase = WorkerPhase.NEW
        self._restore_lock = threading.Lock()
        self._external_restore: list[_CompletedGroup] = []
        self._retry_groups: list[_ActiveGroup] = []
        self._rollout_id_lock = threading.Lock()
        self._rollout_id = -1

    def start(self) -> None:
        with self._phase_lock:
            if self._phase is not WorkerPhase.NEW:
                raise RuntimeError("fully-async worker has already been started")
            self._phase = WorkerPhase.QUIESCED
        self.worker_thread = threading.Thread(
            target=self._thread_main,
            name="fully-async-rollout",
            daemon=True,
        )
        self.worker_thread.start()

    def resume(self, rollout_id: int) -> None:
        self.ensure_healthy()
        with self._phase_lock:
            if self._phase is WorkerPhase.QUIESCING:
                raise RuntimeError("cannot resume fully-async worker while it is quiescing")
            if self._phase not in (WorkerPhase.QUIESCED, WorkerPhase.RUNNING):
                raise RuntimeError(f"cannot resume fully-async worker in phase {self._phase.name}")
            self._phase = WorkerPhase.RUNNING
        with self._rollout_id_lock:
            self._rollout_id = rollout_id
        self._pause_requested.clear()
        self._paused.clear()
        self._wake_requested.set()

    def quiesce(self, timeout: float = FULLY_ASYNC_QUIESCE_TIMEOUT_SECONDS) -> None:
        if timeout <= 0:
            raise ValueError("fully-async quiesce timeout must be greater than zero")
        with self._phase_lock:
            if self._phase is WorkerPhase.RUNNING:
                self._phase = WorkerPhase.QUIESCING
            elif self._phase not in (
                WorkerPhase.QUIESCING,
                WorkerPhase.QUIESCED,
                WorkerPhase.FAILED,
            ):
                self.ensure_healthy()
                raise RuntimeError(f"cannot quiesce fully-async worker in phase {self._phase.name}")
        self._pause_requested.set()
        self._wake_requested.set()
        if not self._paused.wait(timeout):
            raise TimeoutError(f"fully-async worker did not quiesce within {timeout:g} seconds")
        self.ensure_healthy()

    def shutdown(self, timeout: float = FULLY_ASYNC_QUIESCE_TIMEOUT_SECONDS) -> None:
        if timeout <= 0:
            raise ValueError("fully-async shutdown timeout must be greater than zero")
        with self._phase_lock:
            if self._phase is WorkerPhase.NEW:
                self._phase = WorkerPhase.STOPPED
                self._stopped.set()
                return
            if self._phase is not WorkerPhase.STOPPED:
                self._phase = WorkerPhase.STOPPING
        self._shutdown_requested.set()
        self._pause_requested.set()
        self._wake_requested.set()
        thread = self.worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise TimeoutError(f"fully-async worker did not stop within {timeout:g} seconds")
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("fully-async rollout worker failed during shutdown") from failure

    def stop(self, timeout: float = FULLY_ASYNC_QUIESCE_TIMEOUT_SECONDS) -> None:
        """Backward-compatible alias for terminal worker shutdown."""
        self.shutdown(timeout=timeout)

    def ensure_healthy(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("fully-async rollout worker failed") from failure
        if self._shutdown_requested.is_set() or self._stopped.is_set():
            raise RuntimeError("fully-async rollout worker is stopped")
        thread = self.worker_thread
        if thread is not None and not thread.is_alive():
            raise RuntimeError("fully-async rollout worker exited unexpectedly")

    def is_alive(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def has_pending_resources(self) -> bool:
        """Keep a failed worker addressable while its extension owns resources."""
        return False

    def get_completed_groups(self, limit: int) -> list[_CompletedGroup]:
        if limit <= 0:
            return []
        completed = []
        for _ in range(limit):
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def request_restore(self, completed: list[_CompletedGroup]) -> None:
        if not completed:
            return
        with self._restore_lock:
            self._external_restore.extend(completed)
        self._wake_requested.set()

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def _thread_main(self) -> None:
        try:
            run(self._loop())
        except BaseException as exc:
            self._set_failure(exc)
        finally:
            with self._phase_lock:
                self._phase = WorkerPhase.STOPPED
            self._paused.set()
            self._stopped.set()

    async def _loop(self) -> None:
        self.state = GenerateState(self.args)
        active: dict[asyncio.Task, _ActiveGroup] = {}
        next_gid = 0

        try:
            while not self._shutdown_requested.is_set():
                if self._pause_requested.is_set():
                    await self._quiesce_active(active)
                    with self._phase_lock:
                        if self._phase not in (WorkerPhase.FAILED, WorkerPhase.STOPPING):
                            self._phase = WorkerPhase.QUIESCED
                    self._paused.set()
                    await self._wait_until_resumed_or_shutdown()
                    continue

                self._paused.clear()
                source_empty = False
                while len(active) < self.concurrency and len(active) + self.output_queue.qsize() < self.prefetch_limit and not self._pause_requested.is_set() and not self._shutdown_requested.is_set():
                    if self._retry_groups:
                        retry = self._retry_groups.pop(0)
                        work_items = [retry]
                    else:
                        work_items, next_gid = self._take_work(next_gid)
                        if not work_items:
                            source_empty = True
                            break
                    for item in work_items:
                        original = item.original
                        generated = item.generated if item.generated is not None else copy.deepcopy(original)
                        item.generated = None
                        task = asyncio.create_task(
                            self._run_group(item, generated, self.state.sampling_params.copy())
                        )
                        active[task] = item

                next_gid = self._prefetch_work(next_gid)

                if source_empty and not active and self.output_queue.empty():
                    raise RuntimeError("fully-async data source returned no samples")

                if not active:
                    await self._sleep_until_woken()
                    continue

                done, _ = await asyncio.wait(
                    active,
                    timeout=WORKER_POLL_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    item = active.pop(task)
                    try:
                        result = task.result()
                        self._validate_generated_group(result)
                    except asyncio.CancelledError:
                        self._retry_groups.append(item)
                        self._retry_groups.sort(key=lambda value: value.gid)
                        continue
                    except BaseException as exc:
                        self._retry_groups.append(item)
                        self._retry_groups.sort(key=lambda value: value.gid)
                        self._set_failure(exc)
                        self._pause_requested.set()
                        break
                    if self._contains_aborted_sample(result):
                        self._retry_groups.append(item)
                        self._retry_groups.sort(key=lambda value: value.gid)
                    else:
                        self.output_queue.put_nowait(_CompletedGroup(item.gid, result, item.original))
        finally:
            await self._quiesce_active(active)
            self._paused.set()

    def _take_work(self, next_gid: int) -> tuple[list[_ActiveGroup], int]:
        groups = self.data_buffer.get_samples(1)
        return ([_ActiveGroup(next_gid + offset, group) for offset, group in enumerate(groups)],
                next_gid + len(groups))

    def _prefetch_work(self, next_gid: int) -> int:
        return next_gid

    async def _run_group(self, item: _ActiveGroup, generated: list[Sample], sampling_params: dict):
        return await generate_and_rm_group(self.args, generated,
            sampling_params=sampling_params, evaluation=False)

    async def _quiesce_resources(self, restore: list[_ActiveGroup]) -> None:
        """Join extension resources before the source owner checkpoints."""

    async def _quiesce_active(
        self,
        active: dict[asyncio.Task, _ActiveGroup],
    ) -> None:
        restore = [*self._retry_groups, *active.values()]
        self._retry_groups = []
        active_tasks = list(active)
        active.clear()

        if active_tasks:
            assert self.state is not None
            if not self.state.aborted:
                try:
                    await asyncio.wait_for(
                        abort(
                            self.args,
                            self._current_rollout_id(),
                            raise_on_engine_error=True,
                        ),
                        timeout=FULLY_ASYNC_ABORT_TIMEOUT_SECONDS,
                    )
                except BaseException as exc:
                    self._set_failure(
                        exc,
                        context="failed to abort fully-async rollout requests",
                    )
            for task in active_tasks:
                task.cancel()
            done, pending = await asyncio.wait(
                active_tasks,
                timeout=FULLY_ASYNC_TASK_DRAIN_TIMEOUT_SECONDS,
            )
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                for task in pending:
                    task.add_done_callback(self._consume_task_result)
                self._set_failure(TimeoutError("fully-async generation tasks did not stop after abort"))

        while True:
            try:
                completed = self.output_queue.get_nowait()
            except queue.Empty:
                break
            restore.append(_ActiveGroup(completed.gid, completed.original))
        with self._restore_lock:
            external_restore = self._external_restore
            self._external_restore = []
        restore.extend(_ActiveGroup(completed.gid, completed.original) for completed in external_restore)
        try:
            await self._quiesce_resources(restore)
        except BaseException as exc:
            self._set_failure(exc, context="failed to release fully-async worker resources")
        self._restore_groups(restore)
        if self.state is not None:
            self.state.reset()

    async def _wait_until_resumed_or_shutdown(self) -> None:
        while self._pause_requested.is_set() and not self._shutdown_requested.is_set():
            await self._sleep_until_woken()

    async def _sleep_until_woken(self) -> None:
        self._wake_requested.clear()
        await asyncio.sleep(WORKER_POLL_INTERVAL_SECONDS)

    def _restore_groups(self, groups: list[_ActiveGroup]) -> None:
        if not groups:
            return
        ordered = [item.original for item in sorted(groups, key=lambda item: item.gid)]
        try:
            self.data_buffer.prepend_samples(ordered)
        except BaseException as exc:
            self._set_failure(
                exc,
                context="failed to restore fully-async source samples",
            )

    def _current_rollout_id(self) -> int:
        with self._rollout_id_lock:
            return self._rollout_id

    @staticmethod
    def _consume_task_result(task: asyncio.Task) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass

    def _set_failure(self, exc: BaseException, *, context: str | None = None) -> None:
        failure = exc
        if context is not None:
            failure = RuntimeError(f"{context}: {type(exc).__name__}: {exc}")
            failure.__cause__ = exc
        recorded = False
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure
                recorded = True
        if recorded:
            with self._phase_lock:
                self._phase = WorkerPhase.FAILED
            logger.error(
                "fully-async rollout worker failed",
                exc_info=(type(failure), failure, failure.__traceback__),
            )
        self._pause_requested.set()
        self._wake_requested.set()

    @classmethod
    def _validate_generated_group(cls, result: Any) -> None:
        if not isinstance(result, list) or not result:
            raise TypeError("generate_and_rm_group must return a non-empty list of samples")
        samples = list(cls._iter_samples(result))
        if not samples:
            raise TypeError("generate_and_rm_group returned an empty nested group")

    @classmethod
    def _contains_aborted_sample(cls, result: GeneratedGroup) -> bool:
        return any(sample.status == Sample.Status.ABORTED for sample in cls._iter_samples(result))

    @classmethod
    def _iter_samples(cls, value):
        for item in value:
            if isinstance(item, list):
                yield from cls._iter_samples(item)
            elif isinstance(item, Sample):
                yield item
            else:
                raise TypeError("generate_and_rm_group returned a non-Sample nested value")


async def _generate_rollout_async(
    args,
    rollout_id: int,
    data_buffer,
    *,
    worker_factory=None,
) -> list[GeneratedGroup]:
    if not args.rollout_global_dataset:
        raise ValueError("fully-async rollout requires a global dataset")
    worker = _get_global_worker(args, data_buffer, worker_factory=worker_factory)
    worker.resume(rollout_id)

    target = args.rollout_batch_size
    collected: list[_CompletedGroup] = []
    started = time.time()
    last_log = started

    try:
        while len(collected) < target:
            worker.ensure_healthy()
            completed = worker.get_completed_groups(target - len(collected))
            collected.extend(completed)
            if not completed:
                await asyncio.sleep(WORKER_POLL_INTERVAL_SECONDS)

            now = time.time()
            if now - last_log > 30.0:
                logger.info(
                    "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                    rollout_id,
                    len(collected),
                    target,
                    worker.queue_size(),
                    now - started,
                )
                last_log = now
        worker.ensure_healthy()
    except BaseException:
        worker.request_restore(collected)
        try:
            worker.quiesce()
        except BaseException:
            logger.exception("failed to quiesce fully-async worker after rollout error")
        raise

    out = [item.result for item in sorted(collected, key=_completed_group_key)]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def _completed_group_key(item: _CompletedGroup) -> int:
    for sample in AsyncRolloutWorker._iter_samples(item.result):
        if sample.index is not None:
            return int(sample.index)
    return item.gid


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False, *, worker_factory=None):
    """Slime rollout entrypoint with bounded cross-batch prefetch."""

    model_routers = getattr(args, "sglang_model_routers", None)
    if isinstance(model_routers, dict) and len(model_routers) > 1:
        raise ValueError(
            "fully-async rollout does not support multiple model routers"
        )
    if evaluation:
        raise ValueError("fully-async rollout does not support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer, worker_factory=worker_factory))
