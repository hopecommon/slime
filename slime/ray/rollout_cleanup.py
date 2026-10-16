"""Explicit selected-rollout release evidence, independent of recipe imports."""

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import ray
from ray.exceptions import ActorDiedError, GetTimeoutError, RayActorError, RayTaskError


@dataclass(frozen=True)
class RolloutCleanupReport:
    """Separate lower release from the manager's destructive cleanup result."""

    selected_closed: bool
    core_state: str = "not_started"
    errors: tuple[tuple[str, str], ...] = ()
    receipts: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.selected_closed) is not bool or self.core_state not in {"not_started", "complete", "failed"}:
            raise ValueError("invalid selected cleanup report")
        if not self.selected_closed and self.core_state != "not_started":
            raise ValueError("core cleanup requires selected release")
        if type(self.errors) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(not isinstance(value, str) or not value.isidentifier() for value in item)
            for item in self.errors
        ):
            raise ValueError("cleanup diagnostics require safe phase and type names")
        if type(self.receipts) is not tuple or any(
            not isinstance(value, str) or not value or len(value) > 4096 for value in self.receipts
        ):
            raise ValueError("invalid cleanup receipt references")


class RolloutCleanupFailed(RuntimeError):
    """A terminal cleanup fault whose separate durable receipt remains available."""

    def __init__(self, receipt_path):
        self.receipt_path = str(receipt_path)
        super().__init__("selected rollout cleanup failed; inspect its retained receipt")


class RolloutOwnerLost(RolloutCleanupFailed):
    """Confirmed actor death leaves remote resources unresolved, never released."""


def _publish(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


class RetainedRolloutCleanup:
    """Driver-local manager ownership across bounded observations of one RPC."""

    def __init__(self, receipt_path, *, context=None, observer_seconds=30, retry_seconds=1):
        if any(
            isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
            for value in (observer_seconds, retry_seconds)
        ):
            raise ValueError("finite positive cleanup observation bounds required")
        self.receipt_path = Path(receipt_path).absolute()
        self.context = dict(context or {})
        if any(
            not isinstance(key, str) or not key.isidentifier() or not isinstance(value, str) or len(value) > 4096
            for key, value in self.context.items()
        ):
            raise ValueError("cleanup context requires nonsecret identity references")
        self.observer_seconds, self.retry_seconds = observer_seconds, retry_seconds
        self.manager = self._ref = self._terminal = self._fault = None
        self._published = False

    def adopt(self, manager):
        """Retain before any creator RPC, await or durable publication."""
        if self.manager is not None and self.manager is not manager:
            raise RuntimeError("a retained manager cannot be replaced")
        self.manager = manager

    def _finish(self):
        if not self._published:
            lost = isinstance(self._fault, RolloutOwnerLost)
            _publish(
                self.receipt_path,
                {
                    "version": 1,
                    "state": "owner_lost_reconciliation_required"
                    if lost
                    else ("core_cleanup_failed" if self._fault is not None else "closed"),
                    "selected_release": "unknown" if lost else "closed",
                    "context": self.context,
                    "report": asdict(self._terminal) if self._terminal is not None else None,
                },
            )
            self._published = True
        if self._fault is not None:
            raise self._fault
        return True

    def wait_closed(self, timeout):
        """Observe one RPC; only settled calls may rejoin the same manager."""
        if self.manager is None:
            return True
        if self._terminal is not None or self._fault is not None:
            return self._finish()
        try:
            if self._ref is None:
                self._ref = self.manager.dispose.remote(retain_unresolved=True)
            report = ray.get(self._ref, timeout=timeout)
        except RayTaskError:
            # ray.get creates dual-inheritance exceptions for user causes,
            # including child ActorDiedError. That is not this manager's death.
            self._ref = None
            return False
        except ActorDiedError:
            self._fault = RolloutOwnerLost(self.receipt_path)
            return self._finish()
        except GetTimeoutError:
            return False
        except RayActorError:
            # A stored terminal ActorUnavailableError never changes on the
            # original ref. Once ready, rejoin the same idempotent owner; an
            # ambiguous nonterminal observation keeps the outstanding ref.
            if self._ref is not None:
                ready, _ = ray.wait([self._ref], timeout=0, fetch_local=False)
                if ready:
                    self._ref = None
            return False
        self._ref = None
        if type(report) is not RolloutCleanupReport or not report.selected_closed:
            return False
        if report.core_state == "not_started":
            return False
        self._terminal = report
        if report.core_state == "failed":
            self._fault = RolloutCleanupFailed(self.receipt_path)
        return self._finish()

    def add_terminal_note(self, error):
        """Attach durable fault type and receipt evidence to an earlier primary."""
        if self._published and self._fault is not None:
            error.add_note(f"rollout_cleanup_terminal: {type(self._fault).__name__}; receipt: {self.receipt_path}")

    def finish_foreground(self):
        """Keep the creating stack alive until checked release or durable loss."""
        observation_error = None
        while True:
            try:
                if self.wait_closed(self.observer_seconds):
                    break
            except RolloutCleanupFailed:
                if observation_error is not None:
                    self.add_terminal_note(observation_error)
                    raise observation_error from None
                raise
            except BaseException as error:
                # Observer cancellation and failed publication never dispose of
                # the live handles. Rejoin before exposing the original error.
                if observation_error is None:
                    observation_error = error
            try:
                time.sleep(self.retry_seconds)
            except BaseException as error:
                if observation_error is None:
                    observation_error = error
        if observation_error is not None:
            raise observation_error
