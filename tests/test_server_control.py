"""Strict quiescence keeps engine failures and cancellation observable."""
import asyncio

import pytest

from slime.backends.sglang_utils import server_control


@pytest.mark.parametrize("failure", ["abort", "load", "malformed"])
def test_strict_abort_rejects_unconfirmed_idle(monkeypatch, failure):
    async def post(url, payload):
        if failure == "abort":
            raise OSError("abort unavailable")
    async def get(url):
        if failure == "load":
            raise OSError("load unavailable")
        return {} if failure == "malformed" else {"num_reqs": 0}
    monkeypatch.setattr(server_control, "post", post)
    monkeypatch.setattr(server_control, "get", get)
    with pytest.raises(RuntimeError):
        asyncio.run(server_control.abort_servers_until_idle(["http://engine"], raise_on_engine_error=True))


def test_strict_abort_joins_all_engines_before_reporting_failure(monkeypatch):
    completed = []
    async def post(url, payload):
        completed.append(url)
        if url.startswith("http://failed"):
            raise OSError("abort unavailable")
    async def get(url):
        return {"loads": [{"num_running_reqs": 0, "num_waiting_reqs": 0}]}
    monkeypatch.setattr(server_control, "post", post)
    monkeypatch.setattr(server_control, "get", get)
    with pytest.raises(RuntimeError, match="failed to abort rollout engines"):
        asyncio.run(server_control.abort_servers_until_idle(["http://failed", "http://idle"], raise_on_engine_error=True))
    assert completed == ["http://failed/abort_request", "http://idle/abort_request"]


def test_abort_retry_remains_cancellable_until_load_reaches_idle(monkeypatch):
    async def check():
        loads_observed = asyncio.Event()
        sleeping = asyncio.Event()
        async def post(url, payload):
            pass
        async def get(url):
            loads_observed.set()
            return {"num_reqs": 1}
        async def sleep(seconds):
            sleeping.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(server_control, "post", post)
        monkeypatch.setattr(server_control, "get", get)
        monkeypatch.setattr(server_control.asyncio, "sleep", sleep)
        task = asyncio.create_task(server_control.abort_servers_until_idle(["http://busy"], raise_on_engine_error=True))
        await loads_observed.wait()
        await sleeping.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(check())


def test_best_effort_abort_preserves_visible_degraded_return(monkeypatch, caplog):
    async def fail(*args):
        raise OSError("unavailable")
    monkeypatch.setattr(server_control, "post", fail)
    monkeypatch.setattr(server_control, "get", fail)
    asyncio.run(server_control.abort_servers_until_idle(["http://engine"]))
    assert "Failed to abort SGLang server" in caplog.text
    assert "Failed to get SGLang server load" in caplog.text


def test_strict_abort_retries_until_idle_is_observed(monkeypatch):
    loads = iter([{"num_reqs": 2}, {"num_reqs": 0}])
    events = []
    async def post(url, payload):
        events.append("abort")
    async def get(url):
        events.append("load")
        return next(loads)
    async def sleep(seconds):
        events.append("retry")
    monkeypatch.setattr(server_control, "post", post)
    monkeypatch.setattr(server_control, "get", get)
    monkeypatch.setattr(server_control.asyncio, "sleep", sleep)
    asyncio.run(server_control.abort_servers_until_idle(["http://engine"], raise_on_engine_error=True))
    assert events == ["abort", "load", "retry", "abort", "load"]
