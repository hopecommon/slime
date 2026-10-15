import asyncio
import logging
from typing import Any

from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

ABORT_RETRY_INTERVAL_SECONDS = 3


def num_requests_from_load(load: Any, *, strict: bool = False) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item, strict=strict) for item in load)

    if not isinstance(load, dict):
        if strict:
            raise ValueError(f"Invalid SGLang load response: {load!r}")
        return 0

    if "loads" in load:
        return num_requests_from_load(load["loads"], strict=strict)

    for key in ("num_reqs", "num_total_reqs", "total_reqs"):
        value = load.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value

    running = load.get("num_running_reqs", load.get("total_running_reqs"))
    waiting = load.get("num_waiting_reqs", load.get("total_waiting_reqs"))
    if strict and not (
        isinstance(running, int) and not isinstance(running, bool) and running >= 0
        and isinstance(waiting, int) and not isinstance(waiting, bool) and waiting >= 0
    ):
        raise ValueError(f"Invalid SGLang load response: {load!r}")
    return (running if isinstance(running, int) else 0) + (waiting if isinstance(waiting, int) else 0)


async def _abort_server_once(url: str, *, raise_on_engine_error: bool) -> None:
    try:
        await post(f"{url}/abort_request", {"abort_all": True})
    except Exception as e:
        if raise_on_engine_error:
            raise RuntimeError(f"Failed to abort SGLang server at {url}") from e
        logger.warning(f"Failed to abort SGLang server at {url}: {e}")


async def _get_server_num_requests(url: str, *, strict: bool) -> int:
    return num_requests_from_load(await get(f"{url}/v1/loads?include=core"), strict=strict)


async def abort_server_until_idle(
    url: str,
    retry_interval: int = ABORT_RETRY_INTERVAL_SECONDS,
    *,
    raise_on_engine_error: bool = False,
) -> None:
    attempt = 1
    while True:
        logger.info(f"Abort request for SGLang server {url}")
        await _abort_server_once(url, raise_on_engine_error=raise_on_engine_error)

        try:
            num_requests = await _get_server_num_requests(url, strict=raise_on_engine_error)
        except Exception as e:
            if raise_on_engine_error:
                raise RuntimeError(f"Failed to get SGLang server load from {url}") from e
            logger.warning(f"Failed to get SGLang server load from {url}: {e}")
            return

        if num_requests <= 0:
            return

        logger.info(
            f"SGLang server {url} still has {num_requests} requests after abort attempt {attempt}; "
            f"retrying in {retry_interval} seconds."
        )
        await asyncio.sleep(retry_interval)
        attempt += 1


async def abort_servers_until_idle(urls: list[str], *, raise_on_engine_error: bool = False) -> None:
    results = await asyncio.gather(
        *(abort_server_until_idle(url, raise_on_engine_error=raise_on_engine_error) for url in urls),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError("failed to abort rollout engines: " + "; ".join(str(error) for error in errors)) from errors[0]
