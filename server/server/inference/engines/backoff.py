import logging
from asyncio import sleep
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_backoff(
    attempts: int,
    initial_ms: int,
    max_ms: int,
    block: Callable[[int], Awaitable[T]],
) -> T:
    ms = initial_ms
    attempt = 0

    while True:
        attempt += 1
        try:
            return await block(attempt)
        except Exception as ex:
            if attempt > attempts:
                raise
            logger.warning(f"Exception: %s. Retrying in %dms...", type(ex).__name__, ms)
            await sleep(ms / 1000)
            ms = min(ms * 2, max_ms)
