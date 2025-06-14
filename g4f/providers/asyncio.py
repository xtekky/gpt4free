from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, runners
from typing import Optional, Callable, AsyncIterator, Iterator

from ..errors import NestAsyncioError

try:
    import nest_asyncio
    has_nest_asyncio = True
except ImportError:
    has_nest_asyncio = False
try:
    import uvloop
    has_uvloop = True
except ImportError:
    has_uvloop = False

def get_running_loop(check_nested: bool) -> Optional[AbstractEventLoop]:
    try:
        loop = asyncio.get_running_loop()
        # Do not patch uvloop loop because its incompatible.
        if has_uvloop:
            if isinstance(loop, uvloop.Loop):
               return loop
        if not hasattr(loop.__class__, "_nest_patched"):
            if has_nest_asyncio:
                nest_asyncio.apply(loop)
            elif check_nested:
                raise NestAsyncioError('Install "nest_asyncio" package | pip install -U nest_asyncio')
        return loop
    except RuntimeError:
        pass

# Fix for RuntimeError: async generator ignored GeneratorExit
async def await_callback(callback: Callable, timeout: Optional[int] = None) -> any:
    return await asyncio.wait_for(callback(), timeout) if timeout is not None else await callback()

async def async_generator_to_list(generator: AsyncIterator) -> list:
    return [item async for item in generator]

def to_sync_generator(generator: AsyncIterator, stream: bool = True, timeout: int = None) -> Iterator:
    loop = get_running_loop(check_nested=False)
    if not stream:
        yield from asyncio.run(async_generator_to_list(generator))
        return
    new_loop = False
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        new_loop = True
    gen = generator.__aiter__()
    try:
        while True:
            yield loop.run_until_complete(await_callback(gen.__anext__, timeout))
    except StopAsyncIteration:
        pass
    finally:
        if new_loop:
            try:
                runners._cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

# Helper function to convert a synchronous iterator to an async iterator
async def to_async_iterator(iterator) -> AsyncIterator:
    if hasattr(iterator, '__aiter__'):
        async for item in iterator:
            yield item
    elif asyncio.iscoroutine(iterator):
        yield await iterator
    else:
        for item in iterator:
            yield item