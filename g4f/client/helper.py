from __future__ import annotations

import re
import queue
import threading
import logging
import asyncio

from typing import AsyncIterator, Iterator, AsyncGenerator

def filter_json(text: str) -> str:
    """
    Parses JSON code block from a string.

    Args:
        text (str): A string containing a JSON code block.

    Returns:
        dict: A dictionary parsed from the JSON code block.
    """
    match = re.search(r"```(json|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    return text

def find_stop(stop, content: str, chunk: str = None):
    first = -1
    word = None
    if stop is not None:
        for word in list(stop):
            first = content.find(word)
            if first != -1:
                content = content[:first]
                break
        if chunk is not None and first != -1:
            first = chunk.find(word)
            if first != -1:
                chunk = chunk[:first]
            else:
                first = 0
    return first, content, chunk

def filter_none(**kwargs) -> dict:
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }

async def safe_aclose(generator: AsyncGenerator) -> None:
    try:
        await generator.aclose()
    except Exception as e:
        logging.warning(f"Error while closing generator: {e}")

# Helper function to convert an async generator to a synchronous iterator
def to_sync_iter(async_gen: AsyncIterator) -> Iterator:
    q = queue.Queue()
    loop = asyncio.new_event_loop()
    done = object()

    def _run():
        asyncio.set_event_loop(loop)

        async def iterate():
            try:
                async for item in async_gen:
                    q.put(item)
            finally:
                q.put(done)

        loop.run_until_complete(iterate())
        loop.close()

    threading.Thread(target=_run).start()

    while True:
        item = q.get()
        if item is done:
            break
        yield item

# Helper function to convert a synchronous iterator to an async iterator
async def to_async_iterator(iterator: Iterator) -> AsyncIterator:
    for item in iterator:
        yield item