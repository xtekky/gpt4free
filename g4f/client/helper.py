from __future__ import annotations

import re
from typing import Iterable, AsyncIterator

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

async def cast_iter_async(iter: Iterable) -> AsyncIterator:
    for chunk in iter:
        yield chunk