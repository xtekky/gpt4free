from __future__ import annotations

import re
import logging

from typing import AsyncIterator, Iterator, AsyncGenerator, Optional

def filter_markdown(text: str, allowed_types=None, default=None) -> str:
    """
    Parses code block from a string.

    Args:
        text (str): A string containing a code block.

    Returns:
        dict: A dictionary parsed from the code block.
    """
    match = re.search(r"```(.+)\n(?P<code>[\S\s]+?)(\n```|$)", text)
    if match:
        if allowed_types is None or match.group(1) in allowed_types:
            return match.group("code")
    return default

def filter_json(text: str) -> str:
    """
    Parses JSON code block from a string.

    Args:
        text (str): A string containing a JSON code block.

    Returns:
        dict: A dictionary parsed from the JSON code block.
    """
    return filter_markdown(text, ["", "json"], text.strip("^\n "))

def find_stop(stop: Optional[list[str]], content: str, chunk: str = None):
    first = -1
    word = None
    if stop is not None:
        content = str(content)
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
        if generator and hasattr(generator, 'aclose'):
            await generator.aclose()
    except Exception as e:
        logging.warning(f"Error while closing generator: {e}")