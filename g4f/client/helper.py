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
        str: Parsed code block content, or default.
    """
    if not isinstance(text, str):
        return default
    start = text.find("```")
    if start == -1:
        return default
    first_nl = text.find("\n", start + 3)
    if first_nl == -1:
        return default
    tag = text[start + 3 : first_nl].strip("\r\n\t ")
    match_tag = tag if tag else None
    end = text.find("\n```", first_nl)
    if end != -1:
        code = text[first_nl + 1 : end]
        if code.endswith("\r"):
            code = code[:-1]
    else:
        code = text[first_nl + 1 :]
    if (
        allowed_types is None
        or match_tag in allowed_types
        or (not match_tag and ("" in allowed_types or None in allowed_types))
    ):
        return code
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
    return {key: value for key, value in kwargs.items() if value is not None}


async def safe_aclose(generator: AsyncGenerator) -> None:
    try:
        if generator and hasattr(generator, "aclose"):
            await generator.aclose()
    except Exception as e:
        logging.warning(f"Error while closing generator: {e}")
