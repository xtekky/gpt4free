from __future__ import annotations

import random
import string

from ..typing import Messages, Cookies, AsyncIterator, Iterator
from .. import debug

def format_prompt(messages: Messages, add_special_tokens: bool = False, do_continue: bool = False) -> str:
    """
    Format a series of messages into a single string, optionally adding special tokens.

    Args:
        messages (Messages): A list of message dictionaries, each containing 'role' and 'content'.
        add_special_tokens (bool): Whether to add special formatting tokens.

    Returns:
        str: A formatted string containing all messages.
    """
    if not add_special_tokens and len(messages) <= 1:
        return messages[0]["content"]
    formatted = "\n".join([
        f'{message["role"].capitalize()}: {message["content"]}'
        for message in messages
    ])
    if do_continue:
        return formatted
    return f"{formatted}\nAssistant:"

def get_last_user_message(messages: Messages) -> str:
    user_messages = []
    last_message = None if len(messages) == 0 else messages[-1]
    while last_message is not None and messages:
        last_message = messages.pop()
        if last_message["role"] == "user":
            if isinstance(last_message["content"], str):
                user_messages.append(last_message["content"].strip())
        else:
            return "\n".join(user_messages[::-1])
    return "\n".join(user_messages[::-1])

def format_image_prompt(messages, prompt: str = None) -> str:
    if prompt is None:
        return get_last_user_message(messages)
    return prompt

def format_prompt_max_length(messages: Messages, max_lenght: int) -> str:
    prompt = format_prompt(messages)
    start = len(prompt)
    if start > max_lenght:
        if len(messages) > 6:
            prompt = format_prompt(messages[:3] + messages[-3:])
        if len(prompt) > max_lenght:
            if len(messages) > 2:
                prompt = format_prompt([m for m in messages if m["role"] == "system"] + messages[-1:])
            if len(prompt) > max_lenght:
                prompt = messages[-1]["content"]
        debug.log(f"Messages trimmed from: {start} to: {len(prompt)}")
    return prompt

def get_random_string(length: int = 10) -> str:
    """
    Generate a random string of specified length, containing lowercase letters and digits.

    Args:
        length (int, optional): Length of the random string to generate. Defaults to 10.

    Returns:
        str: A random string of the specified length.
    """
    return ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(length)
    )

def get_random_hex(length: int = 32) -> str:
    """
    Generate a random hexadecimal string with n length.

    Returns:
        str: A random hexadecimal string of n characters.
    """
    return ''.join(
        random.choice("abcdef" + string.digits)
        for _ in range(length)
    )

def filter_none(**kwargs) -> dict:
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }

async def async_concat_chunks(chunks: AsyncIterator) -> str:
    return concat_chunks([chunk async for chunk in chunks])

def concat_chunks(chunks: Iterator) -> str:
    return "".join([
        str(chunk) for chunk in chunks
        if chunk and not isinstance(chunk, Exception)
    ])

def format_cookies(cookies: Cookies) -> str:
    return "; ".join([f"{k}={v}" for k, v in cookies.items()])