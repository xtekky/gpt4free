from __future__ import annotations

import random
import string

from ..typing import Messages, Cookies
from .. import debug

def format_prompt(messages: Messages, add_special_tokens=False) -> str:
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
    return f"{formatted}\nAssistant:"

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

def format_cookies(cookies: Cookies) -> str:
    return "; ".join([f"{k}={v}" for k, v in cookies.items()])