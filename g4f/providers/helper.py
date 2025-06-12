from __future__ import annotations

import random
import string
from pathlib import Path

from ..typing import Messages, Cookies, AsyncIterator, Iterator
from ..tools.files import get_bucket_dir, read_bucket
from .. import debug

def to_string(value) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        if "text" in value:
            return value["text"]
        elif "name" in value:
            return ""
        elif "bucket_id" in value:
            bucket_dir = Path(get_bucket_dir(value.get("bucket_id")))
            return "".join(read_bucket(bucket_dir))
        return ""
    elif isinstance(value, list):
        return "".join([to_string(v) for v in value if v.get("type", "text") == "text"])
    elif value is None:
        return ""
    return str(value)

def render_messages(messages: Messages) -> Iterator:
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            yield {
                **message,
                "content": to_string(message["content"]),
            }
        else:
            yield message

def format_prompt(messages: Messages, add_special_tokens: bool = False, do_continue: bool = False, include_system: bool = True) -> str:
    """
    Format a series of messages into a single string, optionally adding special tokens.

    Args:
        messages (Messages): A list of message dictionaries, each containing 'role' and 'content'.
        add_special_tokens (bool): Whether to add special formatting tokens.

    Returns:
        str: A formatted string containing all messages.
    """
    if not add_special_tokens and len(messages) <= 1:
        return to_string(messages[0]["content"])
    messages = [
        (message["role"], to_string(message["content"]))
        for message in messages
        if include_system or message.get("role") not in ("developer", "system")
    ]
    formatted = "\n".join([
        f'{role.capitalize()}: {content}'
        for role, content in messages
        if content.strip()
    ])
    if do_continue:
        return formatted
    return f"{formatted}\nAssistant:"

def get_system_prompt(messages: Messages) -> str:
    return "\n".join([m["content"] for m in messages if m["role"] in ("developer", "system")])

def get_last_user_message(messages: Messages) -> str:
    user_messages = []
    last_message = None if len(messages) == 0 else messages[-1]
    messages = messages.copy()
    while last_message is not None and messages:
        last_message = messages.pop()
        if last_message["role"] == "user":
            content = to_string(last_message.get("content")).strip()
            if content:
                user_messages.append(content)
        else:
            return "\n".join(user_messages[::-1])
    return "\n".join(user_messages[::-1])

def get_last_message(messages: Messages, prompt: str = None) -> str:
    if prompt is None:
        for message in messages[::-1]:
            content = to_string(message.get("content")).strip()
            if content:
                prompt = content
    return prompt

def format_media_prompt(messages, prompt: str = None) -> str:
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