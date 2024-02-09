from __future__ import annotations

import random
import secrets
import string
from aiohttp import BaseConnector

from ..typing import Messages, Optional
from ..errors import MissingRequirementsError
from ..cookies import get_cookies

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

def get_random_hex() -> str:
    """
    Generate a random hexadecimal string of a fixed length.

    Returns:
        str: A random hexadecimal string of 32 characters (16 bytes).
    """
    return secrets.token_hex(16).zfill(32)

def get_connector(connector: BaseConnector = None, proxy: str = None) -> Optional[BaseConnector]:
    if proxy and not connector:
        try:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(proxy)
        except ImportError:
            raise MissingRequirementsError('Install "aiohttp_socks" package for proxy support')
    return connector