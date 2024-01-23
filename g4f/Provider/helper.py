from __future__ import annotations

import asyncio
import os
import random
import secrets
import string
from asyncio import AbstractEventLoop, BaseEventLoop
from aiohttp import BaseConnector
from platformdirs import user_config_dir
from browser_cookie3 import (
    chrome, chromium, opera, opera_gx,
    brave, edge, vivaldi, firefox,
    _LinuxPasswordManager, BrowserCookieError
)
from ..typing import Dict, Messages, Optional
from ..errors import AiohttpSocksError
from .. import debug

# Global variable to store cookies
_cookies: Dict[str, Dict[str, str]] = {}

def get_event_loop() -> AbstractEventLoop:
    """
    Get the current asyncio event loop. If the loop is closed or not set, create a new event loop.
    If a loop is running, handle nested event loops. Patch the loop if 'nest_asyncio' is installed.

    Returns:
        AbstractEventLoop: The current or new event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if isinstance(loop, BaseEventLoop):
            loop._check_closed()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        asyncio.get_running_loop()
        if not hasattr(loop.__class__, "_nest_patched"):
            import nest_asyncio
            nest_asyncio.apply(loop)
    except RuntimeError:
        pass
    except ImportError:
        raise RuntimeError(
            'Use "create_async" instead of "create" function in a running event loop. Or install "nest_asyncio" package.'
        )
    return loop

if os.environ.get('DBUS_SESSION_BUS_ADDRESS') == "/dev/null":
    _LinuxPasswordManager.get_password = lambda a, b: b"secret"

def get_cookies(domain_name: str = '') -> Dict[str, str]:
    """
    Load cookies for a given domain from all supported browsers and cache the results.

    Args:
        domain_name (str): The domain for which to load cookies.

    Returns:
        Dict[str, str]: A dictionary of cookie names and values.
    """
    if domain_name in _cookies:
        return _cookies[domain_name]
    
    cookies = _load_cookies_from_browsers(domain_name)
    _cookies[domain_name] = cookies
    return cookies

def _load_cookies_from_browsers(domain_name: str) -> Dict[str, str]:
    """
    Helper function to load cookies from various browsers.

    Args:
        domain_name (str): The domain for which to load cookies.

    Returns:
        Dict[str, str]: A dictionary of cookie names and values.
    """
    cookies = {}
    for cookie_fn in [_g4f, chrome, chromium, opera, opera_gx, brave, edge, vivaldi, firefox]:
        try:
            cookie_jar = cookie_fn(domain_name=domain_name)
            if len(cookie_jar) and debug.logging:
                print(f"Read cookies from {cookie_fn.__name__} for {domain_name}")
            for cookie in cookie_jar:
                if cookie.name not in cookies:
                    cookies[cookie.name] = cookie.value
        except BrowserCookieError:
            pass
        except Exception as e:
            if debug.logging:
                print(f"Error reading cookies from {cookie_fn.__name__} for {domain_name}: {e}")
    return cookies

def _g4f(domain_name: str) -> list:
    """
    Load cookies from the 'g4f' browser (if exists).

    Args:
        domain_name (str): The domain for which to load cookies.

    Returns:
        list: List of cookies.
    """
    user_data_dir = user_config_dir("g4f")
    cookie_file = os.path.join(user_data_dir, "Default", "Cookies")
    return [] if not os.path.exists(cookie_file) else chrome(cookie_file, domain_name)

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
            raise AiohttpSocksError('Install "aiohttp_socks" package for proxy support')
    return connector