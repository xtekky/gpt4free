from __future__ import annotations

import asyncio
import webbrowser
import random
import string
import secrets
import os
from os import path
from asyncio import AbstractEventLoop, BaseEventLoop
from platformdirs import user_config_dir
from browser_cookie3 import (
    chrome,
    chromium,
    opera,
    opera_gx,
    brave,
    edge,
    vivaldi,
    firefox,
    _LinuxPasswordManager
)

from ..typing import Dict, Messages
from .. import debug

# Local Cookie Storage
_cookies: Dict[str, Dict[str, str]] = {}

# If loop closed or not set, create new event loop.
# If event loop is already running, handle nested event loops.
# If "nest_asyncio" is installed, patch the event loop.
def get_event_loop() -> AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if isinstance(loop, BaseEventLoop):
            loop._check_closed()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        # Is running event loop
        asyncio.get_running_loop()
        if not hasattr(loop.__class__, "_nest_patched"):
            import nest_asyncio
            nest_asyncio.apply(loop)
    except RuntimeError:
        # No running event loop
        pass
    except ImportError:
        raise RuntimeError(
            'Use "create_async" instead of "create" function in a running event loop. Or install the "nest_asyncio" package.'
        )
    return loop

def init_cookies():
    urls = [
        'https://chat-gpt.org',
        'https://www.aitianhu.com',
        'https://chatgptfree.ai',
        'https://gptchatly.com',
        'https://bard.google.com',
        'https://huggingface.co/chat',
        'https://open-assistant.io/chat'
    ]

    browsers = ['google-chrome', 'chrome', 'firefox', 'safari']

    def open_urls_in_browser(browser):
        b = webbrowser.get(browser)
        for url in urls:
            b.open(url, new=0, autoraise=True)

    for browser in browsers:
        try:
            open_urls_in_browser(browser)
            break 
        except webbrowser.Error:
            continue

# Check for broken dbus address in docker image   
if os.environ.get('DBUS_SESSION_BUS_ADDRESS') == "/dev/null":
    _LinuxPasswordManager.get_password = lambda a, b: b"secret"
    
# Load cookies for a domain from all supported browsers.
# Cache the results in the "_cookies" variable.
def get_cookies(domain_name=''):
    if domain_name in _cookies:
        return _cookies[domain_name]
    def g4f(domain_name):
        user_data_dir = user_config_dir("g4f")
        cookie_file = path.join(user_data_dir, "Default", "Cookies")
        return [] if not path.exists(cookie_file) else chrome(cookie_file, domain_name)

    cookies = {}
    for cookie_fn in [g4f, chrome, chromium, opera, opera_gx, brave, edge, vivaldi, firefox]:
        try:
            cookie_jar = cookie_fn(domain_name=domain_name)
            if len(cookie_jar) and debug.logging:
                print(f"Read cookies from {cookie_fn.__name__} for {domain_name}")
            for cookie in cookie_jar:
                if cookie.name not in cookies:
                    cookies[cookie.name] = cookie.value
        except:
            pass
    _cookies[domain_name] = cookies
    return _cookies[domain_name]


def format_prompt(messages: Messages, add_special_tokens=False) -> str:
    if not add_special_tokens and len(messages) <= 1:
        return messages[0]["content"]
    formatted = "\n".join([
        f'{message["role"].capitalize()}: {message["content"]}'
        for message in messages
    ])
    return f"{formatted}\nAssistant:"


def get_random_string(length: int = 10) -> str:
    return ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(length)
    )

def get_random_hex() -> str:
    return secrets.token_hex(16).zfill(32)