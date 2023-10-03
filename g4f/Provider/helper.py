from __future__ import annotations

import asyncio, sys
from asyncio import AbstractEventLoop

import browser_cookie3

# Change event loop policy on windows
if sys.platform == 'win32':
    if isinstance(
        asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
    ):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Local Cookie Storage
_cookies: dict[str, dict[str, str]] = {}

# If event loop is already running, handle nested event loops
# If "nest_asyncio" is installed, patch the event loop.
def get_event_loop() -> AbstractEventLoop:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            return asyncio.get_event_loop()
    try:
        event_loop = asyncio.get_event_loop()
        if not hasattr(event_loop.__class__, "_nest_patched"):
            import nest_asyncio
            nest_asyncio.apply(event_loop)
        return event_loop
    except ImportError:
        raise RuntimeError(
            'Use "create_async" instead of "create" function in a running event loop. Or install the "nest_asyncio" package.')

# Load cookies for a domain from all supported browser.
# Cache the results in the "_cookies" variable
def get_cookies(cookie_domain: str) -> dict:
    if cookie_domain not in _cookies:
        _cookies[cookie_domain] = {}
        try:
            for cookie in browser_cookie3.load(cookie_domain):
                _cookies[cookie_domain][cookie.name] = cookie.value
        except:
            pass
    return _cookies[cookie_domain]


def format_prompt(messages: list[dict[str, str]], add_special_tokens=False):
    if add_special_tokens or len(messages) > 1:
        formatted = "\n".join(
            ["%s: %s" % ((message["role"]).capitalize(), message["content"]) for message in messages]
        )
        return f"{formatted}\nAssistant:"
    else:
        return messages[0]["content"]
    

def get_browser(user_data_dir: str = None):
    try:
        from undetected_chromedriver import Chrome
    except ImportError:
        return None

    def get_user_data_dir():
        dirs = [
            '~/.config/google-chrome/Default',
            '~/.var/app/com.google.Chrome/config/google-chrome/Default',
            '%LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default',
            '~/Library/Application Support/Google/Chrome/Default',
        ]
        from os import path
        for dir in dirs:
            dir = path.expandvars(dir)
            if path.exists(dir):
                return dir
    if not user_data_dir:
        user_data_dir = get_user_data_dir()

    return Chrome(user_data_dir=user_data_dir)