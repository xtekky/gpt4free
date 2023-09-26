from __future__ import annotations

import asyncio
import sys
from asyncio import AbstractEventLoop

import browser_cookie3

_cookies: dict[str, dict[str, str]] = {}

# Use own event_loop_policy with a selector event loop on windows.
if sys.platform == 'win32':
    _event_loop_policy = asyncio.WindowsSelectorEventLoopPolicy()
else:
    _event_loop_policy = asyncio.get_event_loop_policy()
    
# If event loop is already running, handle nested event loops
# If "nest_asyncio" is installed, patch the event loop.
def get_event_loop() -> AbstractEventLoop:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _event_loop_policy.get_event_loop()
    try:
        event_loop = _event_loop_policy.get_event_loop()
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