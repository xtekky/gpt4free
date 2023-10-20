from __future__ import annotations

import sys
import asyncio
import webbrowser
import http.cookiejar

from os              import path
from asyncio         import AbstractEventLoop
from ..typing        import Dict, Messages
from browser_cookie3 import chrome, chromium, opera, opera_gx, brave, edge, vivaldi, firefox, BrowserCookieError


# Change event loop policy on windows
if sys.platform == 'win32':
    if isinstance(
        asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
    ):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Local Cookie Storage
_cookies: Dict[str, Dict[str, str]] = {}

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
            'Use "create_async" instead of "create" function in a running event loop. Or install the "nest_asyncio" package.'
        )

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

# Load cookies for a domain from all supported browsers.
# Cache the results in the "_cookies" variable.
def get_cookies(domain_name=''):
    cj = http.cookiejar.CookieJar()
    for cookie_fn in [chrome, chromium, opera, opera_gx, brave, edge, vivaldi, firefox]:
        try:
            for cookie in cookie_fn(domain_name=domain_name):
                cj.set_cookie(cookie)
        except BrowserCookieError:
            pass
    
    _cookies[domain_name] = {cookie.name: cookie.value for cookie in cj}
    
    return _cookies[domain_name]


def format_prompt(messages: Messages, add_special_tokens=False) -> str:
    if add_special_tokens or len(messages) > 1:
        formatted = "\n".join(
            [
                "%s: %s" % ((message["role"]).capitalize(), message["content"])
                for message in messages
            ]
        )
        return f"{formatted}\nAssistant:"
    else:
        return messages[0]["content"]


def get_browser(user_data_dir: str = None):
    from undetected_chromedriver import Chrome
    from platformdirs import user_config_dir

    if not user_data_dir:
        user_data_dir = user_config_dir("g4f")
        user_data_dir = path.join(user_data_dir, "Default")

    return Chrome(user_data_dir=user_data_dir)