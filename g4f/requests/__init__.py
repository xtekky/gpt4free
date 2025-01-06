from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Iterator
from http.cookies import Morsel
try:
    from curl_cffi.requests import Session, Response
    from .curl_cffi import StreamResponse, StreamSession, FormData
    has_curl_cffi = True
except ImportError:
    from typing import Type as Response
    from .aiohttp import StreamResponse, StreamSession, FormData
    has_curl_cffi = False
try:
    import webview
    import asyncio
    has_webview = True
except ImportError:
    has_webview = False
try:
    import nodriver
    from nodriver.cdp.network import CookieParam
    from nodriver.core.config import find_chrome_executable
    from nodriver import Browser
    has_nodriver = True
except ImportError:
    has_nodriver = False
try:
    from platformdirs import user_config_dir
    has_platformdirs = True
except ImportError:
    has_platformdirs = False

from .. import debug
from .raise_for_status import raise_for_status
from ..errors import MissingRequirementsError
from ..typing import Cookies
from .defaults import DEFAULT_HEADERS, WEBVIEW_HAEDERS

if not has_curl_cffi:
    class Session:
        def __init__(self, **kwargs):
            raise MissingRequirementsError('Install "curl_cffi" package | pip install -U curl_cffi')

async def get_args_from_webview(url: str) -> dict:
    if not has_webview:
        raise MissingRequirementsError('Install "webview" package')
    window = webview.create_window("", url, hidden=True)
    await asyncio.sleep(2)
    body = None
    while body is None:
        try:
            await asyncio.sleep(1)
            body = window.dom.get_element("body:not(.no-js)")
        except:
            ...
    headers = {
        **WEBVIEW_HAEDERS,
        "User-Agent": window.evaluate_js("this.navigator.userAgent"),
        "Accept-Language": window.evaluate_js("this.navigator.language"),
        "Referer": window.real_url
    }
    cookies = [list(*cookie.items()) for cookie in window.get_cookies()]
    cookies = {name: cookie.value for name, cookie in cookies}
    window.destroy()
    return {"headers": headers, "cookies": cookies}

def get_cookie_params_from_dict(cookies: Cookies, url: str = None, domain: str = None) -> list[CookieParam]:
    [CookieParam.from_json({
        "name": key,
        "value": value,
        "url": url,
        "domain": domain
    }) for key, value in cookies.items()]

async def get_args_from_nodriver(
    url: str,
    proxy: str = None,
    timeout: int = 120,
    cookies: Cookies = None
) -> dict:
    if not has_nodriver:
        raise MissingRequirementsError('Install "nodriver" package | pip install -U nodriver')
    if debug.logging:
        print(f"Open nodriver with url: {url}")
    browser = await nodriver.start(
        browser_args=None if proxy is None else [f"--proxy-server={proxy}"],
    )
    domain = urlparse(url).netloc
    if cookies is None:
        cookies = {}
    else:
        await browser.cookies.set_all(get_cookie_params_from_dict(cookies, url=url, domain=domain))
    page = await browser.get(url)
    for c in await page.send(nodriver.cdp.network.get_cookies([url])):
        cookies[c.name] = c.value
    user_agent = await page.evaluate("window.navigator.userAgent")
    await page.wait_for("body:not(.no-js)", timeout=timeout)
    for c in await page.send(nodriver.cdp.network.get_cookies([url])):
        cookies[c.name] = c.value
    await page.close()
    browser.stop()
    return {
        "impersonate": "chrome",
        "cookies": cookies,
        "headers": {
            **DEFAULT_HEADERS,
            "user-agent": user_agent,
            "referer": url,
        },
        "proxy": proxy
    }

def merge_cookies(cookies: Iterator[Morsel], response: Response) -> Cookies:
    if cookies is None:
        cookies = {}
    for cookie in response.cookies.jar:
        cookies[cookie.name] = cookie.value

async def get_nodriver(proxy: str = None, user_data_dir = "nodriver", browser_executable_path=None, **kwargs)-> Browser:
    if not has_nodriver:
        raise MissingRequirementsError('Install "nodriver" package | pip install -U nodriver')
    user_data_dir = user_config_dir(f"g4f-{user_data_dir}") if has_platformdirs else None
    if browser_executable_path is None:
        try:
            browser_executable_path = find_chrome_executable()
        except FileNotFoundError:
            # Default to Edge if Chrome is not found
            browser_executable_path = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
            if not os.path.exists(browser_executable_path):
                browser_executable_path = None
    debug.log(f"Open nodriver with user_dir: {user_data_dir}")
    return await nodriver.start(
        user_data_dir=user_data_dir,
        browser_args=None if proxy is None else [f"--proxy-server={proxy}"],
        browser_executable_path=browser_executable_path,
        **kwargs
    )