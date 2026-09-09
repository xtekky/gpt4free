from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from http.cookies import Morsel
from pathlib import Path
from typing import Iterator, AsyncIterator, Optional
from urllib.parse import urlparse

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

    has_webview = True
except ImportError:
    has_webview = False
try:
    from platformdirs import user_config_dir

    has_platformdirs = True
except ImportError:
    has_platformdirs = False
try:
    from .cdp import CDPSession

    has_cdp = True
except ImportError:
    has_cdp = False

from .cdp_browser import (
    CDPBrowser,
    CDPTab,
    CDPElement,
    CookieParam,
    _CdpShim as _cdp,
    get_cookie_params_from_dict as _get_cookie_params_from_dict_cdp,
)

Browser = CDPBrowser

from .. import debug
from .raise_for_status import raise_for_status
from ..errors import MissingRequirementsError
from ..typing import Cookies
from ..cookies import BrowserConfig, get_cookies_dir
from .defaults import DEFAULT_HEADERS, WEBVIEW_HAEDERS
from .aiohttp import get_shared_connector, close_shared_connectors

if not has_curl_cffi:

    class Session:
        def __init__(self, **kwargs):
            raise MissingRequirementsError(
                'Install "curl_cffi" package | pip install -U curl_cffi'
            )


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
        except Exception:
            ...
    headers = {
        **WEBVIEW_HAEDERS,
        "User-Agent": window.evaluate_js("this.navigator.userAgent"),
        "Accept-Language": window.evaluate_js("this.navigator.language"),
        "Referer": window.real_url,
    }
    cookies = [list(*cookie.items()) for cookie in window.get_cookies()]
    cookies = {name: cookie.value for name, cookie in cookies}
    window.destroy()
    return {"headers": headers, "cookies": cookies}


def get_cookie_params_from_dict(
    cookies: Cookies, url: str = None, domain: str = None
) -> list[CookieParam]:
    return [
        CookieParam.from_json(
            {"name": key, "value": value, "url": url, "domain": domain}
        )
        for key, value in cookies.items()
    ]


async def clear_cookies_for_url(
    browser, url: str, ignore_cookies: list[str] = None
):
    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"Bad url: {url}")

    if ignore_cookies is None:
        ignore_cookies = []
    tab = browser.main_tab  # any open tab is fine
    if tab is None:
        tab = await browser.get("about:blank")
    cookies = await browser.cookies.get_all()
    for c in cookies:
        dom = (c.get("domain", "") or "").lstrip(".")
        if dom and (host == dom or host.endswith("." + dom)):
            if c.get("name") in ignore_cookies:
                continue
            await tab.send(
                _cdp.network.delete_cookies(
                    name=c.get("name"),
                    domain=dom,
                    path=c.get("path"),
                )
            )


async def get_args_from_nodriver(
    url: str,
    proxy: str = None,
    timeout: int = 120,
    wait_for: str = None,
    callback: callable = None,
    cookies: Cookies = None,
    browser: Browser = None,
    user_data_dir: str = "nodriver",
    browser_args: list = None,
    clear_cookies_except: list[str] = None,
) -> dict:
    if clear_cookies_except is None:
        clear_cookies_except = []
    if browser is None:
        browser, stop_browser = await get_nodriver(
            proxy=proxy,
            timeout=timeout,
            user_data_dir=user_data_dir,
            browser_args=browser_args,
        )
    else:

        async def stop_browser():
            pass

    try:
        if clear_cookies_except:
            debug.log(f"Clear Cookies for url: {url}")
            await clear_cookies_for_url(browser, url)

        debug.log(f"Open CDP browser with url: {url}")
        if cookies is None:
            cookies = {}
        else:
            domain = urlparse(url).netloc
            await browser.cookies.set_all(
                get_cookie_params_from_dict(cookies, url=url, domain=domain)
            )
        page = await browser.get(url)
        user_agent = await page.evaluate(
            "window.navigator.userAgent", return_by_value=True
        )
        while not await page.evaluate("!!document.querySelector('body:not(.no-js)')"):
            await asyncio.sleep(1)
        if wait_for is not None:
            await page.wait_for(wait_for, timeout=timeout)
        if callback is not None:
            await callback(page)
        result = await asyncio.wait_for(
            page.send(_cdp.network.get_cookies([url])), timeout=timeout
        )
        for c in result.get("cookies", []):
            cookies[c["name"]] = c["value"]
        await stop_browser()
        return {
            "impersonate": "chrome",
            "cookies": cookies,
            "headers": {
                **DEFAULT_HEADERS,
                "user-agent": user_agent,
                "referer": f"{url.rstrip('/')}/",
            },
            "proxy": proxy,
        }
    except Exception:
        await stop_browser()
        raise

async def get_args_from_cdp(
    url: str,
    proxy: str = None,
    timeout: int = 120,
    user_data_dir: str = "cdp",
    headless: Optional[bool] = None,
) -> dict:
    """Use the lightweight CDP client to get auth cookies and user-agent."""
    if not has_cdp:
        raise MissingRequirementsError("Missing CDP requirements")

    debug.log(f"Open CDP session with url: {url}")
    session = CDPSession(user_data_dir=user_data_dir, headless=headless)
    await session.start()

    try:
        await session.navigate(url)

        # Wait for Cloudflare/protection to pass
        for _ in range(timeout):
            title = await session.evaluate_js("document.title") or ""
            content = await session.evaluate_js("document.body.innerText") or ""

            if (
                "Just a moment" not in title
                and "Attention Required" not in title
                and "cf-browser-verification" not in content
            ):
                break
            await asyncio.sleep(1)

        cookies = await session.get_cookies()
        user_agent = await session.get_user_agent()

        return {
            "impersonate": "chrome",
            "cookies": cookies,
            "headers": {
                **DEFAULT_HEADERS,
                "user-agent": user_agent,
                "referer": f"{url.rstrip('/')}/",
            },
            "proxy": proxy,
        }
    finally:
        await session.close()


def merge_cookies(cookies: Iterator[Morsel], response: Response) -> Cookies:
    if cookies is None:
        cookies = {}
    if hasattr(response.cookies, "jar"):
        for cookie in response.cookies.jar:
            cookies[cookie.name] = cookie.value
    else:
        for key, value in response.cookies.items():
            cookies[key] = value
    return cookies



# CDP browser pool — each CDPBrowser opens tabs on the shared Chrome process.
# Multiple providers can open tabs in parallel without any lock.
_shared_cdp_browsers: dict[str, tuple] = {}  # user_data_dir -> (CDPBrowser, refcount)
_shared_cdp_lock = asyncio.Lock()

def _make_cdp_on_stop(user_data_dir: str):
    """Create an on_stop callback that releases the shared CDP browser ref."""
    async def on_stop():
        async with _shared_cdp_lock:
            if user_data_dir in _shared_cdp_browsers:
                browser, refcount = _shared_cdp_browsers[user_data_dir]
                refcount -= 1
                if refcount <= 0:
                    del _shared_cdp_browsers[user_data_dir]
                    debug.log(f"CDP: Released last browser ref for {user_data_dir}")
                else:
                    _shared_cdp_browsers[user_data_dir] = (browser, refcount)
                    debug.log(f"CDP: Released browser ref (#{refcount} remaining for {user_data_dir})")
    return on_stop

def set_browser_executable_path(browser_executable_path: str):
    BrowserConfig.executable_path = browser_executable_path

async def get_nodriver(
    proxy: str = None,
    user_data_dir="nodriver",
    timeout: int = 300,
    browser_executable_path: str = None,
    browser_args: list = None,
    **kwargs,
) -> tuple:
    """Return a CDPBrowser wrapper that emulates the nodriver Browser API.

    Multiple callers share the same CDPBrowser per user_data_dir, but each
    ``browser.get(url)`` call opens an independent tab — so tabs run in
    parallel without serialising on a lock.
    """
    if not has_cdp:
        raise MissingRequirementsError(
            'Chrome/Chromium/Edge executable not found. Install Google Chrome.'
        )

    if browser_executable_path:
        set_browser_executable_path(browser_executable_path)

    ud_key = str(user_data_dir) if user_data_dir else "default"

    async with _shared_cdp_lock:
        if ud_key in _shared_cdp_browsers:
            browser, refcount = _shared_cdp_browsers[ud_key]
            refcount += 1
            _shared_cdp_browsers[ud_key] = (browser, refcount)
            debug.log(f"CDP: Acquired shared browser (#{refcount} active for {ud_key})")
            return browser, _make_cdp_on_stop(ud_key)

    # No shared browser yet — create a new CDPBrowser
    headless = BrowserConfig.headless if BrowserConfig.headless is not None else True
    browser = CDPBrowser(
        headless=headless, proxy=proxy, user_data_dir=user_data_dir,
        browser_args=browser_args,
    )

    async with _shared_cdp_lock:
        _shared_cdp_browsers[ud_key] = (browser, 1)
    debug.log(f"CDP: Started shared browser for {ud_key} (#1 active)")

    on_stop = _make_cdp_on_stop(ud_key)
    BrowserConfig.stop_browser = on_stop
    return browser, on_stop

@asynccontextmanager
async def get_nodriver_session(**kwargs):
    browser, stop_browser = await get_nodriver(**kwargs)
    try:
        yield browser
    finally:
        await stop_browser()



async def sse_stream(iter_lines: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    if hasattr(iter_lines, "content"):
        iter_lines = iter_lines.content
    elif hasattr(iter_lines, "iter_lines"):
        iter_lines = iter_lines.iter_lines()
    async for line in iter_lines:
        if line.startswith(b"data:"):
            rest = line[5:].strip()
            if not rest:
                continue
            if rest.startswith(b"[DONE]"):
                break
            try:
                yield json.loads(rest)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON data: {rest}")


async def iter_lines(iter_response: AsyncIterator[bytes], delimiter=None):
    """
    iterate streaming content line by line, separated by ``\\n``.

    Copied from: https://requests.readthedocs.io/en/latest/_modules/requests/models/
    which is under the License: Apache 2.0
    """
    pending = None

    async for chunk in iter_response:
        if pending is not None:
            chunk = pending + chunk
        lines = chunk.split(delimiter) if delimiter else chunk.splitlines()
        pending = (
            lines.pop()
            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]
            else None
        )

        for line in lines:
            yield line

    if pending is not None:
        yield pending
