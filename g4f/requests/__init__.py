from __future__ import annotations

import os
import time
import random
import json
from urllib.parse import urlparse
from typing import Iterator, AsyncIterator
from http.cookies import Morsel
from pathlib import Path
import asyncio
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
    import nodriver
    from nodriver.cdp.network import CookieParam
    from nodriver.core.config import find_chrome_executable
    from nodriver import Browser, Tab, util
    has_nodriver = True
except ImportError:
    from typing import Type as Browser
    from typing import Type as Tab
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
from ..cookies import get_cookies_dir
from .defaults import DEFAULT_HEADERS, WEBVIEW_HAEDERS

class BrowserConfig:
    stop_browser = lambda: None
    browser_executable_path: str = None

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
    wait_for: str = None,
    callback: callable = None,
    cookies: Cookies = None,
    browser: Browser = None,
    user_data_dir: str = "nodriver"
) -> dict:
    if browser is None:
        browser, stop_browser = await get_nodriver(proxy=proxy, timeout=timeout, user_data_dir=user_data_dir)
    else:
        def stop_browser():
            pass
    try:
        debug.log(f"Open nodriver with url: {url}")
        if cookies is None:
            cookies = {}
        else:
            domain = urlparse(url).netloc
            await browser.cookies.set_all(get_cookie_params_from_dict(cookies, url=url, domain=domain))
        page = await browser.get(url)
        user_agent = await page.evaluate("window.navigator.userAgent", return_by_value=True)
        while not await page.evaluate("document.querySelector('body:not(.no-js)')"):
            await asyncio.sleep(1)
        if wait_for is not None:
            await page.wait_for(wait_for, timeout=timeout)
        if callback is not None:
            await callback(page)
        for c in await page.send(nodriver.cdp.network.get_cookies([url])):
            cookies[c.name] = c.value
        await page.close()
        stop_browser()
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
    except:
        stop_browser()
        raise

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

def set_browser_executable_path(browser_executable_path: str):
    BrowserConfig.browser_executable_path = browser_executable_path

async def get_nodriver(
    proxy: str = None,
    user_data_dir = "nodriver",
    timeout: int = 300,
    browser_executable_path: str = None,
    **kwargs
) -> tuple[Browser, callable]:
    if not has_nodriver:
        raise MissingRequirementsError('Install "nodriver" and "platformdirs" package | pip install -U nodriver platformdirs')
    user_data_dir = user_config_dir(f"g4f-{user_data_dir}") if user_data_dir and has_platformdirs else None
    if browser_executable_path is None:
        browser_executable_path = BrowserConfig.browser_executable_path
    if browser_executable_path is None:
        try:
            browser_executable_path = find_chrome_executable()
        except FileNotFoundError:
            # Default to Edge if Chrome is not available.
            browser_executable_path = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
            if not os.path.exists(browser_executable_path):
                browser_executable_path = None
    debug.log(f"Browser executable path: {browser_executable_path}")
    lock_file = Path(get_cookies_dir()) / ".nodriver_is_open"
    lock_file.parent.mkdir(exist_ok=True)
    # Implement a short delay (milliseconds) to prevent race conditions.
    if user_data_dir:
        await asyncio.sleep(0.1 * random.randint(0, 50))
        if lock_file.exists():
            opend_at = float(lock_file.read_text())
            time_open = time.time() - opend_at
            if timeout * 2 > time_open:
                debug.log(f"Nodriver: Browser is already in use since {time_open} secs.")
                debug.log("Lock file:", lock_file)
                for idx in range(timeout):
                    if lock_file.exists():
                        await asyncio.sleep(1)
                    else:
                        break
                    if idx == timeout - 1:
                        debug.log("Timeout reached, nodriver is still in use.")
                        raise TimeoutError("Nodriver is already in use, please try again later.")
            else:
                debug.log(f"Nodriver: Browser was opened {time_open} secs ago, closing it.")
                BrowserConfig.stop_browser()
                lock_file.unlink(missing_ok=True)
        lock_file.write_text(str(time.time()))
        debug.log(f"Open nodriver with user_dir: {user_data_dir}")
    try:
        browser = await nodriver.start(
            user_data_dir=user_data_dir,
            browser_args=["--no-sandbox"] if proxy is None else ["--no-sandbox", f"--proxy-server={proxy}"],
            browser_executable_path=browser_executable_path,
            **kwargs
        )
    except FileNotFoundError as e:
        raise MissingRequirementsError(e)
    except Exception as e:
        if util.get_registered_instances():
            debug.error(e)
            browser = util.get_registered_instances().pop()
        else:
            raise
    def on_stop():
        try:
            if browser.connection:
                browser.stop()
        except:
            pass
        finally:
            if user_data_dir:
                lock_file.unlink(missing_ok=True)
    BrowserConfig.stop_browser = on_stop
    return browser, on_stop

async def sse_stream(iter_lines: Iterator[bytes]) -> AsyncIterator[dict]:
    if hasattr(iter_lines, "content"):
        iter_lines = iter_lines.content
    elif hasattr(iter_lines, "iter_lines"):
        iter_lines = iter_lines.iter_lines()
    async for line in iter_lines:
        if line.startswith(b"data: "):
            rest = line[6:].strip()
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