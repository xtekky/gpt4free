from __future__ import annotations

from urllib.parse import urlparse

try:
    from curl_cffi.requests import Session
    from .requests_curl_cffi import StreamResponse, StreamSession
    has_curl_cffi = True
except ImportError:
    from typing import Type as Session
    from .requests_aiohttp import StreamResponse, StreamSession
    has_curl_cffi = False

from .webdriver import WebDriver, WebDriverSession, bypass_cloudflare, get_driver_cookies
from .errors import MissingRequirementsError
from .defaults import DEFAULT_HEADERS

def get_args_from_browser(url: str, webdriver: WebDriver = None, proxy: str = None, timeout: int = 120) -> dict:
    """
    Create a Session object using a WebDriver to handle cookies and headers.

    Args:
        url (str): The URL to navigate to using the WebDriver.
        webdriver (WebDriver, optional): The WebDriver instance to use.
        proxy (str, optional): Proxy server to use for the Session.
        timeout (int, optional): Timeout in seconds for the WebDriver.

    Returns:
        Session: A Session object configured with cookies and headers from the WebDriver.
    """
    with WebDriverSession(webdriver, "", proxy=proxy, virtual_display=False) as driver:
        bypass_cloudflare(driver, url, timeout)
        cookies = get_driver_cookies(driver)
        user_agent = driver.execute_script("return navigator.userAgent")
    parse = urlparse(url)
    return {
        'cookies': cookies,
        'headers': {
            **DEFAULT_HEADERS,
            'Authority': parse.netloc,
            'Origin': f'{parse.scheme}://{parse.netloc}',
            'Referer': url,
            'User-Agent': user_agent,
        },
    }

def get_session_from_browser(url: str, webdriver: WebDriver = None, proxy: str = None, timeout: int = 120) -> Session:
    if not has_curl_cffi:
        raise MissingRequirementsError('Install "curl_cffi" package')
    args = get_args_from_browser(url, webdriver, proxy, timeout)
    return Session(
        **args,
        proxies={"https": proxy, "http": proxy},
        timeout=timeout,
        impersonate="chrome110"
    )