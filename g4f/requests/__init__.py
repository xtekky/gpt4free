from __future__ import annotations

from typing import Union
from aiohttp import ClientResponse
from requests import Response as RequestsResponse

try:
    from curl_cffi.requests import Session, Response
    from .curl_cffi import StreamResponse, StreamSession
    has_curl_cffi = True
except ImportError:
    from typing import Type as Session, Type as Response
    from .aiohttp import StreamResponse, StreamSession
    has_curl_cffi = False

from ..webdriver import WebDriver, WebDriverSession
from ..webdriver import bypass_cloudflare, get_driver_cookies
from ..errors import MissingRequirementsError, RateLimitError, ResponseStatusError
from .defaults import DEFAULT_HEADERS

def get_args_from_browser(
    url: str,
    webdriver: WebDriver = None,
    proxy: str = None,
    timeout: int = 120,
    do_bypass_cloudflare: bool = True,
    virtual_display: bool = False
) -> dict:
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
    with WebDriverSession(webdriver, "", proxy=proxy, virtual_display=virtual_display) as driver:
        if do_bypass_cloudflare:
            bypass_cloudflare(driver, url, timeout)
        headers = {
            **DEFAULT_HEADERS,
            'referer': url,
        }
        if not hasattr(driver, "requests"):
            headers["user-agent"] = driver.execute_script("return navigator.userAgent")
        else:
            for request in driver.requests:
                if request.url.startswith(url):
                    for key, value in request.headers.items():
                        if key in (
                            "accept-encoding",
                            "accept-language",
                            "user-agent",
                            "sec-ch-ua",
                            "sec-ch-ua-platform",
                            "sec-ch-ua-arch",
                            "sec-ch-ua-full-version",
                            "sec-ch-ua-platform-version",
                            "sec-ch-ua-bitness"
                        ):
                            headers[key] = value
                    break
        cookies = get_driver_cookies(driver)
    return {
        'cookies': cookies,
        'headers': headers,
    }

def get_session_from_browser(url: str, webdriver: WebDriver = None, proxy: str = None, timeout: int = 120) -> Session:
    if not has_curl_cffi:
        raise MissingRequirementsError('Install "curl_cffi" package')
    args = get_args_from_browser(url, webdriver, proxy, timeout)
    return Session(
        **args,
        proxies={"https": proxy, "http": proxy},
        timeout=timeout,
        impersonate="chrome"
    )

async def raise_for_status_async(response: Union[StreamResponse, ClientResponse], message: str = None):
    if response.status in (429, 402):
        raise RateLimitError(f"Response {response.status}: Rate limit reached")
    message = await response.text() if not response.ok and message is None else message
    if response.status == 403 and "<title>Just a moment...</title>" in message:
        raise ResponseStatusError(f"Response {response.status}: Cloudflare detected")
    elif not response.ok:
        raise ResponseStatusError(f"Response {response.status}: {message}")

def raise_for_status(response: Union[StreamResponse, ClientResponse, Response, RequestsResponse], message: str = None):
    if isinstance(response, StreamSession) or isinstance(response, ClientResponse):
        return raise_for_status_async(response, message)

    if response.status_code in (429, 402):
        raise RateLimitError(f"Response {response.status_code}: Rate limit reached")
    elif response.status_code == 403 and "<title>Just a moment...</title>" in response.text:
        raise ResponseStatusError(f"Response {response.status_code}: Cloudflare detected")
    elif not response.ok:
        raise ResponseStatusError(f"Response {response.status_code}: {response.text if message is None else message}")