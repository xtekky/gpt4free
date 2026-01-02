import asyncio
import base64
import re
import typing
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

import nodriver
from nodriver import cdp
from nodriver.cdp.fetch import HeaderEntry, RequestStage, RequestPattern, RequestPaused
from nodriver.cdp.network import CookieParam
from nodriver.cdp.network import ResourceType

from g4f.debug import log


async def clear_cookies_for_url(browser: nodriver.Browser, url: str, ignore_cookies: list[str] = None):
    log(f"Clearing cookies for {url}")
    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"Bad url: {url}")

    if ignore_cookies is None:
        ignore_cookies = []

    tab = browser.main_tab  # any open tab is fine
    cookies = await browser.cookies.get_all()  # returns CDP cookies :contentReference[oaicite:2]{index=2}
    for c in cookies:
        dom = (c.domain or "").lstrip(".")
        if dom and (host == dom or host.endswith("." + dom)):
            if c.name in ignore_cookies:
                continue
            await tab.send(
                nodriver.cdp.network.delete_cookies(
                    name=c.name,
                    domain=dom,  # exact domain :contentReference[oaicite:3]{index=3}
                    path=c.path,  # exact path :contentReference[oaicite:4]{index=4}
                    # partition_key=c.partition_key,  # if you use partitioned cookies
                )
            )


async def set_cookies_for_browser(browser: nodriver.Browser, cookies: list | dict, url: str):
    if not cookies:
        return
    log(f"Setting cookies for {url}")
    domain = urlparse(url).netloc

    _cookies: list[nodriver.cdp.network.CookieParam] = []
    if isinstance(cookies, list):
        for value in cookies:
            # if isinstance(value, CookieParam):
            #     _cookies.append(value)
            # else:
            value.pop("sameSite", None)
            _cookies.append(CookieParam.from_json(value))
    else:
        _cookies = [CookieParam.from_json({
            "name": key,
            "value": value,
            "url": url,
            "domain": domain
        }) for key, value in cookies.items()]
    await browser.connection.send(nodriver.cdp.storage.set_cookies(_cookies))


async def get_cookies(browser: nodriver.Browser, url: str) -> dict[str, str]:
    cookies = {}
    for c in await browser.main_tab.send(nodriver.cdp.network.get_cookies([url])):
        cookies[c.name] = c.value
    return cookies


async def get_args(browser: nodriver.Browser, url) -> dict[str, str]:
    cookies = await get_cookies(browser, url)
    args = {
        "impersonate": "chrome136",
        "cookies": cookies,
        "headers": {
            "accept": "*/*", "accept-encoding": "gzip, deflate, br", "accept-language": "en-US",
            "referer": "https://lmarena.ai/",
            "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
            "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": "\"Windows\"", "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors", "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        },
        "proxy": None
    }
    return args


async def wait_for_ready_state(
        page,
        until: Literal["loading", "interactive", "complete"] = "interactive",
        timeout: int = 10,
        raise_error: bool = True,
) -> bool:
    """
    Waits for the page to reach a certain ready state.
    :param until: The ready state to wait for. Can be one of "loading", "interactive", or "complete".
    :type until: str
    :param timeout: The maximum number of seconds to wait.
    :type timeout: int
    :raises asyncio.TimeoutError: If the timeout is reached before the ready state is reached.
    :return: True if the ready state is reached.
    :rtype: bool
    """
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    while True:
        ready_state = await page.evaluate("document.readyState")
        if ready_state == until:
            return True

        if loop.time() - start_time > timeout:
            if raise_error:
                raise asyncio.TimeoutError(
                    "time ran out while waiting for load page until %s" % until
                )
            else:
                return False

        await asyncio.sleep(0.1)


def remove_handlers(
        tab,
        event_type: typing.Optional[type] = None,
        handler: typing.Optional[typing.Union[typing.Callable, typing.Awaitable]] = None,  # type: ignore
) -> None:
    """
    remove handlers for given event

    if no event is provided, all handlers will be removed.
    if no handler is provided, all handlers for the event will be removed.

    .. code-block::

            # remove all handlers for all events
            page.remove_handlers()

            # remove all handlers for a specific event
            page.remove_handlers(cdp.network.RequestWillBeSent)

            # remove a specific handler for a specific event
            page.remove_handlers(cdp.network.RequestWillBeSent, handler)
    """

    if handler and not event_type:
        raise ValueError(
            "if handler is provided, event_type should be provided as well"
        )

    if not event_type:
        tab.handlers.clear()
        return

    if not handler:
        del tab.handlers[event_type]
        return

    if handler in tab.handlers[event_type]:
        tab.handlers[event_type].remove(handler)


class RequestInterception:
    """
    Base class to wait for a Fetch response matching a URL pattern.
    Use this to collect and decode a paused fetch response, while keeping
    the use block clean and returning its own result.

    :param tab: The Tab instance to monitor.
    :param url_pattern: The URL pattern to match requests and responses.
    :param request_stage: The stage of the fetch request to intercept (e.g., request or response).
    :param resource_type: The type of resource to intercept (e.g., document, script, etc.).
    """

    def __init__(
            self,
            tab: nodriver.Tab,
            url_pattern: str,
            request_stage: typing.Optional[RequestStage] = None,
            resource_type: typing.Optional[ResourceType] = None,
    ):
        self.tab = tab
        self.url_pattern = url_pattern
        self.request_stage = request_stage
        self.resource_type = resource_type
        self.response_future: asyncio.Future[Request] = asyncio.Future()

    async def _response_handler(self, event: nodriver.cdp.fetch.RequestPaused) -> None:
        """
        Internal handler for response events.
        :param event: The response event.
        :type event: nodriver.cdp.fetch.RequestPaused
        """
        self._remove_response_handler()
        fetch_data = event.__dict__.copy()
        fetch_data['tab'] = self.tab
        fetch_obj = Request(**fetch_data)
        self.response_future.set_result(fetch_obj)

    def _remove_response_handler(self) -> None:
        """
        Remove the response event handler.
        """
        remove_handlers(self.tab, nodriver.cdp.fetch.RequestPaused, self._response_handler)

    async def __aenter__(self) -> "RequestInterception":
        """
        Enter the context manager, adding request and response handlers.
        """
        await self._setup()
        return self

    async def __aexit__(self, *args: typing.Any) -> None:
        """
        Exit the context manager, removing request and response handlers.
        """
        await self._teardown()

    async def _setup(self) -> None:
        await self.tab.send(
            nodriver.cdp.fetch.enable(
                [
                    RequestPattern(
                        url_pattern=self.url_pattern,
                        request_stage=self.request_stage,
                        resource_type=self.resource_type,
                    )
                ]
            )
        )
        self.tab.enabled_domains.append(
            nodriver.cdp.fetch
        )  # trick to avoid another `fetch.enable` call by _register_handlers
        self.tab.add_handler(nodriver.cdp.fetch.RequestPaused, self._response_handler)

    async def _teardown(self) -> None:
        self._remove_response_handler()
        await self.tab.send(nodriver.cdp.fetch.disable())

    async def reset(self) -> None:
        """
        Resets the internal state, allowing the interception to be reused.
        """
        self.response_future = asyncio.Future()
        await self._teardown()
        await self._setup()

    @property
    async def request(self) -> nodriver.cdp.network.Request:
        """
        Get the matched request.
        :return: The matched request.
        :rtype: nodriver.cdp.network.request
        """
        return (await self.response_future).request


@dataclass
class Request(RequestPaused):
    tab: nodriver.Tab

    @property
    async def response_body(self) -> tuple[str, bool]:
        """
        Get the body of the matched response.
        :return: The response body.
        :rtype: str
        """
        request_id = self.request_id
        body = await self.tab.send(nodriver.cdp.fetch.get_response_body(request_id=request_id))
        return body

    @property
    async def response_body_as_stream(self):
        result = await self.tab.send(nodriver.cdp.fetch.take_response_body_as_stream(
            request_id=self.request_id
        ))
        stream_handle = result
        eof = False

        while not eof:
            is_base64, data, eof = await self.tab.send(nodriver.cdp.io.read(handle=stream_handle))
            if data:
                chunk_raw = base64.b64decode(data) if is_base64 else data
                if isinstance(chunk_raw, bytes):
                    chunk_raw = chunk_raw.decode("utf-8")
                yield chunk_raw

        await self.tab.send(nodriver.cdp.io.close(handle=stream_handle))

        await self.tab.send(nodriver.cdp.fetch.fail_request(
            request_id=self.request_id,
            error_reason=nodriver.cdp.network.ErrorReason.BLOCKED_BY_CLIENT
        ))

    async def fail_request(self, error_reason: nodriver.cdp.network.ErrorReason) -> None:
        request_id = self.request_id
        await self.tab.send(
            nodriver.cdp.fetch.fail_request(request_id=request_id, error_reason=error_reason)
        )

    async def continue_request(
            self,
            url: typing.Optional[str] = None,
            method: typing.Optional[str] = None,
            post_data: typing.Optional[str] = None,
            headers: typing.Optional[typing.List[HeaderEntry]] = None,
            intercept_response: typing.Optional[bool] = None,
    ) -> None:
        request_id = self.request_id

        await self.tab.send(
            nodriver.cdp.fetch.continue_request(
                request_id=request_id,
                url=url,
                method=method,
                post_data=post_data,
                headers=headers,
                intercept_response=intercept_response,
            )
        )

    async def fulfill_request(
            self,
            response_code: int,
            response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
            binary_response_headers: typing.Optional[str] = None,
            body: typing.Optional[str] = None,
            response_phrase: typing.Optional[str] = None,
    ) -> None:
        request_id = self.request_id
        await self.tab.send(
            nodriver.cdp.fetch.fulfill_request(
                request_id=request_id,
                response_code=response_code,
                response_headers=response_headers,
                binary_response_headers=binary_response_headers,
                body=body,
                response_phrase=response_phrase,
            )
        )

    async def continue_response(
            self,
            response_code: typing.Optional[int] = None,
            response_phrase: typing.Optional[str] = None,
            response_headers: typing.Optional[typing.List[HeaderEntry]] = None,
            binary_response_headers: typing.Optional[str] = None,
    ) -> None:
        request_id = self.request_id
        await self.tab.send(
            nodriver.cdp.fetch.continue_response(
                request_id=request_id,
                response_code=response_code,
                response_phrase=response_phrase,
                response_headers=response_headers,
                binary_response_headers=binary_response_headers,
            )
        )


class BaseRequestExpectation:
    """
    Base class for handling request and response expectations.
    This class provides a context manager to wait for specific network requests and responses
    based on a URL pattern. It sets up handlers for request and response events and provides
    properties to access the request, response, and response body.
    :param tab: The Tab instance to monitor.
    :type tab: Tab
    :param url_pattern: The URL pattern to match requests and responses.
    :type url_pattern: Union[str, re.Pattern[str]]
    """

    def __init__(self, tab: nodriver.Tab, url_pattern: typing.Union[str, re.Pattern[str]]):
        self.tab = tab
        self.url_pattern = url_pattern
        self.request_future: asyncio.Future[cdp.network.RequestWillBeSent] = (
            asyncio.Future()
        )
        self.response_future: asyncio.Future[cdp.network.ResponseReceived] = (
            asyncio.Future()
        )
        self.loading_finished_future: asyncio.Future[cdp.network.LoadingFinished] = (
            asyncio.Future()
        )
        self.request_id: typing.Union[cdp.network.RequestId, None] = None

    async def _request_handler(self, event: cdp.network.RequestWillBeSent) -> None:
        """
        Internal handler for request events.
        :param event: The request event.
        :type event: cdp.network.RequestWillBeSent
        """
        if re.fullmatch(self.url_pattern, event.request.url):
            self._remove_request_handler()
            self.request_id = event.request_id
            self.request_future.set_result(event)

    async def _response_handler(self, event: cdp.network.ResponseReceived) -> None:
        """
        Internal handler for response events.
        :param event: The response event.
        :type event: cdp.network.ResponseReceived
        """
        if event.request_id == self.request_id:
            self._remove_response_handler()
            self.response_future.set_result(event)

    async def _loading_finished_handler(
            self, event: cdp.network.LoadingFinished
    ) -> None:
        """
        Internal handler for loading finished events.
        :param event: The loading finished event.
        :type event: cdp.network.LoadingFinished
        """
        if event.request_id == self.request_id:
            self._remove_loading_finished_handler()
            self.loading_finished_future.set_result(event)

    def _remove_request_handler(self) -> None:
        """
        Remove the request event handler.
        """
        remove_handlers(self.tab, cdp.network.RequestWillBeSent, self._request_handler)

    def _remove_response_handler(self) -> None:
        """
        Remove the response event handler.
        """
        remove_handlers(self.tab, cdp.network.ResponseReceived, self._response_handler)

    def _remove_loading_finished_handler(self) -> None:
        """
        Remove the loading finished event handler.
        """
        remove_handlers(self.tab,
                        cdp.network.LoadingFinished, self._loading_finished_handler
                        )

    async def __aenter__(self):  # type: ignore
        """
        Enter the context manager, adding request and response handlers.
        """
        await self._setup()
        return self

    async def __aexit__(self, *args: typing.Any) -> None:
        """
        Exit the context manager, removing request and response handlers.
        """
        self._teardown()

    async def _setup(self) -> None:
        self.tab.add_handler(cdp.network.RequestWillBeSent, self._request_handler)
        self.tab.add_handler(cdp.network.ResponseReceived, self._response_handler)
        self.tab.add_handler(
            cdp.network.LoadingFinished, self._loading_finished_handler
        )

    def _teardown(self) -> None:
        self._remove_request_handler()
        self._remove_response_handler()
        self._remove_loading_finished_handler()

    async def reset(self) -> None:
        """
        Resets the internal state, allowing the expectation to be reused.
        """
        self.request_future = asyncio.Future()
        self.response_future = asyncio.Future()
        self.loading_finished_future = asyncio.Future()
        self.request_id = None
        self._teardown()
        await self._setup()

    @property
    async def request(self) -> cdp.network.Request:
        """
        Get the matched request.
        :return: The matched request.
        :rtype: cdp.network.Request
        """
        return (await self.request_future).request

    @property
    async def response(self) -> cdp.network.Response:
        """
        Get the matched response.
        :return: The matched response.
        :rtype: cdp.network.Response
        """
        return (await self.response_future).response

    @property
    async def response_body(self) -> tuple[str, bool]:
        """
        Get the body of the matched response.
        :return: The response body.
        :rtype: str
        """
        request_id = (await self.response_future).request_id
        await (
            self.loading_finished_future
        )  # Ensure the loading is finished before fetching the body
        body = await self.tab.send(cdp.network.get_response_body(request_id=request_id))
        return body
