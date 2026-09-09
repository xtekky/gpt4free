"""
CDP-based browser wrapper that emulates the zendriver/nodriver Browser API.

This module provides ``CDPBrowser`` and ``CDPTab`` classes that expose the
same high-level interface used by g4f providers (``browser.get(url)``,
``page.evaluate()``, ``page.select()``, ``page.send()``, etc.) but are
backed entirely by the lightweight Chrome DevTools Protocol client in
``g4f.requests.cdp`` — no zendriver/nodriver dependency required.

Multiple ``CDPBrowser`` instances share a single Chrome process via the
refcounted shared-browser mechanism in ``cdp.py``, so providers can open
tabs in parallel without serialising on a lock.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from .cdp import CDPSession, acquire_shared_browser_ref, release_shared_browser_ref
from ..cookies import BrowserConfig
from .. import debug


# ---------------------------------------------------------------------------
# CDP "cdp" module shim — emulates ``nodriver.cdp`` sub-modules used by
# providers: ``cdp.network``, ``cdp.fetch``.
# ---------------------------------------------------------------------------

class _CdpNetwork:
    """Shim for ``nodriver.cdp.network`` commands used via ``page.send()``."""

    @staticmethod
    def enable():
        return {"method": "Network.enable", "params": {}}

    @staticmethod
    def get_cookies(urls: Optional[List[str]] = None):
        params = {}
        if urls:
            params["urls"] = urls
        return {"method": "Network.getCookies", "params": params}

    @staticmethod
    def delete_cookies(name: str = None, domain: str = None, path: str = None):
        params = {}
        if name:
            params["name"] = name
        if domain:
            params["domain"] = domain
        if path:
            params["path"] = path
        return {"method": "Network.deleteCookies", "params": params}

    @staticmethod
    def set_cookies(cookies: list = None):
        return {"method": "Network.setCookies", "params": {"cookies": cookies or []}}

    # Event types used by providers as hashable identifiers
    RequestWillBeSent = "Network.requestWillBeSent"
    WebSocketFrameReceived = "Network.webSocketFrameReceived"
    WebSocketCreated = "Network.webSocketCreated"


class _CdpFetch:
    """Shim for ``nodriver.cdp.fetch`` commands used via ``page.send()``."""

    @staticmethod
    def enable(patterns=None, handle_auth_requests=None):
        params = {}
        if patterns:
            params["patterns"] = patterns
        if handle_auth_requests is not None:
            params["handleAuthRequests"] = handle_auth_requests
        return {"method": "Fetch.enable", "params": params}

    @staticmethod
    def disable():
        return {"method": "Fetch.disable", "params": {}}

    @staticmethod
    def fulfill_request(request_id=None, response_code=200, response_headers=None, body=None):
        params = {"requestId": request_id, "responseCode": response_code}
        if response_headers:
            params["responseHeaders"] = response_headers
        if body:
            params["body"] = body
        return {"method": "Fetch.fulfillRequest", "params": params}

    @staticmethod
    def continue_request(request_id=None):
        return {"method": "Fetch.continueRequest", "params": {"requestId": request_id}}

    @staticmethod
    def fail_request(request_id=None, error_reason=None):
        return {"method": "Fetch.failRequest", "params": {"requestId": request_id, "errorReason": error_reason or "Failed"}}

    # Event types
    RequestPaused = "Fetch.requestPaused"

    # Nested types used by GLM captcha solver
    class RequestPattern:
        def __init__(self, request_stage=None, url_pattern=None, resource_type=None):
            self.request_stage = request_stage
            self.url_pattern = url_pattern
            self.resource_type = resource_type

        def to_dict(self):
            d = {}
            if self.request_stage is not None:
                d["requestStage"] = self.request_stage
            if self.url_pattern is not None:
                d["urlPattern"] = self.url_pattern
            if self.resource_type is not None:
                d["resourceType"] = self.resource_type
            return d

    class RequestStage:
        REQUEST = "Request"
        RESPONSE = "Response"

    class HeaderEntry:
        def __init__(self, name: str, value: str):
            self.name = name
            self.value = value

        def to_dict(self):
            return {"name": self.name, "value": self.value}


class _CdpRuntime:
    """Shim for ``nodriver.cdp.runtime`` — used for exception types."""

    class ProtocolException(Exception):
        pass


class _CdpShim:
    """Top-level shim emulating ``nodriver.cdp``.

    Access ``cdp.network``, ``cdp.fetch``, ``cdp.runtime``.
    """

    network = _CdpNetwork
    fetch = _CdpFetch
    runtime = _CdpRuntime


# Public alias — providers do ``from ..requests.cdp_browser import cdp``
cdp = _CdpShim


# ---------------------------------------------------------------------------
# CookieParam shim — emulates ``zendriver.cdp.network.CookieParam``
# ---------------------------------------------------------------------------

class CookieParam:
    """Emulates ``zendriver.cdp.network.CookieParam``."""

    def __init__(self, name=None, value=None, url=None, domain=None, path=None,
                 secure=None, http_only=None, same_site=None, expires=None):
        self.name = name
        self.value = value
        self.url = url
        self.domain = domain
        self.path = path
        self.secure = secure
        self.http_only = http_only
        self.same_site = same_site
        self.expires = expires

    @classmethod
    def from_json(cls, data: dict) -> "CookieParam":
        return cls(
            name=data.get("name"),
            value=data.get("value"),
            url=data.get("url"),
            domain=data.get("domain"),
            path=data.get("path"),
            secure=data.get("secure"),
            http_only=data.get("httpOnly"),
            same_site=data.get("sameSite"),
            expires=data.get("expires"),
        )

    def to_cdp_dict(self) -> dict:
        d = {}
        if self.name is not None:
            d["name"] = self.name
        if self.value is not None:
            d["value"] = self.value
        if self.url is not None:
            d["url"] = self.url
        if self.domain is not None:
            d["domain"] = self.domain
        if self.path is not None:
            d["path"] = self.path
        if self.secure is not None:
            d["secure"] = self.secure
        if self.http_only is not None:
            d["httpOnly"] = self.http_only
        if self.same_site is not None:
            d["sameSite"] = self.same_site
        if self.expires is not None:
            d["expires"] = self.expires
        return d


def get_cookie_params_from_dict(cookies: dict, url: str = None, domain: str = None) -> list:
    """Build CookieParam list from a plain dict (replaces zendriver version)."""
    return [
        CookieParam(name=key, value=value, url=url, domain=domain)
        for key, value in cookies.items()
    ]


# ---------------------------------------------------------------------------
# CDPElement — emulates nodriver element objects (from page.select / page.find)
# ---------------------------------------------------------------------------

class CDPElement:
    """Wraps a DOM element located via CSS selector for interaction."""

    def __init__(self, tab: "CDPTab", object_id: str, node_id: int = None):
        self._tab = tab
        self._object_id = object_id
        self._node_id = node_id

    async def click(self):
        """Scroll into view and click the element."""
        try:
            await self._tab._session.call(
                "Runtime.callFunctionOn",
                objectId=self._object_id,
                functionDeclaration=(
                    "function(){this.scrollIntoView({block:'center'});"
                    "this.click();}"
                ),
            )
            return
        except Exception:
            pass
        # Fallback: dispatch a real mouse click at the element's on-screen position.
        try:
            res = await self._tab._session.call(
                "DOM.requestNode", objectId=self._object_id
            )
            node_id = res.get("nodeId", 0)
            if node_id:
                box = await self._tab._session.call("DOM.getBoxModel", nodeId=node_id)
                coords = box.get("model", {}).get("content", [])
                if len(coords) >= 2:
                    x, y = coords[0], coords[1]
                    await self._tab._session.call(
                        "Input.dispatchMouseEvent",
                        type="mousePressed", button="left", clickCount=1, x=x, y=y,
                    )
                    await self._tab._session.call(
                        "Input.dispatchMouseEvent",
                        type="mouseReleased", button="left", clickCount=1, x=x, y=y,
                    )
        except Exception:
            pass

    async def send_keys(self, text: str):
        """Type text into the element."""
        # Focus via JS first — more reliable than DOM.focus for contenteditable
        # editors (e.g. ChatGPT's ProseMirror-based prompt box).
        try:
            await self._tab._session.call(
                "Runtime.callFunctionOn",
                objectId=self._object_id,
                functionDeclaration="function(){this.focus();}",
            )
        except Exception:
            pass
        try:
            await self._tab._session.call("DOM.focus", objectId=self._object_id)
        except Exception:
            pass
        # keyDown/keyUp alone don't insert text into contenteditable elements —
        # Input.insertText is required to actually mutate the editor content.
        for char in text:
            await self._tab._session.call(
                "Input.dispatchKeyEvent",
                type="rawKeyDown",
                key=char,
                code="",
                windowsVirtualKeyCode=ord(char) if char.isascii() else 0,
            )
            await self._tab._session.call("Input.insertText", text=char)
            await self._tab._session.call(
                "Input.dispatchKeyEvent",
                type="keyUp",
                key=char,
                code="",
                windowsVirtualKeyCode=ord(char) if char.isascii() else 0,
            )


# ---------------------------------------------------------------------------
# CDPTab — emulates nodriver Tab/page objects
# ---------------------------------------------------------------------------

class CDPTab:
    """A single CDP tab/page that emulates the nodriver ``Tab`` interface.

    Created by ``CDPBrowser.get(url)``.  Each tab is backed by its own
    ``CDPSession`` (a separate WebSocket connection to a separate browser
    target), so multiple tabs operate fully in parallel.
    """

    def __init__(self, session: CDPSession):
        self._session = session
        # Expose cdp shim for providers that do ``page.send(cdp.network.enable())``
        self.cdp = _CdpShim

    # -- core navigation --------------------------------------------------

    async def get(self, url: str):
        """Navigate this tab to *url* (emulates ``browser.get(url)``)."""
        await self._session.navigate(url)
        return self

    async def reload(self):
        """Reload the current page and wait for it to finish loading."""
        await self._session.reload()

    async def close(self):
        """Close this tab."""
        await self._session.close()

    # -- evaluation -------------------------------------------------------

    async def evaluate(self, expression: str, return_by_value: bool = True,
                       await_promise: bool = False) -> Any:
        """Evaluate JS — emulates ``page.evaluate(js, return_by_value=, await_promise=)``."""
        params = {
            "expression": expression,
            "returnByValue": return_by_value,
            "awaitPromise": await_promise,
        }
        res = await self._session.call("Runtime.evaluate", **params)
        result = res.get("result", {})
        if result.get("type") == "object" and result.get("subtype") == "error":
            raise RuntimeError(f"JS evaluation error: {result.get('description', '')}")
        if return_by_value:
            return result.get("value")
        # If not returnByValue, wrap in a simple object with .value
        class _EvalResult:
            def __init__(self, val):
                self.value = val
        return _EvalResult(result.get("value"))

    async def evaluate_js(self, expression: str) -> Any:
        """Direct passthrough to CDPSession.evaluate_js."""
        return await self._session.evaluate_js(expression)

    async def js_dumps(self, expression: str) -> Any:
        """Evaluate and return the JS value (emulates ``page.js_dumps``)."""
        return await self.evaluate(expression, return_by_value=True)

    # -- selectors --------------------------------------------------------

    async def select(self, selector: str, timeout: float = 30) -> Optional[CDPElement]:
        """Wait for a CSS selector and return a CDPElement, or None on timeout.

        Emulates ``page.select(selector, timeout)``.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                # Use Runtime.evaluate to find the element and get its objectId
                res = await self._session.call(
                    "Runtime.evaluate",
                    expression=f"document.querySelector({json.dumps(selector)})",
                    returnByValue=False,
                )
                result = res.get("result", {})
                if result.get("objectId") and result.get("type") == "object":
                    return CDPElement(self, result["objectId"])
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return None

    async def find(self, text: str, timeout: float = 30) -> Optional[CDPElement]:
        """Find an element by visible text content.

        Emulates ``page.find(text)``.
        """
        js = f"""(function() {{
            const elements = document.querySelectorAll('*');
            for (const el of elements) {{
                const content = (el.innerText || el.textContent || '').trim();
                if (content === {json.dumps(text)}) return el;
            }}
            return null;
        }})()"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                res = await self._session.call(
                    "Runtime.evaluate", expression=js, returnByValue=False,
                )
                result = res.get("result", {})
                if result.get("objectId") and result.get("type") == "object":
                    return CDPElement(self, result["objectId"])
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return None

    async def wait_for(self, selector: str, timeout: float = 30) -> bool:
        """Wait until a CSS selector appears on the page.

        Emulates ``page.wait_for(selector, timeout)``.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                found = await self._session.evaluate_js(
                    f"!!document.querySelector({json.dumps(selector)})"
                )
                if found:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    async def get_content(self) -> str:
        """Return the page's HTML content (emulates ``page.get_content()``)."""
        return await self._session.evaluate_js("document.documentElement.outerHTML")

    # -- CDP command sending (emulates ``page.send(cdp_command)``) --------

    async def send(self, command):
        """Send a CDP command.

        *command* can be:
        - A dict with ``{"method": ..., "params": ...}`` (from our cdp shim)
        - A generator (from the OpenaiChat ``get_cookies`` pattern)
        - A string (raw method name)
        """
        if isinstance(command, dict):
            method = command.get("method")
            params = command.get("params", {})
            return await self._session.call(method, **params)
        elif isinstance(command, str):
            return await self._session.call(command)
        elif hasattr(command, "__next__") or hasattr(command, "__iter__"):
            # Generator-based command (e.g. OpenaiChat's get_cookies)
            # The generator yields a dict, expects a response, and returns a value
            try:
                cmd_dict = next(command)
                if isinstance(cmd_dict, dict):
                    method = cmd_dict.get("method")
                    params = cmd_dict.get("params", {})
                    response = await self._session.call(method, **params)
                    try:
                        return command.send(response)
                    except StopIteration as e:
                        return e.value
                return None
            except StopIteration as e:
                return e.value
        return None

    # -- event handlers (emulates ``page.add_handler(event_type, callback)``) --

    def add_handler(self, event_type, callback: Callable):
        """Register a persistent event handler.

        *event_type* can be a string (CDP method name) or a class from our
        cdp shim (e.g. ``cdp.network.RequestWillBeSent``).
        """
        method = event_type
        if not isinstance(method, str):
            # Try to get the string value from our shim classes
            method = getattr(event_type, "value", None) or str(event_type)

        queue: asyncio.Queue = asyncio.Queue()
        self._session.add_event_handler(method, queue)

        async def _listener():
            while True:
                try:
                    event = await queue.get()
                    if event is None:
                        break
                    # Build a simple event object that providers can inspect
                    evt = _CdpEvent(event)
                    try:
                        result = callback(evt, page=self)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        debug.error(f"CDP event handler error: {e}")
                except asyncio.CancelledError:
                    break

        self._listener_task = asyncio.create_task(_listener())

    # -- mouse helpers (emulates page.flash_point / page.mouse_click) -----

    async def flash_point(self, x: float, y: float):
        """Move mouse to a point (emulates ``page.flash_point``)."""
        await self._session.call("Input.dispatchMouseEvent", type="mouseMoved", x=int(x), y=int(y))

    async def mouse_click(self, x: float, y: float):
        """Click at coordinates (emulates ``page.mouse_click``)."""
        await self._session.call(
            "Input.dispatchMouseEvent",
            type="mousePressed", button="left", clickCount=1, x=int(x), y=int(y),
        )
        await self._session.call(
            "Input.dispatchMouseEvent",
            type="mouseReleased", button="left", clickCount=1, x=int(x), y=int(y),
        )


class _CdpEvent:
    """Wraps a raw CDP event dict so providers can access fields as attributes.

    E.g. ``event.request.url`` or ``event.request.headers``.
    """

    def __init__(self, data: dict):
        self._raw = data
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                setattr(self, key, _DictAttr(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return self._raw[key]

    def get(self, key, default=None):
        return self._raw.get(key, default)

    def __repr__(self):
        return f"_CdpEvent({self._raw.get('_method', '?')})"


class _DictAttr:
    """Wraps a dict so its keys are accessible as attributes."""

    def __init__(self, data: dict):
        self._raw = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _DictAttr(value))
            elif isinstance(value, list):
                setattr(self, key, [_DictAttr(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return self._raw[key]

    def get(self, key, default=None):
        return self._raw.get(key, default)

    def items(self):
        return self._raw.items()

    def __repr__(self):
        return f"_DictAttr({self._raw})"


# ---------------------------------------------------------------------------
# CDPBrowser — emulates nodriver Browser
# ---------------------------------------------------------------------------

class CDPBrowser:
    """Emulates the nodriver ``Browser`` object using CDP tabs.

    Each call to ``get(url)`` opens a new tab (CDPSession) on the shared
    Chrome process.  Multiple tabs run in parallel without locking.
    """

    def __init__(self, headless: Optional[bool] = None, proxy: str = None,
                 user_data_dir: str = None, browser_args: Optional[List[str]] = None):
        self.headless = headless
        self.proxy = proxy
        self.user_data_dir = user_data_dir
        self.browser_args = browser_args
        self._tabs: List[CDPTab] = []
        self.cdp = _CdpShim
        self.cookies = _BrowserCookies(self)

    @property
    def main_tab(self) -> Optional[CDPTab]:
        """Return the first open tab, if any."""
        return self._tabs[0] if self._tabs else None

    async def get(self, url: str) -> CDPTab:
        """Open a new tab and navigate to *url*.

        Emulates ``browser.get(url)`` from nodriver.
        """
        session = CDPSession(
            headless=self.headless, proxy=self.proxy, browser_args=self.browser_args
        )
        await session.start()
        tab = CDPTab(session)
        self._tabs.append(tab)
        if url and url != "about:blank":
            await tab.get(url)
        return tab

    async def stop(self):
        """Close all tabs and release the shared browser reference."""
        for tab in list(self._tabs):
            try:
                await tab.close()
            except Exception:
                pass
        self._tabs.clear()


class _BrowserCookies:
    """Emulates ``browser.cookies`` from nodriver."""

    def __init__(self, browser: CDPBrowser):
        self._browser = browser

    async def set_all(self, cookie_params: list):
        """Set cookies via a temporary tab."""
        tab = await self._browser.get("about:blank")
        try:
            for param in cookie_params:
                if isinstance(param, CookieParam):
                    params = param.to_cdp_dict()
                elif isinstance(param, dict):
                    params = param
                else:
                    continue
                await tab._session.call("Network.setCookie", **params)
        finally:
            await tab.close()

    async def get_all(self) -> list:
        """Get all cookies as CDP cookie dicts."""
        tab = await self._browser.get("about:blank")
        try:
            res = await tab._session.call("Network.getCookies")
            return res.get("cookies", [])
        finally:
            await tab.close()
