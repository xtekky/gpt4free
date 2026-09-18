from __future__ import annotations

try:
    from curl_cffi.requests import AsyncSession, Response

    has_curl_cffi = True
except ImportError:
    # Fallback for systems where curl_cffi is not available or causes illegal instruction errors
    class AsyncSession:
        def __init__(self, *args, **kwargs):
            raise ImportError("curl_cffi is not available on this platform")

    class Response:
        pass

    has_curl_cffi = False

if has_curl_cffi:
    try:
        from curl_cffi import CurlMime

        has_curl_mime = True
    except ImportError:
        has_curl_mime = False
    try:
        from curl_cffi import CurlWsFlag

        has_curl_ws = True
    except ImportError:
        has_curl_ws = False
else:
    has_curl_mime = False
    has_curl_ws = False
from typing import AsyncGenerator, Any
from functools import partialmethod
import asyncio
import json
from ..cookies import BrowserConfig

if has_curl_cffi:

    class StreamResponse:
        """
        A wrapper class for handling asynchronous streaming responses.

        Attributes:
            inner (Response): The original Response object.
        """

        def __init__(self, inner: Response) -> None:
            """Initialize the StreamResponse with the provided Response object."""
            self.inner: Response = inner

        async def text(self) -> str:
            """Asynchronously get the response text."""
            return await self.inner.atext()

        def raise_for_status(self) -> None:
            """Raise an HTTPError if one occurred."""
            self.inner.raise_for_status()

        async def json(self, **kwargs) -> Any:
            """Asynchronously parse the JSON response content."""
            return json.loads(await self.inner.acontent(), **kwargs)

        def iter_lines(self) -> AsyncGenerator[bytes, None]:
            """Asynchronously iterate over the lines of the response."""
            return self.inner.aiter_lines()

        def iter_content(self) -> AsyncGenerator[bytes, None]:
            """Asynchronously iterate over the response content."""
            return self.inner.aiter_content()

        async def sse(self) -> AsyncGenerator[dict, None]:
            """Asynchronously iterate over the Server-Sent Events of the response."""
            async for line in self.iter_lines():
                if line.startswith(b"data: "):
                    chunk = line[6:]
                    if chunk == b"[DONE]":
                        break
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

        async def __aenter__(self):
            """Asynchronously enter the runtime context for the response object."""
            inner: Response = await self.inner
            self.inner = inner
            self.url = inner.url
            self.method = inner.request.method
            self.request = inner.request
            self.status: int = inner.status_code
            self.reason: str = inner.reason
            self.ok: bool = inner.ok
            self.headers = inner.headers
            self.cookies = inner.cookies
            return self

        async def __aexit__(self, *args):
            """Asynchronously exit the runtime context for the response object."""
            await self.inner.aclose()

    class StreamSession(AsyncSession):
        """
        An asynchronous session class for handling HTTP requests with streaming.

        Inherits from AsyncSession.
        """

        def __init__(self, impersonate: str = None, **kwargs) -> None:
            if impersonate == "chrome":
                impersonate = BrowserConfig.impersonate
            super().__init__(impersonate=impersonate, **kwargs)

        def request(self, method: str, url: str, ssl=None, **kwargs) -> StreamResponse:
            if (
                has_curl_mime
                and kwargs.get("data")
                and isinstance(kwargs.get("data"), CurlMime)
            ):
                kwargs["multipart"] = kwargs.pop("data")
            """Create and return a StreamResponse object for the given HTTP request."""
            return StreamResponse(
                super().request(method, url, stream=True, verify=ssl, **kwargs)
            )

        def ws_connect(self, url, *args, **kwargs):
            return WebSocket(self, url, **kwargs)

        def _ws_connect(self, url, **kwargs):
            return super().ws_connect(url, **kwargs)

        # Defining HTTP methods as partial methods of the request method.
        head = partialmethod(request, "HEAD")
        get = partialmethod(request, "GET")
        post = partialmethod(request, "POST")
        put = partialmethod(request, "PUT")
        patch = partialmethod(request, "PATCH")
        delete = partialmethod(request, "DELETE")
        options = partialmethod(request, "OPTIONS")

else:
    # Fallback classes when curl_cffi is not available
    class StreamResponse:
        def __init__(self, *args, **kwargs):
            raise ImportError("curl_cffi is not available on this platform")

    class StreamSession:
        def __init__(self, *args, **kwargs):
            raise ImportError("curl_cffi is not available on this platform")


if has_curl_cffi and has_curl_mime:

    class FormData(CurlMime):
        def add_field(
            self, name, data=None, content_type: str = None, filename: str = None
        ) -> None:
            self.addpart(name, content_type=content_type, filename=filename, data=data)

else:

    class FormData:
        def __init__(self) -> None:
            raise RuntimeError("curl_cffi FormData is not available on this platform")


if has_curl_cffi and has_curl_ws:

    class WebSocket:
        """WebSocket wrapper compatible with curl_cffi >= 0.16 (and older).

        curl_cffi 0.16 replaced the old recv()/send() API with AsyncWebSocket
        (recv_str/send_str, keyword-only timeout, ``closed`` attribute). The
        wrapper is awaitable directly and also usable as async context manager.
        """

        def __init__(self, session, url, **kwargs) -> None:
            self.session: StreamSession = session
            self.url: str = url
            if "autoping" in kwargs:
                del kwargs["autoping"]
            self.options: dict = kwargs
            self.inner = None
            self._closed: bool = False

        def __await__(self):
            return self._connect().__await__()

        async def _connect(self):
            if self.inner is None:
                self.inner = await self.session._ws_connect(self.url, **self.options)
            return self

        async def __aenter__(self):
            return await self._connect()

        async def __aexit__(self, *args):
            await self.close()

        @property
        def closed(self) -> bool:
            if self._closed:
                return True
            if self.inner is None:
                return False
            closed = getattr(self.inner, "closed", None)
            if callable(closed):
                return closed()
            return bool(closed)

        async def close(self):
            self._closed = True
            if self.inner is not None:
                inner_close = getattr(self.inner, "close", None)
                if inner_close is not None:
                    result = inner_close()
                    if asyncio.iscoroutine(result):
                        await result
                self.inner = None

        async def receive_str(self, **kwargs) -> str:
            await self._connect()
            timeout = kwargs.get("timeout")
            if hasattr(self.inner, "recv_str"):
                # curl_cffi >= 0.16: recv_str(*, timeout=None) -> str
                if timeout:
                    return await self.inner.recv_str(timeout=timeout)
                return await self.inner.recv_str()
            method = (
                self.inner.arecv if hasattr(self.inner, "arecv") else self.inner.recv
            )
            data = await method()
            if isinstance(data, tuple):
                data = data[0]
            return data.decode(errors="ignore") if isinstance(data, bytes) else data

        async def recv(self, **kwargs):
            """Return (data, flags) like the legacy curl_cffi recv API."""
            await self._connect()
            timeout = kwargs.get("timeout")
            if hasattr(self.inner, "recv"):
                if timeout:
                    return await self.inner.recv(timeout=timeout)
                return await self.inner.recv()
            data = await self.receive_str(**kwargs)
            return data, 0

        async def send_str(self, data: str):
            await self._connect()
            if hasattr(self.inner, "send_str"):
                # curl_cffi >= 0.16
                await self.inner.send_str(data)
            else:
                method = (
                    self.inner.asend if hasattr(self.inner, "asend") else self.inner.send
                )
                await method(data.encode(), CurlWsFlag.TEXT)

        async def send(self, data):
            await self._connect()
            if hasattr(self.inner, "send_str"):
                # curl_cffi >= 0.16
                if isinstance(data, str):
                    await self.inner.send_str(data)
                else:
                    await self.inner.send(data)
            else:
                method = (
                    self.inner.asend if hasattr(self.inner, "asend") else self.inner.send
                )
                await method(
                    data if isinstance(data, bytes) else data.encode(), CurlWsFlag.TEXT
                )

else:

    class WebSocket:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("curl_cffi WebSocket is not available on this platform")
