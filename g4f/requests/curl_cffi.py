from __future__ import annotations

from curl_cffi.requests import AsyncSession, Response
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
from typing import AsyncGenerator, Any
from functools import partialmethod
import json

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
        return  self.inner.aiter_lines()

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

    def request(
        self, method: str, url: str, ssl = None, **kwargs
    ) -> StreamResponse:
        if kwargs.get("data") and isinstance(kwargs.get("data"), CurlMime):
            kwargs["multipart"] = kwargs.pop("data")
        """Create and return a StreamResponse object for the given HTTP request."""
        return StreamResponse(super().request(method, url, stream=True, verify=ssl, **kwargs))

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

if not has_curl_mime:
    class FormData():
        def __init__(self) -> None:
            raise RuntimeError("CurlMimi in curl_cffi is missing | pip install -U curl_cffi")
else:
    class FormData(CurlMime):
        def add_field(self, name, data=None, content_type: str = None, filename: str = None) -> None:
            self.addpart(name, content_type=content_type, filename=filename, data=data)

class WebSocket():
    def __init__(self, session, url, **kwargs) -> None:
        if not has_curl_ws:
            raise RuntimeError("CurlWsFlag in curl_cffi is missing | pip install -U curl_cffi")
        self.session: StreamSession = session
        self.url: str = url
        del kwargs["autoping"]
        self.options: dict = kwargs

    async def __aenter__(self):
        self.inner = await self.session._ws_connect(self.url, **self.options)
        return self

    async def __aexit__(self, *args):
        await self.inner.aclose() if hasattr(self.inner, "aclose") else await self.inner.close()

    async def receive_str(self, **kwargs) -> str:
        method = self.inner.arecv if hasattr(self.inner, "arecv") else self.inner.recv
        bytes, _ = await method()
        return bytes.decode(errors="ignore")

    async def send_str(self, data: str):
        method = self.inner.asend if hasattr(self.inner, "asend") else self.inner.send
        await method(data.encode(), CurlWsFlag.TEXT)