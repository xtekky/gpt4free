from __future__ import annotations

from curl_cffi.requests import AsyncSession, Response, CurlMime
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

    async def iter_lines(self) -> AsyncGenerator[bytes, None]:
        """Asynchronously iterate over the lines of the response."""
        async for line in self.inner.aiter_lines():
            yield line

    async def iter_content(self) -> AsyncGenerator[bytes, None]:
        """Asynchronously iterate over the response content."""
        async for chunk in self.inner.aiter_content():
            yield chunk

    async def __aenter__(self):
        """Asynchronously enter the runtime context for the response object."""
        inner: Response = await self.inner
        self.inner = inner
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
        self, method: str, url: str, **kwargs
    ) -> StreamResponse:
        if isinstance(kwargs.get("data"), CurlMime):
            kwargs["multipart"] = kwargs.pop("data")
        """Create and return a StreamResponse object for the given HTTP request."""
        return StreamResponse(super().request(method, url, stream=True, **kwargs))

    # Defining HTTP methods as partial methods of the request method.
    head = partialmethod(request, "HEAD")
    get = partialmethod(request, "GET")
    post = partialmethod(request, "POST")
    put = partialmethod(request, "PUT")
    patch = partialmethod(request, "PATCH")
    delete = partialmethod(request, "DELETE")

class FormData(CurlMime):
    def add_field(self, name, data=None, content_type: str = None, filename: str = None) -> None:
        self.addpart(name, content_type=content_type, filename=filename, data=data)