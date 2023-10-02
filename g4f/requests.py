from __future__ import annotations

import warnings, json, asyncio

from functools import partialmethod
from asyncio import Future, Queue
from typing import AsyncGenerator

from curl_cffi.requests import AsyncSession, Response

import curl_cffi

is_newer_0_5_8 = hasattr(AsyncSession, "_set_cookies") or hasattr(curl_cffi.requests.Cookies, "get_cookies_for_curl")
is_newer_0_5_9 = hasattr(curl_cffi.AsyncCurl, "remove_handle")
is_newer_0_5_10 = hasattr(AsyncSession, "release_curl")

class StreamResponse:
    def __init__(self, inner: Response, queue: Queue):
        self.inner = inner
        self.queue = queue
        self.request = inner.request
        self.status_code = inner.status_code
        self.reason = inner.reason
        self.ok = inner.ok
        self.headers = inner.headers
        self.cookies = inner.cookies

    async def text(self) -> str:
        content = await self.read()
        return content.decode()

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP Error {self.status_code}: {self.reason}")

    async def json(self, **kwargs):
        return json.loads(await self.read(), **kwargs)
    
    async def iter_lines(self, chunk_size=None, decode_unicode=False, delimiter=None) -> AsyncGenerator[bytes]:
        """
        Copied from: https://requests.readthedocs.io/en/latest/_modules/requests/models/
        which is under the License: Apache 2.0
        """
        pending = None

        async for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):
            if pending is not None:
                chunk = pending + chunk
            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()
            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                yield line

        if pending is not None:
            yield pending

    async def iter_content(self, chunk_size=None, decode_unicode=False) -> As:
        if chunk_size:
            warnings.warn("chunk_size is ignored, there is no way to tell curl that.")
        if decode_unicode:
            raise NotImplementedError()
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                return
            yield chunk

    async def read(self) -> bytes:
        return b"".join([chunk async for chunk in self.iter_content()])

class StreamRequest:
    def __init__(self, session: AsyncSession, method: str, url: str, **kwargs):
        self.session = session
        self.loop = session.loop if session.loop else asyncio.get_running_loop()
        self.queue = Queue()
        self.method = method
        self.url = url
        self.options = kwargs
        self.handle = None

    def _on_content(self, data):
        if not self.enter.done():
            self.enter.set_result(None)
        self.queue.put_nowait(data)

    def _on_done(self, task: Future):
        if not self.enter.done():
            self.enter.set_result(None)
        self.queue.put_nowait(None)

        self.loop.call_soon(self.release_curl)

    async def fetch(self) -> StreamResponse:
        if self.handle:
            raise RuntimeError("Request already started")
        self.curl = await self.session.pop_curl()
        self.enter = self.loop.create_future()
        if is_newer_0_5_10:
            request, _, header_buffer, _, _ = self.session._set_curl_options(
                self.curl,
                self.method,
                self.url,
                content_callback=self._on_content,
                **self.options
            )
        else:
            request, _, header_buffer = self.session._set_curl_options(
                self.curl,
                self.method,
                self.url,
                content_callback=self._on_content,
                **self.options
            )
        if is_newer_0_5_9:
             self.handle = self.session.acurl.add_handle(self.curl)
        else:
            await self.session.acurl.add_handle(self.curl, False)
            self.handle = self.session.acurl._curl2future[self.curl]
        self.handle.add_done_callback(self._on_done)
        # Wait for headers
        await self.enter
        # Raise exceptions
        if self.handle.done():
            self.handle.result()
        if is_newer_0_5_8:
            response = self.session._parse_response(self.curl, _, header_buffer)
            response.request = request
        else:
            response = self.session._parse_response(self.curl, request, _, header_buffer)
        return StreamResponse(
            response,
            self.queue
        )
    
    async def __aenter__(self) -> StreamResponse:
        return await self.fetch()

    async def __aexit__(self, *args):
        self.release_curl()

    def release_curl(self):
        if is_newer_0_5_10:
            self.session.release_curl(self.curl)
            return
        if not self.curl:
            return
        self.curl.clean_after_perform()
        if is_newer_0_5_9:
            self.session.acurl.remove_handle(self.curl)
        elif not self.handle.done() and not self.handle.cancelled():
            self.session.acurl.set_result(self.curl)
        self.curl.reset()
        self.session.push_curl(self.curl)
        self.curl = None

class StreamSession(AsyncSession):
    def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> StreamRequest:
        return StreamRequest(self, method, url, **kwargs)
    
    head = partialmethod(request, "HEAD")
    get = partialmethod(request, "GET")
    post = partialmethod(request, "POST")
    put = partialmethod(request, "PUT")
    patch = partialmethod(request, "PATCH")
    delete = partialmethod(request, "DELETE")