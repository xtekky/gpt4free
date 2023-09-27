from __future__ import annotations

import json, sys
from functools import partialmethod

from aiohttp import StreamReader
from aiohttp.base_protocol import BaseProtocol

from curl_cffi.requests import AsyncSession as BaseSession
from curl_cffi.requests import Response


class StreamResponse:
    def __init__(self, inner: Response, content: StreamReader, request):
        self.inner = inner
        self.content = content
        self.request = request
        self.status_code = inner.status_code
        self.reason = inner.reason
        self.ok = inner.ok
        self.headers = inner.headers
        self.cookies = inner.cookies

    async def text(self) -> str:
        content = await self.content.read()
        return content.decode()

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP Error {self.status_code}: {self.reason}")

    async def json(self, **kwargs):
        return json.loads(await self.content.read(), **kwargs)

class StreamRequest:
    def __init__(self, session: AsyncSession, method: str, url: str, **kwargs):
        self.session = session
        self.loop = session.loop
        self.content = StreamReader(
            BaseProtocol(session.loop),
            sys.maxsize,
            loop=session.loop
        )
        self.method = method
        self.url = url
        self.options = kwargs

    def on_content(self, data):
        if not self.enter.done():
            self.enter.set_result(None)
        self.content.feed_data(data)

    def on_done(self, task):
        self.content.feed_eof()
        self.curl.clean_after_perform()
        self.curl.reset()
        self.session.push_curl(self.curl)   

    async def __aenter__(self) -> StreamResponse:
        self.curl = await self.session.pop_curl()
        self.enter = self.loop.create_future()
        request, _, header_buffer = self.session._set_curl_options(
            self.curl,
            self.method,
            self.url,
            content_callback=self.on_content,
            **self.options
        )
        await self.session.acurl.add_handle(self.curl, False)
        self.handle = self.session.acurl._curl2future[self.curl]
        self.handle.add_done_callback(self.on_done)
        await self.enter
        return StreamResponse(
            self.session._parse_response(self.curl, request, _, header_buffer),
            self.content,
            request
        )
            
    async def __aexit__(self, exc_type, exc, tb):
        pass

class AsyncSession(BaseSession):
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