from __future__ import annotations

from aiohttp import ClientSession, ClientResponse, ClientTimeout, BaseConnector, FormData
from typing import AsyncIterator, Any, Optional

from .defaults import DEFAULT_HEADERS
from ..errors import MissingRequirementsError

class StreamResponse(ClientResponse):
    async def iter_lines(self) -> AsyncIterator[bytes]:
        async for line in self.content:
            yield line.rstrip(b"\r\n")

    async def iter_content(self) -> AsyncIterator[bytes]:
        async for chunk in self.content.iter_any():
            yield chunk

    async def json(self, content_type: str = None) -> Any:
        return await super().json(content_type=content_type)

class StreamSession(ClientSession):
    def __init__(
        self,
        headers: dict = {},
        timeout: int = None,
        connector: BaseConnector = None,
        proxy: str = None,
        proxies: dict = {},
        impersonate = None,
        **kwargs
    ):
        if impersonate:
            headers = {
                **DEFAULT_HEADERS,
                **headers
            }
        connect = None
        if isinstance(timeout, tuple):
            connect, timeout = timeout;
        if timeout is not None:
            timeout = ClientTimeout(timeout, connect)
        if proxy is None:
            proxy = proxies.get("all", proxies.get("https"))
        super().__init__(
            **kwargs,
            timeout=timeout,
            response_class=StreamResponse,
            connector=get_connector(connector, proxy),
            headers=headers
        )

def get_connector(connector: BaseConnector = None, proxy: str = None, rdns: bool = False) -> Optional[BaseConnector]:
    if proxy and not connector:
        try:
            from aiohttp_socks import ProxyConnector
            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
                rdns = True
            connector = ProxyConnector.from_url(proxy, rdns=rdns)
        except ImportError:
            raise MissingRequirementsError('Install "aiohttp_socks" package for proxy support')
    return connector