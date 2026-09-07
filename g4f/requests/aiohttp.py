from __future__ import annotations

import json
import asyncio
import weakref
from aiohttp import (
    ClientSession,
    ClientResponse,
    ClientTimeout,
    BaseConnector,
    TCPConnector,
    FormData,
)
from typing import AsyncIterator, Any, Optional

from .defaults import DEFAULT_HEADERS, has_brotli
from ..errors import MissingRequirementsError

_loop_connectors: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, TCPConnector] = weakref.WeakKeyDictionary()


def get_shared_connector() -> Optional[TCPConnector]:
    """Retrieve or create a loop-bound TCPConnector with keep-alive and DNS caching."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    connector = _loop_connectors.get(loop)
    if connector is None or connector.closed:
        connector = TCPConnector(
            limit=100,
            limit_per_host=20,
            keepalive_timeout=30,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        _loop_connectors[loop] = connector
    return connector


async def close_shared_connectors() -> None:
    """Close all open shared connectors."""
    for connector in list(_loop_connectors.values()):
        if not connector.closed:
            await connector.close()
    _loop_connectors.clear()


class StreamResponse(ClientResponse):
    async def iter_lines(self) -> AsyncIterator[bytes]:
        async for line in self.content:
            yield line.rstrip(b"\r\n")

    async def iter_content(self) -> AsyncIterator[bytes]:
        async for chunk in self.content.iter_any():
            yield chunk

    async def json(self, content_type: str = None) -> Any:
        return await super().json(content_type=content_type)

    async def sse(self) -> AsyncIterator[dict]:
        """Asynchronously iterate over the Server-Sent Events of the response."""
        async for line in self.content:
            if line.startswith(b"data: "):
                chunk = line[6:]
                if chunk.startswith(b"[DONE]"):
                    break
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    continue


class StreamSession:
    def __init__(
        self,
        headers=None,
        timeout: int = None,
        connector: BaseConnector = None,
        proxy: str = None,
        proxies=None,
        impersonate=None,
        connector_owner: bool = None,
        **kwargs,
    ):
        if proxies is None:
            proxies = {}
        if headers is None:
            headers = {}
        if impersonate:
            headers = {**DEFAULT_HEADERS, **headers}
        if not has_brotli and "br" in headers.get("accept-encoding", ""):
            headers["accept-encoding"] = "gzip, deflate"
        connect = None
        if isinstance(timeout, tuple):
            connect, timeout = timeout
        if timeout is not None:
            timeout = ClientTimeout(timeout, connect)
        if proxy is None:
            proxy = proxies.get("all", proxies.get("https"))

        actual_connector = get_connector(connector, proxy)
        if actual_connector is None and not proxy:
            actual_connector = get_shared_connector()
            if connector_owner is None and actual_connector is not None:
                connector_owner = False
        if connector_owner is None:
            connector_owner = True

        self.inner = ClientSession(
            **kwargs,
            timeout=timeout,
            response_class=StreamResponse,
            connector=actual_connector,
            connector_owner=connector_owner,
            headers=headers,
        )

    async def __aenter__(self) -> ClientSession:
        return self.inner

    async def __aexit__(self, *args, **kwargs) -> None:
        await self.inner.close()


def get_connector(
    connector: BaseConnector = None, proxy: str = None, rdns: bool = False
) -> Optional[BaseConnector]:
    if proxy and not connector:
        try:
            from aiohttp_socks import ProxyConnector

            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
                rdns = True
            connector = ProxyConnector.from_url(proxy, rdns=rdns)
        except ImportError:
            raise MissingRequirementsError(
                'Install "aiohttp_socks" package for proxy support'
            )
    return connector
