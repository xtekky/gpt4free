"""Reusable reverse proxy for locally running OpenAI-compatible gateways.

Several local AI tools (OmniRoute, Ollama, LM Studio, llama.cpp, ...) expose an
HTTP API on localhost. Proxying them through the g4f API lets a browser reach
them on the same origin (and port) as g4f itself, which avoids CORS setup and
mixed-content problems:

    /<prefix>/v1/models  ->  http://localhost:<port>/v1/models

Only the configured local instance is proxied; arbitrary target URLs are never
accepted, so the route cannot be used as an open proxy.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
_MAX_BODY_SIZE = 64 * 1024 * 1024  # 64 MB
_CHUNK_SIZE = 64 * 1024
_DROP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "g4f-api-key",
}
_DROP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    # The g4f CORS middleware owns these for the browser-facing response.
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-expose-headers",
    "access-control-allow-credentials",
    # Re-added from getall() below so multiple Set-Cookie headers survive.
    "set-cookie",
}


def env_root_url(
    base_url_var: str,
    host_var: str,
    port_var: str,
    default_host: str,
    default_port: str,
    strip_v1: bool = True,
) -> str:
    """Build a gateway root URL from ``<base_url_var>`` or host/port variables.

    ``<base_url_var>`` wins when set. A trailing ``/v1`` is stripped from it
    (unless ``strip_v1`` is false) because the proxy forwards the full path
    itself, so the prefix must not be baked into the target.
    """
    base_url = os.environ.get(base_url_var)
    if base_url:
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        if strip_v1 and path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit(
            (parsed.scheme or "http", parsed.netloc, path, "", "")
        ).rstrip("/")
    host = os.environ.get(host_var) or default_host
    port = os.environ.get(port_var) or default_port
    return f"http://{host}:{port}"


class LocalGatewayProxy:
    """Forward ``/<prefix>/*`` requests to a local gateway instance."""

    def __init__(
        self,
        prefix: str,
        root_url: Callable[[], str],
        label: Optional[str] = None,
        target_prefix: str = "",
    ) -> None:
        self.prefix = "/" + prefix.strip("/")
        self.root_url = root_url
        self.label = label or prefix.strip("/")
        #: Path the target serves the same content under. Empty means the
        #: target's root corresponds to the proxy prefix.
        self.target_prefix = "/" + target_prefix.strip("/") if target_prefix.strip("/") else ""

    def _rewrite_location(self, location: str, root_url: str) -> str:
        """Keep redirects inside the proxy prefix."""
        if location.startswith(self.prefix):
            return location
        if location.startswith("/"):
            return f"{self.prefix}{location}"
        parsed = urlsplit(location)
        root = urlsplit(root_url)
        if parsed.scheme in ("http", "https") and parsed.netloc == root.netloc:
            path = parsed.path or "/"
            return urlunsplit(
                ("", "", f"{self.prefix}{path}", parsed.query, parsed.fragment)
            )
        return location

    async def forward(self, request: Request, path: str):
        root_url = self.root_url()
        target_path = f"{self.target_prefix}/{path}" if path else (self.target_prefix or "")
        target_url = f"{root_url}{target_path}" if target_path else f"{root_url}/"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        body = await request.body()
        if len(body) > _MAX_BODY_SIZE:
            return JSONResponse(
                {"error": {"message": "Request body too large"}},
                status_code=413,
            )

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
        }
        # Standard reverse-proxy headers: the target may need the origin the
        # browser actually used, e.g. to build share links that stay reachable.
        headers["x-forwarded-host"] = request.headers.get("host", "")
        headers["x-forwarded-proto"] = request.url.scheme
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_read=None)
        session = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)
        try:
            upstream = await session.request(
                request.method,
                target_url,
                headers=headers,
                data=body if body else None,
                allow_redirects=False,
            )
        except (aiohttp.ClientError, OSError) as e:
            await session.close()
            logger.warning("%s proxy: %s is not reachable: %s", self.label, root_url, e)
            return JSONResponse(
                {
                    "error": {
                        "message": f"{self.label} is not running at {root_url}",
                        "type": f"{self.label.lower()}_unavailable",
                        "detail": str(e),
                    }
                },
                status_code=502,
            )
        except Exception:
            await session.close()
            raise

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _DROP_RESPONSE_HEADERS
        }
        if "location" in upstream.headers:
            # Header names are case-insensitive; drop any original spelling
            # before adding the rewritten Location.
            response_headers = {
                key: value
                for key, value in response_headers.items()
                if key.lower() != "location"
            }
            response_headers["location"] = self._rewrite_location(
                upstream.headers["location"], root_url
            )

        async def stream_response():
            try:
                async for chunk in upstream.content.iter_chunked(_CHUNK_SIZE):
                    yield chunk
            finally:
                upstream.release()
                await session.close()

        response = StreamingResponse(
            stream_response(),
            status_code=upstream.status,
            headers=response_headers,
        )
        for cookie in upstream.headers.getall("Set-Cookie", []):
            response.raw_headers.append((b"set-cookie", cookie.encode("latin-1")))
        return response

    def register(self, app: FastAPI) -> None:
        """Register ``<prefix>`` and ``<prefix>/{path}`` on ``app``."""
        prefix = self.prefix

        @app.api_route(prefix, methods=_METHODS, include_in_schema=False)
        async def gateway_root(request: Request):
            return await self.forward(request, "")

        @app.api_route(
            f"{prefix}/{{path:path}}", methods=_METHODS, include_in_schema=False
        )
        async def gateway_path(path: str, request: Request):
            return await self.forward(request, path)

    def register_extra_routes(
        self,
        app: FastAPI,
        routes: Iterable[tuple],
    ) -> None:
        """Register gateway web-UI routes that live outside the prefix.

        ``routes`` holds ``(route_path, methods, target_path)`` tuples; the
        target path is forwarded relative to the gateway root. Used for the
        OmniRoute dashboard, whose assets are requested from ``/``. Path
        parameters in ``route_path`` (e.g. ``{path:path}``) are substituted
        into ``target_path``.
        """
        for route_path, methods, target_path in routes:

            def make_handler(target: str):
                async def handler(request: Request):
                    path = target
                    for name, value in request.path_params.items():
                        path = path.replace(f"{{{name}}}", str(value))
                    return await self.forward(request, path)

                return handler

            app.api_route(
                route_path,
                methods=list(methods),
                include_in_schema=False,
            )(make_handler(target_path))