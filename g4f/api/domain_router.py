"""Host-based routing for the sandbox: ``<app>.g4f.local`` and ``<port>.g4f.local``.

The sandbox runs several web apps on localhost ports and reaches some of them
through path prefixes on the g4f API (``/remote_desktop``, ``/desktop``). Most
apps cannot be served from a sub-path because they emit absolute URLs or
because their client derives its WebSocket URL from the page origin, so this
module maps a hostname to each app:

    http://omniroute.g4f.local/v1/models   ->  http://127.0.0.1:20128/v1/models
    http://20128.g4f.local/v1/models       ->  http://127.0.0.1:20128/v1/models
    http://desktop.g4f.local/              ->  /desktop/
    http://litellm.g4f.local/v1/models     ->  http://127.0.0.1:4000/v1/models

Apps that already have a path-prefix proxy are reached by rewriting the request
path to that prefix, so the existing proxy (and its WebSocket bridge) does the
forwarding. Apps without one are forwarded by a root-mounted
:class:`~g4f.api.local_gateway_proxy.LocalGatewayProxy`, which relays both HTTP
requests and WebSocket connections.

Only the hostnames in :func:`routes` are routed, and only to the fixed local
ports they name, so the router cannot be used as an open proxy. The suffix
defaults to ``g4f.local`` and can be changed with ``G4F_DOMAIN_SUFFIX``; set
``G4F_DOMAIN_ROUTER=0`` to disable the router entirely.

The names must resolve to the g4f API inside the container; the sandbox
entrypoint seeds ``/etc/hosts`` from :func:`hosts_entries` (see
``docker/sandbox-entrypoint.sh``).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket

from .local_gateway_proxy import LocalGatewayProxy, env_root_url

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "g4f.local"

#: ``<port>.g4f.local``: 4 to 8 digits, as used by the sandbox menu.
_PORT_LABEL_RE = re.compile(r"^[0-9]{4,8}$")


def domain_suffix() -> str:
    """Return the routing suffix (``G4F_DOMAIN_SUFFIX``, default ``g4f.local``)."""
    suffix = (
        os.environ.get("G4F_DOMAIN_SUFFIX", DEFAULT_SUFFIX).strip().strip(".").lower()
    )
    return suffix or DEFAULT_SUFFIX


def domain_router_enabled() -> bool:
    """Return whether host routing is on (``G4F_DOMAIN_ROUTER``, default on)."""
    return os.environ.get("G4F_DOMAIN_ROUTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _local_root_url(
    base_url_var: str,
    host_var: str,
    port_var: str,
    port: int,
    strip_v1: bool = False,
) -> Callable[[], str]:
    """Build a root-URL callable for an app that has no path-prefix proxy."""
    return lambda: env_root_url(
        base_url_var,
        host_var,
        port_var,
        "127.0.0.1",
        str(port),
        strip_v1=strip_v1,
    )

def _ollama_root_url() -> str:
    """Return the root URL of the local Ollama instance.

    ``OLLAMA_HOST`` follows the Ollama convention and may carry a scheme and
    port (``http://127.0.0.1:11434``); a bare host is accepted too.
    """
    if os.environ.get("OLLAMA_BASE_URL"):
        return env_root_url(
            "OLLAMA_BASE_URL", "OLLAMA_HOST", "OLLAMA_PORT", "localhost", "11434"
        )
    host = os.environ.get("OLLAMA_HOST")
    if host:
        parsed = urlsplit(host if "//" in host else f"//{host}")
        if parsed.hostname:
            scheme = parsed.scheme or "http"
            port = parsed.port or int(os.environ.get("OLLAMA_PORT") or 11434)
            return f"{scheme}://{parsed.hostname}:{port}"
    return env_root_url(
        "OLLAMA_BASE_URL", "OLLAMA_HOST", "OLLAMA_PORT", "localhost", "11434"
    )


@dataclass(frozen=True)
class DomainRoute:
    """One hostname-addressable sandbox app."""

    name: str
    label: str
    port: Optional[int] = None
    #: Path prefix of an existing proxy on the g4f app. ``""`` means the app is
    #: the g4f API itself (no rewrite); ``None`` means it has no path proxy and
    #: is forwarded by a root-mounted LocalGatewayProxy.
    prefix: Optional[str] = None
    #: Path the target serves the same content under (see LocalGatewayProxy).
    target_prefix: str = ""
    #: Root URL of the app, used only when ``prefix`` is ``None``.
    root_url: Optional[Callable[[], str]] = None


def routes() -> Dict[str, DomainRoute]:
    """Return the hostname -> app registry."""
    return {
        "g4f": DomainRoute("g4f", "g4f API", port=8080, prefix=""),
        "omniroute": DomainRoute(
            "omniroute",
            "OmniRoute",
            port=20128,
            root_url=_local_root_url(
                "OMNIROUTE_BASE_URL", "OMNIROUTE_HOST", "OMNIROUTE_PORT", 20128
            ),
        ),
        "ollama": DomainRoute(
            "ollama", "Ollama", port=11434, root_url=_ollama_root_url
        ),
        "remote-desktop": DomainRoute(
            "remote-desktop",
            "Remote desktop",
            port=8765,
            prefix="/remote_desktop",
            target_prefix="/remote_desktop",
        ),
        "desktop": DomainRoute("desktop", "Desktop", prefix="/desktop"),
        "litellm": DomainRoute(
            "litellm",
            "LiteLLM",
            port=4000,
            root_url=_local_root_url(
                "LITELLM_BASE_URL", "LITELLM_HOST", "LITELLM_PORT", 4000
            ),
        ),
        "librechat": DomainRoute(
            "librechat",
            "LibreChat",
            port=3080,
            root_url=_local_root_url(
                "LIBRECHAT_BASE_URL", "LIBRECHAT_HOST", "LIBRECHAT_PORT", 3080
            ),
        ),
        "sillytavern": DomainRoute(
            "sillytavern",
            "SillyTavern",
            port=8601,
            root_url=_local_root_url(
                "SILLYTAVERN_BASE_URL", "SILLYTAVERN_HOST", "SILLYTAVERN_PORT", 8601
            ),
        ),
        "cptr": DomainRoute(
            "cptr",
            "Open WebUI Computer",
            port=8000,
            root_url=_local_root_url("CPTR_BASE_URL", "CPTR_HOST", "CPTR_PORT", 8000),
        ),
        "code-server": DomainRoute(
            "code-server",
            "code-server",
            port=8602,
            root_url=_local_root_url(
                "CODE_SERVER_BASE_URL",
                "CODE_SERVER_HOST",
                "CODE_SERVER_PORT",
                8602,
            ),
        ),
        "assistant-ui": DomainRoute(
            "assistant-ui",
            "assistant-ui",
            port=3000,
            root_url=_local_root_url(
                "ASSISTANT_UI_BASE_URL",
                "ASSISTANT_UI_HOST",
                "ASSISTANT_UI_PORT",
                3000,
            ),
        ),
        "llamacpp": DomainRoute(
            "llamacpp",
            "llama.cpp",
            port=11435,
            root_url=_local_root_url(
                "LLAMACPP_BASE_URL", "LLAMACPP_HOST", "LLAMACPP_PORT", 11435
            ),
        ),
        "pi-freeflow": DomainRoute(
            "pi-freeflow",
            "pi-freeflow",
            port=28180,
            root_url=_local_root_url(
                "PI_FREEFLOW_BASE_URL",
                "PI_FREEFLOW_HOST",
                "PI_FREEFLOW_PORT",
                28180,
            ),
        ),
        "lmstudio": DomainRoute(
            "lmstudio",
            "LM Studio",
            port=1234,
            root_url=_local_root_url(
                "LMSTUDIO_BASE_URL", "LMSTUDIO_HOST", "LMSTUDIO_PORT", 1234
            ),
        ),
    }


def hosts_entries(suffix: Optional[str] = None) -> List[str]:
    """Return ``/etc/hosts`` lines mapping every routed host to 127.0.0.1."""
    suffix = suffix or domain_suffix()
    registry = routes()
    names = sorted({route.name for route in registry.values()})
    ports = sorted({route.port for route in registry.values() if route.port})
    return [f"127.0.0.1 {name}.{suffix}" for name in names] + [
        f"127.0.0.1 {port}.{suffix}" for port in ports
    ]


class DomainRouterMiddleware:
    """Route ``<app>.g4f.local`` / ``<port>.g4f.local`` to the matching app.

    A pure ASGI middleware rather than a route or ``BaseHTTPMiddleware``: it
    must see WebSocket scopes too (the remote-desktop signaling socket and the
    noVNC bridge both live behind a hostname), and it must rewrite the path
    before Starlette's router picks a route.
    """

    def __init__(self, app: ASGIApp, suffix: str = DEFAULT_SUFFIX) -> None:
        self.app = app
        self.suffix = suffix.strip().strip(".").lower() or DEFAULT_SUFFIX
        self.routes = routes()
        self._by_port = {
            route.port: route for route in self.routes.values() if route.port
        }
        self._proxies: Dict[str, LocalGatewayProxy] = {}

    def _match(self, host: str) -> Optional[DomainRoute]:
        """Return the route for ``host``, or ``None`` when it is not routed."""
        host = host.split(":", 1)[0].strip().lower().rstrip(".")
        prefix = f".{self.suffix}"
        if not host.endswith(prefix):
            return None
        label = host[: -len(prefix)]
        if not label or "." in label:
            return None
        route = self.routes.get(label)
        if route is not None:
            return route
        if _PORT_LABEL_RE.match(label):
            return self._by_port.get(int(label))
        return None

    def _proxy_for(self, route: DomainRoute) -> LocalGatewayProxy:
        proxy = self._proxies.get(route.name)
        if proxy is None:
            proxy = LocalGatewayProxy(
                "",
                route.root_url,
                label=route.label,
                target_prefix=route.target_prefix,
            )
            self._proxies[route.name] = proxy
        return proxy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = ""
        for key, value in scope.get("headers") or ():
            if key == b"host":
                host = value.decode("latin-1")
                break
        route = self._match(host) if host else None
        # An empty prefix marks the g4f API itself: nothing to rewrite.
        if route is None or route.prefix == "":
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or "/"
        if route.prefix is not None:
            # Reuse the app's existing path-prefix proxy (and its WebSocket
            # bridge) by rewriting the path it will see.
            new_path = f"{route.prefix}{path}"
            scope = dict(scope)
            scope["path"] = new_path
            scope["raw_path"] = new_path.encode("utf-8")
            logger.debug("Domain router: %s%s -> %s", host, path, new_path)
            await self.app(scope, receive, send)
            return

        proxy = self._proxy_for(route)
        logger.debug("Domain router: %s%s -> %s", host, path, route.label)
        if scope["type"] == "websocket":
            await proxy.forward_websocket(
                WebSocket(scope, receive, send), path.lstrip("/")
            )
            return
        response = await proxy.forward(Request(scope, receive), path.lstrip("/"))
        await response(scope, receive, send)


def register_domain_router(app) -> None:
    """Install the host router on ``app`` unless ``G4F_DOMAIN_ROUTER=0``."""
    if not domain_router_enabled():
        logger.info("Host routing disabled (G4F_DOMAIN_ROUTER=0)")
        return
    suffix = domain_suffix()
    app.add_middleware(DomainRouterMiddleware, suffix=suffix)
    logger.info(
        "Host routing enabled for *.%s (%s)",
        suffix,
        ", ".join(sorted(routes())),
    )