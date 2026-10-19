"""Reverse proxy for a locally running OmniRoute instance.

Requests under ``/omniroute/*`` are forwarded to the local OmniRoute server,
so the gateway can be used through the same origin (and port) as the g4f API:

    /omniroute/v1/models  ->  http://localhost:20128/v1/models

The target is configured with ``OMNIROUTE_BASE_URL`` or ``OMNIROUTE_HOST`` /
``OMNIROUTE_PORT`` and defaults to ``http://localhost:20128``. Only the
configured local instance is proxied; arbitrary target URLs are not accepted.
"""

from __future__ import annotations

from fastapi import FastAPI

from .local_gateway_proxy import LocalGatewayProxy, env_root_url

# OmniRoute's dashboard is served from the gateway root, so its assets are
# requested outside the /omniroute prefix and need their own routes.
_EXTRA_ROUTES = [
    ("/_next/{path:path}", ["GET"], "_next/{path}"),
    ("/dashboard", ["GET"], "dashboard"),
    ("/dashboard/{path:path}", ["GET"], "dashboard/{path}"),
    ("/home", ["GET"], "home"),
    ("/api/auth/login", ["POST"], "api/auth/login"),
    ("/forgot-password", ["GET"], "forgot-password"),
    ("/dashboard", ["GET"], "dashboard"),
    ("/home", ["GET"], "home"),
    ("/forgot-password", ["GET"], "forgot-password"),
    ("/api/v1/ws", ["GET"], "api/v1/ws"),
    ("/api/provider-metrics", ["GET"], "api/provider-metrics"),
    ("/api/health/ping", ["GET"], "api/health/ping"),
    ("/api/usage/call-logs", ["GET"], "api/usage/call-logs"),
    ("/api/sync/cloud", ["GET"], "api/sync/cloud"),
    ("/api/token-health", ["GET"], "api/token-health"),
    ("/api/health/degradation", ["GET"], "api/health/degradation")
]


def omniroute_root_url() -> str:
    """Return the root URL of the local OmniRoute instance."""
    return env_root_url(
        "OMNIROUTE_BASE_URL", "OMNIROUTE_HOST", "OMNIROUTE_PORT", "localhost", "20128"
    )


_proxy = LocalGatewayProxy("omniroute", omniroute_root_url, label="OmniRoute")


def register_omniroute_routes(app: FastAPI) -> None:
    """Register the ``/omniroute`` reverse proxy on ``app``."""
    _proxy.register(app)
    _proxy.register_extra_routes(app, _EXTRA_ROUTES)