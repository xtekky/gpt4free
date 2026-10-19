"""Reverse proxy for a locally running Ollama instance.

Requests under ``/ollama/*`` are forwarded to the local Ollama server, so the
daemon can be used through the same origin (and port) as the g4f API:

    /ollama/api/tags      ->  http://localhost:11434/api/tags
    /ollama/v1/models     ->  http://localhost:11434/v1/models

The target is configured with ``OLLAMA_BASE_URL`` or ``OLLAMA_HOST`` /
``OLLAMA_PORT`` and defaults to ``http://localhost:11434``. Only the configured
local instance is proxied; arbitrary target URLs are not accepted.

``OLLAMA_HOST`` follows the Ollama convention and may carry a scheme and port
(``http://127.0.0.1:11434``); a bare host is accepted too.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import FastAPI

from .local_gateway_proxy import LocalGatewayProxy, env_root_url


def ollama_root_url() -> str:
    """Return the root URL of the local Ollama instance."""
    base_url = os.environ.get("OLLAMA_BASE_URL")
    if base_url:
        return env_root_url(
            "OLLAMA_BASE_URL", "OLLAMA_HOST", "OLLAMA_PORT", "localhost", "11434"
        )
    # OLLAMA_HOST is Ollama's own variable and may include a scheme and port.
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


_proxy = LocalGatewayProxy("ollama", ollama_root_url, label="Ollama")


def register_ollama_routes(app: FastAPI) -> None:
    """Register the ``/ollama`` reverse proxy on ``app``."""
    _proxy.register(app)