from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from ..template import OpenaiTemplate

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "28180"
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1"

TOKEN_ENV_VARS = (
    "FREEFLOW_TOKEN",
    "FREEFLOW_API_KEY",
    "G4F_FREEFLOW_TOKEN",
    "PI_FREEFLOW_API_KEY",
)


def _normalize_base_url(value: str) -> str:
    """Return an OpenAI-compatible base URL with a ``/v1`` suffix."""
    value = value.rstrip("/")
    path = urlsplit(value).path.rstrip("/")
    if not path.endswith("/v1"):
        value += "/v1"
    return value


def get_base_url() -> str:
    """Return the base URL of the local pi-freeflow proxy.

    ``PI_FREEFLOW_BASE_URL`` (also accepted as ``FREEFLOW_BASE_URL``) wins.
    Both the full ``.../v1`` and the daemon root URL are accepted. Otherwise
    the sandbox default, ``http://127.0.0.1:28180/v1``, is used.
    """
    base_url = os.environ.get("PI_FREEFLOW_BASE_URL") or os.environ.get(
        "FREEFLOW_BASE_URL"
    )
    if base_url:
        return _normalize_base_url(base_url)
    host = os.environ.get("PI_FREEFLOW_HOST") or DEFAULT_HOST
    port = os.environ.get("PI_FREEFLOW_PORT") or DEFAULT_PORT
    return f"http://{host}:{port}/v1"


def get_token_file() -> Path:
    """Return the path of the token written by the pi-freeflow daemon."""
    configured = os.environ.get("PI_FREEFLOW_TOKEN_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".pi" / "agent" / "token"


def get_api_key() -> Optional[str]:
    """Return the pi-freeflow bearer token, when one is configured.

    The daemon shares ``~/.pi/agent/token`` with Pi/OMP sessions. Environment
    variables win so a sandbox or container can inject the token directly.
    """
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    try:
        value = get_token_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


class PiFreeflow(OpenaiTemplate):
    """Provider for the local pi-freeflow gateway (default port 28180)."""

    label = "pi-freeflow"
    url = get_base_url()[: -len("/v1")]
    base_url = get_base_url()
    health_url = f"{base_url}/models"
    working = True
    active_by_default = False
    needs_auth = False
    sort_models = False

    default_model: Optional[str] = None
    models: list[str] = []

    @classmethod
    def get_models(
        cls, api_key: str = None, base_url: str = None, timeout: int = 2, **kwargs
    ) -> list[str]:
        if cls.models:
            return list(cls.models)
        if api_key is None:
            api_key = get_api_key()
        if base_url is None:
            base_url = get_base_url()
        try:
            models = super().get_models(
                api_key=api_key, base_url=base_url, timeout=timeout, **kwargs
            )
        except Exception:
            # The proxy is local and may not be running. Stay silent so the
            # provider can be retried once the daemon starts.
            return []
        models = list(models or [])
        if models and not cls.default_model:
            cls.default_model = models[0]
        cls.models = models
        return list(cls.models)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages,
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = get_api_key()
        if base_url is None:
            base_url = get_base_url()
        async for chunk in super().create_async_generator(
            model,
            messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ):
            yield chunk

    @classmethod
    async def get_health(cls, api_key: str = None, **kwargs) -> dict:
        if api_key is None:
            api_key = get_api_key()
        return await cls.probe_health(f"{get_base_url()}/models", api_key)


# ``Freeflow`` is a convenient short alias for callers that do not use the
# ``Pi`` prefix.
Freeflow = PiFreeflow
