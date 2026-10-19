from __future__ import annotations

import os

from ..template import OpenaiTemplate


def get_base_url() -> str:
    """Return the OpenAI-compatible base URL of the local OmniRoute instance."""
    base_url = os.environ.get("OMNIROUTE_BASE_URL")
    if base_url:
        return base_url.rstrip("/")
    host = os.environ.get("OMNIROUTE_HOST") or "localhost"
    port = os.environ.get("OMNIROUTE_PORT") or "20128"
    return f"http://{host}:{port}/v1"


class OmniRoute(OpenaiTemplate):
    """Provider for a locally running OmniRoute gateway (default port 20128)."""

    label = "OmniRoute"
    url = "http://localhost:20128"
    base_url = get_base_url()
    health_url = f"{base_url}/models"
    working = True
    active_by_default = False
    needs_auth = False
    sort_models = False

    default_model = "auto"
    models: list[str] = []

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> list[str]:
        if cls.models:
            return cls.models
        timeout = kwargs.pop("timeout", 2)
        try:
            models = super().get_models(api_key=api_key, timeout=timeout, **kwargs)
        except Exception:
            # OmniRoute is a local service: stay silent when it is not running.
            return [cls.default_model]
        models = list(models or [])
        if cls.default_model not in models:
            models.insert(0, cls.default_model)
        cls.models = models
        return cls.models
