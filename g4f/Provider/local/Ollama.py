from __future__ import annotations

import re
import json
import requests
import os
from typing import Optional

from ..template import OpenaiTemplate
from ...requests import StreamSession, raise_for_status
from ...providers.response import Usage, Reasoning
from ...cookies import get_cookies
from ...tools.run_tools import AuthManager
from ...typing import AsyncResult, Messages
from ...config import AppConfig
from ...errors import MissingAuthError
from ... import debug

class Ollama(OpenaiTemplate):
    label = "Ollama 🦙"
    url = "https://ollama.com"
    base_url = "https://g4f.space/api/ollama"
    login_url = "https://ollama.com/settings/keys"
    needs_auth = False
    working = True
    active_by_default = True
    local_models: list[str] = []
    model_aliases = {
        "gpt-oss-120b": "gpt-oss:120b",
        "gpt-oss-20b": "gpt-oss:20b"
    }
    default_model = "nemotron-3-super"

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None) -> Optional[dict]:
        session_cookie = get_cookies("ollama.com", cache_result=False).get("__Secure-session")
        if not session_cookie:
            return await super().get_quota(api_key=api_key)
        quota = {}
        try:
            async with StreamSession() as session:
                async with session.get(
                    "https://ollama.com/settings",
                    cookies={"__Secure-session": session_cookie},
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                ) as response:
                    await raise_for_status(response)
                    html = await response.text()
                    lower = html.lower()
                    has_sign_in = "sign in to ollama" in lower or "log in to ollama" in lower
                    has_auth_endpoint = "/api/auth/signin" in lower or "/auth/signin" in lower or 'href="/signin"' in lower or 'href="/login"' in lower
                    has_form = "<form" in lower
                    has_password = 'type="password"' in lower or 'name="password"' in lower
                    has_email = 'type="email"' in lower or 'name="email"' in lower
                    if (has_sign_in and has_form and (has_email or has_password or has_auth_endpoint)) \
                            or (has_form and has_auth_endpoint) \
                            or (has_form and has_password and has_email):
                        raise MissingAuthError("Ollama session cookie is invalid or expired.")
                    for label in ("Session usage", "Hourly usage", "Weekly usage"):
                        idx = html.find(label)
                        if idx == -1:
                            continue
                        section = html[idx + len(label):idx + len(label) + 800]
                        pct_match = re.search(r'(\d+(?:\.\d+)?)%\s*used', section)
                        if not pct_match:
                            width_match = re.search(r'width:\s*(\d+(?:\.\d+)?)%', section)
                            if width_match:
                                pct_match = width_match
                        pct = float(pct_match.group(1)) if pct_match else None
                        reset_match = re.search(r'data-time=["\']([^"\']+)["\']', section)
                        reset_time = reset_match.group(1) if reset_match else None
                        key = label.lower().replace(" ", "_")
                        quota[key] = {
                            "used_percent": pct,
                            "reset_time": reset_time,
                        }
                    match = re.search(r'<span class="text-sm">Premium requests</span>\s*<span class="text-sm">(\d+)/(\d+) used</span>\s*</div>', html)
                    if match:
                        used = int(match.group(1))
                        total = int(match.group(2))
                        pct = (used / total) * 100 if total > 0 else None
                        quota["premium_requests"] = {
                            "used": used,
                            "total": total,
                            "used_percent": pct,
                        }
        except Exception as e:
            raise RuntimeError(f"Failed to get quota information from Ollama: {e}")
        if not quota:
            raise RuntimeError("Failed to find quota information in Ollama settings page.")
        return quota

    @classmethod
    def get_models(cls, api_key: str = None, base_url: str = None, **kwargs):
        if not cls.models:
            cls.models = []
            if not api_key or AppConfig.disable_custom_api_key:
                api_key = AuthManager.load_api_key(cls)
            models = requests.get("https://ollama.com/api/tags").json()["models"]
            if models:
                cls.live += 1
            cls.models = [model["name"] for model in models]
            if base_url is None:
                host = os.getenv("OLLAMA_HOST", "localhost")
                port = os.getenv("OLLAMA_PORT", "11434")
                url = f"http://{host}:{port}/api/tags"
            else:
                url = base_url.replace("/v1", "/api/tags")
            try:
                models = requests.get(url).json()["models"]
            except requests.exceptions.RequestException as e:
                return cls.models
            if cls.live == 0 and models:
                cls.live += 1
            cls.local_models = [model["name"] for model in models]
            cls.models = cls.models.copy() + cls.local_models
            cls.default_model = next(iter(cls.models), None)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        base_url: str = None,
        **kwargs
    ) -> AsyncResult:
        if not cls.models:
            cls.get_models(api_key=api_key, base_url=base_url)
        if model in cls.local_models:
            if base_url is None:
                host = os.getenv("OLLAMA_HOST", "localhost")
                port = os.getenv("OLLAMA_PORT", "11434")
                base_url: str = f"http://{host}:{port}/v1"
        else:
            base_url = cls.backup_url
        async for chunk in super().create_async_generator(
            model,
            messages,
            api_key=api_key,
            base_url=cls.backup_url,
            **kwargs
        ):
            yield chunk