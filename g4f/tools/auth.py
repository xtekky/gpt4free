from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

from ..providers.types import ProviderType
from .. import debug


@dataclass(frozen=True)
class ProviderAuthStatus:
    """How a provider can be reached for the current request.

    ``source`` is intentionally descriptive so routing can prefer a provider
    the user has configured (``request``/``env``/``provider``/``cache``/
    ``cookie``) over a free provider (``free``) and over one that may still
    need an interactive login (``unknown``).
    """

    available: bool
    source: str
    api_key: Optional[str] = None

    @property
    def configured(self) -> bool:
        return self.source in ("request", "env", "provider", "cache", "cookie")


class AuthManager:
    """Handles API key management and provider authentication detection."""

    aliases = {
        "GeminiPro": "Gemini",
        "OpenaiAPI": "Openai",
        "PuterJS": "Puter",
    }

    @classmethod
    def load_api_key(cls, provider: ProviderType) -> Optional[str]:
        """Load API key from config file"""
        if not provider.needs_auth and not hasattr(provider, "login_url"):
            return None
        provider_name = provider.get_parent()
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key and provider_name in cls.aliases:
            env_var = f"{cls.aliases[provider_name].upper()}_API_KEY"
            api_key = os.environ.get(env_var)
        if api_key:
            debug.log(
                f"Loading API key for {provider_name} from environment variable {env_var}"
            )
            return api_key
        return None

    @classmethod
    def is_provider_api_key(cls, provider: ProviderType, api_key: Optional[str]) -> bool:
        """Return True for a key that belongs to the provider, not to g4f itself."""
        if not api_key or not isinstance(api_key, str):
            return False
        custom_check = getattr(provider, "is_provider_api_key", None)
        if callable(custom_check):
            try:
                return bool(custom_check(api_key))
            except Exception:
                pass
        return not api_key.startswith(("g4f_", "gfs_"))

    @classmethod
    def _class_api_key(cls, provider: ProviderType) -> Optional[str]:
        """Return a key already stored on the provider class, if any."""
        for attr in ("api_key", "_api_key"):
            api_key = getattr(provider, attr, None)
            if cls.is_provider_api_key(provider, api_key):
                return api_key
        return None

    @classmethod
    def resolve_api_key(
        cls,
        provider: ProviderType,
        api_key: Optional[str | dict] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve the best API key for *provider* and report where it came from.

        ``api_key`` may be a single key (as passed by the client) or a mapping
        of provider names to keys.  A g4f/gfs key is not a provider key and is
        ignored for routing decisions.
        """
        provided = None
        if isinstance(api_key, dict):
            provided = api_key.get(provider.get_parent()) or api_key.get(
                provider.__name__
            )
        elif isinstance(api_key, str):
            provided = api_key
        if cls.is_provider_api_key(provider, provided):
            return provided, "request"

        env_key = cls.load_api_key(provider)
        if env_key:
            return env_key, "env"

        class_key = cls._class_api_key(provider)
        if class_key:
            return class_key, "provider"
        return None, None

    @classmethod
    def get_auth_status(
        cls,
        provider: ProviderType,
        api_key: Optional[str | dict] = None,
    ) -> ProviderAuthStatus:
        """Detect whether *provider* is usable without an interactive login."""
        key, source = cls.resolve_api_key(provider, api_key)
        if key:
            return ProviderAuthStatus(True, source or "provider", key)

        # Cached OAuth/access-token results (AsyncAuthedProvider subclasses).
        get_cache_file = getattr(provider, "get_cache_file", None)
        if callable(get_cache_file):
            try:
                if get_cache_file().exists():
                    return ProviderAuthStatus(True, "cache")
            except Exception:
                pass

        # In-memory auth state used by some providers (e.g. OpenaiChat).
        for attr in ("access_token", "request_config"):
            value = getattr(provider, attr, None)
            if attr == "request_config":
                value = getattr(value, "access_token", None)
            if isinstance(value, str) and value:
                return ProviderAuthStatus(True, "provider")
        cookies = getattr(provider, "_cookies", None)
        if cookies:
            try:
                if len(cookies):
                    return ProviderAuthStatus(True, "cookie")
            except TypeError:
                return ProviderAuthStatus(True, "cookie")

        # Only trust cookies that were already loaded; do not scan browsers
        # during routing (that is slow and has side effects).
        try:
            from ..cookies import CookiesConfig

            cookies = CookiesConfig.cookies
            for raw_domain in (
                getattr(provider, "domain", None),
                getattr(provider, "cookie_domain", None),
                getattr(provider, "url", None),
            ):
                if not raw_domain:
                    continue
                domain = (
                    urlparse(raw_domain).hostname
                    if "://" in str(raw_domain)
                    else str(raw_domain)
                )
                if not domain:
                    continue
                domain = domain.lstrip(".")
                for candidate in (raw_domain, domain, f".{domain}"):
                    if cookies.get(candidate):
                        return ProviderAuthStatus(True, "cookie")
        except Exception:
            pass

        if not getattr(provider, "needs_auth", False):
            return ProviderAuthStatus(True, "free")
        return ProviderAuthStatus(False, "unknown")

    @classmethod
    def is_authed(
        cls,
        provider: ProviderType,
        api_key: Optional[str | dict] = None,
    ) -> bool:
        """True when credentials or a usable cached session are present."""
        return cls.get_auth_status(provider, api_key).configured
