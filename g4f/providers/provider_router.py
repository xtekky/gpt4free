from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Iterable, Optional, Sequence, Tuple, Type

from .. import debug
from ..tools.auth import AuthManager, ProviderAuthStatus
from .retry_provider import ProviderCircuitBreaker

# Provider model lists are discovered lazily.  The cache keeps a request for an
# unknown model from hitting every gateway on every call.
MODELS_TTL = float(os.environ.get("G4F_PROVIDER_MODELS_TTL", "300"))
FAILURE_TTL = 60.0
DEFAULT_PROBE_TIMEOUT = 4.0

# Provider classes that are routing plumbing rather than network backends.
_SKIP_PROVIDERS = {"AnyProvider", "DefaultProvider", "AgentTools"}

_AUTH_SCORES = {
    "request": 5,
    "env": 4,
    "provider": 4,
    "cache": 4,
    "cookie": 3,
    "free": 2,
    "unknown": 0,
}


def _normalize_model(name: str) -> str:
    """Loose comparison key: vendor prefix, tag and punctuation insensitive."""
    if not name:
        return ""
    return (
        str(name)
        .split("/")[-1]
        .split(":")[0]
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )


def _as_model_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, dict):
        value = list(value.keys())
    if isinstance(value, str):
        value = [value]
    return [model for model in value if isinstance(model, str) and model]


def _static_models(provider: Type) -> list[str]:
    """Models a provider advertises without doing network I/O."""
    result: list[str] = []
    for value in (
        getattr(provider, "models", None),
        getattr(provider, "fallback_models", None),
        getattr(provider, "default_model", None),
    ):
        for model in _as_model_list(value):
            if model not in result:
                result.append(model)
    return result


def _is_media_only(provider: Type) -> bool:
    """True for providers that should not be auto-selected for text."""
    name = getattr(provider, "__name__", "").lower()
    module = getattr(provider, "__module__", "").lower()
    if any(f".{kind}." in module for kind in ("audio", "image", "video")):
        return True
    return name.endswith(("image", "images", "audio", "tts", "voice", "video"))


class ProviderRouter:
    """Rank and discover providers for a model request.

    The router is deliberately side-effect-light: it only reads provider
    metadata and a small cache of model lists.  Network calls happen during
    discovery, with a timeout and a cache, so the common path stays fast.
    """

    _models_cache: dict[str, Tuple[float, Tuple[str, ...]]] = {}

    @classmethod
    def reset_cache(cls) -> None:
        cls._models_cache.clear()

    @classmethod
    def is_ignored(cls, provider: Type, ignored: Iterable[str] = ()) -> bool:
        ignored = set(ignored or ())
        name = getattr(provider, "__name__", "")
        parent = provider.get_parent() if hasattr(provider, "get_parent") else name
        return name in ignored or parent in ignored

    @classmethod
    def all_providers(cls) -> list[Type]:
        """Load every provider known to ProviderLoader (names + extra APIs)."""
        from ..Provider import ProviderLoader

        providers: list[Type] = []
        seen: set[str] = set()
        for name in list(ProviderLoader.names) + list(ProviderLoader.extra):
            if name in ProviderLoader.ignored:
                continue
            try:
                provider = ProviderLoader.from_name(name)
            except Exception as e:
                debug.log(f"ProviderRouter: could not load {name}: {e}")
                continue
            name = getattr(provider, "__name__", name)
            if not provider or name in seen or name in _SKIP_PROVIDERS:
                continue
            seen.add(name)
            providers.append(provider)
        return providers

    @classmethod
    def match_model(
        cls,
        provider: Type,
        model: str,
        provider_models: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """Return the provider's model id for *model*, or None if unsupported."""
        if not model:
            default = getattr(provider, "default_model", None)
            if default:
                return default
            return provider_models[0] if provider_models else None

        models = list(provider_models or _static_models(provider))
        aliases = getattr(provider, "model_aliases", None) or getattr(
            provider, "models_aliases", None
        ) or {}
        alias = aliases.get(model)
        if alias is None:
            normalized = _normalize_model(model)
            for alias_key, alias_value in aliases.items():
                if _normalize_model(alias_key) == normalized:
                    alias = alias_value
                    break
        if alias is not None:
            targets = alias if isinstance(alias, list) else [alias]
            for target in targets:
                if target == "":
                    default = getattr(provider, "default_model", None)
                    if default:
                        return default
                    continue
                if (
                    isinstance(target, str)
                    and target
                    and (not models or target in models)
                ):
                    return target
            # The alias names a model this provider does not advertise.  This
            # happens for media-only subclasses (e.g. image endpoints inherit
            # text aliases); fall through and match the requested id directly.

        if model in models:
            return model

        normalized = _normalize_model(model)
        matches = [
            candidate
            for candidate in models
            if _normalize_model(candidate) == normalized
        ]
        if len(matches) == 1:
            return matches[0]

        # Some gateways take a bare model name but list a vendor-qualified id
        # (or vice versa).  Only accept an unambiguous suffix match.
        suffix = [
            candidate
            for candidate in models
            if candidate.lower().endswith(model.lower())
            or model.lower().endswith(candidate.lower())
        ]
        if len(suffix) == 1:
            return suffix[0]
        return None

    @classmethod
    def _get_models_kwargs(cls, provider: Type, api_key: Optional[str]) -> dict:
        method = provider.get_models
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        api_key_param = parameters.get("api_key")
        if accepts_kwargs or (
            api_key_param is not None
            and api_key_param.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            return {"api_key": api_key}
        return {}

    @classmethod
    async def get_provider_models(
        cls,
        provider: Type,
        api_key: Optional[str | dict] = None,
        timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> tuple[str, ...]:
        """Return (and cache) a provider's model list without blocking the loop."""
        name = getattr(provider, "__name__", str(provider))
        now = time.time()
        cached = cls._models_cache.get(name)
        if cached and cached[0] > now:
            return cached[1]

        static = tuple(_static_models(provider))
        status: ProviderAuthStatus = AuthManager.get_auth_status(provider, api_key)
        key = status.api_key
        needs_auth = bool(getattr(provider, "needs_auth", False))

        # Never probe a gateway we have no credentials for.  Static metadata
        # (including aliases/default_model) is still useful for matching.
        if needs_auth and not status.configured:
            return static
        # Free providers usually advertise a complete static catalogue; skip a
        # network round trip unless there is nothing to match against.
        if not key and static:
            return static

        get_models = getattr(provider, "get_models", None)
        if not callable(get_models):
            return static

        try:
            method = provider.get_models
            kwargs = cls._get_models_kwargs(provider, key)
            if inspect.iscoroutinefunction(method):
                result = await asyncio.wait_for(method(**kwargs), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(method, **kwargs), timeout=timeout
                )
            models = tuple(dict.fromkeys(_as_model_list(result)))
            cls._models_cache[name] = (now + MODELS_TTL, models)
            return models
        except Exception as e:
            debug.log(f"ProviderRouter: model discovery failed for {name}: {e}")
            cls._models_cache[name] = (now + FAILURE_TTL, static)
            return static

    @classmethod
    async def _probe(
        cls,
        provider: Type,
        model: str,
        api_key: Optional[str | dict],
        timeout: float,
    ) -> Optional[Tuple[Type, str]]:
        try:
            models = await cls.get_provider_models(provider, api_key, timeout)
            alias = cls.match_model(provider, model, models)
            if alias is None:
                return None
            if model and alias != model:
                aliases = getattr(provider, "model_aliases", None) or getattr(
                    provider, "models_aliases", None
                )
                aliases = dict(aliases) if aliases else {}
                aliases.setdefault(model, alias)
                provider.model_aliases = aliases
            return provider, alias
        except Exception as e:
            debug.log(
                f"ProviderRouter: skipping {getattr(provider, '__name__', provider)}: {e}"
            )
            return None

    @classmethod
    async def discover(
        cls,
        model: str,
        api_key: Optional[str | dict] = None,
        providers: Optional[Sequence[Type]] = None,
        exclude: Iterable[str] = (),
        ignored: Iterable[str] = (),
        broad: bool = False,
        timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> list[Type]:
        """Find providers that expose *model* but are not in the static map.

        ``broad=False`` only checks providers that are either configured
        (API key / cached auth) or advertise a static model list.  ``broad``
        additionally lets free providers fetch their catalogue, which is the
        last-resort path for a model missing from the static map.
        """
        from ..Provider import ProviderLoader

        excluded = set(exclude or ())
        ignored_names = set(ignored or ()) | set(ProviderLoader.ignored)
        candidates = providers if providers is not None else cls.all_providers()

        tasks = []
        for provider in candidates:
            if not getattr(provider, "working", False):
                continue
            name = getattr(provider, "__name__", "")
            if name in excluded or name in _SKIP_PROVIDERS:
                continue
            if cls.is_ignored(provider, ignored_names):
                continue

            status = AuthManager.get_auth_status(provider, api_key)
            static = _static_models(provider)
            if not broad:
                if not status.configured and not static:
                    continue
            elif status.source == "unknown" and not static:
                # It would only raise MissingAuthError.
                continue
            tasks.append(cls._probe(provider, model, api_key, timeout))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        found: list[Type] = []
        for result in results:
            if isinstance(result, tuple):
                found.append(result[0])
            elif isinstance(result, Exception):
                debug.log(f"ProviderRouter: discovery error: {result}")
        return found

    @classmethod
    def authed_providers(
        cls,
        api_key: Optional[str | dict] = None,
        providers: Optional[Sequence[Type]] = None,
    ) -> list[Type]:
        """Configured providers that can serve their default model (for auto)."""
        candidates = providers if providers is not None else cls.all_providers()
        from ..Provider import ProviderLoader

        ignored_names = set(ProviderLoader.ignored)
        result = []
        for provider in candidates:
            if not getattr(provider, "working", False):
                continue
            if cls.is_ignored(provider, ignored_names):
                continue
            if _is_media_only(provider):
                continue
            default_model = getattr(provider, "default_model", None)
            if not isinstance(default_model, str) or not default_model:
                continue
            if AuthManager.get_auth_status(provider, api_key).configured:
                result.append(provider)
        return result

    @classmethod
    def rank(
        cls,
        providers: Sequence[Type],
        model: str = "",
        has_tools: bool = False,
        api_key: Optional[str | dict] = None,
        ignored: Iterable[str] = (),
    ) -> list[Type]:
        """Order providers by availability, tool support and auth readiness."""
        from ..Provider import ProviderLoader

        ignored_names = set(ignored or ()) | set(ProviderLoader.ignored)
        unique: list[Type] = []
        seen: set[str] = set()
        for provider in providers:
            if not getattr(provider, "working", False):
                continue
            if cls.is_ignored(provider, ignored_names):
                continue
            name = getattr(provider, "__name__", str(provider))
            if name in seen:
                continue
            seen.add(name)
            unique.append(provider)

        if has_tools:
            native = [
                provider
                for provider in unique
                if getattr(provider, "supports_native_tools", False)
            ]
            if native:
                debug.log(
                    "ProviderRouter: prioritizing native tool providers: "
                    f"{[getattr(p, '__name__', str(p)) for p in native]}"
                )
            elif unique:
                debug.log(
                    "ProviderRouter: no native tool provider available; "
                    "falling back to the full candidate list"
                )

        statuses = {
            getattr(provider, "__name__", str(provider)): AuthManager.get_auth_status(
                provider, api_key
            )
            for provider in unique
        }

        def score(provider: Type):
            name = getattr(provider, "__name__", str(provider))
            status = statuses[name]
            available = ProviderCircuitBreaker.is_available(name)
            live = int(getattr(provider, "live", 0) or 0)
            return (
                1 if available else 0,
                1
                if has_tools and getattr(provider, "supports_native_tools", False)
                else 0,
                _AUTH_SCORES.get(status.source, 0),
                1 if not getattr(provider, "needs_auth", False) else 0,
                1 if getattr(provider, "active_by_default", False) else 0,
                max(-10, min(live, 10)),
            )

        # sorted() is stable, so providers with equal scores keep the order
        # supplied by the caller (model map order / discovery order).
        return sorted(unique, key=score, reverse=True)
