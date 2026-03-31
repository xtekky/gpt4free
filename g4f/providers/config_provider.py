"""
Configuration-based model routing provider for g4f.

Loads a ``config.yaml`` file from the cookies/config directory and routes
model requests to providers based on availability, quota balance, and
recent error counts.

Example ``config.yaml``::

    models:
      - name: "my-gpt4"
        providers:
          - provider: "OpenaiAccount"
            model: "gpt-4o"
            condition: "balance > 0 or error_count < 3"
          - provider: "PollinationsAI"
            model: "openai-large"
      - name: "yupp-route"
        providers:
          - provider: "Yupp"
            model: "gpt-4o"
            condition: "quota.credits.remaining > 0"
      - name: "fast-model"
        providers:
          - provider: "Gemini"
            model: "gemini-pro"

The ``condition`` field is optional.  When present it is a boolean expression
that can reference the following variables:

* ``quota``        – the full quota dict returned by the provider's
  ``get_quota()`` call.  Each provider returns its own schema, e.g.:

  * ``PollinationsAI``: ``{"balance": float}``
  * ``Yupp``:            ``{"credits": {"remaining": int, "total": int}}``
  * ``PuterJS``:         raw JSON from the provider's metering API.
  * ``GeminiCLI``:       ``{"buckets": [...]}``

  Access nested fields with dot-notation: ``quota.balance``,
  ``quota.credits.remaining``, etc.  Missing keys resolve to ``0.0``.

* ``balance``      – convenience shorthand for ``quota.balance``.
  Kept for backward compatibility with PollinationsAI.
  Equivalent to ``quota.balance`` when the provider is PollinationsAI.

* ``error_count``  – the number of recent errors recorded for the provider
  within a rolling one-hour window.

Supported operators in conditions: ``>``, ``<``, ``>=``, ``<=``, ``==``,
``!=``, as well as ``and`` / ``or`` / ``not``.  Only the variables above
are available; arbitrary Python is **not** evaluated.
"""

from __future__ import annotations

import os
import re
import time
import operator
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False

from ..typing import Messages, AsyncResult
from .base_provider import AsyncGeneratorProvider
from .response import ProviderInfo
from .. import debug
from ..config import AppConfig
from ..tools.auth import AuthManager

# ---------------------------------------------------------------------------
# Quota cache
# ---------------------------------------------------------------------------

class QuotaCache:
    """Thread-safe in-memory cache for provider quota results.

    Quota values are cached for :attr:`ttl` seconds.  The cache entry for a
    provider can be forcibly invalidated (e.g. when a 429 response is
    received) via :meth:`invalidate`.
    """

    ttl: float = 300  # seconds

    _cache: Dict[str, dict] = {}
    _timestamps: Dict[str, float] = {}

    @classmethod
    def get(cls, provider_name: str) -> Optional[dict]:
        """Return the cached quota dict for *provider_name*, or ``None``."""
        if provider_name in cls._cache:
            if time.time() - cls._timestamps.get(provider_name, 0) < cls.ttl:
                return cls._cache[provider_name]
            # Expired – remove stale entry
            cls._cache.pop(provider_name, None)
            cls._timestamps.pop(provider_name, None)
        return None

    @classmethod
    def set(cls, provider_name: str, quota: dict) -> None:
        """Store *quota* for *provider_name*."""
        cls._cache[provider_name] = quota
        cls._timestamps[provider_name] = time.time()

    @classmethod
    def invalidate(cls, provider_name: str) -> None:
        """Invalidate the cached quota for *provider_name*.

        Call this when a 429 (rate-limit) response is received so that
        the next routing decision fetches a fresh quota value.
        """
        cls._cache.pop(provider_name, None)
        cls._timestamps.pop(provider_name, None)

    @classmethod
    def clear(cls) -> None:
        """Remove all cached entries."""
        cls._cache.clear()
        cls._timestamps.clear()


# ---------------------------------------------------------------------------
# Error counter
# ---------------------------------------------------------------------------

class ErrorCounter:
    """Rolling-window error counter for providers.

    Errors are tracked with timestamps so that only errors that occurred
    within the last :attr:`window` seconds are counted.
    """

    window: float = 3600  # 1 hour

    _timestamps: Dict[str, List[float]] = {}

    @classmethod
    def increment(cls, provider_name: str) -> None:
        """Record one error for *provider_name*."""
        now = time.time()
        bucket = cls._timestamps.setdefault(provider_name, [])
        bucket.append(now)
        # Prune timestamps outside the rolling window
        cls._timestamps[provider_name] = [t for t in bucket if now - t < cls.window]

    @classmethod
    def get_count(cls, provider_name: str) -> int:
        """Return the number of errors for *provider_name* in the current window."""
        now = time.time()
        bucket = cls._timestamps.get(provider_name, [])
        # Prune stale entries on read as well
        fresh = [t for t in bucket if now - t < cls.window]
        cls._timestamps[provider_name] = fresh
        return len(fresh)

    @classmethod
    def reset(cls, provider_name: str) -> None:
        """Reset the error counter for *provider_name*."""
        cls._timestamps.pop(provider_name, None)

    @classmethod
    def clear(cls) -> None:
        """Reset all error counters."""
        cls._timestamps.clear()


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

_OPS: Dict[str, "Callable"] = {
    ">":  operator.gt,
    "<":  operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# Tokenizer for simple condition expressions
_TOKEN_RE = re.compile(
    r"(?P<float>-?\d+\.\d+)"       # float literal
    r"|(?P<int>-?\d+)"              # integer literal
    r"|(?P<op>>=|<=|==|!=|>|<)"    # comparison operator
    r"|(?P<kw>and|or|not)"         # logical keywords
    r"|(?P<id>[a-zA-Z_][a-zA-Z0-9_.-]*)"  # identifier (supports dashed keys)
    r"|(?P<lp>\()"                  # left paren
    r"|(?P<rp>\))"                  # right paren
)


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr.strip()):
        kind = m.lastgroup
        tokens.append((kind, m.group()))
    return tokens


def _parse_expr(tokens: List[Tuple[str, str]], pos: int, variables: Dict[str, float]) -> Tuple[bool, int]:
    """Recursive-descent parser for ``and``/``or``/``not``/comparisons."""
    return _parse_or(tokens, pos, variables)


def _parse_or(tokens, pos, variables):
    left, pos = _parse_and(tokens, pos, variables)
    while pos < len(tokens) and tokens[pos] == ("kw", "or"):
        pos += 1
        right, pos = _parse_and(tokens, pos, variables)
        left = left or right
    return left, pos


def _parse_and(tokens, pos, variables):
    left, pos = _parse_not(tokens, pos, variables)
    while pos < len(tokens) and tokens[pos] == ("kw", "and"):
        pos += 1
        right, pos = _parse_not(tokens, pos, variables)
        left = left and right
    return left, pos


def _parse_not(tokens, pos, variables):
    if pos < len(tokens) and tokens[pos] == ("kw", "not"):
        pos += 1
        val, pos = _parse_not(tokens, pos, variables)
        return not val, pos
    return _parse_comparison(tokens, pos, variables)


def _parse_comparison(tokens, pos, variables):
    if pos < len(tokens) and tokens[pos][0] == "lp":
        pos += 1  # consume '('
        val, pos = _parse_or(tokens, pos, variables)
        if pos < len(tokens) and tokens[pos][0] == "rp":
            pos += 1  # consume ')'
        return val, pos

    left_val, pos = _parse_atom(tokens, pos, variables)

    if pos < len(tokens) and tokens[pos][0] == "op":
        op_str = tokens[pos][1]
        pos += 1
        right_val, pos = _parse_atom(tokens, pos, variables)
        return _OPS[op_str](left_val, right_val), pos

    # Bare value – treat as truthy
    return bool(left_val), pos


def _parse_atom(tokens, pos, variables):
    if pos >= len(tokens):
        raise ValueError("Unexpected end of condition expression")

    kind, value = tokens[pos]
    pos += 1

    if kind == "float":
        return float(value), pos
    elif kind == "int":
        return int(value), pos
    elif kind == "id":
        # Legacy alias: "get_quota.balance" → "quota.balance"
        if value == "get_quota.balance":
            value = "quota.balance"

        # Resolve dotted paths: "quota.credits.remaining", "balance", etc.
        parts = value.split(".")
        root = parts[0]
        if root not in variables:
            raise ValueError(f"Unknown variable in condition: {root!r}")

        result = variables[root]
        for part in parts[1:]:
            if isinstance(result, dict):
                result = result.get(part)
                if result is None:
                    result = 0.0
                    break
            else:
                raise ValueError(
                    f"Cannot access field {part!r} on non-dict value "
                    f"while resolving {value!r}"
                )

        return float(result) if result is not None else 0.0, pos
    else:
        raise ValueError(f"Unexpected token {kind!r}={value!r} in condition expression")


def evaluate_condition(
    condition: str,
    quota: Optional[Dict],
    error_count: int,
) -> bool:
    """Evaluate a provider condition string.

    The condition may reference:

    * ``quota``              – the full quota dict returned by ``get_quota()``.
      Each provider returns its own schema.  Access nested fields with
      dot-notation, e.g. ``quota.balance``, ``quota.credits.remaining``.
      Missing keys resolve to ``0.0``.
    * ``balance``            – shorthand alias for ``quota.balance``.
      Kept for backward compatibility; equivalent to ``quota.balance``
      for providers that return ``{"balance": float}`` (e.g. PollinationsAI).
    * ``error_count``        – recent error count (int).

    If *quota* is ``None`` the ``quota`` variable resolves to ``{}`` and
    ``balance`` resolves to ``0.0``.

    Returns ``True`` if the provider should be used, ``False`` otherwise.
    Raises :class:`ValueError` on parse errors.
    """
    quota_dict = quota if isinstance(quota, dict) else {}
    variables: Dict[str, object] = {
        # Full quota dict – supports quota.balance, quota.credits.remaining, etc.
        "quota": quota_dict,
        # Convenience shorthand: "balance" → quota["balance"] (PollinationsAI compat)
        "balance": float(quota_dict.get("balance", 0.0)),
        "error_count": float(error_count),
    }
    tokens = _tokenize(condition)
    if not tokens:
        return True
    result, _ = _parse_expr(tokens, 0, variables)
    return result


# ---------------------------------------------------------------------------
# Config data structures
# ---------------------------------------------------------------------------

@dataclass
class ProviderRouteConfig:
    """A single provider entry inside a model route."""

    provider: str
    """Provider class name (e.g. ``"OpenaiAccount"``)."""

    model: str = ""
    """Model name passed to the provider.  Defaults to the route model name."""

    condition: Optional[str] = None
    """Optional boolean expression.  If absent the provider is always eligible."""


@dataclass
class ModelRouteConfig:
    """Routing configuration for a single model name."""

    name: str
    """The model name as seen by the client (e.g. ``"my-gpt4"``)."""

    providers: List[ProviderRouteConfig] = field(default_factory=list)
    """Ordered list of provider candidates."""


# ---------------------------------------------------------------------------
# Global router state
# ---------------------------------------------------------------------------

class RouterConfig:
    """Singleton holding the active routing configuration."""

    routes: Dict[str, ModelRouteConfig] = {}
    """Mapping from model name → :class:`ModelRouteConfig`."""

    @classmethod
    def load(cls, path: str) -> None:
        """Load and parse a ``config.yaml`` file at *path*.

        Silently skips the file if PyYAML is not installed or the file does
        not exist.
        """
        if not has_yaml:
            debug.error("config.yaml: PyYAML is not installed – skipping config.yaml")
            return
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as e:
            debug.error(f"config.yaml: Failed to parse {path}:", e)
            return

        if not isinstance(data, dict):
            debug.error(f"config.yaml: Expected a mapping at top level in {path}")
            return

        new_routes: Dict[str, ModelRouteConfig] = {}
        for entry in data.get("models", []):
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            model_name = entry["name"]
            provider_list: List[ProviderRouteConfig] = []
            for pentry in entry.get("providers", []):
                if not isinstance(pentry, dict) or "provider" not in pentry:
                    continue
                provider_list.append(
                    ProviderRouteConfig(
                        provider=pentry["provider"],
                        model=pentry.get("model", model_name),
                        condition=pentry.get("condition"),
                    )
                )
            if provider_list:
                new_routes[model_name] = ModelRouteConfig(
                    name=model_name,
                    providers=provider_list,
                )

        cls.routes = new_routes
        debug.log(f"config.yaml: Loaded {len(new_routes)} model route(s) from {path}")

    @classmethod
    def clear(cls) -> None:
        """Remove all loaded routes."""
        cls.routes.clear()

    @classmethod
    def get(cls, model_name: str) -> Optional[ModelRouteConfig]:
        """Return the :class:`ModelRouteConfig` for *model_name*, or ``None``."""
        return cls.routes.get(model_name)


# ---------------------------------------------------------------------------
# Config-based provider
# ---------------------------------------------------------------------------

def _resolve_provider(provider_name: str):
    """Resolve a provider name string to a provider class."""
    from .. import Provider
    from ..Provider import ProviderUtils

    if provider_name in ProviderUtils.convert:
        return ProviderUtils.convert[provider_name]

    # Try direct attribute lookup on the Provider module
    provider = getattr(Provider, provider_name, None)
    if provider is not None:
        return provider

    raise ValueError(f"Provider not found: {provider_name!r}")


async def _get_quota_cached(provider) -> Optional[dict]:
    """Return quota info for *provider*, using the cache when possible."""
    name = getattr(provider, "__name__", str(provider))
    cached = QuotaCache.get(name)
    if cached is not None:
        return cached
    if not hasattr(provider, "get_quota"):
        return None
    try:
        quota = await provider.get_quota()
        if quota is not None:
            QuotaCache.set(name, quota)
        return quota
    except Exception as e:
        debug.error(f"config.yaml: get_quota failed for {name}:", e)
        return None


def _check_condition(
    route_cfg: ProviderRouteConfig,
    provider,
    quota: Optional[dict],
) -> bool:
    """Return ``True`` if the provider satisfies the route condition."""
    if not route_cfg.condition:
        return True
    provider_name = getattr(provider, "__name__", str(provider))
    error_count = ErrorCounter.get_count(provider_name)
    try:
        return evaluate_condition(route_cfg.condition, quota, error_count)
    except ValueError as e:
        debug.error(f"config.yaml: Invalid condition {route_cfg.condition!r}:", e)
        return False  # Default to skip on parse error


class ConfigModelProvider(AsyncGeneratorProvider):
    """An async generator provider that routes requests using ``config.yaml``.

    This provider is instantiated per model name and tries each configured
    provider in order, skipping those that fail their condition check.  On a
    429 error the quota cache for the failing provider is invalidated so that
    the next call fetches a fresh quota value.
    """

    working = True
    supports_stream = True
    supports_message_history = True

    def __init__(self, route_config: ModelRouteConfig) -> None:
        self._route_config = route_config
        self.__name__ = f"ConfigRouter[{route_config.name}]"

    # Make it usable as an instance (not just a class)
    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        api_key: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncResult:
        """Yield response chunks, routing through configured providers."""
        last_exception: Optional[Exception] = None
        tried: List[str] = []

        for prc in self._route_config.providers:
            try:
                provider = _resolve_provider(prc.provider)
            except ValueError as e:
                debug.error(f"config.yaml: {e}")
                continue

            provider_name = getattr(provider, "__name__", prc.provider)

            # Fetch quota (cached)
            quota = await _get_quota_cached(provider)

            # Evaluate condition
            if not _check_condition(prc, provider, quota):
                debug.log(
                    f"config.yaml: Skipping {provider_name} "
                    f"(condition not met: {prc.condition!r})"
                )
                continue

            target_model = prc.model or model
            tried.append(provider_name)

            yield ProviderInfo(
                name=provider_name,
                url=getattr(provider, "url", ""),
                label=getattr(provider, "label", None),
                model=target_model,
            )

            extra_body = kwargs.copy()
            current_api_key = api_key.get(provider.get_parent()) if isinstance(api_key, dict) else api_key
            if not current_api_key or AppConfig.disable_custom_api_key:
                current_api_key = AuthManager.load_api_key(provider)
            if current_api_key:
                extra_body["api_key"] = current_api_key

            try:
                if hasattr(provider, "create_async_generator"):
                    async for chunk in provider.create_async_generator(
                        target_model, messages, **extra_body
                    ):
                        yield chunk
                elif hasattr(provider, "create_completion"):
                    for chunk in provider.create_completion(
                        target_model, messages, **extra_body
                    ):
                        yield chunk
                else:
                    raise NotImplementedError(
                        f"{provider_name} has no supported create method"
                    )
                debug.log(f"config.yaml: {provider_name} succeeded for model {model!r}")
                return  # Success
            except Exception as e:
                # On rate-limit errors invalidate the quota cache
                from ..errors import RateLimitError
                if isinstance(e, RateLimitError) or "429" in str(e):
                    debug.log(
                        f"config.yaml: Rate-limited by {provider_name}, "
                        "invalidating quota cache"
                    )
                    QuotaCache.invalidate(provider_name)

                ErrorCounter.increment(provider_name)
                last_exception = e
                debug.error(f"config.yaml: {provider_name} failed:", e)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(
            f"config.yaml: No provider succeeded for model {model!r}. "
            f"Tried: {tried}"
        )
