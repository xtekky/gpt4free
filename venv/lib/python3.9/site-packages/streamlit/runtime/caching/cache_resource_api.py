# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""@st.cache_resource implementation"""

from __future__ import annotations

import math
import threading
import types
from datetime import timedelta
from typing import Any, Callable, TypeVar, cast, overload

from cachetools import TTLCache
from pympler import asizeof
from typing_extensions import TypeAlias

import streamlit as st
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
    Cache,
    CachedFuncInfo,
    make_cached_func_wrapper,
    ttl_to_seconds,
)
from streamlit.runtime.caching.cached_message_replay import (
    CachedMessageReplayContext,
    CachedResult,
    ElementMsgData,
    MsgData,
    MultiCacheResults,
)
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider

_LOGGER = get_logger(__name__)


CACHE_RESOURCE_MESSAGE_REPLAY_CTX = CachedMessageReplayContext(CacheType.RESOURCE)

ValidateFunc: TypeAlias = Callable[[Any], bool]


def _equal_validate_funcs(a: ValidateFunc | None, b: ValidateFunc | None) -> bool:
    """True if the two validate functions are equal for the purposes of
    determining whether a given function cache needs to be recreated.
    """
    # To "properly" test for function equality here, we'd need to compare function bytecode.
    # For performance reasons, We've decided not to do that for now.
    return (a is None and b is None) or (a is not None and b is not None)


class ResourceCaches(CacheStatsProvider):
    """Manages all ResourceCache instances"""

    def __init__(self):
        self._caches_lock = threading.Lock()
        self._function_caches: dict[str, ResourceCache] = {}

    def get_cache(
        self,
        key: str,
        display_name: str,
        max_entries: int | float | None,
        ttl: float | timedelta | None,
        validate: ValidateFunc | None,
        allow_widgets: bool,
    ) -> ResourceCache:
        """Return the mem cache for the given key.

        If it doesn't exist, create a new one with the given params.
        """
        if max_entries is None:
            max_entries = math.inf

        ttl_seconds = ttl_to_seconds(ttl)

        # Get the existing cache, if it exists, and validate that its params
        # haven't changed.
        with self._caches_lock:
            cache = self._function_caches.get(key)
            if (
                cache is not None
                and cache.ttl_seconds == ttl_seconds
                and cache.max_entries == max_entries
                and _equal_validate_funcs(cache.validate, validate)
            ):
                return cache

            # Create a new cache object and put it in our dict
            _LOGGER.debug("Creating new ResourceCache (key=%s)", key)
            cache = ResourceCache(
                key=key,
                display_name=display_name,
                max_entries=max_entries,
                ttl_seconds=ttl_seconds,
                validate=validate,
                allow_widgets=allow_widgets,
            )
            self._function_caches[key] = cache
            return cache

    def clear_all(self) -> None:
        """Clear all resource caches."""
        with self._caches_lock:
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        with self._caches_lock:
            # Shallow-clone our caches. We don't want to hold the global
            # lock during stats-gathering.
            function_caches = self._function_caches.copy()

        stats: list[CacheStat] = []
        for cache in function_caches.values():
            stats.extend(cache.get_stats())
        return stats


# Singleton ResourceCaches instance
_resource_caches = ResourceCaches()


def get_resource_cache_stats_provider() -> CacheStatsProvider:
    """Return the StatsProvider for all @st.cache_resource functions."""
    return _resource_caches


class CachedResourceFuncInfo(CachedFuncInfo):
    """Implements the CachedFuncInfo interface for @st.cache_resource"""

    def __init__(
        self,
        func: types.FunctionType,
        show_spinner: bool | str,
        max_entries: int | None,
        ttl: float | timedelta | None,
        validate: ValidateFunc | None,
        allow_widgets: bool,
    ):
        super().__init__(
            func,
            show_spinner=show_spinner,
            allow_widgets=allow_widgets,
        )
        self.max_entries = max_entries
        self.ttl = ttl
        self.validate = validate

    @property
    def cache_type(self) -> CacheType:
        return CacheType.RESOURCE

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        return CACHE_RESOURCE_MESSAGE_REPLAY_CTX

    @property
    def display_name(self) -> str:
        """A human-readable name for the cached function"""
        return f"{self.func.__module__}.{self.func.__qualname__}"

    def get_function_cache(self, function_key: str) -> Cache:
        return _resource_caches.get_cache(
            key=function_key,
            display_name=self.display_name,
            max_entries=self.max_entries,
            ttl=self.ttl,
            validate=self.validate,
            allow_widgets=self.allow_widgets,
        )


class CacheResourceAPI:
    """Implements the public st.cache_resource API: the @st.cache_resource decorator,
    and st.cache_resource.clear().
    """

    def __init__(
        self, decorator_metric_name: str, deprecation_warning: str | None = None
    ):
        """Create a CacheResourceAPI instance.

        Parameters
        ----------
        decorator_metric_name
            The metric name to record for decorator usage. `@st.experimental_singleton` is
            deprecated, but we're still supporting it and tracking its usage separately
            from `@st.cache_resource`.

        deprecation_warning
            An optional deprecation warning to show when the API is accessed.
        """

        # Parameterize the decorator metric name.
        # (Ignore spurious mypy complaints - https://github.com/python/mypy/issues/2427)
        self._decorator = gather_metrics(decorator_metric_name, self._decorator)  # type: ignore
        self._deprecation_warning = deprecation_warning

    # Type-annotate the decorator function.
    # (See https://mypy.readthedocs.io/en/stable/generics.html#decorator-factories)

    F = TypeVar("F", bound=Callable[..., Any])

    # Bare decorator usage
    @overload
    def __call__(self, func: F) -> F:
        ...

    # Decorator with arguments
    @overload
    def __call__(
        self,
        *,
        ttl: float | timedelta | None = None,
        max_entries: int | None = None,
        show_spinner: bool | str = True,
        validate: ValidateFunc | None = None,
        experimental_allow_widgets: bool = False,
    ) -> Callable[[F], F]:
        ...

    def __call__(
        self,
        func: F | None = None,
        *,
        ttl: float | timedelta | None = None,
        max_entries: int | None = None,
        show_spinner: bool | str = True,
        validate: ValidateFunc | None = None,
        experimental_allow_widgets: bool = False,
    ):
        return self._decorator(
            func,
            ttl=ttl,
            max_entries=max_entries,
            show_spinner=show_spinner,
            validate=validate,
            experimental_allow_widgets=experimental_allow_widgets,
        )

    def _decorator(
        self,
        func: F | None,
        *,
        ttl: float | timedelta | None,
        max_entries: int | None,
        show_spinner: bool | str,
        validate: ValidateFunc | None,
        experimental_allow_widgets: bool,
    ):
        """Decorator to cache functions that return global resources (e.g. database connections, ML models).

        Cached objects are shared across all users, sessions, and reruns. They
        must be thread-safe because they can be accessed from multiple threads
        concurrently. If thread safety is an issue, consider using ``st.session_state``
        to store resources per session instead.

        You can clear a function's cache with ``func.clear()`` or clear the entire
        cache with ``st.cache_resource.clear()``.

        To cache data, use ``st.cache_data`` instead. Learn more about caching at
        https://docs.streamlit.io/library/advanced-features/caching.

        Parameters
        ----------
        func : callable
            The function that creates the cached resource. Streamlit hashes the
            function's source code.

        ttl : float or timedelta or None
            The maximum number of seconds to keep an entry in the cache, or
            None if cache entries should not expire. The default is None.

        max_entries : int or None
            The maximum number of entries to keep in the cache, or None
            for an unbounded cache. (When a new entry is added to a full cache,
            the oldest cached entry will be removed.) The default is None.

        show_spinner : boolean or string
            Enable the spinner. Default is True to show a spinner when there is
            a "cache miss" and the cached resource is being created. If string,
            value of show_spinner param will be used for spinner text.

        validate : callable or None
            An optional validation function for cached data. ``validate`` is called
            each time the cached value is accessed. It receives the cached value as
            its only parameter and it must return a boolean. If ``validate`` returns
            False, the current cached value is discarded, and the decorated function
            is called to compute a new value. This is useful e.g. to check the
            health of database connections.

        experimental_allow_widgets : boolean
            Allow widgets to be used in the cached function. Defaults to False.
            Support for widgets in cached functions is currently experimental.
            Setting this parameter to True may lead to excessive memory use since the
            widget value is treated as an additional input parameter to the cache.
            We may remove support for this option at any time without notice.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(url):
        ...     # Create a database session object that points to the URL.
        ...     return session
        ...
        >>> s1 = get_database_session(SESSION_URL_1)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> s2 = get_database_session(SESSION_URL_1)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value. This means that now the connection object in s1 is the same as in s2.
        >>>
        >>> s3 = get_database_session(SESSION_URL_2)
        >>> # This is a different URL, so the function executes.

        By default, all parameters to a cache_resource function must be hashable.
        Any parameter whose name begins with ``_`` will not be hashed. You can use
        this as an "escape hatch" for parameters that are not hashable:

        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(_sessionmaker, url):
        ...     # Create a database connection object that points to the URL.
        ...     return connection
        ...
        >>> s1 = get_database_session(create_sessionmaker(), DATA_URL_1)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> s2 = get_database_session(create_sessionmaker(), DATA_URL_1)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value - even though the _sessionmaker parameter was different
        >>> # in both calls.

        A cache_resource function's cache can be procedurally cleared:

        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(_sessionmaker, url):
        ...     # Create a database connection object that points to the URL.
        ...     return connection
        ...
        >>> get_database_session.clear()
        >>> # Clear all cached entries for this function.

        """
        self._maybe_show_deprecation_warning()

        # Support passing the params via function decorator, e.g.
        # @st.cache_resource(show_spinner=False)
        if func is None:
            return lambda f: make_cached_func_wrapper(
                CachedResourceFuncInfo(
                    func=f,
                    show_spinner=show_spinner,
                    max_entries=max_entries,
                    ttl=ttl,
                    validate=validate,
                    allow_widgets=experimental_allow_widgets,
                )
            )

        return make_cached_func_wrapper(
            CachedResourceFuncInfo(
                func=cast(types.FunctionType, func),
                show_spinner=show_spinner,
                max_entries=max_entries,
                ttl=ttl,
                validate=validate,
                allow_widgets=experimental_allow_widgets,
            )
        )

    @gather_metrics("clear_resource_caches")
    def clear(self) -> None:
        """Clear all cache_resource caches."""
        self._maybe_show_deprecation_warning()
        _resource_caches.clear_all()

    def _maybe_show_deprecation_warning(self):
        """If the API is being accessed with the deprecated `st.experimental_singleton` name,
        show a deprecation warning.
        """
        if self._deprecation_warning is not None:
            show_deprecation_warning(self._deprecation_warning)


class ResourceCache(Cache):
    """Manages cached values for a single st.cache_resource function."""

    def __init__(
        self,
        key: str,
        max_entries: float,
        ttl_seconds: float,
        validate: ValidateFunc | None,
        display_name: str,
        allow_widgets: bool,
    ):
        super().__init__()
        self.key = key
        self.display_name = display_name
        self._mem_cache: TTLCache[str, MultiCacheResults] = TTLCache(
            maxsize=max_entries, ttl=ttl_seconds, timer=cache_utils.TTLCACHE_TIMER
        )
        self._mem_cache_lock = threading.Lock()
        self.validate = validate
        self.allow_widgets = allow_widgets

    @property
    def max_entries(self) -> float:
        return cast(float, self._mem_cache.maxsize)

    @property
    def ttl_seconds(self) -> float:
        return cast(float, self._mem_cache.ttl)

    def read_result(self, key: str) -> CachedResult:
        """Read a value and associated messages from the cache.
        Raise `CacheKeyNotFoundError` if the value doesn't exist.
        """
        with self._mem_cache_lock:
            if key not in self._mem_cache:
                # key does not exist in cache.
                raise CacheKeyNotFoundError()

            multi_results: MultiCacheResults = self._mem_cache[key]

            ctx = get_script_run_ctx()
            if not ctx:
                # ScriptRunCtx does not exist (we're probably running in "raw" mode).
                raise CacheKeyNotFoundError()

            widget_key = multi_results.get_current_widget_key(ctx, CacheType.RESOURCE)
            if widget_key not in multi_results.results:
                # widget_key does not exist in cache (this combination of widgets hasn't been
                # seen for the value_key yet).
                raise CacheKeyNotFoundError()

            result = multi_results.results[widget_key]

            if self.validate is not None and not self.validate(result.value):
                # Validate failed: delete the entry and raise an error.
                del multi_results.results[widget_key]
                raise CacheKeyNotFoundError()

            return result

    @gather_metrics("_cache_resource_object")
    def write_result(self, key: str, value: Any, messages: list[MsgData]) -> None:
        """Write a value and associated messages to the cache."""
        ctx = get_script_run_ctx()
        if ctx is None:
            return

        main_id = st._main.id
        sidebar_id = st.sidebar.id
        if self.allow_widgets:
            widgets = {
                msg.widget_metadata.widget_id
                for msg in messages
                if isinstance(msg, ElementMsgData) and msg.widget_metadata is not None
            }
        else:
            widgets = set()

        with self._mem_cache_lock:
            try:
                multi_results = self._mem_cache[key]
            except KeyError:
                multi_results = MultiCacheResults(widget_ids=widgets, results={})

            multi_results.widget_ids.update(widgets)
            widget_key = multi_results.get_current_widget_key(ctx, CacheType.RESOURCE)

            result = CachedResult(value, messages, main_id, sidebar_id)
            multi_results.results[widget_key] = result
            self._mem_cache[key] = multi_results

    def _clear(self) -> None:
        with self._mem_cache_lock:
            self._mem_cache.clear()

    def get_stats(self) -> list[CacheStat]:
        # Shallow clone our cache. Computing item sizes is potentially
        # expensive, and we want to minimize the time we spend holding
        # the lock.
        with self._mem_cache_lock:
            cache_entries = list(self._mem_cache.values())

        return [
            CacheStat(
                category_name="st_cache_resource",
                cache_name=self.display_name,
                byte_length=asizeof.asizeof(entry),
            )
            for entry in cache_entries
        ]
