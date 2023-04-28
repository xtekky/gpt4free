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

"""Common cache logic shared by st.cache_data and st.cache_resource."""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from datetime import timedelta
from typing import Any, Callable, overload

from typing_extensions import Literal

from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
    CacheError,
    CacheKeyNotFoundError,
    UnevaluatedDataFrameError,
    UnhashableParamError,
    UnhashableTypeError,
    UnserializableReturnValueError,
    get_cached_func_name_md,
)
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
    CachedMessageReplayContext,
    CachedResult,
    MsgData,
    replay_cached_messages,
)
from streamlit.runtime.caching.hashing import update_hash

_LOGGER = get_logger(__name__)

# The timer function we use with TTLCache. This is the default timer func, but
# is exposed here as a constant so that it can be patched in unit tests.
TTLCACHE_TIMER = time.monotonic


@overload
def ttl_to_seconds(
    ttl: float | timedelta | None, *, coerce_none_to_inf: Literal[False]
) -> float | None:
    ...


@overload
def ttl_to_seconds(ttl: float | timedelta | None) -> float:
    ...


def ttl_to_seconds(
    ttl: float | timedelta | None, *, coerce_none_to_inf: bool = True
) -> float | None:
    """
    Convert a ttl value to a float representing "number of seconds".
    """
    if coerce_none_to_inf and ttl is None:
        return math.inf
    if isinstance(ttl, timedelta):
        return ttl.total_seconds()
    return ttl


# We show a special "UnevaluatedDataFrame" warning for cached funcs
# that attempt to return one of these unserializable types:
UNEVALUATED_DATAFRAME_TYPES = (
    "snowflake.snowpark.table.Table",
    "snowflake.snowpark.dataframe.DataFrame",
    "pyspark.sql.dataframe.DataFrame",
)


class Cache:
    """Function cache interface. Caches persist across script runs."""

    def __init__(self):
        self._value_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._value_locks_lock = threading.Lock()

    @abstractmethod
    def read_result(self, value_key: str) -> CachedResult:
        """Read a value and associated messages from the cache.

        Raises
        ------
        CacheKeyNotFoundError
            Raised if value_key is not in the cache.

        """
        raise NotImplementedError

    @abstractmethod
    def write_result(self, value_key: str, value: Any, messages: list[MsgData]) -> None:
        """Write a value and associated messages to the cache, overwriting any existing
        result that uses the value_key.
        """
        # We *could* `del self._value_locks[value_key]` here, since nobody will be taking
        # a compute_value_lock for this value_key after the result is written.
        raise NotImplementedError

    def compute_value_lock(self, value_key: str) -> threading.Lock:
        """Return the lock that should be held while computing a new cached value.
        In a popular app with a cache that hasn't been pre-warmed, many sessions may try
        to access a not-yet-cached value simultaneously. We use a lock to ensure that
        only one of those sessions computes the value, and the others block until
        the value is computed.
        """
        with self._value_locks_lock:
            return self._value_locks[value_key]

    def clear(self):
        """Clear all values from this cache."""
        with self._value_locks_lock:
            self._value_locks.clear()
        self._clear()

    @abstractmethod
    def _clear(self) -> None:
        """Subclasses must implement this to perform cache-clearing logic."""
        raise NotImplementedError


class CachedFuncInfo:
    """Encapsulates data for a cached function instance.

    CachedFuncInfo instances are scoped to a single script run - they're not
    persistent.
    """

    def __init__(
        self,
        func: types.FunctionType,
        show_spinner: bool | str,
        allow_widgets: bool,
    ):
        self.func = func
        self.show_spinner = show_spinner
        self.allow_widgets = allow_widgets

    @property
    def cache_type(self) -> CacheType:
        raise NotImplementedError

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        raise NotImplementedError

    def get_function_cache(self, function_key: str) -> Cache:
        """Get or create the function cache for the given key."""
        raise NotImplementedError


def make_cached_func_wrapper(info: CachedFuncInfo) -> Callable[..., Any]:
    """Create a callable wrapper around a CachedFunctionInfo.

    Calling the wrapper will return the cached value if it's already been
    computed, and will call the underlying function to compute and cache the
    value otherwise.

    The wrapper also has a `clear` function that can be called to clear
    all of the wrapper's cached values.
    """
    cached_func = CachedFunc(info)

    # We'd like to simply return `cached_func`, which is already a Callable.
    # But using `functools.update_wrapper` on the CachedFunc instance
    # itself results in errors when our caching decorators are used to decorate
    # member functions. (See https://github.com/streamlit/streamlit/issues/6109)

    @functools.wraps(info.func)
    def wrapper(*args, **kwargs):
        return cached_func(*args, **kwargs)

    # Give our wrapper its `clear` function.
    # (This results in a spurious mypy error that we suppress.)
    wrapper.clear = cached_func.clear  # type: ignore

    return wrapper


class CachedFunc:
    def __init__(self, info: CachedFuncInfo):
        self._info = info
        self._function_key = _make_function_key(info.cache_type, info.func)

    def __call__(self, *args, **kwargs) -> Any:
        """The wrapper. We'll only call our underlying function on a cache miss."""

        name = self._info.func.__qualname__

        if isinstance(self._info.show_spinner, bool):
            if len(args) == 0 and len(kwargs) == 0:
                message = f"Running `{name}()`."
            else:
                message = f"Running `{name}(...)`."
        else:
            message = self._info.show_spinner

        if self._info.show_spinner or isinstance(self._info.show_spinner, str):
            with spinner(message):
                return self._get_or_create_cached_value(args, kwargs)
        else:
            return self._get_or_create_cached_value(args, kwargs)

    def _get_or_create_cached_value(
        self, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]
    ) -> Any:
        # Retrieve the function's cache object. We must do this "just-in-time"
        # (as opposed to in the constructor), because caches can be invalidated
        # at any time.
        cache = self._info.get_function_cache(self._function_key)

        # Generate the key for the cached value. This is based on the
        # arguments passed to the function.
        value_key = _make_value_key(
            cache_type=self._info.cache_type,
            func=self._info.func,
            func_args=func_args,
            func_kwargs=func_kwargs,
        )

        try:
            cached_result = cache.read_result(value_key)
            return self._handle_cache_hit(cached_result)
        except CacheKeyNotFoundError:
            return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)

    def _handle_cache_hit(self, result: CachedResult) -> Any:
        """Handle a cache hit: replay the result's cached messages, and return its value."""
        replay_cached_messages(
            result,
            self._info.cache_type,
            self._info.func,
        )
        return result.value

    def _handle_cache_miss(
        self,
        cache: Cache,
        value_key: str,
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
    ) -> Any:
        """Handle a cache miss: compute a new cached value, write it back to the cache,
        and return that newly-computed value.
        """

        # Implementation notes:
        # - We take a "compute_value_lock" before computing our value. This ensures that
        #   multiple sessions don't try to compute the same value simultaneously.
        #
        # - We use a different lock for each value_key, as opposed to a single lock for
        #   the entire cache, so that unrelated value computations don't block on each other.
        #
        # - When retrieving a cache entry that may not yet exist, we use a "double-checked locking"
        #   strategy: first we try to retrieve the cache entry without taking a value lock. (This
        #   happens in `_get_or_create_cached_value()`.) If that fails because the value hasn't
        #   been computed yet, we take the value lock and then immediately try to retrieve cache entry
        #   *again*, while holding the lock. If the cache entry exists at this point, it means that
        #   another thread computed the value before us.
        #
        #   This means that the happy path ("cache entry exists") is a wee bit faster because
        #   no lock is acquired. But the unhappy path ("cache entry needs to be recomputed") is
        #   a wee bit slower, because we do two lookups for the entry.

        with cache.compute_value_lock(value_key):
            # We've acquired the lock - but another thread may have acquired it first
            # and already computed the value. So we need to test for a cache hit again,
            # before computing.
            try:
                cached_result = cache.read_result(value_key)
                # Another thread computed the value before us. Early exit!
                return self._handle_cache_hit(cached_result)

            except CacheKeyNotFoundError:
                # We acquired the lock before any other thread. Compute the value!
                with self._info.cached_message_replay_ctx.calling_cached_function(
                    self._info.func, self._info.allow_widgets
                ):
                    computed_value = self._info.func(*func_args, **func_kwargs)

                # We've computed our value, and now we need to write it back to the cache
                # along with any "replay messages" that were generated during value computation.
                messages = self._info.cached_message_replay_ctx._most_recent_messages
                try:
                    cache.write_result(value_key, computed_value, messages)
                    return computed_value
                except (CacheError, RuntimeError):
                    # An exception was thrown while we tried to write to the cache. Report it to the user.
                    # (We catch `RuntimeError` here because it will be raised by Apache Spark if we do not
                    # collect dataframe before using `st.cache_data`.)
                    if True in [
                        type_util.is_type(computed_value, type_name)
                        for type_name in UNEVALUATED_DATAFRAME_TYPES
                    ]:
                        raise UnevaluatedDataFrameError(
                            f"""
                            The function {get_cached_func_name_md(self._info.func)} is decorated with `st.cache_data` but it returns an unevaluated dataframe
                            of type `{type_util.get_fqn_type(computed_value)}`. Please call `collect()` or `to_pandas()` on the dataframe before returning it,
                            so `st.cache_data` can serialize and cache it."""
                        )
                    raise UnserializableReturnValueError(
                        return_value=computed_value, func=self._info.func
                    )

    def clear(self):
        """Clear the wrapped function's associated cache."""
        cache = self._info.get_function_cache(self._function_key)
        cache.clear()


def _make_value_key(
    cache_type: CacheType,
    func: types.FunctionType,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
) -> str:
    """Create the key for a value within a cache.

    This key is generated from the function's arguments. All arguments
    will be hashed, except for those named with a leading "_".

    Raises
    ------
    StreamlitAPIException
        Raised (with a nicely-formatted explanation message) if we encounter
        an un-hashable arg.
    """

    # Create a (name, value) list of all *args and **kwargs passed to the
    # function.
    arg_pairs: list[tuple[str | None, Any]] = []
    for arg_idx in range(len(func_args)):
        arg_name = _get_positional_arg_name(func, arg_idx)
        arg_pairs.append((arg_name, func_args[arg_idx]))

    for kw_name, kw_val in func_kwargs.items():
        # **kwargs ordering is preserved, per PEP 468
        # https://www.python.org/dev/peps/pep-0468/, so this iteration is
        # deterministic.
        arg_pairs.append((kw_name, kw_val))

    # Create the hash from each arg value, except for those args whose name
    # starts with "_". (Underscore-prefixed args are deliberately excluded from
    # hashing.)
    args_hasher = hashlib.new("md5")
    for arg_name, arg_value in arg_pairs:
        if arg_name is not None and arg_name.startswith("_"):
            _LOGGER.debug("Not hashing %s because it starts with _", arg_name)
            continue

        try:
            update_hash(
                (arg_name, arg_value),
                hasher=args_hasher,
                cache_type=cache_type,
            )
        except UnhashableTypeError as exc:
            raise UnhashableParamError(cache_type, func, arg_name, arg_value, exc)

    value_key = args_hasher.hexdigest()
    _LOGGER.debug("Cache key: %s", value_key)

    return value_key


def _make_function_key(cache_type: CacheType, func: types.FunctionType) -> str:
    """Create the unique key for a function's cache.

    A function's key is stable across reruns of the app, and changes when
    the function's source code changes.
    """
    func_hasher = hashlib.new("md5")

    # Include the function's __module__ and __qualname__ strings in the hash.
    # This means that two identical functions in different modules
    # will not share a hash; it also means that two identical *nested*
    # functions in the same module will not share a hash.
    update_hash(
        (func.__module__, func.__qualname__),
        hasher=func_hasher,
        cache_type=cache_type,
    )

    # Include the function's source code in its hash. If the source code can't
    # be retrieved, fall back to the function's bytecode instead.
    source_code: str | bytes
    try:
        source_code = inspect.getsource(func)
    except OSError as e:
        _LOGGER.debug(
            "Failed to retrieve function's source code when building its key; falling back to bytecode. err={0}",
            e,
        )
        source_code = func.__code__.co_code

    update_hash(
        source_code,
        hasher=func_hasher,
        cache_type=cache_type,
    )

    cache_key = func_hasher.hexdigest()
    return cache_key


def _get_positional_arg_name(func: types.FunctionType, arg_index: int) -> str | None:
    """Return the name of a function's positional argument.

    If arg_index is out of range, or refers to a parameter that is not a
    named positional argument (e.g. an *args, **kwargs, or keyword-only param),
    return None instead.
    """
    if arg_index < 0:
        return None

    params: list[inspect.Parameter] = list(inspect.signature(func).parameters.values())
    if arg_index >= len(params):
        return None

    if params[arg_index].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    ):
        return params[arg_index].name

    return None
