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

"""A library of caching utilities."""

import contextlib
import functools
import hashlib
import inspect
import math
import os
import pickle
import shutil
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from cachetools import TTLCache
from pympler.asizeof import asizeof

import streamlit as st
from streamlit import config, file_util, util
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.elements.spinner import spinner
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.errors import StreamlitAPIWarning
from streamlit.logger import get_logger
from streamlit.runtime.caching import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
from streamlit.runtime.legacy_caching.hashing import (
    HashFuncsDict,
    HashReason,
    update_hash,
)
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.stats import CacheStat, CacheStatsProvider

_LOGGER = get_logger(__name__)

# The timer function we use with TTLCache. This is the default timer func, but
# is exposed here as a constant so that it can be patched in unit tests.
_TTLCACHE_TIMER = time.monotonic


_CacheEntry = namedtuple("_CacheEntry", ["value", "hash"])
_DiskCacheEntry = namedtuple("_DiskCacheEntry", ["value"])

# When we show the "st.cache is deprecated" warning, we make a recommendation about which new
# cache decorator to switch to for the following data types:
NEW_CACHE_FUNC_RECOMMENDATIONS: Dict[str, CacheType] = {
    # cache_data recommendations:
    "str": CacheType.DATA,
    "float": CacheType.DATA,
    "int": CacheType.DATA,
    "bytes": CacheType.DATA,
    "bool": CacheType.DATA,
    "datetime.datetime": CacheType.DATA,
    "pandas.DataFrame": CacheType.DATA,
    "pandas.Series": CacheType.DATA,
    "numpy.bool_": CacheType.DATA,
    "numpy.bool8": CacheType.DATA,
    "numpy.ndarray": CacheType.DATA,
    "numpy.float_": CacheType.DATA,
    "numpy.float16": CacheType.DATA,
    "numpy.float32": CacheType.DATA,
    "numpy.float64": CacheType.DATA,
    "numpy.float96": CacheType.DATA,
    "numpy.float128": CacheType.DATA,
    "numpy.int_": CacheType.DATA,
    "numpy.int8": CacheType.DATA,
    "numpy.int16": CacheType.DATA,
    "numpy.int32": CacheType.DATA,
    "numpy.int64": CacheType.DATA,
    "numpy.intp": CacheType.DATA,
    "numpy.uint8": CacheType.DATA,
    "numpy.uint16": CacheType.DATA,
    "numpy.uint32": CacheType.DATA,
    "numpy.uint64": CacheType.DATA,
    "numpy.uintp": CacheType.DATA,
    "PIL.Image.Image": CacheType.DATA,
    "plotly.graph_objects.Figure": CacheType.DATA,
    "matplotlib.figure.Figure": CacheType.DATA,
    "altair.Chart": CacheType.DATA,
    # cache_resource recommendations:
    "pyodbc.Connection": CacheType.RESOURCE,
    "pymongo.mongo_client.MongoClient": CacheType.RESOURCE,
    "mysql.connector.MySQLConnection": CacheType.RESOURCE,
    "psycopg2.connection": CacheType.RESOURCE,
    "psycopg2.extensions.connection": CacheType.RESOURCE,
    "snowflake.connector.connection.SnowflakeConnection": CacheType.RESOURCE,
    "snowflake.snowpark.sessions.Session": CacheType.RESOURCE,
    "sqlalchemy.engine.base.Engine": CacheType.RESOURCE,
    "sqlite3.Connection": CacheType.RESOURCE,
    "torch.nn.Module": CacheType.RESOURCE,
    "tensorflow.keras.Model": CacheType.RESOURCE,
    "tensorflow.Module": CacheType.RESOURCE,
    "tensorflow.compat.v1.Session": CacheType.RESOURCE,
    "transformers.Pipeline": CacheType.RESOURCE,
    "transformers.PreTrainedTokenizer": CacheType.RESOURCE,
    "transformers.PreTrainedTokenizerFast": CacheType.RESOURCE,
    "transformers.PreTrainedTokenizerBase": CacheType.RESOURCE,
    "transformers.PreTrainedModel": CacheType.RESOURCE,
    "transformers.TFPreTrainedModel": CacheType.RESOURCE,
    "transformers.FlaxPreTrainedModel": CacheType.RESOURCE,
}


def _make_deprecation_warning(cached_value: Any) -> str:
    """Build a deprecation warning string for a cache function that has returned the given
    value.
    """
    typename = type(cached_value).__qualname__
    cache_type_rec = NEW_CACHE_FUNC_RECOMMENDATIONS.get(typename)
    if cache_type_rec is not None:
        # We have a recommended cache func for the cached value:
        return (
            f"`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n"
            f"`st.cache_data` or `st.cache_resource`. Based on this function's return value\n"
            f"of type `{typename}`, we recommend using `st.{get_decorator_api_name(cache_type_rec)}`.\n\n"
            f"More information [in our docs]({CACHE_DOCS_URL})."
        )

    # We do not have a recommended cache func for the cached value:
    return (
        f"`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n"
        f"`st.cache_data` or `st.cache_resource`.\n\n"
        f"More information [in our docs]({CACHE_DOCS_URL})."
    )


@dataclass
class MemCache:
    cache: TTLCache
    display_name: str


class _MemCaches(CacheStatsProvider):
    """Manages all in-memory st.cache caches"""

    def __init__(self):
        # Contains a cache object for each st.cache'd function
        self._lock = threading.RLock()
        self._function_caches: Dict[str, MemCache] = {}

    def __repr__(self) -> str:
        return util.repr_(self)

    def get_cache(
        self,
        key: str,
        max_entries: Optional[float],
        ttl: Optional[float],
        display_name: str = "",
    ) -> MemCache:
        """Return the mem cache for the given key.

        If it doesn't exist, create a new one with the given params.
        """

        if max_entries is None:
            max_entries = math.inf
        if ttl is None:
            ttl = math.inf

        if not isinstance(max_entries, (int, float)):
            raise RuntimeError("max_entries must be an int")
        if not isinstance(ttl, (int, float)):
            raise RuntimeError("ttl must be a float")

        # Get the existing cache, if it exists, and validate that its params
        # haven't changed.
        with self._lock:
            mem_cache = self._function_caches.get(key)
            if (
                mem_cache is not None
                and mem_cache.cache.ttl == ttl
                and mem_cache.cache.maxsize == max_entries
            ):
                return mem_cache

            # Create a new cache object and put it in our dict
            _LOGGER.debug(
                "Creating new mem_cache (key=%s, max_entries=%s, ttl=%s)",
                key,
                max_entries,
                ttl,
            )
            ttl_cache = TTLCache(maxsize=max_entries, ttl=ttl, timer=_TTLCACHE_TIMER)
            mem_cache = MemCache(ttl_cache, display_name)
            self._function_caches[key] = mem_cache
            return mem_cache

    def clear(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._function_caches = {}

    def get_stats(self) -> List[CacheStat]:
        with self._lock:
            # Shallow-clone our caches. We don't want to hold the global
            # lock during stats-gathering.
            function_caches = self._function_caches.copy()

        stats = [
            CacheStat("st_cache", cache.display_name, asizeof(c))
            for cache in function_caches.values()
            for c in cache.cache
        ]
        return stats


# Our singleton _MemCaches instance
_mem_caches = _MemCaches()


# A thread-local counter that's incremented when we enter @st.cache
# and decremented when we exit.
class ThreadLocalCacheInfo(threading.local):
    def __init__(self):
        self.cached_func_stack: List[Callable[..., Any]] = []
        self.suppress_st_function_warning = 0

    def __repr__(self) -> str:
        return util.repr_(self)


_cache_info = ThreadLocalCacheInfo()


@contextlib.contextmanager
def _calling_cached_function(func: Callable[..., Any]) -> Iterator[None]:
    _cache_info.cached_func_stack.append(func)
    try:
        yield
    finally:
        _cache_info.cached_func_stack.pop()


@contextlib.contextmanager
def suppress_cached_st_function_warning() -> Iterator[None]:
    _cache_info.suppress_st_function_warning += 1
    try:
        yield
    finally:
        _cache_info.suppress_st_function_warning -= 1
        assert _cache_info.suppress_st_function_warning >= 0


def _show_cached_st_function_warning(
    dg: "st.delta_generator.DeltaGenerator",
    st_func_name: str,
    cached_func: Callable[..., Any],
) -> None:
    # Avoid infinite recursion by suppressing additional cached
    # function warnings from within the cached function warning.
    with suppress_cached_st_function_warning():
        e = CachedStFunctionWarning(st_func_name, cached_func)
        dg.exception(e)


def maybe_show_cached_st_function_warning(
    dg: "st.delta_generator.DeltaGenerator", st_func_name: str
) -> None:
    """If appropriate, warn about calling st.foo inside @cache.

    DeltaGenerator's @_with_element and @_widget wrappers use this to warn
    the user when they're calling st.foo() from within a function that is
    wrapped in @st.cache.

    Parameters
    ----------
    dg : DeltaGenerator
        The DeltaGenerator to publish the warning to.

    st_func_name : str
        The name of the Streamlit function that was called.

    """
    if (
        len(_cache_info.cached_func_stack) > 0
        and _cache_info.suppress_st_function_warning <= 0
    ):
        cached_func = _cache_info.cached_func_stack[-1]
        _show_cached_st_function_warning(dg, st_func_name, cached_func)


def _read_from_mem_cache(
    mem_cache: MemCache,
    key: str,
    allow_output_mutation: bool,
    func_or_code: Callable[..., Any],
    hash_funcs: Optional[HashFuncsDict],
) -> Any:
    cache = mem_cache.cache
    if key in cache:
        entry = cache[key]

        if not allow_output_mutation:
            computed_output_hash = _get_output_hash(
                entry.value, func_or_code, hash_funcs
            )
            stored_output_hash = entry.hash

            if computed_output_hash != stored_output_hash:
                _LOGGER.debug("Cached object was mutated: %s", key)
                raise CachedObjectMutationError(entry.value, func_or_code)

        _LOGGER.debug("Memory cache HIT: %s", type(entry.value))
        return entry.value

    else:
        _LOGGER.debug("Memory cache MISS: %s", key)
        raise CacheKeyNotFoundError("Key not found in mem cache")


def _write_to_mem_cache(
    mem_cache: MemCache,
    key: str,
    value: Any,
    allow_output_mutation: bool,
    func_or_code: Callable[..., Any],
    hash_funcs: Optional[HashFuncsDict],
) -> None:
    if allow_output_mutation:
        hash = None
    else:
        hash = _get_output_hash(value, func_or_code, hash_funcs)

    mem_cache.display_name = f"{func_or_code.__module__}.{func_or_code.__qualname__}"
    mem_cache.cache[key] = _CacheEntry(value=value, hash=hash)


def _get_output_hash(
    value: Any, func_or_code: Callable[..., Any], hash_funcs: Optional[HashFuncsDict]
) -> bytes:
    hasher = hashlib.new("md5")
    update_hash(
        value,
        hasher=hasher,
        hash_funcs=hash_funcs,
        hash_reason=HashReason.CACHING_FUNC_OUTPUT,
        hash_source=func_or_code,
    )
    return hasher.digest()


def _read_from_disk_cache(key: str) -> Any:
    path = file_util.get_streamlit_file_path("cache", "%s.pickle" % key)
    try:
        with file_util.streamlit_read(path, binary=True) as input:
            entry = pickle.load(input)
            value = entry.value
            _LOGGER.debug("Disk cache HIT: %s", type(value))
    except util.Error as e:
        _LOGGER.error(e)
        raise CacheError("Unable to read from cache: %s" % e)

    except FileNotFoundError:
        raise CacheKeyNotFoundError("Key not found in disk cache")
    return value


def _write_to_disk_cache(key: str, value: Any) -> None:
    path = file_util.get_streamlit_file_path("cache", "%s.pickle" % key)

    try:
        with file_util.streamlit_write(path, binary=True) as output:
            entry = _DiskCacheEntry(value=value)
            pickle.dump(entry, output, pickle.HIGHEST_PROTOCOL)
    except util.Error as e:
        _LOGGER.debug(e)
        # Clean up file so we don't leave zero byte files.
        try:
            os.remove(path)
        except (FileNotFoundError, IOError, OSError):
            # If we can't remove the file, it's not a big deal.
            pass
        raise CacheError("Unable to write to cache: %s" % e)


def _read_from_cache(
    mem_cache: MemCache,
    key: str,
    persist: bool,
    allow_output_mutation: bool,
    func_or_code: Callable[..., Any],
    hash_funcs: Optional[HashFuncsDict] = None,
) -> Any:
    """Read a value from the cache.

    Our goal is to read from memory if possible. If the data was mutated (hash
    changed), we show a warning. If reading from memory fails, we either read
    from disk or rerun the code.
    """
    try:
        return _read_from_mem_cache(
            mem_cache, key, allow_output_mutation, func_or_code, hash_funcs
        )

    except CachedObjectMutationError as e:
        handle_uncaught_app_exception(CachedObjectMutationWarning(e))
        return e.cached_value

    except CacheKeyNotFoundError as e:
        if persist:
            value = _read_from_disk_cache(key)
            _write_to_mem_cache(
                mem_cache, key, value, allow_output_mutation, func_or_code, hash_funcs
            )
            return value
        raise e


@gather_metrics("_cache_object")
def _write_to_cache(
    mem_cache: MemCache,
    key: str,
    value: Any,
    persist: bool,
    allow_output_mutation: bool,
    func_or_code: Callable[..., Any],
    hash_funcs: Optional[HashFuncsDict] = None,
):
    _write_to_mem_cache(
        mem_cache, key, value, allow_output_mutation, func_or_code, hash_funcs
    )
    if persist:
        _write_to_disk_cache(key, value)


F = TypeVar("F", bound=Callable[..., Any])


@overload
def cache(
    func: F,
    persist: bool = False,
    allow_output_mutation: bool = False,
    show_spinner: bool = True,
    suppress_st_warning: bool = False,
    hash_funcs: Optional[HashFuncsDict] = None,
    max_entries: Optional[int] = None,
    ttl: Optional[float] = None,
) -> F:
    ...


@overload
def cache(
    func: None = None,
    persist: bool = False,
    allow_output_mutation: bool = False,
    show_spinner: bool = True,
    suppress_st_warning: bool = False,
    hash_funcs: Optional[HashFuncsDict] = None,
    max_entries: Optional[int] = None,
    ttl: Optional[float] = None,
) -> Callable[[F], F]:
    ...


def cache(
    func: Optional[F] = None,
    persist: bool = False,
    allow_output_mutation: bool = False,
    show_spinner: bool = True,
    suppress_st_warning: bool = False,
    hash_funcs: Optional[HashFuncsDict] = None,
    max_entries: Optional[int] = None,
    ttl: Optional[float] = None,
) -> Union[Callable[[F], F], F]:
    """Function decorator to memoize function executions.

    Parameters
    ----------
    func : callable
        The function to cache. Streamlit hashes the function and dependent code.

    persist : boolean
        Whether to persist the cache on disk.

    allow_output_mutation : boolean
        Streamlit shows a warning when return values are mutated, as that
        can have unintended consequences. This is done by hashing the return value internally.

        If you know what you're doing and would like to override this warning, set this to True.

    show_spinner : boolean
        Enable the spinner. Default is True to show a spinner when there is
        a cache miss.

    suppress_st_warning : boolean
        Suppress warnings about calling Streamlit commands from within
        the cached function.

    hash_funcs : dict or None
        Mapping of types or fully qualified names to hash functions. This is used to override
        the behavior of the hasher inside Streamlit's caching mechanism: when the hasher
        encounters an object, it will first check to see if its type matches a key in this
        dict and, if so, will use the provided function to generate a hash for it. See below
        for an example of how this can be used.

    max_entries : int or None
        The maximum number of entries to keep in the cache, or None
        for an unbounded cache. (When a new entry is added to a full cache,
        the oldest cached entry will be removed.) The default is None.

    ttl : float or None
        The maximum number of seconds to keep an entry in the cache, or
        None if cache entries should not expire. The default is None.

    Example
    -------
    >>> import streamlit as st
    >>>
    >>> @st.cache
    ... def fetch_and_clean_data(url):
    ...     # Fetch data from URL here, and then clean it up.
    ...     return data
    ...
    >>> d1 = fetch_and_clean_data(DATA_URL_1)
    >>> # Actually executes the function, since this is the first time it was
    >>> # encountered.
    >>>
    >>> d2 = fetch_and_clean_data(DATA_URL_1)
    >>> # Does not execute the function. Instead, returns its previously computed
    >>> # value. This means that now the data in d1 is the same as in d2.
    >>>
    >>> d3 = fetch_and_clean_data(DATA_URL_2)
    >>> # This is a different URL, so the function executes.

    To set the ``persist`` parameter, use this command as follows:

    >>> @st.cache(persist=True)
    ... def fetch_and_clean_data(url):
    ...     # Fetch data from URL here, and then clean it up.
    ...     return data

    To disable hashing return values, set the ``allow_output_mutation`` parameter to ``True``:

    >>> @st.cache(allow_output_mutation=True)
    ... def fetch_and_clean_data(url):
    ...     # Fetch data from URL here, and then clean it up.
    ...     return data


    To override the default hashing behavior, pass a custom hash function.
    You can do that by mapping a type (e.g. ``MongoClient``) to a hash function (``id``) like this:

    >>> @st.cache(hash_funcs={MongoClient: id})
    ... def connect_to_database(url):
    ...     return MongoClient(url)

    Alternatively, you can map the type's fully-qualified name
    (e.g. ``"pymongo.mongo_client.MongoClient"``) to the hash function instead:

    >>> @st.cache(hash_funcs={"pymongo.mongo_client.MongoClient": id})
    ... def connect_to_database(url):
    ...     return MongoClient(url)

    """
    _LOGGER.debug("Entering st.cache: %s", func)

    # Support passing the params via function decorator, e.g.
    # @st.cache(persist=True, allow_output_mutation=True)
    if func is None:

        def wrapper(f: F) -> F:
            return cache(
                func=f,
                persist=persist,
                allow_output_mutation=allow_output_mutation,
                show_spinner=show_spinner,
                suppress_st_warning=suppress_st_warning,
                hash_funcs=hash_funcs,
                max_entries=max_entries,
                ttl=ttl,
            )

        return wrapper
    else:
        # To make mypy type narrow Optional[F] -> F
        non_optional_func = func

    cache_key = None

    @functools.wraps(non_optional_func)
    def wrapped_func(*args, **kwargs):
        """Wrapper function that only calls the underlying function on a cache miss.

        Cached objects are stored in the cache/ directory.
        """

        if not config.get_option("client.caching"):
            _LOGGER.debug("Purposefully skipping cache")
            return non_optional_func(*args, **kwargs)

        name = non_optional_func.__qualname__

        if len(args) == 0 and len(kwargs) == 0:
            message = "Running `%s()`." % name
        else:
            message = "Running `%s(...)`." % name

        def get_or_create_cached_value():
            nonlocal cache_key
            if cache_key is None:
                # Delay generating the cache key until the first call.
                # This way we can see values of globals, including functions
                # defined after this one.
                # If we generated the key earlier we would only hash those
                # globals by name, and miss changes in their code or value.
                cache_key = _hash_func(non_optional_func, hash_funcs)

            # First, get the cache that's attached to this function.
            # This cache's key is generated (above) from the function's code.
            mem_cache = _mem_caches.get_cache(cache_key, max_entries, ttl)

            # Next, calculate the key for the value we'll be searching for
            # within that cache. This key is generated from both the function's
            # code and the arguments that are passed into it. (Even though this
            # key is used to index into a per-function cache, it must be
            # globally unique, because it is *also* used for a global on-disk
            # cache that is *not* per-function.)
            value_hasher = hashlib.new("md5")

            if args:
                update_hash(
                    args,
                    hasher=value_hasher,
                    hash_funcs=hash_funcs,
                    hash_reason=HashReason.CACHING_FUNC_ARGS,
                    hash_source=non_optional_func,
                )

            if kwargs:
                update_hash(
                    kwargs,
                    hasher=value_hasher,
                    hash_funcs=hash_funcs,
                    hash_reason=HashReason.CACHING_FUNC_ARGS,
                    hash_source=non_optional_func,
                )

            value_key = value_hasher.hexdigest()

            # Avoid recomputing the body's hash by just appending the
            # previously-computed hash to the arg hash.
            value_key = "%s-%s" % (value_key, cache_key)

            _LOGGER.debug("Cache key: %s", value_key)

            try:
                return_value = _read_from_cache(
                    mem_cache=mem_cache,
                    key=value_key,
                    persist=persist,
                    allow_output_mutation=allow_output_mutation,
                    func_or_code=non_optional_func,
                    hash_funcs=hash_funcs,
                )
                _LOGGER.debug("Cache hit: %s", non_optional_func)

            except CacheKeyNotFoundError:
                _LOGGER.debug("Cache miss: %s", non_optional_func)

                with _calling_cached_function(non_optional_func):
                    if suppress_st_warning:
                        with suppress_cached_st_function_warning():
                            return_value = non_optional_func(*args, **kwargs)
                    else:
                        return_value = non_optional_func(*args, **kwargs)

                _write_to_cache(
                    mem_cache=mem_cache,
                    key=value_key,
                    value=return_value,
                    persist=persist,
                    allow_output_mutation=allow_output_mutation,
                    func_or_code=non_optional_func,
                    hash_funcs=hash_funcs,
                )

            # st.cache is deprecated. We show a warning every time it's used.
            show_deprecation_warning(_make_deprecation_warning(return_value))

            return return_value

        if show_spinner:
            with spinner(message):
                return get_or_create_cached_value()
        else:
            return get_or_create_cached_value()

    # Make this a well-behaved decorator by preserving important function
    # attributes.
    try:
        wrapped_func.__dict__.update(non_optional_func.__dict__)
    except AttributeError:
        # For normal functions this should never happen, but if so it's not problematic.
        pass

    return cast(F, wrapped_func)


def _hash_func(func: Callable[..., Any], hash_funcs: Optional[HashFuncsDict]) -> str:
    # Create the unique key for a function's cache. The cache will be retrieved
    # from inside the wrapped function.
    #
    # A naive implementation would involve simply creating the cache object
    # right in the wrapper, which in a normal Python script would be executed
    # only once. But in Streamlit, we reload all modules related to a user's
    # app when the app is re-run, which means that - among other things - all
    # function decorators in the app will be re-run, and so any decorator-local
    # objects will be recreated.
    #
    # Furthermore, our caches can be destroyed and recreated (in response to
    # cache clearing, for example), which means that retrieving the function's
    # cache in the decorator (so that the wrapped function can save a lookup)
    # is incorrect: the cache itself may be recreated between
    # decorator-evaluation time and decorated-function-execution time. So we
    # must retrieve the cache object *and* perform the cached-value lookup
    # inside the decorated function.
    func_hasher = hashlib.new("md5")

    # Include the function's __module__ and __qualname__ strings in the hash.
    # This means that two identical functions in different modules
    # will not share a hash; it also means that two identical *nested*
    # functions in the same module will not share a hash.
    # We do not pass `hash_funcs` here, because we don't want our function's
    # name to get an unexpected hash.
    update_hash(
        (func.__module__, func.__qualname__),
        hasher=func_hasher,
        hash_funcs=None,
        hash_reason=HashReason.CACHING_FUNC_BODY,
        hash_source=func,
    )

    # Include the function's body in the hash. We *do* pass hash_funcs here,
    # because this step will be hashing any objects referenced in the function
    # body.
    update_hash(
        func,
        hasher=func_hasher,
        hash_funcs=hash_funcs,
        hash_reason=HashReason.CACHING_FUNC_BODY,
        hash_source=func,
    )
    cache_key = func_hasher.hexdigest()
    _LOGGER.debug(
        "mem_cache key for %s.%s: %s", func.__module__, func.__qualname__, cache_key
    )
    return cache_key


def clear_cache() -> bool:
    """Clear the memoization cache.

    Returns
    -------
    boolean
        True if the disk cache was cleared. False otherwise (e.g. cache file
        doesn't exist on disk).
    """
    _clear_mem_cache()
    return _clear_disk_cache()


def get_cache_path() -> str:
    return file_util.get_streamlit_file_path("cache")


def _clear_disk_cache() -> bool:
    # TODO: Only delete disk cache for functions related to the user's current
    # script.
    cache_path = get_cache_path()
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
        return True
    return False


def _clear_mem_cache() -> None:
    _mem_caches.clear()


class CacheError(Exception):
    pass


class CacheKeyNotFoundError(Exception):
    pass


class CachedObjectMutationError(ValueError):
    # This is used internally, but never shown to the user.
    # Users see CachedObjectMutationWarning instead.

    def __init__(self, cached_value, func_or_code):
        self.cached_value = cached_value
        if inspect.iscode(func_or_code):
            self.cached_func_name = "a code block"
        else:
            self.cached_func_name = _get_cached_func_name_md(func_or_code)

    def __repr__(self) -> str:
        return util.repr_(self)


class CachedStFunctionWarning(StreamlitAPIWarning):
    def __init__(self, st_func_name, cached_func):
        msg = self._get_message(st_func_name, cached_func)
        super(CachedStFunctionWarning, self).__init__(msg)

    def _get_message(self, st_func_name, cached_func):
        args = {
            "st_func_name": "`st.%s()` or `st.write()`" % st_func_name,
            "func_name": _get_cached_func_name_md(cached_func),
        }

        return (
            """
Your script uses %(st_func_name)s to write to your Streamlit app from within
some cached code at %(func_name)s. This code will only be called when we detect
a cache "miss", which can lead to unexpected results.

How to fix this:
* Move the %(st_func_name)s call outside %(func_name)s.
* Or, if you know what you're doing, use `@st.cache(suppress_st_warning=True)`
to suppress the warning.
            """
            % args
        ).strip("\n")


class CachedObjectMutationWarning(StreamlitAPIWarning):
    def __init__(self, orig_exc):
        msg = self._get_message(orig_exc)
        super(CachedObjectMutationWarning, self).__init__(msg)

    def _get_message(self, orig_exc):
        return (
            """
Return value of %(func_name)s was mutated between runs.

By default, Streamlit's cache should be treated as immutable, or it may behave
in unexpected ways. You received this warning because Streamlit detected
that an object returned by %(func_name)s was mutated outside of %(func_name)s.

How to fix this:
* If you did not mean to mutate that return value:
  - If possible, inspect your code to find and remove that mutation.
  - Otherwise, you could also clone the returned value so you can freely
    mutate it.
* If you actually meant to mutate the return value and know the consequences of
doing so, annotate the function with `@st.cache(allow_output_mutation=True)`.

For more information and detailed solutions check out [our documentation.]
(https://docs.streamlit.io/library/advanced-features/caching)
            """
            % {"func_name": orig_exc.cached_func_name}
        ).strip("\n")


def _get_cached_func_name_md(func: Callable[..., Any]) -> str:
    """Get markdown representation of the function name."""
    if hasattr(func, "__name__"):
        return "`%s()`" % func.__name__
    else:
        return "a cached function"
