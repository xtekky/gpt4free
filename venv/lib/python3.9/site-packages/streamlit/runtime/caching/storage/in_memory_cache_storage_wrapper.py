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
from __future__ import annotations

import math
import threading

from cachetools import TTLCache

from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.storage.cache_storage_protocol import (
    CacheStorage,
    CacheStorageContext,
    CacheStorageKeyNotFoundError,
)
from streamlit.runtime.stats import CacheStat

_LOGGER = get_logger(__name__)


class InMemoryCacheStorageWrapper(CacheStorage):
    """
    In-memory cache storage wrapper.

    This class wraps a cache storage and adds an in-memory cache front layer,
    which is used to reduce the number of calls to the storage.

    The in-memory cache is a TTL cache, which means that the entries are
    automatically removed if a given time to live (TTL) has passed.

    The in-memory cache is also an LRU cache, which means that the entries
    are automatically removed if the cache size exceeds a given maxsize.

    If the storage implements its strategy for maxsize, it is recommended
    (but not necessary) that the storage implement the same LRU strategy,
    otherwise a situation may arise when different items are deleted from
    the memory cache and from the storage.

    Notes
    -----
    Threading: in-memory caching layer is thread safe: we hold self._mem_cache_lock for
    working with this self._mem_cache object.
    However, we do not hold this lock when calling into the underlying storage,
    so it is the responsibility of the that storage to ensure that it is safe to use
    it from multiple threads.
    """

    def __init__(self, persist_storage: CacheStorage, context: CacheStorageContext):
        self.function_key = context.function_key
        self.function_display_name = context.function_display_name
        self._ttl_seconds = context.ttl_seconds
        self._max_entries = context.max_entries
        self._mem_cache: TTLCache[str, bytes] = TTLCache(
            maxsize=self.max_entries,
            ttl=self.ttl_seconds,
            timer=cache_utils.TTLCACHE_TIMER,
        )
        self._mem_cache_lock = threading.Lock()
        self._persist_storage = persist_storage

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds if self._ttl_seconds is not None else math.inf

    @property
    def max_entries(self) -> float:
        return float(self._max_entries) if self._max_entries is not None else math.inf

    def get(self, key: str) -> bytes:
        """
        Returns the stored value for the key or raise CacheStorageKeyNotFoundError if
        the key is not found
        """
        try:
            entry_bytes = self._read_from_mem_cache(key)
        except CacheStorageKeyNotFoundError:
            entry_bytes = self._persist_storage.get(key)
            self._write_to_mem_cache(key, entry_bytes)
        return entry_bytes

    def set(self, key: str, value: bytes) -> None:
        """Sets the value for a given key"""
        self._write_to_mem_cache(key, value)
        self._persist_storage.set(key, value)

    def delete(self, key: str) -> None:
        """Delete a given key"""
        self._remove_from_mem_cache(key)
        self._persist_storage.delete(key)

    def clear(self) -> None:
        """Delete all keys for the in memory cache, and also the persistent storage"""
        with self._mem_cache_lock:
            self._mem_cache.clear()
        self._persist_storage.clear()

    def get_stats(self) -> list[CacheStat]:
        """Returns a list of stats in bytes for the cache memory storage per item"""
        stats = []

        with self._mem_cache_lock:
            for item in self._mem_cache.values():
                stats.append(
                    CacheStat(
                        category_name="st_cache_data",
                        cache_name=self.function_display_name,
                        byte_length=len(item),
                    )
                )
        return stats

    def close(self) -> None:
        """Closes the cache storage"""
        self._persist_storage.close()

    def _read_from_mem_cache(self, key: str) -> bytes:
        with self._mem_cache_lock:
            if key in self._mem_cache:
                entry = bytes(self._mem_cache[key])
                _LOGGER.debug("Memory cache HIT: %s", key)
                return entry

            else:
                _LOGGER.debug("Memory cache MISS: %s", key)
                raise CacheStorageKeyNotFoundError("Key not found in mem cache")

    def _write_to_mem_cache(self, key: str, entry_bytes: bytes) -> None:
        with self._mem_cache_lock:
            self._mem_cache[key] = entry_bytes

    def _remove_from_mem_cache(self, key: str) -> None:
        with self._mem_cache_lock:
            self._mem_cache.pop(key, None)
