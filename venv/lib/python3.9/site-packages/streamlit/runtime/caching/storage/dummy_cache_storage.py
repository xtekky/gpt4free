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

from streamlit.runtime.caching.storage.cache_storage_protocol import (
    CacheStorage,
    CacheStorageContext,
    CacheStorageKeyNotFoundError,
    CacheStorageManager,
)
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
    InMemoryCacheStorageWrapper,
)


class MemoryCacheStorageManager(CacheStorageManager):
    def create(self, context: CacheStorageContext) -> CacheStorage:
        """Creates a new cache storage instance wrapped with in-memory cache layer"""
        persist_storage = DummyCacheStorage()
        return InMemoryCacheStorageWrapper(
            persist_storage=persist_storage, context=context
        )

    def clear_all(self) -> None:
        raise NotImplementedError

    def check_context(self, context: CacheStorageContext) -> None:
        pass


class DummyCacheStorage(CacheStorage):
    def get(self, key: str) -> bytes:
        """
        Dummy gets the value for a given key,
        always raises an CacheStorageKeyNotFoundError
        """
        raise CacheStorageKeyNotFoundError("Key not found in dummy cache")

    def set(self, key: str, value: bytes) -> None:
        pass

    def delete(self, key: str) -> None:
        pass

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass
