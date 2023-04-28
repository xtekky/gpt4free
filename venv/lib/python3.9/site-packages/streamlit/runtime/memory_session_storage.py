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

from typing import List, MutableMapping, Optional

from cachetools import TTLCache

from streamlit.runtime.session_manager import SessionInfo, SessionStorage


class MemorySessionStorage(SessionStorage):
    """A SessionStorage that stores sessions in memory.

    At most maxsize sessions are stored with a TTL of ttl seconds. This class is really
    just a thin wrapper around cachetools.TTLCache that complies with the SessionStorage
    protocol.
    """

    # NOTE: The defaults for maxsize and ttl are chosen arbitrarily for now. These
    # numbers are reasonable as the main problems we're trying to solve at the moment are
    # caused by transient disconnects that are usually just short network blips. In the
    # future, we may want to increase both to support use cases such as saving state for
    # much longer periods of time. For example, we may want session state to persist if
    # a user closes their laptop lid and comes back to an app hours later.
    def __init__(
        self,
        maxsize: int = 128,
        ttl_seconds: int = 2 * 60,  # 2 minutes
    ) -> None:
        """Instantiate a new MemorySessionStorage.

        Parameters
        ----------
        maxsize
            The maximum number of sessions we allow to be stored in this
            MemorySessionStorage. If an entry needs to be removed because we have
            exceeded this number, either
              * an expired entry is removed, or
              * the least recently used entry is removed (if no entries have expired).

        ttl_seconds
            The time in seconds for an entry added to a MemorySessionStorage to live.
            After this amount of time has passed for a given entry, it becomes
            inaccessible and will be removed eventually.
        """

        self._cache: MutableMapping[str, SessionInfo] = TTLCache(
            maxsize=maxsize, ttl=ttl_seconds
        )

    def get(self, session_id: str) -> Optional[SessionInfo]:
        return self._cache.get(session_id, None)

    def save(self, session_info: SessionInfo) -> None:
        self._cache[session_info.session.id] = session_info

    def delete(self, session_id: str) -> None:
        del self._cache[session_id]

    def list(self) -> List[SessionInfo]:
        return list(self._cache.values())
