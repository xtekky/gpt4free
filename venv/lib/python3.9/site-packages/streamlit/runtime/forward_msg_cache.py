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

import hashlib
from typing import TYPE_CHECKING, Dict, List, MutableMapping, Optional
from weakref import WeakKeyDictionary

from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider

if TYPE_CHECKING:
    from streamlit.runtime.app_session import AppSession

LOGGER = get_logger(__name__)


def populate_hash_if_needed(msg: ForwardMsg) -> str:
    """Computes and assigns the unique hash for a ForwardMsg.

    If the ForwardMsg already has a hash, this is a no-op.

    Parameters
    ----------
    msg : ForwardMsg

    Returns
    -------
    string
        The message's hash, returned here for convenience. (The hash
        will also be assigned to the ForwardMsg; callers do not need
        to do this.)

    """
    if msg.hash == "":
        # Move the message's metadata aside. It's not part of the
        # hash calculation.
        metadata = msg.metadata
        msg.ClearField("metadata")

        # MD5 is good enough for what we need, which is uniqueness.
        hasher = hashlib.md5()
        hasher.update(msg.SerializeToString())
        msg.hash = hasher.hexdigest()

        # Restore metadata.
        msg.metadata.CopyFrom(metadata)

    return msg.hash


def create_reference_msg(msg: ForwardMsg) -> ForwardMsg:
    """Create a ForwardMsg that refers to the given message via its hash.

    The reference message will also get a copy of the source message's
    metadata.

    Parameters
    ----------
    msg : ForwardMsg
        The ForwardMsg to create the reference to.

    Returns
    -------
    ForwardMsg
        A new ForwardMsg that "points" to the original message via the
        ref_hash field.

    """
    ref_msg = ForwardMsg()
    ref_msg.ref_hash = populate_hash_if_needed(msg)
    ref_msg.metadata.CopyFrom(msg.metadata)
    return ref_msg


class ForwardMsgCache(CacheStatsProvider):
    """A cache of ForwardMsgs.

    Large ForwardMsgs (e.g. those containing big DataFrame payloads) are
    stored in this cache. The server can choose to send a ForwardMsg's hash,
    rather than the message itself, to a client. Clients can then
    request messages from this cache via another endpoint.

    This cache is *not* thread safe. It's intended to only be accessed by
    the server thread.

    """

    class Entry:
        """Cache entry.

        Stores the cached message, and the set of AppSessions
        that we've sent the cached message to.

        """

        def __init__(self, msg: ForwardMsg):
            self.msg = msg
            self._session_script_run_counts: MutableMapping[
                "AppSession", int
            ] = WeakKeyDictionary()

        def __repr__(self) -> str:
            return util.repr_(self)

        def add_session_ref(self, session: "AppSession", script_run_count: int) -> None:
            """Adds a reference to a AppSession that has referenced
            this Entry's message.

            Parameters
            ----------
            session : AppSession
            script_run_count : int
                The session's run count at the time of the call

            """
            prev_run_count = self._session_script_run_counts.get(session, 0)
            if script_run_count < prev_run_count:
                LOGGER.error(
                    "New script_run_count (%s) is < prev_run_count (%s). "
                    "This should never happen!" % (script_run_count, prev_run_count)
                )
                script_run_count = prev_run_count
            self._session_script_run_counts[session] = script_run_count

        def has_session_ref(self, session: "AppSession") -> bool:
            return session in self._session_script_run_counts

        def get_session_ref_age(
            self, session: "AppSession", script_run_count: int
        ) -> int:
            """The age of the given session's reference to the Entry,
            given a new script_run_count.

            """
            return script_run_count - self._session_script_run_counts[session]

        def remove_session_ref(self, session: "AppSession") -> None:
            del self._session_script_run_counts[session]

        def has_refs(self) -> bool:
            """True if this Entry has references from any AppSession.

            If not, it can be removed from the cache.
            """
            return len(self._session_script_run_counts) > 0

    def __init__(self):
        self._entries: Dict[str, "ForwardMsgCache.Entry"] = {}

    def __repr__(self) -> str:
        return util.repr_(self)

    def add_message(
        self, msg: ForwardMsg, session: "AppSession", script_run_count: int
    ) -> None:
        """Add a ForwardMsg to the cache.

        The cache will also record a reference to the given AppSession,
        so that it can track which sessions have already received
        each given ForwardMsg.

        Parameters
        ----------
        msg : ForwardMsg
        session : AppSession
        script_run_count : int
            The number of times the session's script has run

        """
        populate_hash_if_needed(msg)
        entry = self._entries.get(msg.hash, None)
        if entry is None:
            entry = ForwardMsgCache.Entry(msg)
            self._entries[msg.hash] = entry
        entry.add_session_ref(session, script_run_count)

    def get_message(self, hash: str) -> Optional[ForwardMsg]:
        """Return the message with the given ID if it exists in the cache.

        Parameters
        ----------
        hash : string
            The id of the message to retrieve.

        Returns
        -------
        ForwardMsg | None

        """
        entry = self._entries.get(hash, None)
        return entry.msg if entry else None

    def has_message_reference(
        self, msg: ForwardMsg, session: "AppSession", script_run_count: int
    ) -> bool:
        """Return True if a session has a reference to a message."""
        populate_hash_if_needed(msg)

        entry = self._entries.get(msg.hash, None)
        if entry is None or not entry.has_session_ref(session):
            return False

        # Ensure we're not expired
        age = entry.get_session_ref_age(session, script_run_count)
        return age <= int(config.get_option("global.maxCachedMessageAge"))

    def remove_expired_session_entries(
        self, session: "AppSession", script_run_count: int
    ) -> None:
        """Remove any cached messages that have expired from the given session.

        This should be called each time a AppSession finishes executing.

        Parameters
        ----------
        session : AppSession
        script_run_count : int
            The number of times the session's script has run

        """
        max_age = config.get_option("global.maxCachedMessageAge")

        # Operate on a copy of our entries dict.
        # We may be deleting from it.
        for msg_hash, entry in self._entries.copy().items():
            if not entry.has_session_ref(session):
                continue

            age = entry.get_session_ref_age(session, script_run_count)
            if age > max_age:
                LOGGER.debug(
                    "Removing expired entry [session=%s, hash=%s, age=%s]",
                    id(session),
                    msg_hash,
                    age,
                )
                entry.remove_session_ref(session)
                if not entry.has_refs():
                    # The entry has no more references. Remove it from
                    # the cache completely.
                    del self._entries[msg_hash]

    def clear(self) -> None:
        """Remove all entries from the cache"""
        self._entries.clear()

    def get_stats(self) -> List[CacheStat]:
        stats: List[CacheStat] = []
        for entry_hash, entry in self._entries.items():
            stats.append(
                CacheStat(
                    category_name="ForwardMessageCache",
                    cache_name="",
                    byte_length=entry.msg.ByteSize(),
                )
            )
        return stats
