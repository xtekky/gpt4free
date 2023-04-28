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

from typing import Any, Dict, List, Optional, Tuple

from streamlit.logger import get_logger
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

LOGGER = get_logger(__name__)


class ForwardMsgQueue:
    """Accumulates a session's outgoing ForwardMsgs.

    Each AppSession adds messages to its queue, and the Server periodically
    flushes all session queues and delivers their messages to the appropriate
    clients.

    ForwardMsgQueue is not thread-safe - a queue should only be used from
    a single thread.
    """

    def __init__(self):
        self._queue: List[ForwardMsg] = []
        # A mapping of (delta_path -> _queue.indexof(msg)) for each
        # Delta message in the queue. We use this for coalescing
        # redundant outgoing Deltas (where a newer Delta supersedes
        # an older Delta, with the same delta_path, that's still in the
        # queue).
        self._delta_index_map: Dict[Tuple[int, ...], int] = dict()

    def get_debug(self) -> Dict[str, Any]:
        from google.protobuf.json_format import MessageToDict

        return {
            "queue": [MessageToDict(m) for m in self._queue],
            "ids": list(self._delta_index_map.keys()),
        }

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def enqueue(self, msg: ForwardMsg) -> None:
        """Add message into queue, possibly composing it with another message."""
        if not _is_composable_message(msg):
            self._queue.append(msg)
            return

        # If there's a Delta message with the same delta_path already in
        # the queue - meaning that it refers to the same location in
        # the app - we attempt to combine this new Delta into the old
        # one. This is an optimization that prevents redundant Deltas
        # from being sent to the frontend.
        delta_key = tuple(msg.metadata.delta_path)
        if delta_key in self._delta_index_map:
            index = self._delta_index_map[delta_key]
            old_msg = self._queue[index]
            composed_delta = _maybe_compose_deltas(old_msg.delta, msg.delta)
            if composed_delta is not None:
                new_msg = ForwardMsg()
                new_msg.delta.CopyFrom(composed_delta)
                new_msg.metadata.CopyFrom(msg.metadata)
                self._queue[index] = new_msg
                return

        # No composition occurred. Append this message to the queue, and
        # store its index for potential future composition.
        self._delta_index_map[delta_key] = len(self._queue)
        self._queue.append(msg)

    def clear(self) -> None:
        """Clear the queue."""
        self._queue = []
        self._delta_index_map = dict()

    def flush(self) -> List[ForwardMsg]:
        """Clear the queue and return a list of the messages it contained
        before being cleared.
        """
        queue = self._queue
        self.clear()
        return queue


def _is_composable_message(msg: ForwardMsg) -> bool:
    """True if the ForwardMsg is potentially composable with other ForwardMsgs."""
    if not msg.HasField("delta"):
        # Non-delta messages are never composable.
        return False

    # We never compose add_rows messages in Python, because the add_rows
    # operation can raise errors, and we don't have a good way of handling
    # those errors in the message queue.
    delta_type = msg.delta.WhichOneof("type")
    return delta_type != "add_rows" and delta_type != "arrow_add_rows"


def _maybe_compose_deltas(old_delta: Delta, new_delta: Delta) -> Optional[Delta]:
    """Combines new_delta onto old_delta if possible.

    If the combination takes place, the function returns a new Delta that
    should replace old_delta in the queue.

    If the new_delta is incompatible with old_delta, the function returns None.
    In this case, the new_delta should just be appended to the queue as normal.
    """
    old_delta_type = old_delta.WhichOneof("type")
    if old_delta_type == "add_block":
        # We never replace add_block deltas, because blocks can have
        # other dependent deltas later in the queue. For example:
        #
        #   placeholder = st.empty()
        #   placeholder.columns(1)
        #   placeholder.empty()
        #
        # The call to "placeholder.columns(1)" creates two blocks, a parent
        # container with delta_path (0, 0), and a column child with
        # delta_path (0, 0, 0). If the final "placeholder.empty()" Delta
        # is composed with the parent container Delta, the frontend will
        # throw an error when it tries to add that column child to what is
        # now just an element, and not a block.
        return None

    new_delta_type = new_delta.WhichOneof("type")
    if new_delta_type == "new_element":
        return new_delta

    if new_delta_type == "add_block":
        return new_delta

    return None
