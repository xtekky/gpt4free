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

import contextlib
from typing import Any, Iterator, Union

from google.protobuf.message import Message

from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_data_api import (
    CACHE_DATA_MESSAGE_REPLAY_CTX,
    CacheDataAPI,
    _data_caches,
)
from streamlit.runtime.caching.cache_errors import CACHE_DOCS_URL as CACHE_DOCS_URL
from streamlit.runtime.caching.cache_resource_api import (
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX,
    CacheResourceAPI,
    _resource_caches,
)
from streamlit.runtime.state.common import WidgetMetadata


def save_element_message(
    delta_type: str,
    element_proto: Message,
    invoked_dg_id: str,
    used_dg_id: str,
    returned_dg_id: str,
) -> None:
    """Save the message for an element to a thread-local callstack, so it can
    be used later to replay the element when a cache-decorated function's
    execution is skipped.
    """
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_element_message(
        delta_type, element_proto, invoked_dg_id, used_dg_id, returned_dg_id
    )
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_element_message(
        delta_type, element_proto, invoked_dg_id, used_dg_id, returned_dg_id
    )


def save_block_message(
    block_proto: Block,
    invoked_dg_id: str,
    used_dg_id: str,
    returned_dg_id: str,
) -> None:
    """Save the message for a block to a thread-local callstack, so it can
    be used later to replay the block when a cache-decorated function's
    execution is skipped.
    """
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_block_message(
        block_proto, invoked_dg_id, used_dg_id, returned_dg_id
    )
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_block_message(
        block_proto, invoked_dg_id, used_dg_id, returned_dg_id
    )


def save_widget_metadata(metadata: WidgetMetadata[Any]) -> None:
    """Save a widget's metadata to a thread-local callstack, so the widget
    can be registered again when that widget is replayed.
    """
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_widget_metadata(metadata)
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_widget_metadata(metadata)


def save_media_data(
    image_data: Union[bytes, str], mimetype: str, image_id: str
) -> None:
    CACHE_DATA_MESSAGE_REPLAY_CTX.save_image_data(image_data, mimetype, image_id)
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.save_image_data(image_data, mimetype, image_id)


def maybe_show_cached_st_function_warning(dg, st_func_name: str) -> None:
    CACHE_DATA_MESSAGE_REPLAY_CTX.maybe_show_cached_st_function_warning(
        dg, st_func_name
    )
    CACHE_RESOURCE_MESSAGE_REPLAY_CTX.maybe_show_cached_st_function_warning(
        dg, st_func_name
    )


@contextlib.contextmanager
def suppress_cached_st_function_warning() -> Iterator[None]:
    with CACHE_DATA_MESSAGE_REPLAY_CTX.suppress_cached_st_function_warning(), CACHE_RESOURCE_MESSAGE_REPLAY_CTX.suppress_cached_st_function_warning():
        yield


# Explicitly export public symbols
from streamlit.runtime.caching.cache_data_api import (
    get_data_cache_stats_provider as get_data_cache_stats_provider,
)
from streamlit.runtime.caching.cache_resource_api import (
    get_resource_cache_stats_provider as get_resource_cache_stats_provider,
)

# Create and export public API singletons.
cache_data = CacheDataAPI(decorator_metric_name="cache_data")
cache_resource = CacheResourceAPI(decorator_metric_name="cache_resource")

# Deprecated singletons
_MEMO_WARNING = (
    f"`st.experimental_memo` is deprecated. Please use the new command `st.cache_data` instead, "
    f"which has the same behavior. More information [in our docs]({CACHE_DOCS_URL})."
)

experimental_memo = CacheDataAPI(
    decorator_metric_name="experimental_memo", deprecation_warning=_MEMO_WARNING
)

_SINGLETON_WARNING = (
    f"`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, "
    f"which has the same behavior. More information [in our docs]({CACHE_DOCS_URL})."
)

experimental_singleton = CacheResourceAPI(
    decorator_metric_name="experimental_singleton",
    deprecation_warning=_SINGLETON_WARNING,
)
