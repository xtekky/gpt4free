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

import enum


class CacheType(enum.Enum):
    """The function cache types we implement."""

    DATA = "DATA"
    RESOURCE = "RESOURCE"


def get_decorator_api_name(cache_type: CacheType) -> str:
    """Return the name of the public decorator API for the given CacheType."""
    if cache_type is CacheType.DATA:
        return "cache_data"
    if cache_type is CacheType.RESOURCE:
        return "cache_resource"
    raise RuntimeError(f"Unrecognized CacheType '{cache_type}'")
