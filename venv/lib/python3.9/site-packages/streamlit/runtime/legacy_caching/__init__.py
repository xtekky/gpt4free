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

from streamlit.runtime.legacy_caching.caching import cache as cache
from streamlit.runtime.legacy_caching.caching import clear_cache as clear_cache
from streamlit.runtime.legacy_caching.caching import get_cache_path as get_cache_path
from streamlit.runtime.legacy_caching.caching import (
    maybe_show_cached_st_function_warning as maybe_show_cached_st_function_warning,
)
