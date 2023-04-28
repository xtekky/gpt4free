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

# `html` and `iframe` are part of Custom Components, so they appear in this
# `streamlit.components.v1` namespace.
import streamlit

# Modules that the user should have access to. These are imported with "as"
# syntax pass mypy checking with implicit_reexport disabled.
from streamlit.components.v1.components import declare_component as declare_component

html = streamlit._main._html
iframe = streamlit._main._iframe
