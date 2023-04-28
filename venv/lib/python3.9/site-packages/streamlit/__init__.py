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

# isort: skip_file

"""Streamlit.

How to use Streamlit in 3 seconds:

    1. Write an app
    >>> import streamlit as st
    >>> st.write(anything_you_want)

    2. Run your app
    $ streamlit run my_script.py

    3. Use your app
    A new tab will open on your browser. That's your Streamlit app!

    4. Modify your code, save it, and watch changes live on your browser.

Take a look at the other commands in this module to find out what else
Streamlit can do:

    >>> dir(streamlit)

Or try running our "Hello World":

    $ streamlit hello

For more detailed info, see https://docs.streamlit.io.
"""

# IMPORTANT: Prefix with an underscore anything that the user shouldn't see.

# Must be at the top, to avoid circular dependency.
from streamlit import logger as _logger
from streamlit import config as _config
from streamlit.version import STREAMLIT_VERSION_STRING as _STREAMLIT_VERSION_STRING

# Give the package a version.
__version__ = _STREAMLIT_VERSION_STRING

from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
from streamlit.proto.RootContainer_pb2 import RootContainer as _RootContainer
from streamlit.runtime.caching import (
    cache_resource as _cache_resource,
    cache_data as _cache_data,
    experimental_singleton as _experimental_singleton,
    experimental_memo as _experimental_memo,
)
from streamlit.runtime.metrics_util import gather_metrics as _gather_metrics
from streamlit.runtime.secrets import secrets_singleton as _secrets_singleton
from streamlit.runtime.state import SessionStateProxy as _SessionStateProxy
from streamlit.user_info import UserInfoProxy as _UserInfoProxy
from streamlit.commands.query_params import (
    get_query_params as _get_query_params,
    set_query_params as _set_query_params,
)
from streamlit.elements.show import show as _show

# Modules that the user should have access to. These are imported with "as"
# syntax pass mypy checking with implicit_reexport disabled.

from streamlit.echo import echo as echo
from streamlit.runtime.legacy_caching import cache as _cache
from streamlit.elements.spinner import spinner as spinner
from streamlit.commands.page_config import set_page_config as set_page_config
from streamlit.commands.execution_control import (
    stop as stop,
    rerun as _rerun,
)

# We add the metrics tracking for caching here,
# since the actual cache function calls itself recursively
cache = _gather_metrics("cache", _cache)


def _update_logger() -> None:
    _logger.set_log_level(_config.get_option("logger.level").upper())
    _logger.update_formatter()
    _logger.init_tornado_logs()


# Make this file only depend on config option in an asynchronous manner. This
# avoids a race condition when another file (such as a test file) tries to pass
# in an alternative config.
_config.on_config_parsed(_update_logger, True)


_main = _DeltaGenerator(root_container=_RootContainer.MAIN)
sidebar = _DeltaGenerator(root_container=_RootContainer.SIDEBAR, parent=_main)

secrets = _secrets_singleton

# DeltaGenerator methods:

altair_chart = _main.altair_chart
area_chart = _main.area_chart
audio = _main.audio
balloons = _main.balloons
bar_chart = _main.bar_chart
bokeh_chart = _main.bokeh_chart
button = _main.button
caption = _main.caption
camera_input = _main.camera_input
checkbox = _main.checkbox
code = _main.code
columns = _main.columns
tabs = _main.tabs
container = _main.container
dataframe = _main.dataframe
date_input = _main.date_input
divider = _main.divider
download_button = _main.download_button
expander = _main.expander
pydeck_chart = _main.pydeck_chart
empty = _main.empty
error = _main.error
exception = _main.exception
file_uploader = _main.file_uploader
form = _main.form
form_submit_button = _main.form_submit_button
graphviz_chart = _main.graphviz_chart
header = _main.header
help = _main.help
image = _main.image
info = _main.info
json = _main.json
latex = _main.latex
line_chart = _main.line_chart
map = _main.map
markdown = _main.markdown
metric = _main.metric
multiselect = _main.multiselect
number_input = _main.number_input
plotly_chart = _main.plotly_chart
progress = _main.progress
pyplot = _main.pyplot
radio = _main.radio
selectbox = _main.selectbox
select_slider = _main.select_slider
slider = _main.slider
snow = _main.snow
subheader = _main.subheader
success = _main.success
table = _main.table
text = _main.text
text_area = _main.text_area
text_input = _main.text_input
time_input = _main.time_input
title = _main.title
vega_lite_chart = _main.vega_lite_chart
video = _main.video
warning = _main.warning
write = _main.write
color_picker = _main.color_picker

# Legacy
_legacy_dataframe = _main._legacy_dataframe
_legacy_table = _main._legacy_table
_legacy_altair_chart = _main._legacy_altair_chart
_legacy_area_chart = _main._legacy_area_chart
_legacy_bar_chart = _main._legacy_bar_chart
_legacy_line_chart = _main._legacy_line_chart
_legacy_vega_lite_chart = _main._legacy_vega_lite_chart

# Apache Arrow
_arrow_dataframe = _main._arrow_dataframe
_arrow_table = _main._arrow_table
_arrow_altair_chart = _main._arrow_altair_chart
_arrow_area_chart = _main._arrow_area_chart
_arrow_bar_chart = _main._arrow_bar_chart
_arrow_line_chart = _main._arrow_line_chart
_arrow_vega_lite_chart = _main._arrow_vega_lite_chart

# Config
get_option = _config.get_option
# We add the metrics tracking here, since importing
# gather_metrics in config causes a circular dependency
set_option = _gather_metrics("set_option", _config.set_user_option)

# Session State
session_state = _SessionStateProxy()

# Caching
cache_data = _cache_data
cache_resource = _cache_resource

# Beta APIs
beta_container = _gather_metrics("beta_container", _main.beta_container)
beta_expander = _gather_metrics("beta_expander", _main.beta_expander)
beta_columns = _gather_metrics("beta_columns", _main.beta_columns)

# Experimental APIs
experimental_user = _UserInfoProxy()
experimental_singleton = _experimental_singleton
experimental_memo = _experimental_memo
experimental_get_query_params = _get_query_params
experimental_set_query_params = _set_query_params
experimental_show = _show
experimental_rerun = _rerun
experimental_data_editor = _main.experimental_data_editor
