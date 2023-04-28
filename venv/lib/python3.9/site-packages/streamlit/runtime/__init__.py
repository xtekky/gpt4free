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

# Explicitly re-export public symbols from runtime.py and session_manager.py
from streamlit.runtime.runtime import Runtime as Runtime
from streamlit.runtime.runtime import RuntimeConfig as RuntimeConfig
from streamlit.runtime.runtime import RuntimeState as RuntimeState
from streamlit.runtime.session_manager import SessionClient as SessionClient
from streamlit.runtime.session_manager import (
    SessionClientDisconnectedError as SessionClientDisconnectedError,
)


def get_instance() -> Runtime:
    """Return the singleton Runtime instance. Raise an Error if the
    Runtime hasn't been created yet.
    """
    return Runtime.instance()


def exists() -> bool:
    """True if the singleton Runtime instance has been created.

    When a Streamlit app is running in "raw mode" - that is, when the
    app is run via `python app.py` instead of `streamlit run app.py` -
    the Runtime will not exist, and various Streamlit functions need
    to adapt.
    """
    return Runtime.exists()
