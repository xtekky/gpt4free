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

from typing import Iterator, Mapping, NoReturn, Optional

from streamlit.errors import StreamlitAPIException
from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import UserInfo


def _get_user_info() -> UserInfo:
    ctx = _get_script_run_ctx()
    if ctx is None:
        # TODO: Add appropriate warnings when ctx is missing
        return {}
    return ctx.user_info


class UserInfoProxy(Mapping[str, Optional[str]]):
    """A dict like proxy object for accessing information about current user."""

    def __getitem__(self, key: str) -> Optional[str]:
        return _get_user_info()[key]

    def __getattr__(self, key: str) -> Optional[str]:
        try:
            return _get_user_info()[key]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name: str, value: Optional[str]) -> NoReturn:
        raise StreamlitAPIException("st.experimental_user cannot be modified")

    def __setitem__(self, name: str, value: Optional[str]) -> NoReturn:
        raise StreamlitAPIException("st.experimental_user cannot be modified")

    def __iter__(self) -> Iterator[str]:
        return iter(_get_user_info())

    def __len__(self) -> int:
        return len(_get_user_info())

    def to_dict(self) -> UserInfo:
        return _get_user_info()
