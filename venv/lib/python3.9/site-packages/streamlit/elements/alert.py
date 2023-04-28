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

from typing import TYPE_CHECKING, Optional, cast

from streamlit.errors import StreamlitAPIException
from streamlit.proto.Alert_pb2 import Alert as AlertProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text, is_emoji

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.type_util import SupportsStr


def validate_emoji(maybe_emoji: Optional[str]) -> str:
    if maybe_emoji is None:
        return ""
    elif is_emoji(maybe_emoji):
        return maybe_emoji
    else:
        raise StreamlitAPIException(
            f'The value "{maybe_emoji}" is not a valid emoji. Shortcodes are not allowed, please use a single character instead.'
        )


class AlertMixin:
    @gather_metrics("error")
    def error(
        self,
        body: "SupportsStr",
        *,  # keyword-only args:
        icon: Optional[str] = None,
    ) -> "DeltaGenerator":
        """Display error message.

        Parameters
        ----------
        body : str
            The error text to display.
        icon : str or None
            An optional, keyword-only argument that specifies an emoji to use as
            the icon for the alert. Shortcodes are not allowed, please use a
            single character instead. E.g. "🚨", "🔥", "🤖", etc.
            Defaults to None, which means no icon is displayed.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.error('This is an error', icon="🚨")

        """
        alert_proto = AlertProto()
        alert_proto.icon = validate_emoji(icon)
        alert_proto.body = clean_text(body)
        alert_proto.format = AlertProto.ERROR
        return self.dg._enqueue("alert", alert_proto)

    @gather_metrics("warning")
    def warning(
        self,
        body: "SupportsStr",
        *,  # keyword-only args:
        icon: Optional[str] = None,
    ) -> "DeltaGenerator":
        """Display warning message.

        Parameters
        ----------
        body : str
            The warning text to display.
        icon : str or None
            An optional, keyword-only argument that specifies an emoji to use as
            the icon for the alert. Shortcodes are not allowed, please use a
            single character instead. E.g. "🚨", "🔥", "🤖", etc.
            Defaults to None, which means no icon is displayed.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.warning('This is a warning', icon="⚠️")

        """
        alert_proto = AlertProto()
        alert_proto.body = clean_text(body)
        alert_proto.icon = validate_emoji(icon)
        alert_proto.format = AlertProto.WARNING
        return self.dg._enqueue("alert", alert_proto)

    @gather_metrics("info")
    def info(
        self,
        body: "SupportsStr",
        *,  # keyword-only args:
        icon: Optional[str] = None,
    ) -> "DeltaGenerator":
        """Display an informational message.

        Parameters
        ----------
        body : str
            The info text to display.
        icon : str or None
            An optional, keyword-only argument that specifies an emoji to use as
            the icon for the alert. Shortcodes are not allowed, please use a
            single character instead. E.g. "🚨", "🔥", "🤖", etc.
            Defaults to None, which means no icon is displayed.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.info('This is a purely informational message', icon="ℹ️")

        """

        alert_proto = AlertProto()
        alert_proto.body = clean_text(body)
        alert_proto.icon = validate_emoji(icon)
        alert_proto.format = AlertProto.INFO
        return self.dg._enqueue("alert", alert_proto)

    @gather_metrics("success")
    def success(
        self,
        body: "SupportsStr",
        *,  # keyword-only args:
        icon: Optional[str] = None,
    ) -> "DeltaGenerator":
        """Display a success message.

        Parameters
        ----------
        body : str
            The success text to display.
        icon : str or None
            An optional, keyword-only argument that specifies an emoji to use as
            the icon for the alert. Shortcodes are not allowed, please use a
            single character instead. E.g. "🚨", "🔥", "🤖", etc.
            Defaults to None, which means no icon is displayed.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.success('This is a success message!', icon="✅")

        """
        alert_proto = AlertProto()
        alert_proto.body = clean_text(body)
        alert_proto.icon = validate_emoji(icon)
        alert_proto.format = AlertProto.SUCCESS
        return self.dg._enqueue("alert", alert_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)
