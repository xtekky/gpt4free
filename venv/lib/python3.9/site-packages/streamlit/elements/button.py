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

import io
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, BinaryIO, Optional, TextIO, Union, cast

from typing_extensions import Final, Literal

from streamlit import runtime
from streamlit.elements.form import current_form_id, is_in_form
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.DownloadButton_pb2 import DownloadButton as DownloadButtonProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    register_widget,
)
from streamlit.type_util import Key, to_key

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


FORM_DOCS_INFO: Final = """

For more information, refer to the
[documentation for forms](https://docs.streamlit.io/library/api-reference/control-flow/st.form).
"""

DownloadButtonDataType = Union[str, bytes, TextIO, BinaryIO, io.RawIOBase]


@dataclass
class ButtonSerde:
    def serialize(self, v: bool) -> bool:
        return bool(v)

    def deserialize(self, ui_value: Optional[bool], widget_id: str = "") -> bool:
        return ui_value or False


class ButtonMixin:
    @gather_metrics("button")
    def button(
        self,
        label: str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_click: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        type: Literal["primary", "secondary"] = "secondary",
        disabled: bool = False,
        use_container_width: bool = False,
    ) -> bool:
        r"""Display a button widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this button is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, and Emojis.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\. Not an ordered list``.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed when the button is
            hovered over.
        on_click : callable
            An optional callback invoked when this button is clicked.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        type : "secondary" or "primary"
            An optional string that specifies the button type. Can be "primary" for a
            button with additional emphasis or "secondary" for a normal button. This
            argument can only be supplied by keyword. Defaults to "secondary".
        disabled : bool
            An optional boolean, which disables the button if set to True. The
            default is False. This argument can only be supplied by keyword.
        use_container_width: bool
            An optional boolean, which makes the button stretch its width to match the parent container.

        Returns
        -------
        bool
            True if the button was clicked on the last run of the app,
            False otherwise.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> if st.button('Say hello'):
        ...     st.write('Why hello there')
        ... else:
        ...     st.write('Goodbye')

        .. output::
           https://doc-buton.streamlitapp.com/
           height: 220px

        """
        key = to_key(key)
        ctx = get_script_run_ctx()

        # Checks whether the entered button type is one of the allowed options - either "primary" or "secondary"
        if type not in ["primary", "secondary"]:
            raise StreamlitAPIException(
                'The type argument to st.button must be "primary" or "secondary". \n'
                f'The argument passed was "{type}".'
            )

        return self.dg._button(
            label,
            key,
            help,
            is_form_submitter=False,
            on_click=on_click,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            type=type,
            use_container_width=use_container_width,
            ctx=ctx,
        )

    @gather_metrics("download_button")
    def download_button(
        self,
        label: str,
        data: DownloadButtonDataType,
        file_name: Optional[str] = None,
        mime: Optional[str] = None,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_click: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        use_container_width: bool = False,
    ) -> bool:
        r"""Display a download button widget.

        This is useful when you would like to provide a way for your users
        to download a file directly from your app.

        Note that the data to be downloaded is stored in-memory while the
        user is connected, so it's a good idea to keep file sizes under a
        couple hundred megabytes to conserve memory.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this button is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, and Emojis.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\. Not an ordered list``.
        data : str or bytes or file
            The contents of the file to be downloaded. See example below for
            caching techniques to avoid recomputing this data unnecessarily.
        file_name: str
            An optional string to use as the name of the file to be downloaded,
            such as 'my_file.csv'. If not specified, the name will be
            automatically generated.
        mime : str or None
            The MIME type of the data. If None, defaults to "text/plain"
            (if data is of type *str* or is a textual *file*) or
            "application/octet-stream" (if data is of type *bytes* or is a
            binary *file*).
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed when the button is
            hovered over.
        on_click : callable
            An optional callback invoked when this button is clicked.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the download button if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        use_container_width: bool
            An optional boolean, which makes the button stretch its width to match the parent container.


        Returns
        -------
        bool
            True if the button was clicked on the last run of the app,
            False otherwise.

        Examples
        --------
        Download a large DataFrame as a CSV:

        >>> import streamlit as st
        >>>
        >>> @st.cache
        ... def convert_df(df):
        ...     # IMPORTANT: Cache the conversion to prevent computation on every rerun
        ...     return df.to_csv().encode('utf-8')
        >>>
        >>> csv = convert_df(my_large_df)
        >>>
        >>> st.download_button(
        ...     label="Download data as CSV",
        ...     data=csv,
        ...     file_name='large_df.csv',
        ...     mime='text/csv',
        ... )

        Download a string as a file:

        >>> import streamlit as st
        >>>
        >>> text_contents = '''This is some text'''
        >>> st.download_button('Download some text', text_contents)

        Download a binary file:

        >>> import streamlit as st
        >>>
        >>> binary_contents = b'example content'
        >>> # Defaults to 'application/octet-stream'
        >>> st.download_button('Download binary file', binary_contents)

        Download an image:

        >>> import streamlit as st
        >>>
        >>> with open("flower.png", "rb") as file:
        ...     btn = st.download_button(
        ...             label="Download image",
        ...             data=file,
        ...             file_name="flower.png",
        ...             mime="image/png"
        ...           )

        .. output::
           https://doc-download-buton.streamlitapp.com/
           height: 335px

        """
        ctx = get_script_run_ctx()
        return self._download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
            help=help,
            on_click=on_click,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            use_container_width=use_container_width,
            ctx=ctx,
        )

    def _download_button(
        self,
        label: str,
        data: DownloadButtonDataType,
        file_name: Optional[str] = None,
        mime: Optional[str] = None,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_click: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        use_container_width: bool = False,
        ctx: Optional[ScriptRunContext] = None,
    ) -> bool:

        key = to_key(key)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        if is_in_form(self.dg):
            raise StreamlitAPIException(
                f"`st.download_button()` can't be used in an `st.form()`.{FORM_DOCS_INFO}"
            )

        download_button_proto = DownloadButtonProto()

        download_button_proto.use_container_width = use_container_width
        download_button_proto.label = label
        download_button_proto.default = False
        marshall_file(
            self.dg._get_delta_path_str(), data, download_button_proto, mime, file_name
        )

        if help is not None:
            download_button_proto.help = dedent(help)

        serde = ButtonSerde()

        button_state = register_widget(
            "download_button",
            download_button_proto,
            user_key=key,
            on_change_handler=on_click,
            args=args,
            kwargs=kwargs,
            deserializer=serde.deserialize,
            serializer=serde.serialize,
            ctx=ctx,
        )

        # This needs to be done after register_widget because we don't want
        # the following proto fields to affect a widget's ID.
        download_button_proto.disabled = disabled

        self.dg._enqueue("download_button", download_button_proto)
        return button_state.value

    def _button(
        self,
        label: str,
        key: Optional[str],
        help: Optional[str],
        is_form_submitter: bool,
        on_click: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        type: Literal["primary", "secondary"] = "secondary",
        disabled: bool = False,
        use_container_width: bool = False,
        ctx: Optional[ScriptRunContext] = None,
    ) -> bool:
        if not is_form_submitter:
            check_callback_rules(self.dg, on_click)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)

        # It doesn't make sense to create a button inside a form (except
        # for the "Form Submitter" button that's automatically created in
        # every form). We throw an error to warn the user about this.
        # We omit this check for scripts running outside streamlit, because
        # they will have no script_run_ctx.
        if runtime.exists():
            if is_in_form(self.dg) and not is_form_submitter:
                raise StreamlitAPIException(
                    f"`st.button()` can't be used in an `st.form()`.{FORM_DOCS_INFO}"
                )
            elif not is_in_form(self.dg) and is_form_submitter:
                raise StreamlitAPIException(
                    f"`st.form_submit_button()` must be used inside an `st.form()`.{FORM_DOCS_INFO}"
                )

        button_proto = ButtonProto()
        button_proto.label = label
        button_proto.default = False
        button_proto.is_form_submitter = is_form_submitter
        button_proto.form_id = current_form_id(self.dg)
        button_proto.type = type
        button_proto.use_container_width = use_container_width
        if help is not None:
            button_proto.help = dedent(help)

        serde = ButtonSerde()

        button_state = register_widget(
            "button",
            button_proto,
            user_key=key,
            on_change_handler=on_click,
            args=args,
            kwargs=kwargs,
            deserializer=serde.deserialize,
            serializer=serde.serialize,
            ctx=ctx,
        )

        # This needs to be done after register_widget because we don't want
        # the following proto fields to affect a widget's ID.
        button_proto.disabled = disabled

        self.dg._enqueue("button", button_proto)

        return button_state.value

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall_file(
    coordinates: str,
    data: DownloadButtonDataType,
    proto_download_button: DownloadButtonProto,
    mimetype: Optional[str],
    file_name: Optional[str] = None,
) -> None:
    data_as_bytes: bytes
    if isinstance(data, str):
        data_as_bytes = data.encode()
        mimetype = mimetype or "text/plain"
    elif isinstance(data, io.TextIOWrapper):
        string_data = data.read()
        data_as_bytes = string_data.encode()
        mimetype = mimetype or "text/plain"
    # Assume bytes; try methods until we run out.
    elif isinstance(data, bytes):
        data_as_bytes = data
        mimetype = mimetype or "application/octet-stream"
    elif isinstance(data, io.BytesIO):
        data.seek(0)
        data_as_bytes = data.getvalue()
        mimetype = mimetype or "application/octet-stream"
    elif isinstance(data, io.BufferedReader):
        data.seek(0)
        data_as_bytes = data.read()
        mimetype = mimetype or "application/octet-stream"
    elif isinstance(data, io.RawIOBase):
        data.seek(0)
        data_as_bytes = data.read() or b""
        mimetype = mimetype or "application/octet-stream"
    else:
        raise RuntimeError("Invalid binary data format: %s" % type(data))

    if runtime.exists():
        file_url = runtime.get_instance().media_file_mgr.add(
            data_as_bytes,
            mimetype,
            coordinates,
            file_name=file_name,
            is_for_static_download=True,
        )
    else:
        # When running in "raw mode", we can't access the MediaFileManager.
        file_url = ""

    proto_download_button.url = file_url
