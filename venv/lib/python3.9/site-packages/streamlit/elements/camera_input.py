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

from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, List, Optional, cast

from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
    check_callback_rules,
    check_session_state_rules,
    get_label_visibility_proto_value,
)
from streamlit.proto.CameraInput_pb2 import CameraInput as CameraInputProto
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    register_widget,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

SomeUploadedSnapshotFile = Optional[UploadedFile]


def _get_file_recs_for_camera_input_widget(
    widget_id: str, widget_value: Optional[FileUploaderStateProto]
) -> List[UploadedFileRec]:
    if widget_value is None:
        return []

    ctx = get_script_run_ctx()
    if ctx is None:
        return []

    uploaded_file_info = widget_value.uploaded_file_info
    if len(uploaded_file_info) == 0:
        return []

    active_file_ids = [f.id for f in uploaded_file_info]

    # Grab the files that correspond to our active file IDs.
    return ctx.uploaded_file_mgr.get_files(
        session_id=ctx.session_id,
        widget_id=widget_id,
        file_ids=active_file_ids,
    )


@dataclass
class CameraInputSerde:
    def serialize(
        self,
        snapshot: SomeUploadedSnapshotFile,
    ) -> FileUploaderStateProto:
        state_proto = FileUploaderStateProto()

        ctx = get_script_run_ctx()
        if ctx is None:
            return state_proto

        # ctx.uploaded_file_mgr._file_id_counter stores the id to use for
        # the *next* uploaded file, so the current highest file id is the
        # counter minus 1.
        state_proto.max_file_id = ctx.uploaded_file_mgr._file_id_counter - 1

        if not snapshot:
            return state_proto

        file_info: UploadedFileInfoProto = state_proto.uploaded_file_info.add()
        file_info.id = snapshot.id
        file_info.name = snapshot.name
        file_info.size = snapshot.size

        return state_proto

    def deserialize(
        self, ui_value: Optional[FileUploaderStateProto], widget_id: str
    ) -> SomeUploadedSnapshotFile:
        file_recs = _get_file_recs_for_camera_input_widget(widget_id, ui_value)

        if len(file_recs) == 0:
            return_value = None
        else:
            return_value = UploadedFile(file_recs[0])
        return return_value


class CameraInputMixin:
    @gather_metrics("camera_input")
    def camera_input(
        self,
        label: str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
    ) -> SomeUploadedSnapshotFile:
        r"""Display a widget that returns pictures from the user's webcam.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this widget is used for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

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

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.

        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.

        help : str
            A tooltip that gets displayed next to the camera input.

        on_change : callable
            An optional callback invoked when this camera_input's value
            changes.

        args : tuple
            An optional tuple of args to pass to the callback.

        kwargs : dict
            An optional dict of kwargs to pass to the callback.

        disabled : bool
            An optional boolean, which disables the camera input if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        label_visibility : "visible" or "hidden" or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible". This argument can only be supplied by keyword.

        Returns
        -------
        None or UploadedFile
            The UploadedFile class is a subclass of BytesIO, and therefore
            it is "file-like". This means you can pass them anywhere where
            a file is expected.

        Examples
        --------
        >>> import streamlit as st
        >>>
        >>> picture = st.camera_input("Take a picture")
        >>>
        >>> if picture:
        ...     st.image(picture)

        """
        ctx = get_script_run_ctx()
        return self._camera_input(
            label=label,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility=label_visibility,
            ctx=ctx,
        )

    def _camera_input(
        self,
        label: str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        ctx: Optional[ScriptRunContext] = None,
    ) -> SomeUploadedSnapshotFile:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        maybe_raise_label_warnings(label, label_visibility)

        camera_input_proto = CameraInputProto()
        camera_input_proto.label = label
        camera_input_proto.form_id = current_form_id(self.dg)

        if help is not None:
            camera_input_proto.help = dedent(help)

        serde = CameraInputSerde()

        camera_input_state = register_widget(
            "camera_input",
            camera_input_proto,
            user_key=key,
            on_change_handler=on_change,
            args=args,
            kwargs=kwargs,
            deserializer=serde.deserialize,
            serializer=serde.serialize,
            ctx=ctx,
        )

        # This needs to be done after register_widget because we don't want
        # the following proto fields to affect a widget's ID.
        camera_input_proto.disabled = disabled
        camera_input_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )

        ctx = get_script_run_ctx()
        camera_image_input_state = serde.serialize(camera_input_state.value)

        uploaded_shapshot_info = camera_image_input_state.uploaded_file_info

        if ctx is not None and len(uploaded_shapshot_info) != 0:
            newest_file_id = camera_image_input_state.max_file_id
            active_file_ids = [f.id for f in uploaded_shapshot_info]

            ctx.uploaded_file_mgr.remove_orphaned_files(
                session_id=ctx.session_id,
                widget_id=camera_input_proto.id,
                newest_file_id=newest_file_id,
                active_file_ids=active_file_ids,
            )

        self.dg._enqueue("camera_input", camera_input_proto)
        return camera_input_state.value

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)
