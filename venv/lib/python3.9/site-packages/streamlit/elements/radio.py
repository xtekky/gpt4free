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
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Sequence, cast

from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
    check_callback_rules,
    check_session_state_rules,
    get_label_visibility_proto_value,
)
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Radio_pb2 import Radio as RadioProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    register_widget,
)
from streamlit.type_util import (
    Key,
    LabelVisibility,
    OptionSequence,
    T,
    ensure_indexable,
    maybe_raise_label_warnings,
    to_key,
)
from streamlit.util import index_

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


@dataclass
class RadioSerde(Generic[T]):
    options: Sequence[T]
    index: int

    def serialize(self, v: object) -> int:
        if len(self.options) == 0:
            return 0
        return index_(self.options, v)

    def deserialize(
        self,
        ui_value: Optional[int],
        widget_id: str = "",
    ) -> Optional[T]:
        idx = ui_value if ui_value is not None else self.index

        return (
            self.options[idx]
            if len(self.options) > 0 and self.options[idx] is not None
            else None
        )


class RadioMixin:
    @gather_metrics("radio")
    def radio(
        self,
        label: str,
        options: OptionSequence[T],
        index: int = 0,
        format_func: Callable[[Any], Any] = str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only args:
        disabled: bool = False,
        horizontal: bool = False,
        label_visibility: LabelVisibility = "visible",
    ) -> Optional[T]:
        r"""Display a radio button widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this radio group is for.
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
        options : Sequence, numpy.ndarray, pandas.Series, pandas.DataFrame, or pandas.Index
            Labels for the radio options. This will be cast to str internally
            by default. For pandas.DataFrame, the first column is selected.
        index : int
            The index of the preselected option on first render.
        format_func : function
            Function to modify the display of radio options. It receives
            the raw option as an argument and should output the label to be
            shown for that option. This has no impact on the return value of
            the radio.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the radio.
        on_change : callable
            An optional callback invoked when this radio's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the radio button if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        horizontal : bool
            An optional boolean, which orients the radio group horizontally.
            The default is false (vertical buttons). This argument can only
            be supplied by keyword.

        label_visibility : "visible" or "hidden" or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible". This argument can only be supplied by keyword.

        Returns
        -------
        any
            The selected option.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> genre = st.radio(
        ...     "What\'s your favorite movie genre",
        ...     ('Comedy', 'Drama', 'Documentary'))
        >>>
        >>> if genre == 'Comedy':
        ...     st.write('You selected comedy.')
        ... else:
        ...     st.write("You didn\'t select comedy.")

        .. output::
           https://doc-radio.streamlitapp.com/
           height: 260px

        """
        ctx = get_script_run_ctx()
        return self._radio(
            label=label,
            options=options,
            index=index,
            format_func=format_func,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            horizontal=horizontal,
            ctx=ctx,
            label_visibility=label_visibility,
        )

    def _radio(
        self,
        label: str,
        options: OptionSequence[T],
        index: int = 0,
        format_func: Callable[[Any], Any] = str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only args:
        disabled: bool = False,
        horizontal: bool = False,
        label_visibility: LabelVisibility = "visible",
        ctx: Optional[ScriptRunContext],
    ) -> Optional[T]:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None if index == 0 else index, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        opt = ensure_indexable(options)

        if not isinstance(index, int):
            raise StreamlitAPIException(
                "Radio Value has invalid type: %s" % type(index).__name__
            )

        if len(opt) > 0 and not 0 <= index < len(opt):
            raise StreamlitAPIException(
                "Radio index must be between 0 and length of options"
            )

        radio_proto = RadioProto()
        radio_proto.label = label
        radio_proto.default = index
        radio_proto.options[:] = [str(format_func(option)) for option in opt]
        radio_proto.form_id = current_form_id(self.dg)
        radio_proto.horizontal = horizontal
        if help is not None:
            radio_proto.help = dedent(help)

        serde = RadioSerde(opt, index)

        widget_state = register_widget(
            "radio",
            radio_proto,
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
        radio_proto.disabled = disabled
        radio_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )

        if widget_state.value_changed:
            radio_proto.value = serde.serialize(widget_state.value)
            radio_proto.set_value = True

        self.dg._enqueue("radio", radio_proto)
        return widget_state.value

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)
