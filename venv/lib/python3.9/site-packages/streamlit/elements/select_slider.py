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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from typing_extensions import TypeGuard

from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
    check_callback_rules,
    check_session_state_rules,
    get_label_visibility_proto_value,
)
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
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


def _is_range_value(value: Union[T, Sequence[T]]) -> TypeGuard[Sequence[T]]:
    return isinstance(value, (list, tuple))


@dataclass
class SelectSliderSerde(Generic[T]):
    options: Sequence[T]
    value: List[int]
    is_range_value: bool

    def serialize(self, v: object) -> List[int]:
        return self._as_index_list(v)

    def deserialize(
        self,
        ui_value: Optional[List[int]],
        widget_id: str = "",
    ) -> Union[T, Tuple[T, T]]:
        if not ui_value:
            # Widget has not been used; fallback to the original value,
            ui_value = self.value

        # The widget always returns floats, so convert to ints before indexing
        return_value: Tuple[T, T] = cast(
            Tuple[T, T],
            tuple(map(lambda x: self.options[int(x)], ui_value)),
        )

        # If the original value was a list/tuple, so will be the output (and vice versa)
        return return_value if self.is_range_value else return_value[0]

    def _as_index_list(self, v: object) -> List[int]:
        if _is_range_value(v):
            slider_value = [index_(self.options, val) for val in v]
            start, end = slider_value
            if start > end:
                slider_value = [end, start]
            return slider_value
        else:
            return [index_(self.options, v)]


class SelectSliderMixin:
    @gather_metrics("select_slider")
    def select_slider(
        self,
        label: str,
        options: OptionSequence[T] = (),
        value: object = None,
        format_func: Callable[[Any], Any] = str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
    ) -> Union[T, Tuple[T, T]]:
        r"""
        Display a slider widget to select items from a list.

        This also allows you to render a range slider by passing a two-element
        tuple or list as the `value`.

        The difference between `st.select_slider` and `st.slider` is that
        `select_slider` accepts any datatype and takes an iterable set of
        options, while `slider` only accepts numerical or date/time data and
        takes a range as input.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this slider is for.
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
            Labels for the slider options. All options will be cast to str
            internally by default. For pandas.DataFrame, the first column is
            selected.
        value : a supported type or a tuple/list of supported types or None
            The value of the slider when it first renders. If a tuple/list
            of two values is passed here, then a range slider with those lower
            and upper bounds is rendered. For example, if set to `(1, 10)` the
            slider will have a selectable range between 1 and 10.
            Defaults to first option.
        format_func : function
            Function to modify the display of the labels from the options.
            argument. It receives the option as an argument and its output
            will be cast to str.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the select slider.
        on_change : callable
            An optional callback invoked when this select_slider's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the select slider if set to True.
            The default is False. This argument can only be supplied by keyword.
        label_visibility : "visible" or "hidden" or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible". This argument can only be supplied by keyword.

        Returns
        -------
        any value or tuple of any value
            The current value of the slider widget. The return type will match
            the data type of the value parameter.

        Examples
        --------
        >>> import streamlit as st
        >>>
        >>> color = st.select_slider(
        ...     'Select a color of the rainbow',
        ...     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
        >>> st.write('My favorite color is', color)

        And here's an example of a range select slider:

        >>> import streamlit as st
        >>>
        >>> start_color, end_color = st.select_slider(
        ...     'Select a range of color wavelength',
        ...     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        ...     value=('red', 'blue'))
        >>> st.write('You selected wavelengths between', start_color, 'and', end_color)

        .. output::
           https://doc-select-slider.streamlitapp.com/
           height: 450px

        """
        ctx = get_script_run_ctx()
        return self._select_slider(
            label=label,
            options=options,
            value=value,
            format_func=format_func,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility=label_visibility,
            ctx=ctx,
        )

    def _select_slider(
        self,
        label: str,
        options: OptionSequence[T] = (),
        value: object = None,
        format_func: Callable[[Any], Any] = str,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        ctx: Optional[ScriptRunContext] = None,
    ) -> Union[T, Tuple[T, T]]:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        opt = ensure_indexable(options)

        if len(opt) == 0:
            raise StreamlitAPIException("The `options` argument needs to be non-empty")

        def as_index_list(v: object) -> List[int]:
            if _is_range_value(v):
                slider_value = [index_(opt, val) for val in v]
                start, end = slider_value
                if start > end:
                    slider_value = [end, start]
                return slider_value
            else:
                # Simplify future logic by always making value a list
                try:
                    return [index_(opt, v)]
                except ValueError:
                    if value is not None:
                        raise

                    return [0]

        # Convert element to index of the elements
        slider_value = as_index_list(value)

        slider_proto = SliderProto()
        slider_proto.type = SliderProto.Type.SELECT_SLIDER
        slider_proto.label = label
        slider_proto.format = "%s"
        slider_proto.default[:] = slider_value
        slider_proto.min = 0
        slider_proto.max = len(opt) - 1
        slider_proto.step = 1  # default for index changes
        slider_proto.data_type = SliderProto.INT
        slider_proto.options[:] = [str(format_func(option)) for option in opt]
        slider_proto.form_id = current_form_id(self.dg)
        if help is not None:
            slider_proto.help = dedent(help)

        serde = SelectSliderSerde(opt, slider_value, _is_range_value(value))

        widget_state = register_widget(
            "slider",
            slider_proto,
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
        slider_proto.disabled = disabled
        slider_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )
        if widget_state.value_changed:
            slider_proto.value[:] = serde.serialize(widget_state.value)
            slider_proto.set_value = True

        self.dg._enqueue("slider", slider_proto)
        return widget_state.value

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)
