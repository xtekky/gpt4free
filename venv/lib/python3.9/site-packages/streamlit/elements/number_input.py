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

import numbers
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional, Union, cast

import streamlit
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
    check_callback_rules,
    check_session_state_rules,
    get_label_visibility_proto_value,
)
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    NoValue,
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    register_widget,
)
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key

Number = Union[int, float]


@dataclass
class NumberInputSerde:
    value: Union[int, float]
    data_type: int

    def serialize(self, v: Number) -> Number:
        return v

    def deserialize(self, ui_value: Optional[Number], widget_id: str = "") -> Number:
        if ui_value is not None:
            val = ui_value
        else:
            # Widget has not been used; fallback to the original value,
            val = self.value

        if self.data_type == NumberInputProto.INT:
            val = int(val)

        return val


class NumberInputMixin:
    @gather_metrics("number_input")
    def number_input(
        self,
        label: str,
        min_value: Optional[Number] = None,
        max_value: Optional[Number] = None,
        value: Union[NoValue, Number, None] = NoValue(),
        step: Optional[Number] = None,
        format: Optional[str] = None,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
    ) -> Number:
        r"""Display a numeric input widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this input is for.
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
        min_value : int or float or None
            The minimum permitted value.
            If None, there will be no minimum.
        max_value : int or float or None
            The maximum permitted value.
            If None, there will be no maximum.
        value : int or float or None
            The value of this widget when it first renders.
            Defaults to min_value, or 0.0 if min_value is None
        step : int or float or None
            The stepping interval.
            Defaults to 1 if the value is an int, 0.01 otherwise.
            If the value is not specified, the format parameter will be used.
        format : str or None
            A printf-style format string controlling how the interface should
            display numbers. Output must be purely numeric. This does not impact
            the return value. Valid formatters: %d %e %f %g %i %u
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the input.
        on_change : callable
            An optional callback invoked when this number_input's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the number input if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        label_visibility : "visible" or "hidden" or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible". This argument can only be supplied by keyword.

        Returns
        -------
        int or float
            The current value of the numeric input widget. The return type
            will match the data type of the value parameter.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> number = st.number_input('Insert a number')
        >>> st.write('The current number is ', number)

        .. output::
           https://doc-number-input.streamlitapp.com/
           height: 260px

        """
        ctx = get_script_run_ctx()
        return self._number_input(
            label=label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            format=format,
            key=key,
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility=label_visibility,
            ctx=ctx,
        )

    def _number_input(
        self,
        label: str,
        min_value: Optional[Number] = None,
        max_value: Optional[Number] = None,
        value: Union[NoValue, Number, None] = NoValue(),
        step: Optional[Number] = None,
        format: Optional[str] = None,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        ctx: Optional[ScriptRunContext] = None,
    ) -> Number:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(
            default_value=None if isinstance(value, NoValue) else value, key=key
        )
        maybe_raise_label_warnings(label, label_visibility)
        # Ensure that all arguments are of the same type.
        number_input_args = [min_value, max_value, value, step]

        int_args = all(
            isinstance(a, (numbers.Integral, type(None), NoValue))
            for a in number_input_args
        )

        float_args = all(
            isinstance(a, (float, type(None), NoValue)) for a in number_input_args
        )

        if not int_args and not float_args:
            raise StreamlitAPIException(
                "All numerical arguments must be of the same type."
                f"\n`value` has {type(value).__name__} type."
                f"\n`min_value` has {type(min_value).__name__} type."
                f"\n`max_value` has {type(max_value).__name__} type."
                f"\n`step` has {type(step).__name__} type."
            )

        if isinstance(value, NoValue):
            if min_value is not None:
                value = min_value
            elif int_args and float_args:
                value = 0.0  # if no values are provided, defaults to float
            elif int_args:
                value = 0
            else:
                value = 0.0

        int_value = isinstance(value, numbers.Integral)
        float_value = isinstance(value, float)

        if value is None:
            raise StreamlitAPIException(
                "Default value for number_input should be an int or a float."
            )
        else:
            if format is None:
                format = "%d" if int_value else "%0.2f"

            # Warn user if they format an int type as a float or vice versa.
            if format in ["%d", "%u", "%i"] and float_value:
                import streamlit as st

                st.warning(
                    "Warning: NumberInput value below has type float,"
                    f" but format {format} displays as integer."
                )
            elif format[-1] == "f" and int_value:
                import streamlit as st

                st.warning(
                    "Warning: NumberInput value below has type int so is"
                    f" displayed as int despite format string {format}."
                )

        if step is None:
            step = 1 if int_value else 0.01

        try:
            float(format % 2)
        except (TypeError, ValueError):
            raise StreamlitAPIException(
                "Format string for st.number_input contains invalid characters: %s"
                % format
            )

        # Ensure that the value matches arguments' types.
        all_ints = int_value and int_args

        if (min_value and min_value > value) or (max_value and max_value < value):
            raise StreamlitAPIException(
                "The default `value` of %(value)s "
                "must lie between the `min_value` of %(min)s "
                "and the `max_value` of %(max)s, inclusively."
                % {"value": value, "min": min_value, "max": max_value}
            )

        # Bounds checks. JSNumber produces human-readable exceptions that
        # we simply re-package as StreamlitAPIExceptions.
        try:
            if all_ints:
                if min_value is not None:
                    JSNumber.validate_int_bounds(
                        min_value, "`min_value`"  # type: ignore
                    )
                if max_value is not None:
                    JSNumber.validate_int_bounds(
                        max_value, "`max_value`"  # type: ignore
                    )
                if step is not None:
                    JSNumber.validate_int_bounds(step, "`step`")  # type: ignore
                JSNumber.validate_int_bounds(value, "`value`")  # type: ignore
            else:
                if min_value is not None:
                    JSNumber.validate_float_bounds(min_value, "`min_value`")
                if max_value is not None:
                    JSNumber.validate_float_bounds(max_value, "`max_value`")
                if step is not None:
                    JSNumber.validate_float_bounds(step, "`step`")
                JSNumber.validate_float_bounds(value, "`value`")
        except JSNumberBoundsException as e:
            raise StreamlitAPIException(str(e))

        data_type = NumberInputProto.INT if all_ints else NumberInputProto.FLOAT

        number_input_proto = NumberInputProto()
        number_input_proto.data_type = data_type
        number_input_proto.label = label
        number_input_proto.default = value
        number_input_proto.form_id = current_form_id(self.dg)
        if help is not None:
            number_input_proto.help = dedent(help)

        if min_value is not None:
            number_input_proto.min = min_value
            number_input_proto.has_min = True

        if max_value is not None:
            number_input_proto.max = max_value
            number_input_proto.has_max = True

        if step is not None:
            number_input_proto.step = step

        if format is not None:
            number_input_proto.format = format

        serde = NumberInputSerde(value, data_type)
        widget_state = register_widget(
            "number_input",
            number_input_proto,
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
        number_input_proto.disabled = disabled
        number_input_proto.label_visibility.value = get_label_visibility_proto_value(
            label_visibility
        )

        if widget_state.value_changed:
            number_input_proto.value = widget_state.value
            number_input_proto.set_value = True

        self.dg._enqueue("number_input", number_input_proto)
        return widget_state.value

    @property
    def dg(self) -> "streamlit.delta_generator.DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("streamlit.delta_generator.DeltaGenerator", self)
