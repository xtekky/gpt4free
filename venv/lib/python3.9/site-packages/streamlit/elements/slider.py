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
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import Final, TypeAlias

from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
    check_callback_rules,
    check_session_state_rules,
    get_label_visibility_proto_value,
)
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
    get_session_state,
    register_widget,
)
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

SliderScalarT = TypeVar("SliderScalarT", int, float, date, time, datetime)

Step: TypeAlias = Union[int, float, timedelta]
SliderScalar: TypeAlias = Union[int, float, date, time, datetime]

SliderValueGeneric: TypeAlias = Union[
    SliderScalarT,
    Sequence[SliderScalarT],
]
SliderValue: TypeAlias = Union[
    SliderValueGeneric[int],
    SliderValueGeneric[float],
    SliderValueGeneric[date],
    SliderValueGeneric[time],
    SliderValueGeneric[datetime],
]

SliderReturnGeneric: TypeAlias = Union[
    SliderScalarT,
    Tuple[SliderScalarT],
    Tuple[SliderScalarT, SliderScalarT],
]
SliderReturn: TypeAlias = Union[
    SliderReturnGeneric[int],
    SliderReturnGeneric[float],
    SliderReturnGeneric[date],
    SliderReturnGeneric[time],
    SliderReturnGeneric[datetime],
]

SECONDS_TO_MICROS: Final = 1000 * 1000
DAYS_TO_MICROS: Final = 24 * 60 * 60 * SECONDS_TO_MICROS

UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _time_to_datetime(time_: time) -> datetime:
    # Note, here we pick an arbitrary date well after Unix epoch.
    # This prevents pre-epoch timezone issues (https://bugs.python.org/issue36759)
    # We're dropping the date from datetime later, anyway.
    return datetime.combine(date(2000, 1, 1), time_)


def _date_to_datetime(date_: date) -> datetime:
    return datetime.combine(date_, time())


def _delta_to_micros(delta: timedelta) -> int:
    return (
        delta.microseconds
        + delta.seconds * SECONDS_TO_MICROS
        + delta.days * DAYS_TO_MICROS
    )


def _datetime_to_micros(dt: datetime) -> int:
    # The frontend is not aware of timezones and only expects a UTC-based
    # timestamp (in microseconds). Since we want to show the date/time exactly
    # as it is in the given datetime object, we just set the tzinfo to UTC and
    # do not do any timezone conversions. Only the backend knows about
    # original timezone and will replace the UTC timestamp in the deserialization.
    utc_dt = dt.replace(tzinfo=timezone.utc)
    return _delta_to_micros(utc_dt - UTC_EPOCH)


def _micros_to_datetime(micros: int, orig_tz: Optional[tzinfo]) -> datetime:
    """Restore times/datetimes to original timezone (dates are always naive)"""
    utc_dt = UTC_EPOCH + timedelta(microseconds=micros)
    # Add the original timezone. No conversion is required here,
    # since in the serialization, we also just replace the timestamp with UTC.
    return utc_dt.replace(tzinfo=orig_tz)


@dataclass
class SliderSerde:
    value: List[float]
    data_type: int
    single_value: bool
    orig_tz: Optional[tzinfo]

    def deserialize(self, ui_value: Optional[List[float]], widget_id: str = ""):
        if ui_value is not None:
            val: Any = ui_value
        else:
            # Widget has not been used; fallback to the original value,
            val = self.value

        # The widget always returns a float array, so fix the return type if necessary
        if self.data_type == SliderProto.INT:
            val = [int(v) for v in val]
        if self.data_type == SliderProto.DATETIME:
            val = [_micros_to_datetime(int(v), self.orig_tz) for v in val]
        if self.data_type == SliderProto.DATE:
            val = [_micros_to_datetime(int(v), self.orig_tz).date() for v in val]
        if self.data_type == SliderProto.TIME:
            val = [
                _micros_to_datetime(int(v), self.orig_tz)
                .time()
                .replace(tzinfo=self.orig_tz)
                for v in val
            ]
        return val[0] if self.single_value else tuple(val)

    def serialize(self, v: Any) -> List[Any]:
        range_value = isinstance(v, (list, tuple))
        value = list(v) if range_value else [v]
        if self.data_type == SliderProto.DATE:
            value = [_datetime_to_micros(_date_to_datetime(v)) for v in value]
        if self.data_type == SliderProto.TIME:
            value = [_datetime_to_micros(_time_to_datetime(v)) for v in value]
        if self.data_type == SliderProto.DATETIME:
            value = [_datetime_to_micros(v) for v in value]
        return value


class SliderMixin:
    @gather_metrics("slider")
    def slider(
        self,
        label: str,
        min_value: Optional[SliderScalar] = None,
        max_value: Optional[SliderScalar] = None,
        value: Optional[SliderValue] = None,
        step: Optional[Step] = None,
        format: Optional[str] = None,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
        *,  # keyword-only arguments:
        disabled: bool = False,
        label_visibility: LabelVisibility = "visible",
        # TODO(harahu): Add overload definitions. The return type is
        #  `SliderReturn`, in reality, but the return type is left as `Any`
        #  until we have proper overload definitions in place. Otherwise the
        #  user would have to cast the return value more often than not, which
        #  can be annoying.
    ) -> Any:
        r"""Display a slider widget.

        This supports int, float, date, time, and datetime types.

        This also allows you to render a range slider by passing a two-element
        tuple or list as the `value`.

        The difference between `st.slider` and `st.select_slider` is that
        `slider` only accepts numerical or date/time data and takes a range as
        input, while `select_slider` accepts any datatype and takes an iterable
        set of options.

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
        min_value : a supported type or None
            The minimum permitted value.
            Defaults to 0 if the value is an int, 0.0 if a float,
            value - timedelta(days=14) if a date/datetime, time.min if a time
        max_value : a supported type or None
            The maximum permitted value.
            Defaults to 100 if the value is an int, 1.0 if a float,
            value + timedelta(days=14) if a date/datetime, time.max if a time
        value : a supported type or a tuple/list of supported types or None
            The value of the slider when it first renders. If a tuple/list
            of two values is passed here, then a range slider with those lower
            and upper bounds is rendered. For example, if set to `(1, 10)` the
            slider will have a selectable range between 1 and 10.
            Defaults to min_value.
        step : int/float/timedelta or None
            The stepping interval.
            Defaults to 1 if the value is an int, 0.01 if a float,
            timedelta(days=1) if a date/datetime, timedelta(minutes=15) if a time
            (or if max_value - min_value < 1 day)
        format : str or None
            A printf-style format string controlling how the interface should
            display numbers. This does not impact the return value.
            Formatter for int/float supports: %d %e %f %g %i
            Formatter for date/time/datetime uses Moment.js notation:
            https://momentjs.com/docs/#/displaying/format/
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the slider.
        on_change : callable
            An optional callback invoked when this slider's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the slider if set to True. The
            default is False. This argument can only be supplied by keyword.
        label_visibility : "visible" or "hidden" or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible". This argument can only be supplied by keyword.


        Returns
        -------
        int/float/date/time/datetime or tuple of int/float/date/time/datetime
            The current value of the slider widget. The return type will match
            the data type of the value parameter.

        Examples
        --------
        >>> import streamlit as st
        >>>
        >>> age = st.slider('How old are you?', 0, 130, 25)
        >>> st.write("I'm ", age, 'years old')

        And here's an example of a range slider:

        >>> import streamlit as st
        >>>
        >>> values = st.slider(
        ...     'Select a range of values',
        ...     0.0, 100.0, (25.0, 75.0))
        >>> st.write('Values:', values)

        This is a range time slider:

        >>> import streamlit as st
        >>> from datetime import time
        >>>
        >>> appointment = st.slider(
        ...     "Schedule your appointment:",
        ...     value=(time(11, 30), time(12, 45)))
        >>> st.write("You're scheduled for:", appointment)

        Finally, a datetime slider:

        >>> import streamlit as st
        >>> from datetime import datetime
        >>>
        >>> start_time = st.slider(
        ...     "When do you start?",
        ...     value=datetime(2020, 1, 1, 9, 30),
        ...     format="MM/DD/YY - hh:mm")
        >>> st.write("Start time:", start_time)

        .. output::
           https://doc-slider.streamlitapp.com/
           height: 300px

        """
        ctx = get_script_run_ctx()
        return self._slider(
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

    def _slider(
        self,
        label: str,
        min_value=None,
        max_value=None,
        value=None,
        step: Optional[Step] = None,
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
    ) -> SliderReturn:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value, key=key)

        maybe_raise_label_warnings(label, label_visibility)

        if value is None:
            # Set value from session_state if exists.
            session_state = get_session_state().filtered_state

            # we look first to session_state value of the widget because
            # depending on the value (single value or list/tuple) the slider should be
            # initializing differently (either as range or single value slider)
            if key is not None and key in session_state:
                value = session_state[key]
            else:
                # Set value default.
                value = min_value if min_value is not None else 0

        SUPPORTED_TYPES = {
            int: SliderProto.INT,
            float: SliderProto.FLOAT,
            datetime: SliderProto.DATETIME,
            date: SliderProto.DATE,
            time: SliderProto.TIME,
        }
        TIMELIKE_TYPES = (SliderProto.DATETIME, SliderProto.TIME, SliderProto.DATE)

        # Ensure that the value is either a single value or a range of values.
        single_value = isinstance(value, tuple(SUPPORTED_TYPES.keys()))
        range_value = isinstance(value, (list, tuple)) and len(value) in (0, 1, 2)
        if not single_value and not range_value:
            raise StreamlitAPIException(
                "Slider value should either be an int/float/datetime or a list/tuple of "
                "0 to 2 ints/floats/datetimes"
            )

        # Simplify future logic by always making value a list
        if single_value:
            value = [value]

        def all_same_type(items):
            return len(set(map(type, items))) < 2

        if not all_same_type(value):
            raise StreamlitAPIException(
                "Slider tuple/list components must be of the same type.\n"
                f"But were: {list(map(type, value))}"
            )

        if len(value) == 0:
            data_type = SliderProto.INT
        else:
            data_type = SUPPORTED_TYPES[type(value[0])]

        datetime_min = time.min
        datetime_max = time.max
        if data_type == SliderProto.TIME:
            datetime_min = time.min.replace(tzinfo=value[0].tzinfo)
            datetime_max = time.max.replace(tzinfo=value[0].tzinfo)
        if data_type in (SliderProto.DATETIME, SliderProto.DATE):
            datetime_min = value[0] - timedelta(days=14)
            datetime_max = value[0] + timedelta(days=14)

        DEFAULTS = {
            SliderProto.INT: {
                "min_value": 0,
                "max_value": 100,
                "step": 1,
                "format": "%d",
            },
            SliderProto.FLOAT: {
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.01,
                "format": "%0.2f",
            },
            SliderProto.DATETIME: {
                "min_value": datetime_min,
                "max_value": datetime_max,
                "step": timedelta(days=1),
                "format": "YYYY-MM-DD",
            },
            SliderProto.DATE: {
                "min_value": datetime_min,
                "max_value": datetime_max,
                "step": timedelta(days=1),
                "format": "YYYY-MM-DD",
            },
            SliderProto.TIME: {
                "min_value": datetime_min,
                "max_value": datetime_max,
                "step": timedelta(minutes=15),
                "format": "HH:mm",
            },
        }

        if min_value is None:
            min_value = DEFAULTS[data_type]["min_value"]
        if max_value is None:
            max_value = DEFAULTS[data_type]["max_value"]
        if step is None:
            step = cast(Step, DEFAULTS[data_type]["step"])
            if data_type in (
                SliderProto.DATETIME,
                SliderProto.DATE,
            ) and max_value - min_value < timedelta(days=1):
                step = timedelta(minutes=15)
        if format is None:
            format = cast(str, DEFAULTS[data_type]["format"])

        if step == 0:
            raise StreamlitAPIException(
                "Slider components cannot be passed a `step` of 0."
            )

        # Ensure that all arguments are of the same type.
        slider_args = [min_value, max_value, step]
        int_args = all(map(lambda a: isinstance(a, int), slider_args))
        float_args = all(map(lambda a: isinstance(a, float), slider_args))
        # When min and max_value are the same timelike, step should be a timedelta
        timelike_args = (
            data_type in TIMELIKE_TYPES
            and isinstance(step, timedelta)
            and type(min_value) == type(max_value)
        )

        if not int_args and not float_args and not timelike_args:
            raise StreamlitAPIException(
                "Slider value arguments must be of matching types."
                "\n`min_value` has %(min_type)s type."
                "\n`max_value` has %(max_type)s type."
                "\n`step` has %(step)s type."
                % {
                    "min_type": type(min_value).__name__,
                    "max_type": type(max_value).__name__,
                    "step": type(step).__name__,
                }
            )

        # Ensure that the value matches arguments' types.
        all_ints = data_type == SliderProto.INT and int_args
        all_floats = data_type == SliderProto.FLOAT and float_args
        all_timelikes = data_type in TIMELIKE_TYPES and timelike_args

        if not all_ints and not all_floats and not all_timelikes:
            raise StreamlitAPIException(
                "Both value and arguments must be of the same type."
                "\n`value` has %(value_type)s type."
                "\n`min_value` has %(min_type)s type."
                "\n`max_value` has %(max_type)s type."
                % {
                    "value_type": type(value).__name__,
                    "min_type": type(min_value).__name__,
                    "max_type": type(max_value).__name__,
                }
            )

        # Ensure that min <= value(s) <= max, adjusting the bounds as necessary.
        min_value = min(min_value, max_value)
        max_value = max(min_value, max_value)
        if len(value) == 1:
            min_value = min(value[0], min_value)
            max_value = max(value[0], max_value)
        elif len(value) == 2:
            start, end = value
            if start > end:
                # Swap start and end, since they seem reversed
                start, end = end, start
                value = start, end
            min_value = min(start, min_value)
            max_value = max(end, max_value)
        else:
            # Empty list, so let's just use the outer bounds
            value = [min_value, max_value]

        # Bounds checks. JSNumber produces human-readable exceptions that
        # we simply re-package as StreamlitAPIExceptions.
        # (We check `min_value` and `max_value` here; `value` and `step` are
        # already known to be in the [min_value, max_value] range.)
        try:
            if all_ints:
                JSNumber.validate_int_bounds(min_value, "`min_value`")
                JSNumber.validate_int_bounds(max_value, "`max_value`")
            elif all_floats:
                JSNumber.validate_float_bounds(min_value, "`min_value`")
                JSNumber.validate_float_bounds(max_value, "`max_value`")
            elif all_timelikes:
                # No validation yet. TODO: check between 0001-01-01 to 9999-12-31
                pass
        except JSNumberBoundsException as e:
            raise StreamlitAPIException(str(e))

        orig_tz = None
        # Convert dates or times into datetimes
        if data_type == SliderProto.TIME:
            value = list(map(_time_to_datetime, value))
            min_value = _time_to_datetime(min_value)
            max_value = _time_to_datetime(max_value)

        if data_type == SliderProto.DATE:
            value = list(map(_date_to_datetime, value))
            min_value = _date_to_datetime(min_value)
            max_value = _date_to_datetime(max_value)

        # Now, convert to microseconds (so we can serialize datetime to a long)
        if data_type in TIMELIKE_TYPES:
            # Restore times/datetimes to original timezone (dates are always naive)
            orig_tz = (
                value[0].tzinfo
                if data_type in (SliderProto.TIME, SliderProto.DATETIME)
                else None
            )

            value = list(map(_datetime_to_micros, value))
            min_value = _datetime_to_micros(min_value)
            max_value = _datetime_to_micros(max_value)
            step = _delta_to_micros(cast(timedelta, step))

        # It would be great if we could guess the number of decimal places from
        # the `step` argument, but this would only be meaningful if step were a
        # decimal. As a possible improvement we could make this function accept
        # decimals and/or use some heuristics for floats.

        slider_proto = SliderProto()
        slider_proto.type = SliderProto.Type.SLIDER
        slider_proto.label = label
        slider_proto.format = format
        slider_proto.default[:] = value
        slider_proto.min = min_value
        slider_proto.max = max_value
        slider_proto.step = cast(float, step)
        slider_proto.data_type = data_type
        slider_proto.options[:] = []
        slider_proto.form_id = current_form_id(self.dg)
        if help is not None:
            slider_proto.help = dedent(help)

        serde = SliderSerde(value, data_type, single_value, orig_tz)

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
        return cast(SliderReturn, widget_state.value)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)
