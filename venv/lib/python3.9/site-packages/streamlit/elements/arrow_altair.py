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

"""A Python wrapper around Altair.
Altair is a Python visualization library based on Vega-Lite,
a nice JSON schema for expressing graphs and charts.
"""

from datetime import date
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import altair as alt
import pandas as pd
from altair.vegalite.v4.api import Chart
from pandas.api.types import infer_dtype, is_integer_dtype
from typing_extensions import Literal

import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
    ArrowVegaLiteChart as ArrowVegaLiteChartProto,
)
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

# Create and enable streamlit theme
STREAMLIT_THEME = {"embedOptions": {"theme": "streamlit"}}

# This allows to use alt.themes.enable("streamlit") to activate Streamlit theme.
alt.themes.register("streamlit", lambda: {"usermeta": STREAMLIT_THEME})

# no theme applied to charts
alt.themes.enable("none")


class ChartType(Enum):
    AREA = "area"
    BAR = "bar"
    LINE = "line"


class ArrowAltairMixin:
    @gather_metrics("_arrow_line_chart")
    def _arrow_line_chart(
        self,
        data: Data = None,
        *,
        x: Union[str, None] = None,
        y: Union[str, Sequence[str], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display a line chart.

        This is syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_line_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, dict or None
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.
            This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings, draws several series
            on the same chart by melting your wide-format table into a long-format table behind
            the scenes. If None, draws the data of all remaining columns as data series.
            This argument can only be supplied by keyword.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.
            This argument can only be supplied by keyword.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> st._arrow_line_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.50.0-td2L/index.html?id=BdxXG3MmrVBfJyqS2R2ki8
           height: 220px

        """
        proto = ArrowVegaLiteChartProto()
        chart = _generate_chart(ChartType.LINE, data, x, y, width, height)
        marshall(proto, chart, use_container_width, theme="streamlit")
        last_index = last_index_for_melted_dataframes(data)

        return self.dg._enqueue("arrow_line_chart", proto, last_index=last_index)

    @gather_metrics("_arrow_area_chart")
    def _arrow_area_chart(
        self,
        data: Data = None,
        *,
        x: Union[str, None] = None,
        y: Union[str, Sequence[str], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display an area chart.

        This is just syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_area_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.
            This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings, draws several series
            on the same chart by melting your wide-format table into a long-format table behind
            the scenes. If None, draws the data of all remaining columns as data series.
            This argument can only be supplied by keyword.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> st._arrow_area_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.50.0-td2L/index.html?id=Pp65STuFj65cJRDfhGh4Jt
           height: 220px

        """

        proto = ArrowVegaLiteChartProto()
        chart = _generate_chart(ChartType.AREA, data, x, y, width, height)
        marshall(proto, chart, use_container_width, theme="streamlit")
        last_index = last_index_for_melted_dataframes(data)

        return self.dg._enqueue("arrow_area_chart", proto, last_index=last_index)

    @gather_metrics("_arrow_bar_chart")
    def _arrow_bar_chart(
        self,
        data: Data = None,
        *,
        x: Union[str, None] = None,
        y: Union[str, Sequence[str], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display a bar chart.

        This is just syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_bar_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.
            This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings, draws several series
            on the same chart by melting your wide-format table into a long-format table behind
            the scenes. If None, draws the data of all remaining columns as data series.
            This argument can only be supplied by keyword.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.
            This argument can only be supplied by keyword.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(50, 3),
        ...     columns=["a", "b", "c"])
        ...
        >>> st._arrow_bar_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.66.0-2BLtg/index.html?id=GaYDn6vxskvBUkBwsGVEaL
           height: 220px

        """

        proto = ArrowVegaLiteChartProto()
        chart = _generate_chart(ChartType.BAR, data, x, y, width, height)
        marshall(proto, chart, use_container_width, theme="streamlit")
        last_index = last_index_for_melted_dataframes(data)

        return self.dg._enqueue("arrow_bar_chart", proto, last_index=last_index)

    @gather_metrics("_arrow_altair_chart")
    def _arrow_altair_chart(
        self,
        altair_chart: Chart,
        use_container_width: bool = False,
        theme: Union[None, Literal["streamlit"]] = "streamlit",
    ) -> "DeltaGenerator":
        """Display a chart using the Altair library.

        Parameters
        ----------
        altair_chart : altair.vegalite.v2.api.Chart
            The Altair chart object to display.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over Altair's native `width` value.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>> import altair as alt
        >>>
        >>> df = pd.DataFrame(
        ...     np.random.randn(200, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> c = alt.Chart(df).mark_circle().encode(
        ...     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        >>>
        >>> st._arrow_altair_chart(c, use_container_width=True)

        .. output::
           https://static.streamlit.io/0.25.0-2JkNY/index.html?id=8jmmXR8iKoZGV4kXaKGYV5
           height: 200px

        Examples of Altair charts can be found at
        https://altair-viz.github.io/gallery/.

        """
        if theme != "streamlit" and theme != None:
            raise StreamlitAPIException(
                f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.'
            )
        proto = ArrowVegaLiteChartProto()
        marshall(
            proto,
            altair_chart,
            use_container_width=use_container_width,
            theme=theme,
        )

        return self.dg._enqueue("arrow_vega_lite_chart", proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def _is_date_column(df: pd.DataFrame, name: str) -> bool:
    """True if the column with the given name stores datetime.date values.

    This function just checks the first value in the given column, so
    it's meaningful only for columns whose values all share the same type.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        The column name

    Returns
    -------
    bool

    """
    column = df[name]
    if column.size == 0:
        return False

    return isinstance(column.iloc[0], date)


def _melt_data(
    data_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Converts a wide-format dataframe to a long-format dataframe."""

    data_df = pd.melt(
        data_df,
        id_vars=[x_column],
        value_vars=value_columns,
        var_name=color_column,
        value_name=y_column,
    )

    y_series = data_df[y_column]
    if (
        y_series.dtype == "object"
        and "mixed" in infer_dtype(y_series)
        and len(y_series.unique()) > 100
    ):
        raise StreamlitAPIException(
            "The columns used for rendering the chart contain too many values with mixed types. Please select the columns manually via the y parameter."
        )

    # Arrow has problems with object types after melting two different dtypes
    # pyarrow.lib.ArrowTypeError: "Expected a <TYPE> object, got a object"
    data_df = type_util.fix_arrow_incompatible_column_types(
        data_df, selected_columns=[x_column, color_column, y_column]
    )

    return data_df


def _maybe_melt(
    data_df: pd.DataFrame,
    x: Union[str, None] = None,
    y: Union[str, Sequence[str], None] = None,
) -> Tuple[pd.DataFrame, str, str, str, str, Optional[str], Optional[str]]:
    """Determines based on the selected x & y parameter, if the data needs to
    be converted to a long-format dataframe. If so, it returns the melted dataframe
    and the x, y, and color columns used for rendering the chart.
    """

    color_column: Optional[str]
    # This has to contain an empty space, otherwise the
    # full y-axis disappears (maybe a bug in vega-lite)?
    color_title: Optional[str] = " "

    y_column = "value"
    y_title = ""

    if x and isinstance(x, str):
        # x is a single string -> use for x-axis
        x_column = x
        x_title = x
        if x_column not in data_df.columns:
            raise StreamlitAPIException(
                f"{x_column} (x parameter) was not found in the data columns or keys”."
            )
    else:
        # use index for x-axis
        x_column = data_df.index.name or "index"
        x_title = ""
        data_df = data_df.reset_index()

    if y and type_util.is_sequence(y) and len(y) == 1:
        # Sequence is only a single element
        y = str(y[0])

    if y and isinstance(y, str):
        # y is a single string -> use for y-axis
        y_column = y
        y_title = y
        if y_column not in data_df.columns:
            raise StreamlitAPIException(
                f"{y_column} (y parameter) was not found in the data columns or keys”."
            )

        # Set var name to None since it should not be used
        color_column = None
    elif y and type_util.is_sequence(y):
        color_column = "variable"
        # y is a list -> melt dataframe into value vars provided in y
        value_columns: List[str] = []
        for col in y:
            if str(col) not in data_df.columns:
                raise StreamlitAPIException(
                    f"{str(col)} in y parameter was not found in the data columns or keys”."
                )
            value_columns.append(str(col))

        if x_column in [y_column, color_column]:
            raise StreamlitAPIException(
                f"Unable to melt the table. Please rename the columns used for x ({x_column}) or y ({y_column})."
            )

        data_df = _melt_data(data_df, x_column, y_column, color_column, value_columns)
    else:
        color_column = "variable"
        # -> data will be melted into the value prop for y
        data_df = _melt_data(data_df, x_column, y_column, color_column)

    relevant_columns = []
    if x_column and x_column not in relevant_columns:
        relevant_columns.append(x_column)
    if color_column and color_column not in relevant_columns:
        relevant_columns.append(color_column)
    if y_column and y_column not in relevant_columns:
        relevant_columns.append(y_column)
    # Only select the relevant columns required for the chart
    # Other columns can be ignored
    data_df = data_df[relevant_columns]
    return data_df, x_column, x_title, y_column, y_title, color_column, color_title


def _generate_chart(
    chart_type: ChartType,
    data: Data,
    x: Union[str, None] = None,
    y: Union[str, Sequence[str], None] = None,
    width: int = 0,
    height: int = 0,
) -> Chart:
    """Function to use the chart's type, data columns and indices to figure out the chart's spec."""

    if data is None:
        # Use an empty-ish dict because if we use None the x axis labels rotate
        # 90 degrees. No idea why. Need to debug.
        data = {"": []}

    if not isinstance(data, pd.DataFrame):
        data = type_util.convert_anything_to_df(data)

    data, x_column, x_title, y_column, y_title, color_column, color_title = _maybe_melt(
        data, x, y
    )

    opacity = None
    if chart_type == ChartType.AREA and color_column:
        opacity = {y_column: 0.7}
    # Set the X and Y axes' scale to "utc" if they contain date values.
    # This causes time data to be displayed in UTC, rather the user's local
    # time zone. (By default, vega-lite displays time data in the browser's
    # local time zone, regardless of which time zone the data specifies:
    # https://vega.github.io/vega-lite/docs/timeunit.html#output).
    x_scale = (
        alt.Scale(type="utc") if _is_date_column(data, x_column) else alt.Undefined
    )
    y_scale = (
        alt.Scale(type="utc") if _is_date_column(data, y_column) else alt.Undefined
    )

    x_type = alt.Undefined
    # Bar charts should have a discrete (ordinal) x-axis, UNLESS type is date/time
    # https://github.com/streamlit/streamlit/pull/2097#issuecomment-714802475
    if chart_type == ChartType.BAR and not _is_date_column(data, x_column):
        x_type = "ordinal"

    # Use a max tick size of 1 for integer columns (prevents zoom into float numbers)
    # and deactivate grid lines for x-axis
    x_axis_config = alt.Axis(
        tickMinStep=1 if is_integer_dtype(data[x_column]) else alt.Undefined, grid=False
    )
    y_axis_config = alt.Axis(
        tickMinStep=1 if is_integer_dtype(data[y_column]) else alt.Undefined
    )

    tooltips = [
        alt.Tooltip(x_column, title=x_column),
        alt.Tooltip(y_column, title=y_column),
    ]
    color = None

    if color_column:
        color = alt.Color(
            color_column,
            title=color_title,
            type="nominal",
            legend=alt.Legend(titlePadding=0, offset=10, orient="bottom"),
        )
        tooltips.append(alt.Tooltip(color_column, title="label"))

    chart = getattr(
        alt.Chart(data, width=width, height=height),
        "mark_" + chart_type.value,
    )().encode(
        x=alt.X(
            x_column,
            title=x_title,
            scale=x_scale,
            type=x_type,
            axis=x_axis_config,
        ),
        y=alt.Y(y_column, title=y_title, scale=y_scale, axis=y_axis_config),
        tooltip=tooltips,
    )

    if color:
        chart = chart.encode(color=color)

    if opacity:
        chart = chart.encode(opacity=opacity)

    return chart.interactive()


def marshall(
    vega_lite_chart: ArrowVegaLiteChartProto,
    altair_chart: Chart,
    use_container_width: bool = False,
    theme: Union[None, Literal["streamlit"]] = "streamlit",
    **kwargs: Any,
) -> None:
    """Marshall chart's data into proto."""
    import altair as alt

    # Normally altair_chart.to_dict() would transform the dataframe used by the
    # chart into an array of dictionaries. To avoid that, we install a
    # transformer that replaces datasets with a reference by the object id of
    # the dataframe. We then fill in the dataset manually later on.

    datasets = {}

    def id_transform(data) -> Dict[str, str]:
        """Altair data transformer that returns a fake named dataset with the
        object id.
        """
        datasets[id(data)] = data
        return {"name": str(id(data))}

    alt.data_transformers.register("id", id_transform)

    with alt.data_transformers.enable("id"):
        chart_dict = altair_chart.to_dict()

        # Put datasets back into the chart dict but note how they weren't
        # transformed.
        chart_dict["datasets"] = datasets

        arrow_vega_lite.marshall(
            vega_lite_chart,
            chart_dict,
            use_container_width=use_container_width,
            theme=theme,
            **kwargs,
        )
