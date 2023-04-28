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

"""A Python wrapper around Bokeh."""

import hashlib
import json
from typing import TYPE_CHECKING, cast

from typing_extensions import Final

from streamlit.errors import StreamlitAPIException
from streamlit.proto.BokehChart_pb2 import BokehChart as BokehChartProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from bokeh.plotting.figure import Figure

    from streamlit.delta_generator import DeltaGenerator

ST_BOKEH_VERSION: Final = "2.4.3"


class BokehMixin:
    @gather_metrics("bokeh_chart")
    def bokeh_chart(
        self,
        figure: "Figure",
        use_container_width: bool = False,
    ) -> "DeltaGenerator":
        """Display an interactive Bokeh chart.

        Bokeh is a charting library for Python. The arguments to this function
        closely follow the ones for Bokeh's `show` function. You can find
        more about Bokeh at https://bokeh.pydata.org.

        To show Bokeh charts in Streamlit, call `st.bokeh_chart`
        wherever you would call Bokeh's `show`.

        Parameters
        ----------
        figure : bokeh.plotting.figure.Figure
            A Bokeh figure to plot.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over Bokeh's native `width` value.

        Example
        -------
        >>> import streamlit as st
        >>> from bokeh.plotting import figure
        >>>
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [6, 7, 2, 4, 5]
        >>>
        >>> p = figure(
        ...     title='simple line example',
        ...     x_axis_label='x',
        ...     y_axis_label='y')
        ...
        >>> p.line(x, y, legend_label='Trend', line_width=2)
        >>>
        >>> st.bokeh_chart(p, use_container_width=True)

        .. output::
           https://doc-bokeh-chart.streamlitapp.com/
           height: 700px

        """
        import bokeh

        if bokeh.__version__ != ST_BOKEH_VERSION:
            raise StreamlitAPIException(
                f"Streamlit only supports Bokeh version {ST_BOKEH_VERSION}, "
                f"but you have version {bokeh.__version__} installed. Please "
                f"run `pip install --force-reinstall --no-deps bokeh=="
                f"{ST_BOKEH_VERSION}` to install the correct version."
            )

        # Generate element ID from delta path
        delta_path = self.dg._get_delta_path_str()
        element_id = hashlib.md5(delta_path.encode()).hexdigest()

        bokeh_chart_proto = BokehChartProto()
        marshall(bokeh_chart_proto, figure, use_container_width, element_id)
        return self.dg._enqueue("bokeh_chart", bokeh_chart_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall(
    proto: BokehChartProto,
    figure: "Figure",
    use_container_width: bool,
    element_id: str,
) -> None:
    """Construct a Bokeh chart object.

    See DeltaGenerator.bokeh_chart for docs.
    """
    from bokeh.embed import json_item

    data = json_item(figure)
    proto.figure = json.dumps(data)
    proto.use_container_width = use_container_width
    proto.element_id = element_id
