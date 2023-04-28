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

"""Streamlit support for Plotly charts."""

import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union, cast

from typing_extensions import Final, Literal, TypeAlias

from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    import matplotlib
    import plotly.graph_objs as go
    from plotly.basedatatypes import BaseFigure

    from streamlit.delta_generator import DeltaGenerator


try:
    import plotly.io as pio

    import streamlit.elements.lib.streamlit_plotly_theme

    pio.templates.default = "streamlit"
except ModuleNotFoundError:
    # We have imports here because it takes too loo long to load the template default for the first graph to load
    # We do nothing if Plotly is not installed. This is expected since Plotly is an optional dependency.
    pass

LOGGER: Final = get_logger(__name__)

SharingMode: TypeAlias = Literal["streamlit", "private", "public", "secret"]

SHARING_MODES: Set[SharingMode] = {
    # This means the plot will be sent to the Streamlit app rather than to
    # Plotly.
    "streamlit",
    # The three modes below are for plots that should be hosted in Plotly.
    # These are the names Plotly uses for them.
    "private",
    "public",
    "secret",
}

_AtomicFigureOrData: TypeAlias = Union[
    "go.Figure",
    "go.Data",
]
FigureOrData: TypeAlias = Union[
    _AtomicFigureOrData,
    List[_AtomicFigureOrData],
    # It is kind of hard to figure out exactly what kind of dict is supported
    # here, as plotly hasn't embraced typing yet. This version is chosen to
    # align with the docstring.
    Dict[str, _AtomicFigureOrData],
    "BaseFigure",
    "matplotlib.figure.Figure",
]


class PlotlyMixin:
    @gather_metrics("plotly_chart")
    def plotly_chart(
        self,
        figure_or_data: FigureOrData,
        use_container_width: bool = False,
        sharing: SharingMode = "streamlit",
        theme: Union[None, Literal["streamlit"]] = "streamlit",
        **kwargs: Any,
    ) -> "DeltaGenerator":
        """Display an interactive Plotly chart.

        Plotly is a charting library for Python. The arguments to this function
        closely follow the ones for Plotly's `plot()` function. You can find
        more about Plotly at https://plot.ly/python.

        To show Plotly charts in Streamlit, call `st.plotly_chart` wherever you
        would call Plotly's `py.plot` or `py.iplot`.

        Parameters
        ----------
        figure_or_data : plotly.graph_objs.Figure, plotly.graph_objs.Data,
            dict/list of plotly.graph_objs.Figure/Data

            See https://plot.ly/python/ for examples of graph descriptions.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the figure's native `width` value.

        sharing : {'streamlit', 'private', 'secret', 'public'}
            Use 'streamlit' to insert the plot and all its dependencies
            directly in the Streamlit app using plotly's offline mode (default).
            Use any other sharing mode to send the chart to Plotly chart studio, which
            requires an account. See https://plot.ly/python/chart-studio/ for more information.

        theme : "streamlit" or None
            The theme of the chart. Currently, we only support "streamlit" for the Streamlit
            defined design or None to fallback to the default behavior of the library.

        **kwargs
            Any argument accepted by Plotly's `plot()` function.

        Example
        -------
        The example below comes straight from the examples at
        https://plot.ly/python:

        >>> import streamlit as st
        >>> import numpy as np
        >>> import plotly.figure_factory as ff
        >>>
        >>> # Add histogram data
        >>> x1 = np.random.randn(200) - 2
        >>> x2 = np.random.randn(200)
        >>> x3 = np.random.randn(200) + 2
        >>>
        >>> # Group data together
        >>> hist_data = [x1, x2, x3]
        >>>
        >>> group_labels = ['Group 1', 'Group 2', 'Group 3']
        >>>
        >>> # Create distplot with custom bin_size
        >>> fig = ff.create_distplot(
        ...         hist_data, group_labels, bin_size=[.1, .25, .5])
        >>>
        >>> # Plot!
        >>> st.plotly_chart(fig, use_container_width=True)

        .. output::
           https://doc-plotly-chart.streamlitapp.com/
           height: 400px

        """
        # NOTE: "figure_or_data" is the name used in Plotly's .plot() method
        # for their main parameter. I don't like the name, but it's best to
        # keep it in sync with what Plotly calls it.

        plotly_chart_proto = PlotlyChartProto()
        if theme != "streamlit" and theme != None:
            raise StreamlitAPIException(
                f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.'
            )
        marshall(
            plotly_chart_proto,
            figure_or_data,
            use_container_width,
            sharing,
            theme,
            **kwargs,
        )
        return self.dg._enqueue("plotly_chart", plotly_chart_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall(
    proto: PlotlyChartProto,
    figure_or_data: FigureOrData,
    use_container_width: bool,
    sharing: SharingMode,
    theme: Union[None, Literal["streamlit"]],
    **kwargs: Any,
) -> None:
    """Marshall a proto with a Plotly spec.

    See DeltaGenerator.plotly_chart for docs.
    """
    # NOTE: "figure_or_data" is the name used in Plotly's .plot() method
    # for their main parameter. I don't like the name, but its best to keep
    # it in sync with what Plotly calls it.

    import plotly.tools

    if type_util.is_type(figure_or_data, "matplotlib.figure.Figure"):
        figure = plotly.tools.mpl_to_plotly(figure_or_data)

    else:
        figure = plotly.tools.return_figure_from_figure_or_data(
            figure_or_data, validate_figure=True
        )

    if not isinstance(sharing, str) or sharing.lower() not in SHARING_MODES:
        raise ValueError("Invalid sharing mode for Plotly chart: %s" % sharing)

    proto.use_container_width = use_container_width

    if sharing == "streamlit":
        import plotly.utils

        config = dict(kwargs.get("config", {}))
        # Copy over some kwargs to config dict. Plotly does the same in plot().
        config.setdefault("showLink", kwargs.get("show_link", False))
        config.setdefault("linkText", kwargs.get("link_text", False))

        proto.figure.spec = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
        proto.figure.config = json.dumps(config)

    else:
        url = _plot_to_url_or_load_cached_url(
            figure, sharing=sharing, auto_open=False, **kwargs
        )
        proto.url = _get_embed_url(url)
    proto.theme = theme or ""


@caching.cache
def _plot_to_url_or_load_cached_url(*args: Any, **kwargs: Any) -> "go.Figure":
    """Call plotly.plot wrapped in st.cache.

    This is so we don't unnecessarily upload data to Plotly's SASS if nothing
    changed since the previous upload.
    """
    try:
        # Plotly 4 changed its main package.
        import chart_studio.plotly as ply
    except ImportError:
        import plotly.plotly as ply

    return ply.plot(*args, **kwargs)


def _get_embed_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)

    # Plotly's embed URL is the normal URL plus ".embed".
    # (Note that our use namedtuple._replace is fine because that's not a
    # private method! It just has an underscore to avoid clashing with the
    # tuple field names)
    parsed_embed_url = parsed_url._replace(path=parsed_url.path + ".embed")

    return urllib.parse.urlunparse(parsed_embed_url)
