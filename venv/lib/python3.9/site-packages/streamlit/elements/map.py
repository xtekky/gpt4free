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

"""A wrapper for simple PyDeck scatter charts."""

import copy
import json
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union, cast

import pandas as pd
from typing_extensions import Final, TypeAlias

import streamlit.elements.deck_gl_json_chart as deck_gl_json_chart
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as DeckGlJsonChartProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

    from streamlit.delta_generator import DeltaGenerator


Data: TypeAlias = Union[
    pd.DataFrame,
    "Styler",
    Iterable[Any],
    Dict[Any, Any],
    None,
]

# Map used as the basis for st.map.
_DEFAULT_MAP: Final[Dict[str, Any]] = dict(deck_gl_json_chart.EMPTY_MAP)

# Other default parameters for st.map.
_DEFAULT_COLOR: Final = [200, 30, 0, 160]
_DEFAULT_ZOOM_LEVEL: Final = 12
_ZOOM_LEVELS: Final = [
    360,
    180,
    90,
    45,
    22.5,
    11.25,
    5.625,
    2.813,
    1.406,
    0.703,
    0.352,
    0.176,
    0.088,
    0.044,
    0.022,
    0.011,
    0.005,
    0.003,
    0.001,
    0.0005,
    0.00025,
]


class MapMixin:
    @gather_metrics("map")
    def map(
        self,
        data: Data = None,
        zoom: Optional[int] = None,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display a map with points on it.

        This is a wrapper around ``st.pydeck_chart`` to quickly create
        scatterplot charts on top of a map, with auto-centering and auto-zoom.

        When using this command, Mapbox provides the map tiles to render map
        content. Note that Mapbox is a third-party product, the use of which is
        governed by Mapbox's Terms of Use.

        Mapbox requires users to register and provide a token before users can
        request map tiles. Currently, Streamlit provides this token for you, but
        this could change at any time. We strongly recommend all users create and
        use their own personal Mapbox token to avoid any disruptions to their
        experience. You can do this with the ``mapbox.token`` config option.

        To get a token for yourself, create an account at https://mapbox.com.
        For more info on how to set config options, see
        https://docs.streamlit.io/library/advanced-features/configuration

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, dict, or None
            The data to be plotted. Must have two columns:

            - latitude called 'lat', 'latitude', 'LAT', 'LATITUDE'
            - longitude called 'lon', 'longitude', 'LON', 'LONGITUDE'.

        zoom : int
            Zoom level as specified in
            https://wiki.openstreetmap.org/wiki/Zoom_levels
        use_container_width: bool

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        ...     columns=['lat', 'lon'])
        >>>
        >>> st.map(df)

        .. output::
           https://doc-map.streamlitapp.com/
           height: 650px

        """
        map_proto = DeckGlJsonChartProto()
        map_proto.json = to_deckgl_json(data, zoom)
        map_proto.use_container_width = use_container_width
        return self.dg._enqueue("deck_gl_json_chart", map_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def _get_zoom_level(distance: float) -> int:
    """Get the zoom level for a given distance in degrees.

    See https://wiki.openstreetmap.org/wiki/Zoom_levels for reference.

    Parameters
    ----------
    distance : float
        How many degrees of longitude should fit in the map.

    Returns
    -------
    int
        The zoom level, from 0 to 20.

    """
    for i in range(len(_ZOOM_LEVELS) - 1):
        if _ZOOM_LEVELS[i + 1] < distance <= _ZOOM_LEVELS[i]:
            return i

    # For small number of points the default zoom level will be used.
    return _DEFAULT_ZOOM_LEVEL


def to_deckgl_json(data: Data, zoom: Optional[int]) -> str:
    if data is None:
        return json.dumps(_DEFAULT_MAP)

    # TODO(harahu): iterables don't have the empty attribute. This is either
    # a bug, or the documented data type is too broad. One or the other
    # should be addressed
    if hasattr(data, "empty") and data.empty:
        return json.dumps(_DEFAULT_MAP)

    data = type_util.convert_anything_to_df(data)
    formmated_column_names = ", ".join(map(repr, list(data.columns)))

    allowed_lat_columns = {"lat", "latitude", "LAT", "LATITUDE"}
    lat = next((d for d in allowed_lat_columns if d in data), None)

    if not lat:
        formatted_allowed_column_name = ", ".join(
            map(repr, sorted(allowed_lat_columns))
        )
        raise StreamlitAPIException(
            f"Map data must contain a latitude column named: {formatted_allowed_column_name}. "
            f"Existing columns: {formmated_column_names}"
        )

    allowed_lon_columns = {"lon", "longitude", "LON", "LONGITUDE"}
    lon = next((d for d in allowed_lon_columns if d in data), None)

    if not lon:
        formatted_allowed_column_name = ", ".join(
            map(repr, sorted(allowed_lon_columns))
        )
        raise StreamlitAPIException(
            f"Map data must contain a longitude column named: {formatted_allowed_column_name}. "
            f"Existing columns: {formmated_column_names}"
        )

    if data[lon].isnull().values.any() or data[lat].isnull().values.any():
        raise StreamlitAPIException("Latitude and longitude data must be numeric.")

    min_lat = data[lat].min()
    max_lat = data[lat].max()
    min_lon = data[lon].min()
    max_lon = data[lon].max()
    center_lat = (max_lat + min_lat) / 2.0
    center_lon = (max_lon + min_lon) / 2.0
    range_lon = abs(max_lon - min_lon)
    range_lat = abs(max_lat - min_lat)

    if zoom is None:
        if range_lon > range_lat:
            longitude_distance = range_lon
        else:
            longitude_distance = range_lat
        zoom = _get_zoom_level(longitude_distance)

    # "+1" because itertuples includes the row index.
    lon_col_index = data.columns.get_loc(lon) + 1
    lat_col_index = data.columns.get_loc(lat) + 1
    final_data = []
    for row in data.itertuples():
        final_data.append(
            {"lon": float(row[lon_col_index]), "lat": float(row[lat_col_index])}
        )

    default = copy.deepcopy(_DEFAULT_MAP)
    default["initialViewState"]["latitude"] = center_lat
    default["initialViewState"]["longitude"] = center_lon
    default["initialViewState"]["zoom"] = zoom
    default["layers"] = [
        {
            "@@type": "ScatterplotLayer",
            "getPosition": "@@=[lon, lat]",
            "getRadius": 10,
            "radiusScale": 10,
            "radiusMinPixels": 3,
            "getFillColor": _DEFAULT_COLOR,
            "data": final_data,
        }
    ]
    return json.dumps(default)
