"""
Wind Vector Map
---------------
An example showing a vector array map showing wind speed and direction using ``wedge``
as shape for ``mark_point`` and ``angle`` encoding for the wind direction.
This is adapted from this corresponding Vega-Lite Example:
`Wind Vector Map <https://vega.github.io/vega-lite/examples/point_angle_windvector.html>`_.
"""
# category: scatter plots

import altair as alt
from vega_datasets import data

source = data.windvectors()

alt.Chart(source).mark_point(shape="wedge", filled=True).encode(
    latitude="latitude",
    longitude="longitude",
    color=alt.Color(
        "dir", scale=alt.Scale(domain=[0, 360], scheme="rainbow"), legend=None
    ),
    angle=alt.Angle("dir", scale=alt.Scale(domain=[0, 360], range=[180, 540])),
    size=alt.Size("speed", scale=alt.Scale(rangeMax=500)),
).project("equalEarth")
