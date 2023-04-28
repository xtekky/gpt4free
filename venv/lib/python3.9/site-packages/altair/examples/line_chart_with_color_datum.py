"""
Line Chart with datum for color
-------------------------------
An example of using ``datum`` and ``repeat`` to color a multi-series line chart.
This is adapted from this corresponding Vega-Lite Example:
`Repeat and Layer to Show Different Movie Measures <https://vega.github.io/vega-lite/examples/repeat_layer.html>`_.
"""
# category: line charts

import altair as alt
from vega_datasets import data

source = data.movies()

alt.Chart(source).mark_line().encode(
    x=alt.X("IMDB_Rating", bin=True),
    y=alt.Y(
        alt.repeat("layer"), aggregate="mean", title="Mean of US and Worldwide Gross"
    ),
    color=alt.datum(alt.repeat("layer")),
).repeat(layer=["US_Gross", "Worldwide_Gross"])
