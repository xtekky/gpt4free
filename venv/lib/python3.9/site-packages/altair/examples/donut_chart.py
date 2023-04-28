"""
Donut Chart
-----------
This example shows how to make a Donut Chart using ``mark_arc``.
This is adapted from a corresponding Vega-Lite Example:
`Donut Chart <https://vega.github.io/vega-lite/examples/arc_donut.html>`_.
"""
# category: circular plots

import pandas as pd
import altair as alt

source = pd.DataFrame({"category": [1, 2, 3, 4, 5, 6], "value": [4, 6, 10, 3, 7, 8]})

alt.Chart(source).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="value", type="quantitative"),
    color=alt.Color(field="category", type="nominal"),
)
