"""
Pie Chart with Labels
---------------------
This example shows how to layer text over arc marks (``mark_arc``) to label pie charts.
This is adapted from a corresponding Vega-Lite Example:
`Pie Chart with Labels <https://vega.github.io/vega-lite/examples/layer_arc_label.html>`_.
"""
# category: circular plots

import pandas as pd
import altair as alt

source = pd.DataFrame(
    {"category": ["a", "b", "c", "d", "e", "f"], "value": [4, 6, 10, 3, 7, 8]}
)

base = alt.Chart(source).encode(
    theta=alt.Theta("value:Q", stack=True), color=alt.Color("category:N", legend=None)
)

pie = base.mark_arc(outerRadius=120)
text = base.mark_text(radius=140, size=20).encode(text="category:N")

pie + text
