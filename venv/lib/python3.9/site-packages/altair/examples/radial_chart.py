"""
Radial Chart
------------
This radial plot uses both angular and radial extent to convey multiple dimensions of data.
This is adapted from a corresponding Vega-Lite Example:
`Radial Plot <https://vega.github.io/vega-lite/examples/arc_radial.html>`_.
"""
# category: circular plots

import pandas as pd
import altair as alt

source = pd.DataFrame({"values": [12, 23, 47, 6, 52, 19]})

base = alt.Chart(source).encode(
    theta=alt.Theta("values:Q", stack=True),
    radius=alt.Radius("values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
    color="values:N",
)

c1 = base.mark_arc(innerRadius=20, stroke="#fff")

c2 = base.mark_text(radiusOffset=10).encode(text="values:Q")

c1 + c2
