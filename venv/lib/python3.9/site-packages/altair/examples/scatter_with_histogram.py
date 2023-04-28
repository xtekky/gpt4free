"""
Scatter Plot and Histogram with Interval Selection
==================================================

This example shows how to link a scatter plot and a histogram
together such that an interval selection in the histogram will
plot the selected values in the scatter plot.

Note that both subplots need to know about the `mbin` field created
by the `transform_bin` method. In order to achieve this, the data is
not passed to the `Chart()` instances creating the subplots, but
directly in the `hconcat()` function, which joins the two plots together.
"""
# category: interactive charts

import altair as alt
import pandas as pd
import numpy as np

x = np.random.normal(size=100)
y = np.random.normal(size=100)

m = np.random.normal(15, 1, size=100)

source = pd.DataFrame({"x": x, "y":y, "m":m})

# interval selection in the scatter plot
pts = alt.selection(type="interval", encodings=["x"])

# left panel: scatter plot
points = alt.Chart().mark_point(filled=True, color="black").encode(
    x='x',
    y='y'
).transform_filter(
    pts
).properties(
    width=300,
    height=300
)

# right panel: histogram
mag = alt.Chart().mark_bar().encode(
    x='mbin:N',
    y="count()",
    color=alt.condition(pts, alt.value("black"), alt.value("lightgray"))
).properties(
    width=300,
    height=300
).add_selection(pts)

# build the chart:
alt.hconcat(
    points,
    mag,
    data=source
).transform_bin(
    "mbin",
    field="m",
    bin=alt.Bin(maxbins=20)
)
