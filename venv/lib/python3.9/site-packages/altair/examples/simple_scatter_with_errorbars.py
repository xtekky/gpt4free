"""
Simple Scatter Plot with Errorbars
----------------------------------
A simple scatter plot of a data set with errorbars.
"""
# category: scatter plots
import altair as alt
import pandas as pd
import numpy as np

# generate some data points with uncertainties
np.random.seed(0)
x = [1, 2, 3, 4, 5]
y = np.random.normal(10, 0.5, size=len(x))
yerr = 0.2

# set up data frame
source = pd.DataFrame({"x": x, "y": y, "yerr": yerr})

# the base chart
base = alt.Chart(source).transform_calculate(
    ymin="datum.y-datum.yerr",
    ymax="datum.y+datum.yerr"
)

# generate the points
points = base.mark_point(
    filled=True,
    size=50,
    color='black'
).encode(
    x=alt.X('x', scale=alt.Scale(domain=(0, 6))),
    y=alt.Y('y', scale=alt.Scale(zero=False))
)

# generate the error bars
errorbars = base.mark_errorbar().encode(
    x="x",
    y="ymin:Q",
    y2="ymax:Q"
)

points + errorbars
