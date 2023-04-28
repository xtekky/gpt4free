"""
Horizontal Bar Chart
--------------------
This example is a bar chart drawn horizontally by putting the quantitative value on the x axis.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.wheat()

alt.Chart(source).mark_bar().encode(
    x='wheat:Q',
    y="year:O"
).properties(height=700)
