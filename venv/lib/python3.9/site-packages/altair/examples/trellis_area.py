"""
Trellis Area Chart
------------------
This example shows small multiples of an area chart.
"""
# category: area charts
import altair as alt
from vega_datasets import data

source = data.iowa_electricity()

alt.Chart(source).mark_area().encode(
    x="year:T",
    y="net_generation:Q",
    color="source:N",
    row="source:N"
).properties(
    height=100
)
