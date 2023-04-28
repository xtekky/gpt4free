"""
Simple Stacked Area Chart
-------------------------
This example shows how to make a simple stacked area chart.
"""
# category: simple charts
import altair as alt
from vega_datasets import data

source = data.iowa_electricity()

alt.Chart(source).mark_area().encode(
    x="year:T",
    y="net_generation:Q",
    color="source:N"
)
