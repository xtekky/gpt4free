"""
Multiple Marks
==============
This example demonstrates creating a single chart with multiple markers
representing the same data.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.stocks()

alt.Chart(source).mark_line(point=True).encode(
    x='date:T',
    y='price:Q',
    color='symbol:N'
)
