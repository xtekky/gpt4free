"""
Slope Graph
-----------------------
This example shows how to make Slope Graph.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_line().encode(
    x='year:O',
    y='median(yield)',
    color='site'
)
