"""
Line Chart with Varying Size
----------------------------
This is example of using the ``trail`` marker to vary the size of a line.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.wheat()

alt.Chart(source).mark_trail().encode(
    x='year:T',
    y='wheat:Q',
    size='wheat:Q'
)
