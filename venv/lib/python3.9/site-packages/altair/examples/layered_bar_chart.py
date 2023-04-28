"""
Layered Bar Chart
-----------------
This example shows a segmented bar chart that is layered rather than stacked.  
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.iowa_electricity()

alt.Chart(source).mark_bar(opacity=0.7).encode(
    x='year:O',
    y=alt.Y('net_generation:Q', stack=None),
    color="source",
)
