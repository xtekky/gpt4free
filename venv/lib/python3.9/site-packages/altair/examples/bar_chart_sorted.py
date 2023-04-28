"""
Sorted Bar Chart
================
This example shows a bar chart sorted by a calculated value.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_bar().encode(
    x='sum(yield):Q',
    y=alt.Y('site:N', sort='-x')
)
