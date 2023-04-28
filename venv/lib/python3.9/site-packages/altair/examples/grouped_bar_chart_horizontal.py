"""
Horizontal Grouped Bar Chart
----------------------------
This example shows a horizontal grouped bar chart.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_bar().encode(
    x='sum(yield):Q',
    y='year:O',
    color='year:N',
    row='site:N'
)
