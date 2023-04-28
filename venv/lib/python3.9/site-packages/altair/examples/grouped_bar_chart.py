"""
Grouped Bar Chart
-----------------
This example shows a grouped bar chart.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_bar().encode(
    x='year:O',
    y='sum(yield):Q',
    color='year:N',
    column='site:N'
)
