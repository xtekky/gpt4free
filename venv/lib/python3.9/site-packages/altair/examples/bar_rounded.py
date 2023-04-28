"""
Bar Chart with rounded edges
----------------------------
This example shows how to create a bar chart with rounded edges.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.seattle_weather()

alt.Chart(source).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x='month(date):O',
    y='count():Q',
    color='weather:N'
)
