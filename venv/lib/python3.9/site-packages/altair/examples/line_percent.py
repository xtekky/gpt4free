"""
Line Chart with Percent axis
----------------------------
This example shows how to format the tick labels of the y-axis of a chart as percentages.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.jobs.url

alt.Chart(source).mark_line().encode(
    alt.X('year:O'),
    alt.Y('perc:Q', axis=alt.Axis(format='%')),
    color='sex:N'
).transform_filter(
    alt.datum.job == 'Welder'
)
