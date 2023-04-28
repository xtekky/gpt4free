"""
Bar Chart with Rolling Mean
---------------------------
A bar chart overlayed with a rolling mean. In this example the average of values over the previous decade is displayed as a line.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.wheat()

bar = alt.Chart(source).mark_bar().encode(
    x='year:O',
    y='wheat:Q'
)

line = alt.Chart(source).mark_line(color='red').transform_window(
    # The field to average
    rolling_mean='mean(wheat)',
    # The number of values before and after the current value to include.
    frame=[-9, 0]
).encode(
    x='year:O',
    y='rolling_mean:Q'
)

(bar + line).properties(width=600)
