"""
Scatter Plot with Rolling Mean
------------------------------
A scatter plot with a rolling mean overlay. In this example a 30 day window
is used to calculate the mean of the maximum temperature around each date.
"""
# category: scatter plots

import altair as alt
from vega_datasets import data

source = data.seattle_weather()

line = alt.Chart(source).mark_line(
    color='red', 
    size=3
).transform_window(
    rolling_mean='mean(temp_max)',
    frame=[-15, 15]
).encode(
    x='date:T',
    y='rolling_mean:Q'
)

points = alt.Chart(source).mark_point().encode(
    x='date:T', 
    y=alt.Y('temp_max:Q', 
            axis=alt.Axis(title='Max Temp'))
)

points + line
