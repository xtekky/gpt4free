"""
Sorted Error Bars showing Confidence Interval
=============================================
This example shows how to show error bars using confidence intervals, while also sorting the y-axis based on x-axis values. 
"""
# category: other charts

import altair as alt
from vega_datasets import data

source = data.barley()

points = alt.Chart(source).mark_point(
    filled=True,
    color='black'
).encode(
    x=alt.X('mean(yield)', title='Barley Yield'),
    y=alt.Y(
        'variety',
         sort=alt.EncodingSortField(
             field='yield',
             op='mean',
             order='descending'
         )
    )
).properties(
    width=400,
    height=250
)

error_bars = points.mark_rule().encode(
    x='ci0(yield)',
    x2='ci1(yield)',
)

points + error_bars
