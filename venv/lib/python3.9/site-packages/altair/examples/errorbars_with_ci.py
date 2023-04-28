"""
Error Bars showing Confidence Interval
======================================
This example shows how to show error bars using confidence intervals.
The confidence intervals are computed internally in vega by a non-parametric
`bootstrap of the mean <https://github.com/vega/vega-statistics/blob/master/src/bootstrapCI.js>`_.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.barley()

error_bars = alt.Chart(source).mark_errorbar(extent='ci').encode(
  x=alt.X('yield:Q', scale=alt.Scale(zero=False)),
  y=alt.Y('variety:N')
)

points = alt.Chart(source).mark_point(filled=True, color='black').encode(
  x=alt.X('yield:Q', aggregate='mean'),
  y=alt.Y('variety:N'),
)

error_bars + points
