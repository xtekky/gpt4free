"""
Error Bar with Standard Deviation
---------------------------------
This example shows how to show error bars with standard deviation using crop yields data of different
in the years of 1930s.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.barley()

error_bars = alt.Chart(source).mark_errorbar(extent='stdev').encode(
  x=alt.X('yield:Q', scale=alt.Scale(zero=False)),
  y=alt.Y('variety:N')
)

points = alt.Chart(source).mark_point(filled=True, color='black').encode(
  x=alt.X('yield:Q', aggregate='mean'),
  y=alt.Y('variety:N'),
)

error_bars + points
