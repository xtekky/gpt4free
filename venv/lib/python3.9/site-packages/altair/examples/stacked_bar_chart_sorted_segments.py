"""
Stacked Bar Chart with Sorted Segments
--------------------------------------
This is an example of a stacked-bar chart with the segments of each bar resorted.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_bar().encode(
    x='sum(yield)',
    y='variety',
    color='site',
    order=alt.Order(
      # Sort the segments of the bars by this field
      'site',
      sort='ascending'
    )
)
