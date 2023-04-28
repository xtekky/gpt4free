"""
Stacked Bar Chart
-----------------

This is an example of a stacked bar chart using data which contains crop yields over different regions and different years in the 1930s.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source).mark_bar().encode(
    x='variety',
    y='sum(yield)',
    color='site'
)
