"""
Bar Chart with Line on Dual Axis
--------------------------------
This example shows how to combine two plots and keep their axes.

For a more polished version of this chart, see :ref:`gallery_wheat_wages`.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.wheat()

base = alt.Chart(source).encode(x='year:O')

bar = base.mark_bar().encode(y='wheat:Q')

line =  base.mark_line(color='red').encode(
    y='wages:Q'
)

(bar + line).properties(width=600)
