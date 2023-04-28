"""
Bar Chart with Line at Mean
---------------------------
This example shows the mean value overlayed on a bar chart.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source = data.wheat()

bar = alt.Chart(source).mark_bar().encode(
    x='year:O',
    y='wheat:Q'
)

rule = alt.Chart(source).mark_rule(color='red').encode(
    y='mean(wheat):Q'
)

(bar + rule).properties(width=600)
