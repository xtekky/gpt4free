"""
Bar Chart with Highlighted Segment
----------------------------------
This example shows a bar chart that highlights values beyond a threshold.
"""
import altair as alt
import pandas as pd
from vega_datasets import data

source = data.wheat()
threshold = pd.DataFrame([{"threshold": 90}])

bars = alt.Chart(source).mark_bar().encode(
    x="year:O",
    y="wheat:Q",
)

highlight = alt.Chart(source).mark_bar(color="#e45755").encode(
    x='year:O',
    y='baseline:Q',
    y2='wheat:Q'
).transform_filter(
    alt.datum.wheat > 90
).transform_calculate("baseline", "90")

rule = alt.Chart(threshold).mark_rule().encode(
    y='threshold:Q'
)

(bars + highlight + rule).properties(width=600)
