"""
Line Chart with Layered Aggregates
----------------------------------
This example shows how to make a multi-series line chart of the daily closing
stock prices for AAPL, AMZN, GOOG, IBM, and MSFT between 2000 and 2010, along
with a layered rule showing the average values.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.stocks()

base = alt.Chart(source).properties(width=550)

line = base.mark_line().encode(
    x='date',
    y='price',
    color='symbol'
)

rule = base.mark_rule().encode(
    y='average(price)',
    color='symbol',
    size=alt.value(2)
)

line + rule
