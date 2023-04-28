'''
Trellis Area Sort Chart
-----------------------
This example shows small multiples of an area chart.
Stock prices of four large companies
sorted by `['MSFT', 'AAPL', 'IBM', 'AMZN']`
'''
# category: area charts
import altair as alt
from vega_datasets import data

source = data.stocks()

alt.Chart(source).transform_filter(
    alt.datum.symbol != 'GOOG'
).mark_area().encode(
    x='date:T',
    y='price:Q',
    color='symbol:N',
    row=alt.Row('symbol:N', sort=['MSFT', 'AAPL', 'IBM', 'AMZN'])
).properties(height=50, width=400)
