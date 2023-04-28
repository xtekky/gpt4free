"""
Bump Chart
----------
This example shows a bump chart.  The data is first grouped into six-month
intervals using pandas. The ranks are computed by Altair using a 
window transform.
"""
# category: line charts

import altair as alt
from vega_datasets import data
import pandas as pd

stocks = data.stocks()
source = stocks.groupby([pd.Grouper(key="date", freq="6M"),"symbol"]).mean().reset_index()

alt.Chart(source).mark_line(point = True).encode(
    x = alt.X("date:O", timeUnit="yearmonth", title="date"),
    y="rank:O",
    color=alt.Color("symbol:N")
).transform_window(
    rank="rank()",
    sort=[alt.SortField("price", order="descending")],
    groupby=["date"]
).properties(
    title="Bump Chart for Stock Prices",
    width=600,
    height=150,
)