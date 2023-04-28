"""
Step Chart
----------
This example shows Google's stock price over time.
This uses the "step-after" interpolation scheme.
The full list of interpolation options includes 'linear',
'linear-closed', 'step', 'step-before', 'step-after', 'basis',
'basis-open', 'basis-closed', 'cardinal', 'cardinal-open',
'cardinal-closed', 'bundle', and 'monotone'.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.stocks()

alt.Chart(source).mark_line(interpolate='step-after').encode(
    x='date',
    y='price'
).transform_filter(
    alt.datum.symbol == 'GOOG'
)
