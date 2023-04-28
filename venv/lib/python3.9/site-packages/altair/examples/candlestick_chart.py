"""
Candlestick Chart
=================
A candlestick chart inspired from `Protovis <http://mbostock.github.io/protovis/ex/candlestick.html>`_. 
This example shows the performance of the Chicago Board Options Exchange `Volatility Index <https://en.wikipedia.org/wiki/VIX>`_ (VIX) 
in the summer of 2009. The thick bar represents the opening and closing prices, 
while the thin bar shows intraday high and low prices; if the index closed higher on a given day, the bars are colored green rather than red.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.ohlc()

open_close_color = alt.condition("datum.open <= datum.close",
                                 alt.value("#06982d"),
                                 alt.value("#ae1325"))

base = alt.Chart(source).encode(
    alt.X('date:T',
          axis=alt.Axis(
              format='%m/%d', 
              labelAngle=-45, 
              title='Date in 2009'
          )
    ),
    color=open_close_color
)

rule = base.mark_rule().encode(
    alt.Y(
        'low:Q',
        title='Price',
        scale=alt.Scale(zero=False),
    ),
    alt.Y2('high:Q')
)

bar = base.mark_bar().encode(
    alt.Y('open:Q'),
    alt.Y2('close:Q')
)

rule + bar