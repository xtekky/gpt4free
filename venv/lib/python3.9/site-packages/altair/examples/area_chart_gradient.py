"""
Area Chart with Gradient
------------------------
This example shows how to make an area chart with a gradient fill. 
For more information about gradient options see the Vega-Lite `Gradient documentation <https://vega.github.io/vega-lite/docs/types.html#gradient>`_.
"""
# category: area charts

import altair as alt
from vega_datasets import data

source = data.stocks()

alt.Chart(source).transform_filter(
    'datum.symbol==="GOOG"'
).mark_area(
    line={'color':'darkgreen'},
    color=alt.Gradient(
        gradient='linear', 
        stops=[alt.GradientStop(color='white', offset=0), 
               alt.GradientStop(color='darkgreen', offset=1)], 
        x1=1, 
        x2=1, 
        y1=1, 
        y2=0
    )
).encode(
    alt.X('date:T'), 
    alt.Y('price:Q')
)