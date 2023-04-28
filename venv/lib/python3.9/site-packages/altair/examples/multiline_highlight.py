"""
Multi-Line Highlight
====================
This multi-line chart uses an invisible Voronoi tessellation to handle mouseover to
identify the nearest point and then highlight the line on which the point falls.
It is adapted from the Vega-Lite example found at
https://bl.ocks.org/amitkaps/fe4238e716db53930b2f1a70d3401701
"""
# category: interactive charts
import altair as alt
from vega_datasets import data

source = data.stocks()

highlight = alt.selection(type='single', on='mouseover',
                          fields=['symbol'], nearest=True)

base = alt.Chart(source).encode(
    x='date:T',
    y='price:Q',
    color='symbol:N'
)

points = base.mark_circle().encode(
    opacity=alt.value(0)
).add_selection(
    highlight
).properties(
    width=600
)

lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3))
)

points + lines
