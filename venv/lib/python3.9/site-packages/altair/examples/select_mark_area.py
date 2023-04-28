"""
Using Selection Interval with mark_area
=========================================

Because area is considered one object, just using the plain 
selector will select the entire area instead of just one part of it.

This example shows how to use two areas, one on top of the other, and a
`transform_filter` to fake out this effect.

"""
# category: interactive charts
import altair as alt
from vega_datasets import data

source = data.unemployment_across_industries.url

base = alt.Chart(source).mark_area(
    color='goldenrod',
    opacity=0.3
).encode(
    x='yearmonth(date):T',
    y='sum(count):Q',
)

brush = alt.selection_interval(encodings=['x'],empty='all')
background = base.add_selection(brush)
selected = base.transform_filter(brush).mark_area(color='goldenrod')

background + selected
