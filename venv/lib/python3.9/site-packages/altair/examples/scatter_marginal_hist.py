"""
Facetted Scatterplot with marginal histograms
---------------------------------------------
This example demonstrates how to generate a facetted scatterplot,
with marginal facetted histograms, and how to share their respective
- x,some y-limits.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.iris()

base = alt.Chart(source)

xscale = alt.Scale(domain=(4.0, 8.0))
yscale = alt.Scale(domain=(1.9, 4.55))

bar_args = {'opacity': .3, 'binSpacing': 0}

points = base.mark_circle().encode(
    alt.X('sepalLength', scale=xscale),
    alt.Y('sepalWidth', scale=yscale),
    color='species',
)

top_hist = base.mark_bar(**bar_args).encode(
    alt.X('sepalLength:Q',
          # when using bins, the axis scale is set through
          # the bin extent, so we do not specify the scale here
          # (which would be ignored anyway)
          bin=alt.Bin(maxbins=20, extent=xscale.domain),
          stack=None,
          title=''
         ),
    alt.Y('count()', stack=None, title=''),
    alt.Color('species:N'),
).properties(height=60)

right_hist = base.mark_bar(**bar_args).encode(
    alt.Y('sepalWidth:Q',
          bin=alt.Bin(maxbins=20, extent=yscale.domain),
          stack=None,
          title='',
         ),
    alt.X('count()', stack=None, title=''),
    alt.Color('species:N'),
).properties(width=60)

top_hist & (points | right_hist)
