"""
Line Chart with datum
---------------------------------
An example of using ``datum`` to highlight certain values, including a ``DateTime`` value.
This is adapted from two corresponding Vega-Lite Examples:
`Highlight a Specific Value <https://vega.github.io/vega-lite/docs/datum.html#highlight-a-specific-data-value>`_.
"""
# category: line charts

import altair as alt
from vega_datasets import data

source = data.stocks()

lines = (
    alt.Chart(source)
    .mark_line()
    .encode(x="date", y="price", color="symbol")
)

xrule = (
    alt.Chart()
    .mark_rule(color="cyan", strokeWidth=2)
    .encode(x=alt.datum(alt.DateTime(year=2006, month="November")))
)

yrule = (
    alt.Chart().mark_rule(strokeDash=[12, 6], size=2).encode(y=alt.datum(350))
)


lines + yrule + xrule
