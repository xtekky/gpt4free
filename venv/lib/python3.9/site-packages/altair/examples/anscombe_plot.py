"""
Anscombe's Quartet
------------------

This example shows how to use the column channel to make a trellis plot. Anscombe's Quartet is a famous dataset constructed by Francis Anscombe. Common summary statistics are identical for each subset of the data, despite the subsets having vastly different characteristics.
"""
# category: case studies
import altair as alt
from vega_datasets import data

source = data.anscombe()

alt.Chart(source).mark_circle().encode(
    alt.X('X', scale=alt.Scale(zero=False)),
    alt.Y('Y', scale=alt.Scale(zero=False)),
    alt.Facet('Series', columns=2),
).properties(
    width=180,
    height=180,
)
