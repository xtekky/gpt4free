"""
Line Chart with Logarithmic Scale
---------------------------------
How to make a line chart on a `Logarithmic scale <https://en.wikipedia.org/wiki/Logarithmic_scale>`_.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.population()

alt.Chart(source).mark_line().encode(
    x='year:O',
    y=alt.Y(
        'sum(people)',
        scale=alt.Scale(type="log")  # Here the scale is applied
    )
)