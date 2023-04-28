"""
US Population: Wrapped Facet
============================
This chart visualizes the age distribution of the US population over time,
using a wrapped faceting of the data by decade.
"""
# category: case studies
import altair as alt
from vega_datasets import data

source = data.population.url

alt.Chart(source).mark_area().encode(
    x='age:O',
    y=alt.Y(
        'sum(people):Q',
        title='Population',
        axis=alt.Axis(format='~s')
    ),
    facet=alt.Facet('year:O', columns=5),
).properties(
    title='US Age Distribution By Year',
    width=90,
    height=80
)