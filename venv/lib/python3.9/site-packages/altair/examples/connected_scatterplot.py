"""
Connected Scatterplot (Lines with Custom Paths)
-----------------------------------------------

This example show how the order encoding can be used to draw a custom path. The dataset tracks miles driven per capita along with gas prices annually from 1956 to 2010.
It is based on Hannah Fairfield's article 'Driving Shifts Into Reverse'. See https://archive.nytimes.com/www.nytimes.com/imagepages/2010/05/02/business/02metrics.html for the original.
"""
# category: scatter plots
import altair as alt
from vega_datasets import data

source = data.driving()

alt.Chart(source).mark_line(point=True).encode(
    alt.X('miles', scale=alt.Scale(zero=False)),
    alt.Y('gas', scale=alt.Scale(zero=False)),
    order='year'
)
