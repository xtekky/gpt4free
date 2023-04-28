"""
Trellis Scatter Plot
-----------------------
This example shows how to make a trellis scatter plot.
"""
# category: scatter plots
import altair as alt
from vega_datasets import data

source = data.cars()

alt.Chart(source).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    row='Origin:N'
)
