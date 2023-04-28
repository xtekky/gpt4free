"""
Simple Strip Plot
-----------------
A simple example of how to make a strip plot. 
"""
# category: simple charts
import altair as alt
from vega_datasets import data

source = data.cars()

alt.Chart(source).mark_tick().encode(
    x='Horsepower:Q',
    y='Cylinders:O'
)
