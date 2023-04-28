"""
Scatter Plot with Href
----------------------
This example shows a scatter plot with an ``href`` encoding constructed from
the car name. With this encoding, you can click on any of the points to open
a google search for the car name.
"""
# category: scatter plots

import altair as alt
from vega_datasets import data

source = data.cars()

alt.Chart(source).transform_calculate(
    url='https://www.google.com/search?q=' + alt.datum.Name
).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color='Origin:N',
    href='url:N',
    tooltip=['Name:N', 'url:N']
)
