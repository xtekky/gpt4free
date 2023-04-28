"""
Multifeature Scatter Plot
=========================
This example shows how to make a scatter plot with multiple feature encodings.
"""
# category: scatter plots
import altair as alt
from vega_datasets import data

source = data.iris()

alt.Chart(source).mark_circle().encode(
    alt.X('sepalLength', scale=alt.Scale(zero=False)),
    alt.Y('sepalWidth', scale=alt.Scale(zero=False, padding=1)),
    color='species',
    size='petalWidth'
)
