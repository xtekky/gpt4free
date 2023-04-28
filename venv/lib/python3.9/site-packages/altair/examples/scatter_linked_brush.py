"""
Multi-panel Scatter Plot with Linked Brushing
---------------------------------------------
This is an example of using an interval selection to control the color of
points across multiple panels.
"""
# category: interactive charts
import altair as alt
from vega_datasets import data

source = data.cars()

brush = alt.selection(type='interval', resolve='global')

base = alt.Chart(source).mark_point().encode(
    y='Miles_per_Gallon',
    color=alt.condition(brush, 'Origin', alt.ColorValue('gray')),
).add_selection(
    brush
).properties(
    width=250,
    height=250
)

base.encode(x='Horsepower') | base.encode(x='Acceleration')
