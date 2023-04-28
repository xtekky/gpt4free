"""
Line Chart with Confidence Interval Band
----------------------------------------
How to make a line chart with a bootstrapped 95% confidence interval band.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.cars()

line = alt.Chart(source).mark_line().encode(
    x='Year',
    y='mean(Miles_per_Gallon)'
)

band = alt.Chart(source).mark_errorband(extent='ci').encode(
    x='Year',
    y=alt.Y('Miles_per_Gallon', title='Miles/Gallon'),
)

band + line
