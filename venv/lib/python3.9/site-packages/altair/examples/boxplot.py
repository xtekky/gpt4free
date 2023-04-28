"""
Boxplot with Min/Max Whiskers
------------------------------
This example shows how to make a boxplot using US Population data from 2000.  
Note that the default value of the `extent` property is 1.5,
which represents the convention of extending the whiskers
to the furthest points within 1.5 * IQR from the first and third quartile.
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.population.url

alt.Chart(source).mark_boxplot(extent='min-max').encode(
    x='age:O',
    y='people:Q'
)
