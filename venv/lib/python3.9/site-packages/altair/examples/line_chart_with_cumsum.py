"""
Line Chart with Cumulative Sum
------------------------------
This chart creates a simple line chart from the cumulative sum of a fields.
"""
# category: line charts
import altair as alt
from vega_datasets import data

source = data.wheat()

alt.Chart(source).mark_line().transform_window(
    # Sort the data chronologically
    sort=[{'field': 'year'}],
    # Include all previous records before the current record and none after
    # (This is the default value so you could skip it and it would still work.)
    frame=[None, 0],
    # What to add up as you go
    cumulative_wheat='sum(wheat)'
).encode(
    x='year:O',
    # Plot the calculated field created by the transformation
    y='cumulative_wheat:Q'
).properties(width=600)