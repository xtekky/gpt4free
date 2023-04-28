"""
Parallel Coordinates Example
----------------------------
A `Parallel Coordinates <https://en.wikipedia.org/wiki/Parallel_coordinates>`_
chart is a chart that lets you visualize the individual data points by drawing
a single line for each of them.
Such a chart can be created in Altair by first transforming the data into a
suitable representation.
This example shows a parallel coordinates chart with the Iris dataset.
"""
# category: other charts

import altair as alt
from vega_datasets import data

source = data.iris()

alt.Chart(source).transform_window(
    index='count()'  
).transform_fold(
    ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']
).mark_line().encode(
    x='key:N',
    y='value:Q',
    color='species:N',
    detail='index:N',
    opacity=alt.value(0.5)
).properties(width=500)
