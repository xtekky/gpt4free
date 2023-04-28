"""
Normalized Parallel Coordinates Example
---------------------------------------
A `Parallel Coordinates <https://en.wikipedia.org/wiki/Parallel_coordinates>`_
chart is a chart that lets you visualize the individual data points by drawing
a single line for each of them.

Such a chart can be created in Altair by first transforming the data into a
suitable representation.

This example shows a modified parallel coordinates chart with the Iris dataset,
where the y-axis shows the value after min-max rather than the raw value. It's a
simplified Altair version of `the VegaLite version <https://vega.github.io/vega-lite/examples/parallel_coordinate.html>`_
"""
# category: other charts
import altair as alt
from vega_datasets import data
from altair import datum

source = data.iris()

alt.Chart(source).transform_window(
    index='count()'
).transform_fold(
    ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']
).transform_joinaggregate(
     min='min(value)', 
     max='max(value)',
     groupby=['key']
).transform_calculate(
    minmax_value=(datum.value-datum.min)/(datum.max-datum.min), 
    mid=(datum.min+datum.max)/2
).mark_line().encode(
    x='key:N',
    y='minmax_value:Q',
    color='species:N',
    detail='index:N',
    opacity=alt.value(0.5)
).properties(width=500)


