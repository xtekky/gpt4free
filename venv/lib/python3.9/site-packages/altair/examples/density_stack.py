"""
Stacked Density Estimates
-------------------------
To plot a stacked graph of estimates, use a shared ``extent`` and a fixed
number of subdivision ``steps`` to ensure that the points for each area align
well.  Density estimates of measurements for each iris flower feature are plot
in a stacked method.  In addition, setting ``counts`` to true multiplies the
densities by the number of data points in each group, preserving proportional
differences.
"""
# category: area charts

import altair as alt
from vega_datasets import data

source = data.iris()

alt.Chart(source).transform_fold(
    ['petalWidth', 
     'petalLength', 
     'sepalWidth', 
     'sepalLength'], 
    as_ = ['Measurement_type', 'value']
).transform_density(
    density='value', 
    bandwidth=0.3, 
    groupby=['Measurement_type'], 
    extent= [0, 8], 
    counts = True, 
    steps=200
).mark_area().encode(
    alt.X('value:Q'), 
    alt.Y('density:Q', stack='zero'),
    alt.Color('Measurement_type:N')
).properties(width=400, height=100)
