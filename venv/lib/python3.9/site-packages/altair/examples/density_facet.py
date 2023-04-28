"""
Faceted Density Estimates
-------------------------
Density estimates of measurements for each iris flower feature
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
    extent= [0, 8]
).mark_area().encode(
    alt.X('value:Q'), 
    alt.Y('density:Q'),
    alt.Row('Measurement_type:N')
).properties(width=300, height=50)
