"""
Line Chart with Sequence Generator
----------------------------------
This examples shows how to create multiple lines using the sequence generator.
"""
# category: line charts

import altair as alt

source = alt.sequence(start=0, stop=12.7, step=0.1, as_='x')

alt.Chart(source).mark_line().transform_calculate(
    sin='sin(datum.x)',
    cos='cos(datum.x)'
).transform_fold(
    ['sin', 'cos']
).encode(
    x='x:Q', 
    y='value:Q', 
    color='key:N'
)
