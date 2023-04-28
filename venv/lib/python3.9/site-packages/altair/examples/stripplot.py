"""
Stripplot
---------
This example shows how to make a Stripplot.
"""
# category: scatter plots
import altair as alt
from vega_datasets import data

source = data.movies.url

stripplot =  alt.Chart(source, width=40).mark_circle(size=8).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('IMDB_Rating:Q'),
    color=alt.Color('Major_Genre:N', legend=None),
    column=alt.Column(
        'Major_Genre:N',
        header=alt.Header(
            labelAngle=-90,
            titleOrient='top',
            labelOrient='bottom',
            labelAlign='right',
            labelPadding=3,
        ),
    ),
).transform_calculate(
    # Generate Gaussian jitter with a Box-Muller transform
    jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
).configure_facet(
    spacing=0
).configure_view(
    stroke=None
)

stripplot
