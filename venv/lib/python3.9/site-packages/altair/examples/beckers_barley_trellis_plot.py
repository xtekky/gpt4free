"""
Becker's Barley Trellis Plot
----------------------------
The example demonstrates the trellis charts created by Richard Becker, William Cleveland and others in the 1990s. Using the visualization technique below they identified an anomoly in a widely used agriculatural dataset, which they termed `"The Morris Mistake." <http://ml.stat.purdue.edu/stat695t/writings/Trellis.User.pdf>`_. It became their favored way of showcasing the power of this pioneering plot.
"""
# category: case studies
import altair as alt
from vega_datasets import data

source = data.barley()

alt.Chart(source, title="The Morris Mistake").mark_point().encode(
    alt.X(
        'yield:Q',
        title="Barley Yield (bushels/acre)",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(grid=False)
    ),
    alt.Y(
        'variety:N',
        title="",
        sort='-x',
        axis=alt.Axis(grid=True)
    ),
    color=alt.Color('year:N', legend=alt.Legend(title="Year")),
    row=alt.Row(
        'site:N',
        title="",
        sort=alt.EncodingSortField(field='yield', op='sum', order='descending'),
    )
).properties(
    height=alt.Step(20)
).configure_view(stroke="transparent")
