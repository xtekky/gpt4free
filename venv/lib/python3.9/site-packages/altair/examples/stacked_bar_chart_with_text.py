"""
Stacked Bar Chart with Text Overlay
===================================
This example shows how to overlay text on a stacked bar chart. For both the
bar and text marks, we use the ``stack`` argument in the ``x`` encoding to
cause the values to be stacked horizontally.
"""
# category: bar charts
import altair as alt
from vega_datasets import data

source=data.barley()

bars = alt.Chart(source).mark_bar().encode(
    x=alt.X('sum(yield):Q', stack='zero'),
    y=alt.Y('variety:N'),
    color=alt.Color('site')
)

text = alt.Chart(source).mark_text(dx=-15, dy=3, color='white').encode(
    x=alt.X('sum(yield):Q', stack='zero'),
    y=alt.Y('variety:N'),
    detail='site:N',
    text=alt.Text('sum(yield):Q', format='.1f')
)

bars + text
