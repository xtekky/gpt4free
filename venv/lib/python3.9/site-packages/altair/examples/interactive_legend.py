"""
Interactive Legend
------------------
The following shows how to create a chart with an interactive legend, by
binding the selection to ``"legend"``. Such a binding only works with
``selection_single`` or ``selection_multi`` when projected over a single
field or encoding.
"""
# category: interactive charts
import altair as alt
from vega_datasets import data

source = data.unemployment_across_industries.url

selection = alt.selection_multi(fields=['series'], bind='legend')

alt.Chart(source).mark_area().encode(
    alt.X('yearmonth(date):T', axis=alt.Axis(domain=False, format='%Y', tickSize=0)),
    alt.Y('sum(count):Q', stack='center', axis=None),
    alt.Color('series:N', scale=alt.Scale(scheme='category20b')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).add_selection(
    selection
)