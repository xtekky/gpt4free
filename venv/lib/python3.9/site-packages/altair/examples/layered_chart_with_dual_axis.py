"""
Layered chart with Dual-Axis
----------------------------
This example shows how to create a second independent y axis.
"""
# category: other charts

import altair as alt
from vega_datasets import data

source = data.seattle_weather()

base = alt.Chart(source).encode(
    alt.X('month(date):T', axis=alt.Axis(title=None))
)

area = base.mark_area(opacity=0.3, color='#57A44C').encode(
    alt.Y('average(temp_max)',
          axis=alt.Axis(title='Avg. Temperature (°C)', titleColor='#57A44C')),
    alt.Y2('average(temp_min)')
)

line = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
    alt.Y('average(precipitation)',
          axis=alt.Axis(title='Precipitation (inches)', titleColor='#5276A7'))
)

alt.layer(area, line).resolve_scale(
    y = 'independent'
)
