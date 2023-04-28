"""
Atmospheric CO2 Concentration
-----------------------------
This example is a fully developed line chart that uses a window transformation.
It was inspired by `Gregor Aisch's work at datawrapper
<https://www.datawrapper.de/_/OHgEm/>`_.
"""
# category: case studies
import altair as alt
from vega_datasets import data

source = data.co2_concentration.url

base = alt.Chart(
    source,
    title="Carbon Dioxide in the Atmosphere"
).transform_calculate(
    year="year(datum.Date)"
).transform_calculate(
    decade="floor(datum.year / 10)"
).transform_calculate(
    scaled_date="(datum.year % 10) + (month(datum.Date)/12)"
).transform_window(
    first_date='first_value(scaled_date)',
    last_date='last_value(scaled_date)',
    sort=[{"field": "scaled_date", "order": "ascending"}],
    groupby=['decade'],
    frame=[None, None]
).transform_calculate(
  end="datum.first_date === datum.scaled_date ? 'first' : datum.last_date === datum.scaled_date ? 'last' : null"
).encode(
    x=alt.X(
        "scaled_date:Q",
        axis=alt.Axis(title="Year into Decade", tickCount=11)
    ),
    y=alt.Y(
        "CO2:Q",
        title="CO2 concentration in ppm",
        scale=alt.Scale(zero=False)
    )
)

line = base.mark_line().encode(
    color=alt.Color(
        "decade:O",
        scale=alt.Scale(scheme="magma"),
        legend=None
    )
)

text = base.encode(text="year:N")

start_year = text.transform_filter(
  alt.datum.end == 'first'
).mark_text(baseline="top")

end_year = text.transform_filter(
  alt.datum.end == 'last'
).mark_text(baseline="bottom")

(line + start_year + end_year).configure_text(
    align="left",
    dx=1,
    dy=3
).properties(width=600, height=375)
