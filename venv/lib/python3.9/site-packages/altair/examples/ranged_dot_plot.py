"""
Ranged Dot Plot
-----------------
This example shows a ranged dot plot that uses 'layer' to convey changing life expectancy for the five most populous countries (between 1955 and 2000).
"""
# category: other charts
import altair as alt
from vega_datasets import data

source = data.countries.url

chart = alt.layer(
    data=source
).transform_filter(
    filter={"field": 'country',
            "oneOf": ["China", "India", "United States", "Indonesia", "Brazil"]}
).transform_filter(
    filter={'field': 'year',
            "oneOf": [1955, 2000]}
)

chart += alt.Chart().mark_line(color='#db646f').encode(
    x='life_expect:Q',
    y='country:N',
    detail='country:N'
)
# Add points for life expectancy in 1955 & 2000
chart += alt.Chart().mark_point(
    size=100,
    opacity=1,
    filled=True
).encode(
    x='life_expect:Q',
    y='country:N',
    color=alt.Color('year:O',
        scale=alt.Scale(
            domain=['1955', '2000'],
            range=['#e6959c', '#911a24']
        )
    )
).interactive()

chart
