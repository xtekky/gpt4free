"""
Horizon Graph
-------------
This example shows how to make a Horizon Graph with 2 layers. (See https://idl.cs.washington.edu/papers/horizon/ for more details on Horizon Graphs.)
"""
# category: area charts
import altair as alt
import pandas as pd

source = pd.DataFrame([
    {"x": 1,  "y": 28}, {"x": 2,  "y": 55},
    {"x": 3,  "y": 43}, {"x": 4,  "y": 91},
    {"x": 5,  "y": 81}, {"x": 6,  "y": 53},
    {"x": 7,  "y": 19}, {"x": 8,  "y": 87},
    {"x": 9,  "y": 52}, {"x": 10, "y": 48},
    {"x": 11, "y": 24}, {"x": 12, "y": 49},
    {"x": 13, "y": 87}, {"x": 14, "y": 66},
    {"x": 15, "y": 17}, {"x": 16, "y": 27},
    {"x": 17, "y": 68}, {"x": 18, "y": 16},
    {"x": 19, "y": 49}, {"x": 20, "y": 15}
])

area1 = alt.Chart(source).mark_area(
    clip=True,
    interpolate='monotone'
).encode(
    alt.X('x', scale=alt.Scale(zero=False, nice=False)),
    alt.Y('y', scale=alt.Scale(domain=[0, 50]), title='y'),
    opacity=alt.value(0.6)
).properties(
    width=500,
    height=75
)

area2 = area1.encode(
    alt.Y('ny:Q', scale=alt.Scale(domain=[0, 50]))
).transform_calculate(
    "ny", alt.datum.y - 50
)

area1 + area2
