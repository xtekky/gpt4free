"""
Bar and Tick Chart
------------------
How to layer a tick chart on top of a bar chart.
"""
# category: bar charts
import altair as alt
import pandas as pd

source = pd.DataFrame({
    'project': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
    'score': [25, 57, 23, 19, 8, 47, 8],
    'goal': [25, 47, 30, 27, 38, 19, 4]
})

bar = alt.Chart(source).mark_bar().encode(
    x='project',
    y='score'
).properties(
    width=alt.Step(40)  # controls width of bar.
)

tick = alt.Chart(source).mark_tick(
    color='red',
    thickness=2,
    size=40 * 0.9,  # controls width of tick.
).encode(
    x='project',
    y='goal'
)

bar + tick
