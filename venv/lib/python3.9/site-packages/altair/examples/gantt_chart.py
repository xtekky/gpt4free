"""
Gantt Chart
-----------------
This example shows how to make a simple Gantt chart.
"""
# category: other charts
import altair as alt
import pandas as pd

source = pd.DataFrame([
    {"task": "A", "start": 1, "end": 3},
    {"task": "B", "start": 3, "end": 8},
    {"task": "C", "start": 8, "end": 10}
])

alt.Chart(source).mark_bar().encode(
    x='start',
    x2='end',
    y='task'
)
