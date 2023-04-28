"""
Compact Trellis Grid of Bar Charts
==================================
This example shows a simple grid of bar charts to compare performance data..
"""
# category: bar charts
import altair as alt
import pandas as pd

source = pd.DataFrame(
    [
        {"a": "a1", "b": "b1", "c": "x", "p": "0.14"},
        {"a": "a1", "b": "b1", "c": "y", "p": "0.60"},
        {"a": "a1", "b": "b1", "c": "z", "p": "0.03"},
        {"a": "a1", "b": "b2", "c": "x", "p": "0.80"},
        {"a": "a1", "b": "b2", "c": "y", "p": "0.38"},
        {"a": "a1", "b": "b2", "c": "z", "p": "0.55"},
        {"a": "a1", "b": "b3", "c": "x", "p": "0.11"},
        {"a": "a1", "b": "b3", "c": "y", "p": "0.58"},
        {"a": "a1", "b": "b3", "c": "z", "p": "0.79"},
        {"a": "a2", "b": "b1", "c": "x", "p": "0.83"},
        {"a": "a2", "b": "b1", "c": "y", "p": "0.87"},
        {"a": "a2", "b": "b1", "c": "z", "p": "0.67"},
        {"a": "a2", "b": "b2", "c": "x", "p": "0.97"},
        {"a": "a2", "b": "b2", "c": "y", "p": "0.84"},
        {"a": "a2", "b": "b2", "c": "z", "p": "0.90"},
        {"a": "a2", "b": "b3", "c": "x", "p": "0.74"},
        {"a": "a2", "b": "b3", "c": "y", "p": "0.64"},
        {"a": "a2", "b": "b3", "c": "z", "p": "0.19"},
        {"a": "a3", "b": "b1", "c": "x", "p": "0.57"},
        {"a": "a3", "b": "b1", "c": "y", "p": "0.35"},
        {"a": "a3", "b": "b1", "c": "z", "p": "0.49"},
        {"a": "a3", "b": "b2", "c": "x", "p": "0.91"},
        {"a": "a3", "b": "b2", "c": "y", "p": "0.38"},
        {"a": "a3", "b": "b2", "c": "z", "p": "0.91"},
        {"a": "a3", "b": "b3", "c": "x", "p": "0.99"},
        {"a": "a3", "b": "b3", "c": "y", "p": "0.80"},
        {"a": "a3", "b": "b3", "c": "z", "p": "0.37"},
    ]
)

alt.Chart(source, width=60, height=alt.Step(8)).mark_bar().encode(
    y=alt.Y("c:N", axis=None),
    x=alt.X("p:Q", title=None, axis=alt.Axis(format="%")),
    color=alt.Color(
        "c:N", title="settings", legend=alt.Legend(orient="bottom", titleOrient="left")
    ),
    row=alt.Row("a:N", title="Factor A", header=alt.Header(labelAngle=0)),
    column=alt.Column("b:N", title="Factor B"),
)
