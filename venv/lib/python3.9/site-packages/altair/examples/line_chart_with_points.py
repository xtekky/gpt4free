"""
Line Chart with Points
----------------------
This chart shows a simple line chart with points marking each value. Use
``point=True`` for points with default appearance or customize it with
``OverlayMarkDef()``.
"""
# category: line charts
import altair as alt
import numpy as np
import pandas as pd

x = np.arange(100)
source = pd.DataFrame({
  'x': x,
  'f(x)': np.sin(x / 5)
})

alt.Chart(source).mark_line(
    point=alt.OverlayMarkDef(color="red")
).encode(
    x='x',
    y='f(x)'
)
