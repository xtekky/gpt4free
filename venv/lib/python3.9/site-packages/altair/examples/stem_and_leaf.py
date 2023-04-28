"""
Stem and Leaf Plot
------------------
This example shows how to make a stem and leaf plot.
"""
# category: other charts
import altair as alt
import pandas as pd
import numpy as np
np.random.seed(42)

# Generating random data
source = pd.DataFrame({'samples': np.random.normal(50, 15, 100).astype(int).astype(str)})

# Splitting stem and leaf
source['stem'] = source['samples'].str[:-1]
source['leaf'] = source['samples'].str[-1]

source = source.sort_values(by=['stem', 'leaf'])

# Determining leaf position
source['position'] = source.groupby('stem').cumcount().add(1)

# Creating stem and leaf plot
alt.Chart(source).mark_text(
    align='left',
    baseline='middle',
    dx=-5
).encode(
    alt.X('position:Q', title='',
        axis=alt.Axis(ticks=False, labels=False, grid=False)
    ),
    alt.Y('stem:N', title='', axis=alt.Axis(tickSize=0)),
    text='leaf:N',
).configure_axis(
    labelFontSize=20
).configure_text(
    fontSize=20
)
