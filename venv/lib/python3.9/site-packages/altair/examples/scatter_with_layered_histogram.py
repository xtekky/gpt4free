"""
Interactive Scatter Plot and Linked Layered Histogram
=====================================================

This example shows how to link a scatter plot and a histogram
together such that clicking on a point in the scatter plot will
isolate the distribution corresponding to that point, and vice versa.
"""
# category: interactive charts

import altair as alt
import pandas as pd
import numpy as np

# generate fake data
source = pd.DataFrame({'gender': ['M']*1000 + ['F']*1000,
               'height':np.concatenate((np.random.normal(69, 7, 1000),
                                       np.random.normal(64, 6, 1000))),
               'weight': np.concatenate((np.random.normal(195.8, 144, 1000),
                                        np.random.normal(167, 100, 1000))),
               'age': np.concatenate((np.random.normal(45, 8, 1000),
                                        np.random.normal(51, 6, 1000)))
        })

selector = alt.selection_single(empty='all', fields=['gender'])

color_scale = alt.Scale(domain=['M', 'F'],
                        range=['#1FC3AA', '#8624F5'])

base = alt.Chart(source).properties(
    width=250,
    height=250
).add_selection(selector)

points = base.mark_point(filled=True, size=200).encode(
    x=alt.X('mean(height):Q',
            scale=alt.Scale(domain=[0,84])),
    y=alt.Y('mean(weight):Q',
            scale=alt.Scale(domain=[0,250])),
    color=alt.condition(selector,
                        'gender:N',
                        alt.value('lightgray'),
                        scale=color_scale),
)

hists = base.mark_bar(opacity=0.5, thickness=100).encode(
    x=alt.X('age',
            bin=alt.Bin(step=5), # step keeps bin size the same
            scale=alt.Scale(domain=[0,100])),
    y=alt.Y('count()',
            stack=None,
            scale=alt.Scale(domain=[0,350])),
    color=alt.Color('gender:N',
                    scale=color_scale)
).transform_filter(
    selector
)


points | hists
