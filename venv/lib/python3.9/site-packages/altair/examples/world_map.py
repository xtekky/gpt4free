"""
World Map
---------

This example shows how to create a world map using data generators for
different background layers.
"""
# category: maps

import altair as alt
from vega_datasets import data

# Data generators for the background
sphere = alt.sphere()
graticule = alt.graticule()

# Source of land data
source = alt.topo_feature(data.world_110m.url, 'countries')

# Layering and configuring the components 
alt.layer(
    alt.Chart(sphere).mark_geoshape(fill='lightblue'), 
    alt.Chart(graticule).mark_geoshape(stroke='white', strokeWidth=0.5), 
    alt.Chart(source).mark_geoshape(fill='ForestGreen', stroke='black')
).project(
    'naturalEarth1'
).properties(width=600, height=400).configure_view(stroke=None)
