"""
Top K Letters
-------------
This example shows how to use a window transform in order to display only the
top K categories by number of entries. In this case, we rank the characters in
the first paragraph of Dickens' *A Tale of Two Cities* by number of occurances.
"""
# category: case studies
import altair as alt
import pandas as pd
import numpy as np

# Excerpt from A Tale of Two Cities; public domain text
text = """
It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of
incredulity, it was the season of Light, it was the season of Darkness, it was
the spring of hope, it was the winter of despair, we had everything before us,
we had nothing before us, we were all going direct to Heaven, we were all going
direct the other way - in short, the period was so far like the present period,
that some of its noisiest authorities insisted on its being received, for good
or for evil, in the superlative degree of comparison only.
"""

source = pd.DataFrame(
    {'letters': np.array([c for c in text if c.isalpha()])}
)

alt.Chart(source).transform_aggregate(
    count='count()',
    groupby=['letters']
).transform_window(
    rank='rank(count)',
    sort=[alt.SortField('count', order='descending')]
).transform_filter(
    alt.datum.rank < 10
).mark_bar().encode(
    y=alt.Y('letters:N', sort='-x'),
    x='count:Q',
)
