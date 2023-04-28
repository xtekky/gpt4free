"""
Comet Chart Example
----------------------------
Inspired by `Zan Armstrong's comet chart <https://www.zanarmstrong.com/infovisresearch>`_
this plot uses ``mark_trail`` to visualize change of grouped data over time.
A more elaborate example and explanation of creating comet charts in Altair
is shown in `this blogpost <https://medium.com/de-dataverbinders/comet-charts-in-python-visualizing-statistical-mix-effects-and-simpsons-paradox-with-altair-6cd51fb58b7c>`_.
"""
# category: other charts

import altair as alt
import vega_datasets

(
    alt.Chart(vega_datasets.data.barley.url)
    .transform_pivot("year", value="yield", groupby=["variety", "site"])
    .transform_fold(["1931", "1932"], as_=["year", "yield"])
    .transform_calculate(calculate="datum['1932'] - datum['1931']", as_="delta")
    .mark_trail()
    .encode(
        x=alt.X('year:O', title=None), 
        y=alt.Y('variety:N', title='Variety'),
        size=alt.Size('yield:Q', scale=alt.Scale(range=[0, 12]), legend=alt.Legend(values=[20, 60], title='Barley Yield (bushels/acre)')),
        color=alt.Color('delta:Q', scale=alt.Scale(domainMid=0), legend=alt.Legend(title='Yield Delta (%)')),
        tooltip=alt.Tooltip(['year:O', 'yield:Q']),
        column=alt.Column('site:N', title='Site')

    )
    .configure_view(stroke=None)
    .configure_legend(orient='bottom', direction='horizontal')
    .properties(title='Barley Yield comparison between 1932 and 1931')
)
