"""
Pyramid Pie Chart
-----------------
Altair reproduction of http://robslink.com/SAS/democd91/pyramid_pie.htm
"""
import altair as alt
import pandas as pd

category = ['Sky', 'Shady side of a pyramid', 'Sunny side of a pyramid']
color = ["#416D9D", "#674028", "#DEAC58"]
df = pd.DataFrame({'category': category, 'value': [75, 10, 15]})

alt.Chart(df).mark_arc(outerRadius=80).encode(
    alt.Theta('value:Q', scale=alt.Scale(range=[2.356, 8.639])),
    alt.Color('category:N',
        scale=alt.Scale(domain=category, range=color),
        legend=alt.Legend(title=None, orient='none', legendX=160, legendY=50)),
    order='value:Q'
).properties(width=150, height=150).configure_view(strokeOpacity=0)