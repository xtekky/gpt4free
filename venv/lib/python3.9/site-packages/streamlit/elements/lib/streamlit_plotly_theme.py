# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import plotly.graph_objects as go
import plotly.io as pio

# This is the streamlit theme for plotly where we pass in a template.data
# and a template.layout.

# Template.data is for changing specific graph properties in a general aspect
# such as Contour plots or Waterfall plots.

# Template.layout is for changing things such as the x axis and fonts and other
# general layout properties for general graphs.

# We pass in temporary colors to the frontend and the frontend will replace
# those colors because we want to change colors based on the background color.

# Start at #0000001 because developers may be likely to use #000000
CATEGORY_0 = "#000001"
CATEGORY_1 = "#000002"
CATEGORY_2 = "#000003"
CATEGORY_3 = "#000004"
CATEGORY_4 = "#000005"
CATEGORY_5 = "#000006"
CATEGORY_6 = "#000007"
CATEGORY_7 = "#000008"
CATEGORY_8 = "#000009"
CATEGORY_9 = "#000010"

SEQUENTIAL_0 = "#000011"
SEQUENTIAL_1 = "#000012"
SEQUENTIAL_2 = "#000013"
SEQUENTIAL_3 = "#000014"
SEQUENTIAL_4 = "#000015"
SEQUENTIAL_5 = "#000016"
SEQUENTIAL_6 = "#000017"
SEQUENTIAL_7 = "#000018"
SEQUENTIAL_8 = "#000019"
SEQUENTIAL_9 = "#000020"

DIVERGING_0 = "#000021"
DIVERGING_1 = "#000022"
DIVERGING_2 = "#000023"
DIVERGING_3 = "#000024"
DIVERGING_4 = "#000025"
DIVERGING_5 = "#000026"
DIVERGING_6 = "#000027"
DIVERGING_7 = "#000028"
DIVERGING_8 = "#000029"
DIVERGING_9 = "#000030"
DIVERGING_10 = "#000031"

INCREASING = "#000032"
DECREASING = "#000033"
TOTAL = "#000034"

GRAY_30 = "#000035"
GRAY_70 = "#000036"
GRAY_90 = "#000037"
BG_COLOR = "#000038"
FADED_TEXT_05 = "#000039"
BG_MIX = "#000040"


# Plotly represents continuous colorscale through an array of pairs.
# The pair's first index is the starting point and the next pair's first index is the end point.
# The pair's second index is the starting color and the next pair's second index is the end color.
# For more information, please refer to https://plotly.com/python/colorscales/

streamlit_colorscale = [
    [0.0, SEQUENTIAL_0],
    [0.1111111111111111, SEQUENTIAL_1],
    [0.2222222222222222, SEQUENTIAL_2],
    [0.3333333333333333, SEQUENTIAL_3],
    [0.4444444444444444, SEQUENTIAL_4],
    [0.5555555555555556, SEQUENTIAL_5],
    [0.6666666666666666, SEQUENTIAL_6],
    [0.7777777777777778, SEQUENTIAL_7],
    [0.8888888888888888, SEQUENTIAL_8],
    [1.0, SEQUENTIAL_9],
]

pio.templates["streamlit"] = go.layout.Template(
    data=go.layout.template.Data(
        candlestick=[
            go.layout.template.data.Candlestick(
                decreasing=go.candlestick.Decreasing(
                    line=go.candlestick.decreasing.Line(color=DECREASING)
                ),
                increasing=go.candlestick.Increasing(
                    line=go.candlestick.increasing.Line(color=INCREASING)
                ),
            )
        ],
        contour=[go.layout.template.data.Contour(colorscale=streamlit_colorscale)],
        contourcarpet=[
            go.layout.template.data.Contourcarpet(colorscale=streamlit_colorscale)
        ],
        heatmap=[go.layout.template.data.Heatmap(colorscale=streamlit_colorscale)],
        histogram2d=[
            go.layout.template.data.Histogram2d(colorscale=streamlit_colorscale)
        ],
        icicle=[
            go.layout.template.data.Icicle(textfont=go.icicle.Textfont(color="white"))
        ],
        sankey=[
            go.layout.template.data.Sankey(textfont=go.sankey.Textfont(color=GRAY_70))
        ],
        scatter=[
            go.layout.template.data.Scatter(
                marker=go.scatter.Marker(line=go.scatter.marker.Line(width=0))
            )
        ],
        table=[
            go.layout.template.data.Table(
                cells=go.table.Cells(
                    fill=go.table.cells.Fill(color=BG_COLOR),
                    font=go.table.cells.Font(color=GRAY_90),
                    line=go.table.cells.Line(color=FADED_TEXT_05),
                ),
                header=go.table.Header(
                    font=go.table.header.Font(color=GRAY_70),
                    line=go.table.header.Line(color=FADED_TEXT_05),
                    fill=go.table.header.Fill(color=BG_MIX),
                ),
            )
        ],
        waterfall=[
            go.layout.template.data.Waterfall(
                increasing=go.waterfall.Increasing(
                    marker=go.waterfall.increasing.Marker(color=INCREASING)
                ),
                decreasing=go.waterfall.Decreasing(
                    marker=go.waterfall.decreasing.Marker(color=DECREASING)
                ),
                totals=go.waterfall.Totals(
                    marker=go.waterfall.totals.Marker(color=TOTAL)
                ),
                connector=go.waterfall.Connector(
                    line=go.waterfall.connector.Line(color=GRAY_70, width=2)
                ),
            )
        ],
    ),
    layout=go.Layout(
        colorway=[
            CATEGORY_0,
            CATEGORY_1,
            CATEGORY_2,
            CATEGORY_3,
            CATEGORY_4,
            CATEGORY_5,
            CATEGORY_6,
            CATEGORY_7,
            CATEGORY_8,
            CATEGORY_9,
        ],
        colorscale=go.layout.Colorscale(
            sequential=streamlit_colorscale,
            sequentialminus=streamlit_colorscale,
            diverging=[
                [0.0, DIVERGING_0],
                [0.1, DIVERGING_1],
                [0.2, DIVERGING_2],
                [0.3, DIVERGING_3],
                [0.4, DIVERGING_4],
                [0.5, DIVERGING_5],
                [0.6, DIVERGING_6],
                [0.7, DIVERGING_7],
                [0.8, DIVERGING_8],
                [0.9, DIVERGING_9],
                [1.0, DIVERGING_10],
            ],
        ),
        coloraxis=go.layout.Coloraxis(colorscale=streamlit_colorscale),
    ),
)
