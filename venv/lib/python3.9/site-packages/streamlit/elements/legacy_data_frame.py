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

"""Helper functions to marshall a pandas.DataFrame into a proto.DataFrame."""

import datetime
import re
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, cast

import pyarrow as pa
import tzlocal
from pandas import DataFrame
from pandas.io.formats.style import Styler
from typing_extensions import Final

from streamlit import errors, type_util
from streamlit.elements.arrow import Data
from streamlit.logger import get_logger
from streamlit.proto.DataFrame_pb2 import DataFrame as DataFrameProto
from streamlit.proto.DataFrame_pb2 import TableStyle as TableStyleProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

LOGGER: Final = get_logger(__name__)


class CSSStyle(NamedTuple):
    property: Any
    value: Any


class LegacyDataFrameMixin:
    @gather_metrics("_legacy_dataframe")
    def _legacy_dataframe(
        self,
        data: Data = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "DeltaGenerator":
        """Display a dataframe as an interactive table.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict,
            or None
            The data to display.

            If 'data' is a pandas.Styler, it will be used to style its
            underlying DataFrame. Streamlit supports custom cell
            values and colors. (It does not support some of the more exotic
            pandas styling features, like bar charts, hovering, and captions.)
            Styler support is experimental!
        width : int or None
            Desired width of the UI element expressed in pixels. If None, a
            default width based on the page width is used.
        height : int or None
            Desired height of the UI element expressed in pixels. If None, a
            default height is used.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...    np.random.randn(50, 20),
        ...    columns=('col %d' % i for i in range(20)))
        ...
        >>> st._legacy_dataframe(df)

        .. output::
           https://static.streamlit.io/0.25.0-2JkNY/index.html?id=165mJbzWdAC8Duf8a4tjyQ
           height: 330px

        >>> st._legacy_dataframe(df, 200, 100)

        You can also pass a Pandas Styler object to change the style of
        the rendered DataFrame:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...    np.random.randn(10, 20),
        ...    columns=('col %d' % i for i in range(20)))
        ...
        >>> st._legacy_dataframe(df.style.highlight_max(axis=0))

        .. output::
           https://static.streamlit.io/0.29.0-dV1Y/index.html?id=Hb6UymSNuZDzojUNybzPby
           height: 285px

        """
        data_frame_proto = DataFrameProto()
        marshall_data_frame(data, data_frame_proto)

        return self.dg._enqueue(
            "data_frame",
            data_frame_proto,
            element_width=width,
            element_height=height,
        )

    @gather_metrics("_legacy_table")
    def _legacy_table(self, data: Data = None) -> "DeltaGenerator":
        """Display a static table.

        This differs from `st._legacy_dataframe` in that the table in this case is
        static: its entire contents are laid out directly on the page.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict,
            or None
            The table data.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame(
        ...    np.random.randn(10, 5),
        ...    columns=('col %d' % i for i in range(5)))
        ...
        >>> st._legacy_table(df)

        .. output::
           https://static.streamlit.io/0.25.0-2JkNY/index.html?id=KfZvDMprL4JFKXbpjD3fpq
           height: 480px

        """
        table_proto = DataFrameProto()
        marshall_data_frame(data, table_proto)
        return self.dg._enqueue("table", table_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall_data_frame(data: Data, proto_df: DataFrameProto) -> None:
    """Convert a pandas.DataFrame into a proto.DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame, numpy.ndarray, Iterable, dict, DataFrame, Styler, or None
        Something that is or can be converted to a dataframe.

    proto_df : proto.DataFrame
        Output. The protobuf for a Streamlit DataFrame proto.
    """
    if isinstance(data, pa.Table):
        raise errors.StreamlitAPIException(
            """
pyarrow tables are not supported  by Streamlit's legacy DataFrame serialization (i.e. with `config.dataFrameSerialization = "legacy"`).

To be able to use pyarrow tables, please enable pyarrow by changing the config setting,
`config.dataFrameSerialization = "arrow"`
"""
        )
    df = type_util.convert_anything_to_df(data)

    # Convert df into an iterable of columns (each of type Series).
    df_data = (df.iloc[:, col] for col in range(len(df.columns)))

    _marshall_table(df_data, proto_df.data)
    _marshall_index(df.columns, proto_df.columns)
    _marshall_index(df.index, proto_df.index)

    styler = data if type_util.is_pandas_styler(data) else None
    _marshall_styles(proto_df.style, df, styler)


def _marshall_styles(
    proto_table_style: TableStyleProto, df: DataFrame, styler: Optional[Styler] = None
) -> None:
    """Adds pandas.Styler styling data to a proto.DataFrame

    Parameters
    ----------
    proto_table_style : proto.TableStyle
    df : pandas.DataFrame
    styler : pandas.Styler holding styling data for the data frame, or
        None if there's no style data to marshall
    """

    # NB: we're using protected members of Styler to get this data,
    # which is non-ideal and could break if Styler's interface changes.

    if styler is not None:
        styler._compute()

        # In Pandas 1.3.0, styler._translate() signature was changed.
        # 2 arguments were added: sparse_index and sparse_columns.
        # The functionality that they provide is not yet supported.
        if type_util.is_pandas_version_less_than("1.3.0"):
            translated_style = styler._translate()
        else:
            translated_style = styler._translate(False, False)

        css_styles = _get_css_styles(translated_style)
        display_values = _get_custom_display_values(translated_style)
    else:
        # If we have no Styler, we just make an empty CellStyle for each cell
        css_styles = {}
        display_values = {}

    nrows, ncols = df.shape
    for col in range(ncols):
        proto_col = proto_table_style.cols.add()
        for row in range(nrows):
            proto_cell_style = proto_col.styles.add()

            for css in css_styles.get((row, col), []):
                proto_css = proto_cell_style.css.add()
                proto_css.property = css.property
                proto_css.value = css.value

            display_value = display_values.get((row, col), None)
            if display_value is not None:
                proto_cell_style.display_value = display_value
                proto_cell_style.has_display_value = True


def _get_css_styles(translated_style: Dict[Any, Any]) -> Dict[Any, Any]:
    """Parses pandas.Styler style dictionary into a
    {(row, col): [CSSStyle]} dictionary
    """
    # In pandas < 1.1.0
    # translated_style["cellstyle"] has the following shape:
    # [
    #   {
    #       "props": [["color", " black"], ["background-color", "orange"], ["", ""]],
    #       "selector": "row0_col0"
    #   }
    #   ...
    # ]
    #
    # In pandas >= 1.1.0
    # translated_style["cellstyle"] has the following shape:
    # [
    #   {
    #       "props": [("color", " black"), ("background-color", "orange"), ("", "")],
    #       "selectors": ["row0_col0"]
    #   }
    #   ...
    # ]

    cell_selector_regex = re.compile(r"row(\d+)_col(\d+)")

    css_styles = {}
    for cell_style in translated_style["cellstyle"]:
        if type_util.is_pandas_version_less_than("1.1.0"):
            cell_selectors = [cell_style["selector"]]
        else:
            cell_selectors = cell_style["selectors"]

        for cell_selector in cell_selectors:
            match = cell_selector_regex.match(cell_selector)
            if not match:
                raise RuntimeError(
                    f'Failed to parse cellstyle selector "{cell_selector}"'
                )
            row = int(match.group(1))
            col = int(match.group(2))
            css_declarations = []
            props = cell_style["props"]
            for prop in props:
                if not isinstance(prop, (tuple, list)) or len(prop) != 2:
                    raise RuntimeError(f'Unexpected cellstyle props "{prop}"')
                name = str(prop[0]).strip()
                value = str(prop[1]).strip()
                if name and value:
                    css_declarations.append(CSSStyle(property=name, value=value))
            css_styles[(row, col)] = css_declarations

    return css_styles


def _get_custom_display_values(translated_style: Dict[Any, Any]) -> Dict[Any, Any]:
    """Parses pandas.Styler style dictionary into a
    {(row, col): display_value} dictionary for cells whose display format
    has been customized.
    """
    # Create {(row, col): display_value} from translated_style['body']
    # translated_style['body'] has the shape:
    # [
    #   [ // row
    #     {  // cell or header
    #       'id': 'level0_row0' (for row header) | 'row0_col0' (for cells)
    #       'value': 1.329212
    #       'display_value': '132.92%'
    #       ...
    #     }
    #   ]
    # ]

    def has_custom_display_value(cell: Dict[Any, Any]) -> bool:
        # We'd prefer to only pass `display_value` data to the frontend
        # when a DataFrame cell has been custom-formatted by the user, to
        # save on bandwidth. However:
        #
        # Panda's Styler's internals are private, and it doesn't give us a
        # consistent way of testing whether a cell has a custom display_value
        # or not. Prior to Pandas 1.4, we could test whether a cell's
        # `display_value` differed from its `value`, and only stick the
        # `display_value` in the protobuf when that was the case. In 1.4, an
        # unmodified Styler will contain `display_value` strings for all
        # cells, regardless of whether any formatting has been applied to
        # that cell, so we no longer have this ability.
        #
        # So we're only testing that a cell's `display_value` is not None.
        # In Pandas 1.4, it seems that `display_value` is never None, so this
        # is purely a defense against future Styler changes.
        return cell.get("display_value") is not None

    cell_selector_regex = re.compile(r"row(\d+)_col(\d+)")
    header_selector_regex = re.compile(r"level(\d+)_row(\d+)")

    display_values = {}
    for row in translated_style["body"]:
        # row is a List[Dict], containing format data for each cell in the row,
        # plus an extra first entry for the row header, which we skip
        found_row_header = False
        for cell in row:
            cell_id = cell["id"]  # a string in the form 'row0_col0'
            if header_selector_regex.match(cell_id):
                if not found_row_header:
                    # We don't care about processing row headers, but as
                    # a sanity check, ensure we only see one per row
                    found_row_header = True
                    continue
                else:
                    raise RuntimeError('Found unexpected row header "%s"' % cell)
            match = cell_selector_regex.match(cell_id)
            if not match:
                raise RuntimeError('Failed to parse cell selector "%s"' % cell_id)

            if has_custom_display_value(cell):
                row = int(match.group(1))
                col = int(match.group(2))
                display_values[(row, col)] = str(cell["display_value"])

    return display_values


def _marshall_index(pandas_index, proto_index) -> None:
    """Convert an pandas.Index into a proto.Index.

    pandas_index - Panda.Index or related (input)
    proto_index  - proto.Index (output)
    """
    import numpy as np
    import pandas as pd

    if type(pandas_index) == pd.Index and pandas_index.dtype.kind not in ["f", "i"]:
        _marshall_any_array(np.array(pandas_index), proto_index.plain_index.data)
    elif type(pandas_index) == pd.RangeIndex:
        min = pandas_index.min()
        max = pandas_index.max()
        if pd.isna(min) or pd.isna(max):
            proto_index.range_index.start = 0
            proto_index.range_index.stop = 0
        else:
            proto_index.range_index.start = min
            proto_index.range_index.stop = max + 1
    elif type(pandas_index) == pd.MultiIndex:
        for level in pandas_index.levels:
            _marshall_index(level, proto_index.multi_index.levels.add())
        if hasattr(pandas_index, "codes"):
            index_codes = pandas_index.codes
        else:
            # Deprecated in Pandas 0.24, do don't bother covering.
            index_codes = pandas_index.labels  # pragma: no cover
        for label in index_codes:
            proto_index.multi_index.labels.add().data.extend(label)
    elif type(pandas_index) == pd.DatetimeIndex:
        if pandas_index.tz is None:
            current_zone = tzlocal.get_localzone()
            pandas_index = pandas_index.tz_localize(current_zone)
        proto_index.datetime_index.data.data.extend(
            pandas_index.map(datetime.datetime.isoformat)
        )
    elif type(pandas_index) == pd.TimedeltaIndex:
        proto_index.timedelta_index.data.data.extend(pandas_index.astype(np.int64))
    elif type_util.is_type(pandas_index, "pandas.core.indexes.numeric.Int64Index") or (
        type(pandas_index) == pd.Index and pandas_index.dtype.kind == "i"
    ):
        proto_index.int_64_index.data.data.extend(pandas_index)
    elif type_util.is_type(
        pandas_index, "pandas.core.indexes.numeric.Float64Index"
    ) or (type(pandas_index) == pd.Index and pandas_index.dtype.kind == "f"):
        proto_index.float_64_index.data.data.extend(pandas_index)
    else:
        raise NotImplementedError("Can't handle %s yet." % type(pandas_index))


def _marshall_table(pandas_table, proto_table) -> None:
    """Convert a sequence of 1D arrays into proto.Table.

    pandas_table - Sequence of 1D arrays which are AnyArray compatible (input).
    proto_table  - proto.Table (output)
    """
    for pandas_array in pandas_table:
        if len(pandas_array) == 0:
            continue
        _marshall_any_array(pandas_array, proto_table.cols.add())


def _marshall_any_array(pandas_array, proto_array) -> None:
    """Convert a 1D numpy.Array into a proto.AnyArray.

    pandas_array - 1D arrays which is AnyArray compatible (input).
    proto_array  - proto.AnyArray (output)
    """
    import numpy as np

    # Convert to np.array as necessary.
    if not hasattr(pandas_array, "dtype"):
        pandas_array = np.array(pandas_array)

    # Only works on 1D arrays.
    if len(pandas_array.shape) != 1:
        raise ValueError("Array must be 1D.")

    # Perform type-conversion based on the array dtype.
    if issubclass(pandas_array.dtype.type, np.floating):
        proto_array.doubles.data.extend(pandas_array)
    elif issubclass(pandas_array.dtype.type, np.timedelta64):
        proto_array.timedeltas.data.extend(pandas_array.astype(np.int64))
    elif issubclass(pandas_array.dtype.type, np.integer):
        proto_array.int64s.data.extend(pandas_array)
    elif pandas_array.dtype == np.bool_:
        proto_array.int64s.data.extend(pandas_array)
    elif pandas_array.dtype == np.object_:
        proto_array.strings.data.extend(map(str, pandas_array))
    # dtype='string', <class 'pandas.core.arrays.string_.StringDtype'>
    # NOTE: StringDtype is considered experimental.
    # The implementation and parts of the API may change without warning.
    elif pandas_array.dtype.name == "string":
        proto_array.strings.data.extend(map(str, pandas_array))
    # Setting a timezone changes (dtype, dtype.type) from
    #   'datetime64[ns]', <class 'numpy.datetime64'>
    # to
    #   datetime64[ns, UTC], <class 'pandas._libs.tslibs.timestamps.Timestamp'>
    elif pandas_array.dtype.name.startswith("datetime64"):
        # Just convert straight to ISO 8601, preserving timezone
        # awareness/unawareness. The frontend will render it correctly.
        proto_array.datetimes.data.extend(pandas_array.map(datetime.datetime.isoformat))
    else:
        raise NotImplementedError("Dtype %s not understood." % pandas_array.dtype)
