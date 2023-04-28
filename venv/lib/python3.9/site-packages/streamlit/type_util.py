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

"""A bunch of useful utilities for dealing with types."""

from __future__ import annotations

import contextlib
import re
import types
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pyarrow as pa
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.api.types import infer_dtype, is_dict_like, is_list_like
from typing_extensions import Final, Literal, Protocol, TypeAlias, TypeGuard, get_args

import streamlit as st
from streamlit import errors
from streamlit import logger as _logger
from streamlit import string_util

if TYPE_CHECKING:
    import graphviz
    import sympy
    from pandas.core.indexing import _iLocIndexer
    from pandas.io.formats.style import Styler
    from plotly.graph_objs import Figure
    from pydeck import Deck


# Maximum number of rows to request from an unevaluated (out-of-core) dataframe
MAX_UNEVALUATED_DF_ROWS = 10000

_LOGGER = _logger.get_logger("root")

# The array value field names are part of the larger set of possible value
# field names. See the explanation for said set below. The message types
# associated with these fields are distinguished by storing data in a `data`
# field in their messages, meaning they need special treatment in certain
# circumstances. Hence, they need their own, dedicated, sub-type.
ArrayValueFieldName: TypeAlias = Literal[
    "double_array_value",
    "int_array_value",
    "string_array_value",
]

# A frozenset containing the allowed values of the ArrayValueFieldName type.
# Useful for membership checking.
ARRAY_VALUE_FIELD_NAMES: Final = frozenset(
    cast(
        "tuple[ArrayValueFieldName, ...]",
        # NOTE: get_args is not recursive, so this only works as long as
        # ArrayValueFieldName remains flat.
        get_args(ArrayValueFieldName),
    )
)

# These are the possible field names that can be set in the `value` oneof-field
# of the WidgetState message (schema found in .proto/WidgetStates.proto).
# We need these as a literal type to ensure correspondence with the protobuf
# schema in certain parts of the python code.
# TODO(harahu): It would be preferable if this type was automatically derived
#  from the protobuf schema, rather than manually maintained. Not sure how to
#  achieve that, though.
ValueFieldName: TypeAlias = Literal[
    ArrayValueFieldName,
    "arrow_value",
    "bool_value",
    "bytes_value",
    "double_value",
    "file_uploader_state_value",
    "int_value",
    "json_value",
    "string_value",
    "trigger_value",
]

V_co = TypeVar(
    "V_co",
    covariant=True,  # https://peps.python.org/pep-0484/#covariance-and-contravariance
)

T = TypeVar("T")


class DataFrameGenericAlias(Protocol[V_co]):
    """Technically not a GenericAlias, but serves the same purpose in
    OptionSequence below, in that it is a type which admits DataFrame,
    but is generic. This allows OptionSequence to be a fully generic type,
    significantly increasing its usefulness.

    We can't use types.GenericAlias, as it is only available from python>=3.9,
    and isn't easily back-ported.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        ...


OptionSequence: TypeAlias = Union[
    Iterable[V_co],
    DataFrameGenericAlias[V_co],
]


Key: TypeAlias = Union[str, int]

LabelVisibility = Literal["visible", "hidden", "collapsed"]


class SupportsStr(Protocol):
    def __str__(self) -> str:
        ...


def is_array_value_field_name(obj: object) -> TypeGuard[ArrayValueFieldName]:
    return obj in ARRAY_VALUE_FIELD_NAMES


@overload
def is_type(
    obj: object, fqn_type_pattern: Literal["pydeck.bindings.deck.Deck"]
) -> TypeGuard[Deck]:
    ...


@overload
def is_type(
    obj: object, fqn_type_pattern: Literal["plotly.graph_objs._figure.Figure"]
) -> TypeGuard[Figure]:
    ...


@overload
def is_type(obj: object, fqn_type_pattern: Union[str, re.Pattern[str]]) -> bool:
    ...


def is_type(obj: object, fqn_type_pattern: Union[str, re.Pattern[str]]) -> bool:
    """Check type without importing expensive modules.

    Parameters
    ----------
    obj : object
        The object to type-check.
    fqn_type_pattern : str or regex
        The fully-qualified type string or a regular expression.
        Regexes should start with `^` and end with `$`.

    Example
    -------

    To check whether something is a Matplotlib Figure without importing
    matplotlib, use:

    >>> is_type(foo, 'matplotlib.figure.Figure')

    """
    fqn_type = get_fqn_type(obj)
    if isinstance(fqn_type_pattern, str):
        return fqn_type_pattern == fqn_type
    else:
        return fqn_type_pattern.match(fqn_type) is not None


def get_fqn(the_type: type) -> str:
    """Get module.type_name for a given type."""
    return f"{the_type.__module__}.{the_type.__qualname__}"


def get_fqn_type(obj: object) -> str:
    """Get module.type_name for a given object."""
    return get_fqn(type(obj))


_PANDAS_DF_TYPE_STR: Final = "pandas.core.frame.DataFrame"
_PANDAS_INDEX_TYPE_STR: Final = "pandas.core.indexes.base.Index"
_PANDAS_SERIES_TYPE_STR: Final = "pandas.core.series.Series"
_PANDAS_STYLER_TYPE_STR: Final = "pandas.io.formats.style.Styler"
_NUMPY_ARRAY_TYPE_STR: Final = "numpy.ndarray"
_SNOWPARK_DF_TYPE_STR: Final = "snowflake.snowpark.dataframe.DataFrame"
_SNOWPARK_DF_ROW_TYPE_STR: Final = "snowflake.snowpark.row.Row"
_SNOWPARK_TABLE_TYPE_STR: Final = "snowflake.snowpark.table.Table"
_PYSPARK_DF_TYPE_STR: Final = "pyspark.sql.dataframe.DataFrame"

_DATAFRAME_LIKE_TYPES: Final[tuple[str, ...]] = (
    _PANDAS_DF_TYPE_STR,
    _PANDAS_INDEX_TYPE_STR,
    _PANDAS_SERIES_TYPE_STR,
    _PANDAS_STYLER_TYPE_STR,
    _NUMPY_ARRAY_TYPE_STR,
)

DataFrameLike: TypeAlias = "Union[DataFrame, Index, Series, Styler]"

_DATAFRAME_COMPATIBLE_TYPES: Final[tuple[type, ...]] = (
    dict,
    list,
    set,
    tuple,
    type(None),
)

_DataFrameCompatible: TypeAlias = Union[dict, list, set, Tuple[Any], None]
DataFrameCompatible: TypeAlias = Union[_DataFrameCompatible, DataFrameLike]

_BYTES_LIKE_TYPES: Final[tuple[type, ...]] = (
    bytes,
    bytearray,
)

BytesLike: TypeAlias = Union[bytes, bytearray]


class DataFormat(Enum):
    """DataFormat is used to determine the format of the data."""

    UNKNOWN = auto()
    EMPTY = auto()  # None
    PANDAS_DATAFRAME = auto()  # pd.DataFrame
    PANDAS_SERIES = auto()  # pd.Series
    PANDAS_INDEX = auto()  # pd.Index
    NUMPY_LIST = auto()  # np.array[Scalar]
    NUMPY_MATRIX = auto()  # np.array[List[Scalar]]
    PYARROW_TABLE = auto()  # pyarrow.Table
    SNOWPARK_OBJECT = auto()  # Snowpark DataFrame, Table, List[Row]
    PYSPARK_OBJECT = auto()  # pyspark.DataFrame
    PANDAS_STYLER = auto()  # pandas Styler
    LIST_OF_RECORDS = auto()  # List[Dict[str, Scalar]]
    LIST_OF_ROWS = auto()  # List[List[Scalar]]
    LIST_OF_VALUES = auto()  # List[Scalar]
    TUPLE_OF_VALUES = auto()  # Tuple[Scalar]
    SET_OF_VALUES = auto()  # Set[Scalar]
    COLUMN_INDEX_MAPPING = auto()  # {column: {index: value}}
    COLUMN_VALUE_MAPPING = auto()  # {column: List[values]}
    COLUMN_SERIES_MAPPING = auto()  # {column: Series(values)}
    KEY_VALUE_DICT = auto()  # {index: value}


def is_dataframe(obj: object) -> TypeGuard[DataFrame]:
    return is_type(obj, _PANDAS_DF_TYPE_STR)


def is_dataframe_like(obj: object) -> TypeGuard[DataFrameLike]:
    return any(is_type(obj, t) for t in _DATAFRAME_LIKE_TYPES)


def is_snowpark_or_pyspark_data_object(obj: object) -> bool:
    """True if if obj is of type snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table or
    True when obj is a list which contains snowflake.snowpark.row.Row or True when obj is of type pyspark.sql.dataframe.DataFrame
    False otherwise.
    """
    return is_snowpark_data_object(obj) or is_pyspark_data_object(obj)


def is_snowpark_data_object(obj: object) -> bool:
    """True if obj is of type snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table or
    True when obj is a list which contains snowflake.snowpark.row.Row,
    False otherwise.
    """
    if is_type(obj, _SNOWPARK_TABLE_TYPE_STR):
        return True
    if is_type(obj, _SNOWPARK_DF_TYPE_STR):
        return True
    if not isinstance(obj, list):
        return False
    if len(obj) < 1:
        return False
    if not hasattr(obj[0], "__class__"):
        return False
    return is_type(obj[0], _SNOWPARK_DF_ROW_TYPE_STR)


def is_pyspark_data_object(obj: object) -> bool:
    """True if obj is of type pyspark.sql.dataframe.DataFrame"""
    return (
        is_type(obj, _PYSPARK_DF_TYPE_STR)
        and hasattr(obj, "toPandas")
        and callable(getattr(obj, "toPandas"))
    )


def is_dataframe_compatible(obj: object) -> TypeGuard[DataFrameCompatible]:
    """True if type that can be passed to convert_anything_to_df."""
    return is_dataframe_like(obj) or type(obj) in _DATAFRAME_COMPATIBLE_TYPES


def is_bytes_like(obj: object) -> TypeGuard[BytesLike]:
    """True if the type is considered bytes-like for the purposes of
    protobuf data marshalling.
    """
    return isinstance(obj, _BYTES_LIKE_TYPES)


def to_bytes(obj: BytesLike) -> bytes:
    """Converts the given object to bytes.

    Only types for which `is_bytes_like` is true can be converted; anything
    else will result in an exception.
    """
    if isinstance(obj, bytearray):
        return bytes(obj)
    elif isinstance(obj, bytes):
        return obj

    raise RuntimeError(f"{obj} is not convertible to bytes")


_SYMPY_RE: Final = re.compile(r"^sympy.*$")


def is_sympy_expession(obj: object) -> TypeGuard[sympy.Expr]:
    """True if input is a SymPy expression."""
    if not is_type(obj, _SYMPY_RE):
        return False

    try:
        import sympy

        return isinstance(obj, sympy.Expr)
    except ImportError:
        return False


_ALTAIR_RE: Final = re.compile(r"^altair\.vegalite\.v\d+\.api\.\w*Chart$")


def is_altair_chart(obj: object) -> bool:
    """True if input looks like an Altair chart."""
    return is_type(obj, _ALTAIR_RE)


def is_keras_model(obj: object) -> bool:
    """True if input looks like a Keras model."""
    return (
        is_type(obj, "keras.engine.sequential.Sequential")
        or is_type(obj, "keras.engine.training.Model")
        or is_type(obj, "tensorflow.python.keras.engine.sequential.Sequential")
        or is_type(obj, "tensorflow.python.keras.engine.training.Model")
    )


def is_list_of_scalars(data: Iterable[Any]) -> bool:
    """Check if the list only contains scalar values."""
    # Overview on all value that are interpreted as scalar:
    # https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_scalar.html
    return infer_dtype(data, skipna=True) not in ["mixed", "unknown-array"]


def is_plotly_chart(obj: object) -> TypeGuard[Union[Figure, list[Any], dict[str, Any]]]:
    """True if input looks like a Plotly chart."""
    return (
        is_type(obj, "plotly.graph_objs._figure.Figure")
        or _is_list_of_plotly_objs(obj)
        or _is_probably_plotly_dict(obj)
    )


def is_graphviz_chart(
    obj: object,
) -> TypeGuard[Union[graphviz.Graph, graphviz.Digraph]]:
    """True if input looks like a GraphViz chart."""
    return (
        # GraphViz < 0.18
        is_type(obj, "graphviz.dot.Graph")
        or is_type(obj, "graphviz.dot.Digraph")
        # GraphViz >= 0.18
        or is_type(obj, "graphviz.graphs.Graph")
        or is_type(obj, "graphviz.graphs.Digraph")
    )


def _is_plotly_obj(obj: object) -> bool:
    """True if input if from a type that lives in plotly.plotly_objs."""
    the_type = type(obj)
    return the_type.__module__.startswith("plotly.graph_objs")


def _is_list_of_plotly_objs(obj: object) -> TypeGuard[list[Any]]:
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    return all(_is_plotly_obj(item) for item in obj)


def _is_probably_plotly_dict(obj: object) -> TypeGuard[dict[str, Any]]:
    if not isinstance(obj, dict):
        return False

    if len(obj.keys()) == 0:
        return False

    if any(k not in ["config", "data", "frames", "layout"] for k in obj.keys()):
        return False

    if any(_is_plotly_obj(v) for v in obj.values()):
        return True

    if any(_is_list_of_plotly_objs(v) for v in obj.values()):
        return True

    return False


def is_function(x: object) -> TypeGuard[types.FunctionType]:
    """Return True if x is a function."""
    return isinstance(x, types.FunctionType)


def is_namedtuple(x: object) -> TypeGuard[NamedTuple]:
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n).__name__ == "str" for n in f)


def is_pandas_styler(obj: object) -> TypeGuard[Styler]:
    return is_type(obj, _PANDAS_STYLER_TYPE_STR)


def is_pydeck(obj: object) -> TypeGuard[Deck]:
    """True if input looks like a pydeck chart."""
    return is_type(obj, "pydeck.bindings.deck.Deck")


def is_iterable(obj: object) -> TypeGuard[Iterable[Any]]:
    try:
        # The ignore statement here is intentional, as this is a
        # perfectly fine way of checking for iterables.
        iter(obj)  # type: ignore[call-overload]
    except TypeError:
        return False
    return True


def is_sequence(seq: Any) -> bool:
    """True if input looks like a sequence."""
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


def convert_anything_to_df(
    data: Any,
    max_unevaluated_rows: int = MAX_UNEVALUATED_DF_ROWS,
    ensure_copy: bool = False,
) -> DataFrame:
    """Try to convert different formats to a Pandas Dataframe.

    Parameters
    ----------
    data : ndarray, Iterable, dict, DataFrame, Styler, pa.Table, None, dict, list, or any

    max_unevaluated_rows: int
        If unevaluated data is detected this func will evaluate it,
        taking max_unevaluated_rows, defaults to 10k and 100 for st.table

    ensure_copy: bool
        If True, make sure to always return a copy of the data. If False, it depends on the
        type of the data. For example, a Pandas DataFrame will be returned as-is.

    Returns
    -------
    pandas.DataFrame

    """
    # This is inefficient as the data will be converted back to Arrow
    # when marshalled to protobuf, but area/bar/line charts need
    # DataFrame magic to generate the correct output.
    if isinstance(data, pa.Table):
        return data.to_pandas()

    if is_type(data, _PANDAS_DF_TYPE_STR):
        return data.copy() if ensure_copy else data

    if is_pandas_styler(data):
        return data.data.copy() if ensure_copy else data.data

    if is_type(data, "numpy.ndarray"):
        if len(data.shape) == 0:
            return DataFrame([])
        return DataFrame(data)

    if (
        is_type(data, _SNOWPARK_DF_TYPE_STR)
        or is_type(data, _SNOWPARK_TABLE_TYPE_STR)
        or is_type(data, _PYSPARK_DF_TYPE_STR)
    ):
        if is_type(data, _PYSPARK_DF_TYPE_STR):
            data = data.limit(max_unevaluated_rows).toPandas()
        else:
            data = DataFrame(data.take(max_unevaluated_rows))
        if data.shape[0] == max_unevaluated_rows:
            st.caption(
                f"⚠️ Showing only {string_util.simplify_number(max_unevaluated_rows)} rows. "
                "Call `collect()` on the dataframe to show more."
            )
        return data

    # Try to convert to pandas.DataFrame. This will raise an error is df is not
    # compatible with the pandas.DataFrame constructor.
    try:

        return DataFrame(data)

    except ValueError as ex:
        if isinstance(data, dict):
            with contextlib.suppress(ValueError):
                # Try to use index orient as back-up to support key-value dicts
                return DataFrame.from_dict(data, orient="index")
        raise errors.StreamlitAPIException(
            f"""
Unable to convert object of type `{type(data)}` to `pandas.DataFrame`.
Offending object:
```py
{data}
```"""
        ) from ex


@overload
def ensure_iterable(obj: Iterable[V_co]) -> Iterable[V_co]:
    ...


@overload
def ensure_iterable(obj: DataFrame) -> Iterable[Any]:
    ...


def ensure_iterable(obj: Union[DataFrame, Iterable[V_co]]) -> Iterable[Any]:
    """Try to convert different formats to something iterable. Most inputs
    are assumed to be iterable, but if we have a DataFrame, we can just
    select the first column to iterate over. If the input is not iterable,
    a TypeError is raised.

    Parameters
    ----------
    obj : list, tuple, numpy.ndarray, pandas.Series, pandas.DataFrame, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame or snowflake.snowpark.table.Table

    Returns
    -------
    iterable

    """
    if is_snowpark_or_pyspark_data_object(obj):
        obj = convert_anything_to_df(obj)

    if is_dataframe(obj):
        # Return first column as a pd.Series
        # The type of the elements in this column is not known up front, hence
        # the Iterable[Any] return type.
        return cast(Iterable[Any], obj.iloc[:, 0])

    if is_iterable(obj):
        return obj

    raise TypeError(
        f"Object is not an iterable and could not be converted to one. Object: {obj}"
    )


def ensure_indexable(obj: OptionSequence[V_co]) -> Sequence[V_co]:
    """Try to ensure a value is an indexable Sequence. If the collection already
    is one, it has the index method that we need. Otherwise, convert it to a list.
    """
    it = ensure_iterable(obj)
    # This is an imperfect check because there is no guarantee that an `index`
    # function actually does the thing we want.
    index_fn = getattr(it, "index", None)
    if callable(index_fn):
        return it  # type: ignore[return-value]
    else:
        return list(it)


def is_pandas_version_less_than(v: str) -> bool:
    """Return True if the current Pandas version is less than the input version.

    Parameters
    ----------
    v : str
        Version string, e.g. "0.25.0"

    Returns
    -------
    bool

    """
    import pandas as pd
    from packaging import version

    return version.parse(pd.__version__) < version.parse(v)


def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    """Serialize pyarrow.Table to bytes using Apache Arrow.

    Parameters
    ----------
    table : pyarrow.Table
        A table to convert.

    """
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return cast(bytes, sink.getvalue().to_pybytes())


def is_colum_type_arrow_incompatible(column: Union[Series, Index]) -> bool:
    """Return True if the column type is known to cause issues during Arrow conversion."""
    if column.dtype.kind in [
        # timedelta is supported by pyarrow but not in the Arrow JS:
        # https://github.com/streamlit/streamlit/issues/4489
        "m",  # timedelta64[ns]
        "c",  # complex64, complex128, complex256
    ]:
        return True

    if column.dtype == "object":
        # The dtype of mixed type columns is always object, the actual type of the column
        # values can be determined via the infer_dtype function:
        # https://pandas.pydata.org/docs/reference/api/pandas.api.types.infer_dtype.html
        inferred_type = infer_dtype(column, skipna=True)

        if inferred_type in [
            "mixed-integer",
            "complex",
            "timedelta",
            "timedelta64",
        ]:
            return True
        elif inferred_type == "mixed":
            # This includes most of the more complex/custom types (objects, dicts, lists, ...)
            if len(column) == 0 or not hasattr(column, "iloc"):
                # The column seems to be invalid, so we assume it is incompatible.
                # But this would most likely never happen since empty columns
                # cannot be mixed.
                return True

            # Get the first value to check if it is a supported list-like type.
            first_value = column.iloc[0]

            if (
                not is_list_like(first_value)
                # dicts are list-like, but have issues in Arrow JS (see comments in Quiver.ts)
                or is_dict_like(first_value)
                # Frozensets are list-like, but are not compatible with pyarrow.
                or isinstance(first_value, frozenset)
            ):
                # This seems to be an incompatible list-like type
                return True
            return False
    # We did not detect an incompatible type, so we assume it is compatible:
    return False


def fix_arrow_incompatible_column_types(
    df: DataFrame, selected_columns: Optional[List[str]] = None
) -> DataFrame:
    """Fix column types that are not supported by Arrow table.

    This includes mixed types (e.g. mix of integers and strings)
    as well as complex numbers (complex128 type). These types will cause
    errors during conversion of the dataframe to an Arrow table.
    It is fixed by converting all values of the column to strings
    This is sufficient for displaying the data on the frontend.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe to fix.

    selected_columns: Optional[List[str]]
        A list of columns to fix. If None, all columns are evaluated.

    Returns
    -------
    The fixed dataframe.
    """
    # Make a copy, but only initialize if necessary to preserve memory.
    df_copy = None
    for col in selected_columns or df.columns:
        if is_colum_type_arrow_incompatible(df[col]):
            if df_copy is None:
                df_copy = df.copy()
            df_copy[col] = df[col].astype(str)

    # The index can also contain mixed types
    # causing Arrow issues during conversion.
    # Skipping multi-indices since they won't return
    # the correct value from infer_dtype
    if not selected_columns and (
        not isinstance(
            df.index,
            MultiIndex,
        )
        and is_colum_type_arrow_incompatible(df.index)
    ):
        if df_copy is None:
            df_copy = df.copy()
        df_copy.index = df.index.astype(str)
    return df_copy if df_copy is not None else df


def data_frame_to_bytes(df: DataFrame) -> bytes:
    """Serialize pandas.DataFrame to bytes using Apache Arrow.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe to convert.

    """
    try:
        table = pa.Table.from_pandas(df)
    except (pa.ArrowTypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as ex:
        _LOGGER.info(
            "Serialization of dataframe to Arrow table was unsuccessful due to: %s. "
            "Applying automatic fixes for column types to make the dataframe Arrow-compatible.",
            ex,
        )
        df = fix_arrow_incompatible_column_types(df)
        table = pa.Table.from_pandas(df)
    return pyarrow_table_to_bytes(table)


def bytes_to_data_frame(source: bytes) -> DataFrame:
    """Convert bytes to pandas.DataFrame.

    Parameters
    ----------
    source : bytes
        A bytes object to convert.

    """
    reader = pa.RecordBatchStreamReader(source)
    return reader.read_pandas()


def determine_data_format(input_data: Any) -> DataFormat:
    """Determine the data format of the input data.

    Parameters
    ----------
    input_data : Any
        The input data to determine the data format of.

    Returns
    -------
    DataFormat
        The data format of the input data.
    """
    if input_data is None:
        return DataFormat.EMPTY
    elif isinstance(input_data, DataFrame):
        return DataFormat.PANDAS_DATAFRAME
    elif isinstance(input_data, np.ndarray):
        if len(input_data.shape) == 1:
            # For technical reasons, we need to distinguish one
            # one-dimensional numpy array from multidimensional ones.
            return DataFormat.NUMPY_LIST
        return DataFormat.NUMPY_MATRIX
    elif isinstance(input_data, pa.Table):
        return DataFormat.PYARROW_TABLE
    elif isinstance(input_data, Series):
        return DataFormat.PANDAS_SERIES
    elif isinstance(input_data, Index):
        return DataFormat.PANDAS_INDEX
    elif is_pandas_styler(input_data):
        return DataFormat.PANDAS_STYLER
    elif is_snowpark_data_object(input_data):
        return DataFormat.SNOWPARK_OBJECT
    elif is_pyspark_data_object(input_data):
        return DataFormat.PYSPARK_OBJECT
    elif isinstance(input_data, (list, tuple, set)):
        if is_list_of_scalars(input_data):
            # -> one-dimensional data structure
            if isinstance(input_data, tuple):
                return DataFormat.TUPLE_OF_VALUES
            if isinstance(input_data, set):
                return DataFormat.SET_OF_VALUES
            return DataFormat.LIST_OF_VALUES
        else:
            # -> Multi-dimensional data structure
            # This should always contain at least one element,
            # otherwise the values type from infer_dtype would have been empty
            first_element = next(iter(input_data))
            if isinstance(first_element, dict):
                return DataFormat.LIST_OF_RECORDS
            if isinstance(first_element, (list, tuple, set)):
                return DataFormat.LIST_OF_ROWS
    elif isinstance(input_data, dict):
        if not input_data:
            return DataFormat.KEY_VALUE_DICT
        if len(input_data) > 0:
            first_value = next(iter(input_data.values()))
            if isinstance(first_value, dict):
                return DataFormat.COLUMN_INDEX_MAPPING
            if isinstance(first_value, (list, tuple)):
                return DataFormat.COLUMN_VALUE_MAPPING
            if isinstance(first_value, Series):
                return DataFormat.COLUMN_SERIES_MAPPING
            # In the future, we could potentially also support the tight & split formats here
            if is_list_of_scalars(input_data.values()):
                # Only use the key-value dict format if the values are only scalar values
                return DataFormat.KEY_VALUE_DICT
    return DataFormat.UNKNOWN


def convert_df_to_data_format(
    df: DataFrame, data_format: DataFormat
) -> Union[
    DataFrame,
    Series,
    Index,
    Styler,
    pa.Table,
    np.ndarray[Any, np.dtype[Any]],
    Tuple[Any],
    List[Any],
    Set[Any],
    Dict[str, Any],
]:
    """Convert a dataframe to the specified data format.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to convert.

    data_format : DataFormat
        The data format to convert to.

    Returns
    -------
    pd.DataFrame, pd.Index, Styler, pa.Table, np.ndarray, tuple, list, set, dict
        The converted dataframe.
    """
    if data_format in [
        DataFormat.EMPTY,
        DataFormat.PANDAS_DATAFRAME,
        DataFormat.SNOWPARK_OBJECT,
        DataFormat.PYSPARK_OBJECT,
        DataFormat.PANDAS_INDEX,
        DataFormat.PANDAS_STYLER,
    ]:
        return df
    elif data_format == DataFormat.NUMPY_LIST:
        # It's a 1-dimensional array, so we only return
        # the first column as numpy array
        # Calling to_numpy() on the full DataFrame would result in:
        # [[1], [2]] instead of [1, 2]
        return np.ndarray(0) if df.empty else df.iloc[:, 0].to_numpy()
    elif data_format == DataFormat.NUMPY_MATRIX:
        return np.ndarray(0) if df.empty else df.to_numpy()
    elif data_format == DataFormat.PYARROW_TABLE:
        return pa.Table.from_pandas(df)
    elif data_format == DataFormat.PANDAS_SERIES:
        # Select first column in dataframe and create a new series based on the values
        if len(df.columns) != 1:
            raise ValueError(
                f"DataFrame is expected to have a single column but has {len(df.columns)}."
            )
        return df[df.columns[0]]
    elif data_format == DataFormat.LIST_OF_RECORDS:
        return df.to_dict(orient="records")
    elif data_format == DataFormat.LIST_OF_ROWS:
        # to_numpy converts the dataframe to a list of rows
        return df.to_numpy().tolist()
    elif data_format == DataFormat.COLUMN_INDEX_MAPPING:
        return df.to_dict(orient="dict")
    elif data_format == DataFormat.COLUMN_VALUE_MAPPING:
        return df.to_dict(orient="list")
    elif data_format == DataFormat.COLUMN_SERIES_MAPPING:
        return df.to_dict(orient="series")
    elif data_format in [
        DataFormat.LIST_OF_VALUES,
        DataFormat.TUPLE_OF_VALUES,
        DataFormat.SET_OF_VALUES,
    ]:
        return_list = []
        if len(df.columns) == 1:
            #  Get the first column and convert to list
            return_list = df[df.columns[0]].tolist()
        elif len(df.columns) >= 1:
            raise ValueError(
                f"DataFrame is expected to have a single column but has {len(df.columns)}."
            )
        if data_format == DataFormat.TUPLE_OF_VALUES:
            return tuple(return_list)
        if data_format == DataFormat.SET_OF_VALUES:
            return set(return_list)
        return return_list
    elif data_format == DataFormat.KEY_VALUE_DICT:
        # The key is expected to be the index -> this will return the first column
        # as a dict with index as key.
        return dict() if df.empty else df.iloc[:, 0].to_dict()

    raise ValueError(f"Unsupported input data format: {data_format}")


@overload
def to_key(key: None) -> None:
    ...


@overload
def to_key(key: Key) -> str:
    ...


def to_key(key: Optional[Key]) -> Optional[str]:
    if key is None:
        return None
    else:
        return str(key)


def maybe_raise_label_warnings(label: Optional[str], label_visibility: Optional[str]):
    if not label:
        _LOGGER.warning(
            "`label` got an empty value. This is discouraged for accessibility "
            "reasons and may be disallowed in the future by raising an exception. "
            "Please provide a non-empty label and hide it with label_visibility "
            "if needed."
        )
    if label_visibility not in ("visible", "hidden", "collapsed"):
        raise errors.StreamlitAPIException(
            f"Unsupported label_visibility option '{label_visibility}'. "
            f"Valid values are 'visible', 'hidden' or 'collapsed'."
        )
