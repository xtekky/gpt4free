from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type as type_t,
    TypeVar,
    Union,
)

import numpy as np

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    import numpy.typing as npt

    from pandas._libs import (
        NaTType,
        Period,
        Timedelta,
        Timestamp,
    )
    from pandas._libs.tslibs import BaseOffset

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Interval
    from pandas.core.arrays.base import ExtensionArray
    from pandas.core.frame import DataFrame
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.generic import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )
    from pandas.core.indexes.base import Index
    from pandas.core.internals import (
        ArrayManager,
        BlockManager,
        SingleArrayManager,
        SingleBlockManager,
    )
    from pandas.core.resample import Resampler
    from pandas.core.series import Series
    from pandas.core.window.rolling import BaseWindow

    from pandas.io.formats.format import EngFormatter

    ScalarLike_co = Union[
        int,
        float,
        complex,
        str,
        bytes,
        np.generic,
    ]

    # numpy compatible types
    NumpyValueArrayLike = Union[ScalarLike_co, npt.ArrayLike]
    # Name "npt._ArrayLikeInt_co" is not defined  [name-defined]
    NumpySorter = Optional[npt._ArrayLikeInt_co]  # type: ignore[name-defined]

else:
    npt: Any = None

HashableT = TypeVar("HashableT", bound=Hashable)

# array-like

ArrayLike = Union["ExtensionArray", np.ndarray]
AnyArrayLike = Union[ArrayLike, "Index", "Series"]

# scalars

PythonScalar = Union[str, float, bool]
DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime]
IntStrT = TypeVar("IntStrT", int, str)


# timestamp and timedelta convertible types

TimestampConvertibleTypes = Union[
    "Timestamp", datetime, np.datetime64, np.int64, float, str
]
TimedeltaConvertibleTypes = Union[
    "Timedelta", timedelta, np.timedelta64, np.int64, float, str
]
Timezone = Union[str, tzinfo]

# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound="NDFrame")

NumpyIndexT = TypeVar("NumpyIndexT", np.ndarray, "Index")

Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Level = Hashable
Shape = Tuple[int, ...]
Suffixes = Tuple[Optional[str], Optional[str]]
Ordered = Optional[bool]
JSONSerializable = Optional[Union[PythonScalar, List, Dict]]
Frequency = Union[str, "BaseOffset"]
Axes = Union[AnyArrayLike, List, range]

RandomState = Union[
    int,
    ArrayLike,
    np.random.Generator,
    np.random.BitGenerator,
    np.random.RandomState,
]

# dtypes
NpDtype = Union[str, np.dtype, type_t[Union[str, complex, bool, object]]]
Dtype = Union["ExtensionDtype", NpDtype]
AstypeArg = Union["ExtensionDtype", "npt.DTypeLike"]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg = Union[Dtype, Dict[Hashable, Dtype]]
DtypeObj = Union[np.dtype, "ExtensionDtype"]

# converters
ConvertersArg = Dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg = Union[
    bool, List[Hashable], List[List[Hashable]], Dict[Hashable, List[Hashable]]
]

# For functions like rename that convert one label to another
Renamer = Union[Mapping[Any, Hashable], Callable[[Any], Hashable]]

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# types of vectorized key functions for DataFrame::sort_values and
# DataFrame::sort_index, among others
ValueKeyFunc = Optional[Callable[["Series"], Union["Series", AnyArrayLike]]]
IndexKeyFunc = Optional[Callable[["Index"], Union["Index", AnyArrayLike]]]

# types of `func` kwarg for DataFrame.aggregate and Series.aggregate
AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = Dict[Hashable, Union[AggFuncTypeBase, List[AggFuncTypeBase]]]
AggFuncType = Union[
    AggFuncTypeBase,
    List[AggFuncTypeBase],
    AggFuncTypeDict,
]
AggObjType = Union[
    "Series",
    "DataFrame",
    "GroupBy",
    "SeriesGroupBy",
    "DataFrameGroupBy",
    "BaseWindow",
    "Resampler",
]

PythonFuncType = Callable[[Any], Any]

# filenames and file-like-objects
AnyStr_cov = TypeVar("AnyStr_cov", str, bytes, covariant=True)
AnyStr_con = TypeVar("AnyStr_con", str, bytes, contravariant=True)


class BaseBuffer(Protocol):
    @property
    def mode(self) -> str:
        # for _get_filepath_or_buffer
        ...

    def fileno(self) -> int:
        # for _MMapWrapper
        ...

    def seek(self, __offset: int, __whence: int = ...) -> int:
        # with one argument: gzip.GzipFile, bz2.BZ2File
        # with two arguments: zip.ZipFile, read_sas
        ...

    def seekable(self) -> bool:
        # for bz2.BZ2File
        ...

    def tell(self) -> int:
        # for zip.ZipFile, read_stata, to_stata
        ...


class ReadBuffer(BaseBuffer, Protocol[AnyStr_cov]):
    def read(self, __n: int | None = ...) -> AnyStr_cov:
        # for BytesIOWrapper, gzip.GzipFile, bz2.BZ2File
        ...


class WriteBuffer(BaseBuffer, Protocol[AnyStr_con]):
    def write(self, __b: AnyStr_con) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...

    def flush(self) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...


class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> AnyStr_cov:
        ...


class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ...) -> int:
        ...


class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol):
    def __iter__(self) -> Iterator[AnyStr_cov]:
        # for engine=python
        ...

    def readline(self) -> AnyStr_cov:
        # for engine=python
        ...

    @property
    def closed(self) -> bool:
        # for enine=pyarrow
        ...


FilePath = Union[str, "PathLike[str]"]

# for arbitrary kwargs passed during reading/writing files
StorageOptions = Optional[Dict[str, Any]]


# compression keywords and compression
CompressionDict = Dict[str, Any]
CompressionOptions = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"], CompressionDict]
]

# types in DataFrameFormatter
FormattersType = Union[
    List[Callable], Tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
ColspaceType = Mapping[Hashable, Union[str, int]]
FloatFormatType = Union[str, Callable, "EngFormatter"]
ColspaceArgType = Union[
    str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]
]

# Arguments for fillna()
FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]

# internals
Manager = Union[
    "ArrayManager", "SingleArrayManager", "BlockManager", "SingleBlockManager"
]
SingleManager = Union["SingleArrayManager", "SingleBlockManager"]
Manager2D = Union["ArrayManager", "BlockManager"]

# indexing
# PositionalIndexer -> valid 1D positional indexer, e.g. can pass
# to ndarray.__getitem__
# ScalarIndexer is for a single value as the index
# SequenceIndexer is for list like or slices (but not tuples)
# PositionalIndexerTuple is extends the PositionalIndexer for 2D arrays
# These are used in various __getitem__ overloads
# TODO(typing#684): add Ellipsis, see
# https://github.com/python/typing/issues/684#issuecomment-548203158
# https://bugs.python.org/issue41810
# Using List[int] here rather than Sequence[int] to disallow tuples.
ScalarIndexer = Union[int, np.integer]
SequenceIndexer = Union[slice, List[int], np.ndarray]
PositionalIndexer = Union[ScalarIndexer, SequenceIndexer]
PositionalIndexerTuple = Tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D = Union[PositionalIndexer, PositionalIndexerTuple]
if TYPE_CHECKING:
    TakeIndexer = Union[Sequence[int], Sequence[np.integer], npt.NDArray[np.integer]]
else:
    TakeIndexer = Any

# Shared by functions such as drop and astype
IgnoreRaise = Literal["ignore", "raise"]

# Windowing rank methods
WindowingRankType = Literal["average", "min", "max"]

# read_csv engines
CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]

# read_xml parsers
XMLParsers = Literal["lxml", "etree"]

# Interval closed type
IntervalLeftRight = Literal["left", "right"]
IntervalClosedType = Union[IntervalLeftRight, Literal["both", "neither"]]

# datetime and NaTType
DatetimeNaTType = Union[datetime, "NaTType"]
DateTimeErrorChoices = Union[IgnoreRaise, Literal["coerce"]]

# sort_index
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition = Literal["first", "last"]

# quantile interpolation
QuantileInterpolation = Literal["linear", "lower", "higher", "midpoint", "nearest"]

# plotting
PlottingOrientation = Literal["horizontal", "vertical"]
