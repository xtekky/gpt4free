"""
Utility routines
"""
from collections.abc import Mapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings

import jsonschema
import pandas as pd
import numpy as np

from .schemapi import SchemaBase, Undefined

try:
    from pandas.api.types import infer_dtype as _infer_dtype
except ImportError:
    # Import for pandas < 0.20.0
    from pandas.lib import infer_dtype as _infer_dtype


def infer_dtype(value):
    """Infer the dtype of the value.

    This is a compatibility function for pandas infer_dtype,
    with skipna=False regardless of the pandas version.
    """
    if not hasattr(infer_dtype, "_supports_skipna"):
        try:
            _infer_dtype([1], skipna=False)
        except TypeError:
            # pandas < 0.21.0 don't support skipna keyword
            infer_dtype._supports_skipna = False
        else:
            infer_dtype._supports_skipna = True
    if infer_dtype._supports_skipna:
        return _infer_dtype(value, skipna=False)
    else:
        return _infer_dtype(value)


TYPECODE_MAP = {
    "ordinal": "O",
    "nominal": "N",
    "quantitative": "Q",
    "temporal": "T",
    "geojson": "G",
}

INV_TYPECODE_MAP = {v: k for k, v in TYPECODE_MAP.items()}


# aggregates from vega-lite version 4.6.0
AGGREGATES = [
    "argmax",
    "argmin",
    "average",
    "count",
    "distinct",
    "max",
    "mean",
    "median",
    "min",
    "missing",
    "product",
    "q1",
    "q3",
    "ci0",
    "ci1",
    "stderr",
    "stdev",
    "stdevp",
    "sum",
    "valid",
    "values",
    "variance",
    "variancep",
]

# window aggregates from vega-lite version 4.6.0
WINDOW_AGGREGATES = [
    "row_number",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "lag",
    "lead",
    "first_value",
    "last_value",
    "nth_value",
]

# timeUnits from vega-lite version 4.17.0
TIMEUNITS = [
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "dayofyear",
    "date",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "yearquarter",
    "yearquartermonth",
    "yearmonth",
    "yearmonthdate",
    "yearmonthdatehours",
    "yearmonthdatehoursminutes",
    "yearmonthdatehoursminutesseconds",
    "yearweek",
    "yearweekday",
    "yearweekdayhours",
    "yearweekdayhoursminutes",
    "yearweekdayhoursminutesseconds",
    "yeardayofyear",
    "quartermonth",
    "monthdate",
    "monthdatehours",
    "monthdatehoursminutes",
    "monthdatehoursminutesseconds",
    "weekday",
    "weeksdayhours",
    "weekdayhoursminutes",
    "weekdayhoursminutesseconds",
    "dayhours",
    "dayhoursminutes",
    "dayhoursminutesseconds",
    "hoursminutes",
    "hoursminutesseconds",
    "minutesseconds",
    "secondsmilliseconds",
    "utcyear",
    "utcquarter",
    "utcmonth",
    "utcweek",
    "utcday",
    "utcdayofyear",
    "utcdate",
    "utchours",
    "utcminutes",
    "utcseconds",
    "utcmilliseconds",
    "utcyearquarter",
    "utcyearquartermonth",
    "utcyearmonth",
    "utcyearmonthdate",
    "utcyearmonthdatehours",
    "utcyearmonthdatehoursminutes",
    "utcyearmonthdatehoursminutesseconds",
    "utcyearweek",
    "utcyearweekday",
    "utcyearweekdayhours",
    "utcyearweekdayhoursminutes",
    "utcyearweekdayhoursminutesseconds",
    "utcyeardayofyear",
    "utcquartermonth",
    "utcmonthdate",
    "utcmonthdatehours",
    "utcmonthdatehoursminutes",
    "utcmonthdatehoursminutesseconds",
    "utcweekday",
    "utcweeksdayhours",
    "utcweekdayhoursminutes",
    "utcweekdayhoursminutesseconds",
    "utcdayhours",
    "utcdayhoursminutes",
    "utcdayhoursminutesseconds",
    "utchoursminutes",
    "utchoursminutesseconds",
    "utcminutesseconds",
    "utcsecondsmilliseconds",
]


def infer_vegalite_type(data):
    """
    From an array-like input, infer the correct vega typecode
    ('ordinal', 'nominal', 'quantitative', or 'temporal')

    Parameters
    ----------
    data: Numpy array or Pandas Series
    """
    # Otherwise, infer based on the dtype of the input
    typ = infer_dtype(data)

    # TODO: Once this returns 'O', please update test_select_x and test_select_y in test_api.py

    if typ in [
        "floating",
        "mixed-integer-float",
        "integer",
        "mixed-integer",
        "complex",
    ]:
        return "quantitative"
    elif typ in ["string", "bytes", "categorical", "boolean", "mixed", "unicode"]:
        return "nominal"
    elif typ in [
        "datetime",
        "datetime64",
        "timedelta",
        "timedelta64",
        "date",
        "time",
        "period",
    ]:
        return "temporal"
    else:
        warnings.warn(
            "I don't know how to infer vegalite type from '{}'.  "
            "Defaulting to nominal.".format(typ)
        )
        return "nominal"


def merge_props_geom(feat):
    """
    Merge properties with geometry
    * Overwrites 'type' and 'geometry' entries if existing
    """

    geom = {k: feat[k] for k in ("type", "geometry")}
    try:
        feat["properties"].update(geom)
        props_geom = feat["properties"]
    except (AttributeError, KeyError):
        # AttributeError when 'properties' equals None
        # KeyError when 'properties' is non-existing
        props_geom = geom

    return props_geom


def sanitize_geo_interface(geo):
    """Santize a geo_interface to prepare it for serialization.

    * Make a copy
    * Convert type array or _Array to list
    * Convert tuples to lists (using json.loads/dumps)
    * Merge properties with geometry
    """

    geo = deepcopy(geo)

    # convert type _Array or array to list
    for key in geo.keys():
        if str(type(geo[key]).__name__).startswith(("_Array", "array")):
            geo[key] = geo[key].tolist()

    # convert (nested) tuples to lists
    geo = json.loads(json.dumps(geo))

    # sanitize features
    if geo["type"] == "FeatureCollection":
        geo = geo["features"]
        if len(geo) > 0:
            for idx, feat in enumerate(geo):
                geo[idx] = merge_props_geom(feat)
    elif geo["type"] == "Feature":
        geo = merge_props_geom(geo)
    else:
        geo = {"type": "Feature", "geometry": geo}

    return geo


def sanitize_dataframe(df):  # noqa: C901
    """Sanitize a DataFrame to prepare it for serialization.

    * Make a copy
    * Convert RangeIndex columns to strings
    * Raise ValueError if column names are not strings
    * Raise ValueError if it has a hierarchical index.
    * Convert categoricals to strings.
    * Convert np.bool_ dtypes to Python bool objects
    * Convert np.int dtypes to Python int objects
    * Convert floats to objects and replace NaNs/infs with None.
    * Convert DateTime dtypes into appropriate string representations
    * Convert Nullable integers to objects and replace NaN with None
    * Convert Nullable boolean to objects and replace NaN with None
    * convert dedicated string column to objects and replace NaN with None
    * Raise a ValueError for TimeDelta dtypes
    """
    df = df.copy()

    if isinstance(df.columns, pd.RangeIndex):
        df.columns = df.columns.astype(str)

    for col in df.columns:
        if not isinstance(col, str):
            raise ValueError(
                "Dataframe contains invalid column name: {0!r}. "
                "Column names must be strings".format(col)
            )

    if isinstance(df.index, pd.MultiIndex):
        raise ValueError("Hierarchical indices not supported")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Hierarchical indices not supported")

    def to_list_if_array(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val

    for col_name, dtype in df.dtypes.items():
        if str(dtype) == "category":
            # XXXX: work around bug in to_json for categorical types
            # https://github.com/pydata/pandas/issues/10778
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif str(dtype) == "string":
            # dedicated string datatype (since 1.0)
            # https://pandas.pydata.org/pandas-docs/version/1.0.0/whatsnew/v1.0.0.html#dedicated-string-data-type
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif str(dtype) == "bool":
            # convert numpy bools to objects; np.bool is not JSON serializable
            df[col_name] = df[col_name].astype(object)
        elif str(dtype) == "boolean":
            # dedicated boolean datatype (since 1.0)
            # https://pandas.io/docs/user_guide/boolean.html
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif str(dtype).startswith("datetime"):
            # Convert datetimes to strings. This needs to be a full ISO string
            # with time, which is why we cannot use ``col.astype(str)``.
            # This is because Javascript parses date-only times in UTC, but
            # parses full ISO-8601 dates as local time, and dates in Vega and
            # Vega-Lite are displayed in local time by default.
            # (see https://github.com/altair-viz/altair/issues/1027)
            df[col_name] = (
                df[col_name].apply(lambda x: x.isoformat()).replace("NaT", "")
            )
        elif str(dtype).startswith("timedelta"):
            raise ValueError(
                'Field "{col_name}" has type "{dtype}" which is '
                "not supported by Altair. Please convert to "
                "either a timestamp or a numerical value."
                "".format(col_name=col_name, dtype=dtype)
            )
        elif str(dtype).startswith("geometry"):
            # geopandas >=0.6.1 uses the dtype geometry. Continue here
            # otherwise it will give an error on np.issubdtype(dtype, np.integer)
            continue
        elif str(dtype) in {
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
        }:  # nullable integer datatypes (since 24.0) and nullable float datatypes (since 1.2.0)
            # https://pandas.pydata.org/pandas-docs/version/0.25/whatsnew/v0.24.0.html#optional-integer-na-support
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif np.issubdtype(dtype, np.integer):
            # convert integers to objects; np.int is not JSON serializable
            df[col_name] = df[col_name].astype(object)
        elif np.issubdtype(dtype, np.floating):
            # For floats, convert to Python float: np.float is not JSON serializable
            # Also convert NaN/inf values to null, as they are not JSON serializable
            col = df[col_name]
            bad_values = col.isnull() | np.isinf(col)
            df[col_name] = col.astype(object).where(~bad_values, None)
        elif dtype == object:
            # Convert numpy arrays saved as objects to lists
            # Arrays are not JSON serializable
            col = df[col_name].apply(to_list_if_array, convert_dtype=False)
            df[col_name] = col.where(col.notnull(), None)
    return df


def parse_shorthand(
    shorthand,
    data=None,
    parse_aggregates=True,
    parse_window_ops=False,
    parse_timeunits=True,
    parse_types=True,
):
    """General tool to parse shorthand values

    These are of the form:

    - "col_name"
    - "col_name:O"
    - "average(col_name)"
    - "average(col_name):O"

    Optionally, a dataframe may be supplied, from which the type
    will be inferred if not specified in the shorthand.

    Parameters
    ----------
    shorthand : dict or string
        The shorthand representation to be parsed
    data : DataFrame, optional
        If specified and of type DataFrame, then use these values to infer the
        column type if not provided by the shorthand.
    parse_aggregates : boolean
        If True (default), then parse aggregate functions within the shorthand.
    parse_window_ops : boolean
        If True then parse window operations within the shorthand (default:False)
    parse_timeunits : boolean
        If True (default), then parse timeUnits from within the shorthand
    parse_types : boolean
        If True (default), then parse typecodes within the shorthand

    Returns
    -------
    attrs : dict
        a dictionary of attributes extracted from the shorthand

    Examples
    --------
    >>> data = pd.DataFrame({'foo': ['A', 'B', 'A', 'B'],
    ...                      'bar': [1, 2, 3, 4]})

    >>> parse_shorthand('name') == {'field': 'name'}
    True

    >>> parse_shorthand('name:Q') == {'field': 'name', 'type': 'quantitative'}
    True

    >>> parse_shorthand('average(col)') == {'aggregate': 'average', 'field': 'col'}
    True

    >>> parse_shorthand('foo:O') == {'field': 'foo', 'type': 'ordinal'}
    True

    >>> parse_shorthand('min(foo):Q') == {'aggregate': 'min', 'field': 'foo', 'type': 'quantitative'}
    True

    >>> parse_shorthand('month(col)') == {'field': 'col', 'timeUnit': 'month', 'type': 'temporal'}
    True

    >>> parse_shorthand('year(col):O') == {'field': 'col', 'timeUnit': 'year', 'type': 'ordinal'}
    True

    >>> parse_shorthand('foo', data) == {'field': 'foo', 'type': 'nominal'}
    True

    >>> parse_shorthand('bar', data) == {'field': 'bar', 'type': 'quantitative'}
    True

    >>> parse_shorthand('bar:O', data) == {'field': 'bar', 'type': 'ordinal'}
    True

    >>> parse_shorthand('sum(bar)', data) == {'aggregate': 'sum', 'field': 'bar', 'type': 'quantitative'}
    True

    >>> parse_shorthand('count()', data) == {'aggregate': 'count', 'type': 'quantitative'}
    True
    """
    if not shorthand:
        return {}

    valid_typecodes = list(TYPECODE_MAP) + list(INV_TYPECODE_MAP)

    units = dict(
        field="(?P<field>.*)",
        type="(?P<type>{})".format("|".join(valid_typecodes)),
        agg_count="(?P<aggregate>count)",
        op_count="(?P<op>count)",
        aggregate="(?P<aggregate>{})".format("|".join(AGGREGATES)),
        window_op="(?P<op>{})".format("|".join(AGGREGATES + WINDOW_AGGREGATES)),
        timeUnit="(?P<timeUnit>{})".format("|".join(TIMEUNITS)),
    )

    patterns = []

    if parse_aggregates:
        patterns.extend([r"{agg_count}\(\)"])
        patterns.extend([r"{aggregate}\({field}\)"])
    if parse_window_ops:
        patterns.extend([r"{op_count}\(\)"])
        patterns.extend([r"{window_op}\({field}\)"])
    if parse_timeunits:
        patterns.extend([r"{timeUnit}\({field}\)"])

    patterns.extend([r"{field}"])

    if parse_types:
        patterns = list(itertools.chain(*((p + ":{type}", p) for p in patterns)))

    regexps = (
        re.compile(r"\A" + p.format(**units) + r"\Z", re.DOTALL) for p in patterns
    )

    # find matches depending on valid fields passed
    if isinstance(shorthand, dict):
        attrs = shorthand
    else:
        attrs = next(
            exp.match(shorthand).groupdict() for exp in regexps if exp.match(shorthand)
        )

    # Handle short form of the type expression
    if "type" in attrs:
        attrs["type"] = INV_TYPECODE_MAP.get(attrs["type"], attrs["type"])

    # counts are quantitative by default
    if attrs == {"aggregate": "count"}:
        attrs["type"] = "quantitative"

    # times are temporal by default
    if "timeUnit" in attrs and "type" not in attrs:
        attrs["type"] = "temporal"

    # if data is specified and type is not, infer type from data
    if isinstance(data, pd.DataFrame) and "type" not in attrs:
        if "field" in attrs and attrs["field"] in data.columns:
            attrs["type"] = infer_vegalite_type(data[attrs["field"]])
    return attrs


def use_signature(Obj):
    """Apply call signature and documentation of Obj to the decorated method"""

    def decorate(f):
        # call-signature of f is exposed via __wrapped__.
        # we want it to mimic Obj.__init__
        f.__wrapped__ = Obj.__init__
        f._uses_signature = Obj

        # Supplement the docstring of f with information from Obj
        if Obj.__doc__:
            doclines = Obj.__doc__.splitlines()
            if f.__doc__:
                doc = f.__doc__ + "\n".join(doclines[1:])
            else:
                doc = "\n".join(doclines)
            try:
                f.__doc__ = doc
            except AttributeError:
                # __doc__ is not modifiable for classes in Python < 3.3
                pass

        return f

    return decorate


def update_subtraits(obj, attrs, **kwargs):
    """Recursively update sub-traits without overwriting other traits"""
    # TODO: infer keywords from args
    if not kwargs:
        return obj

    # obj can be a SchemaBase object or a dict
    if obj is Undefined:
        obj = dct = {}
    elif isinstance(obj, SchemaBase):
        dct = obj._kwds
    else:
        dct = obj

    if isinstance(attrs, str):
        attrs = (attrs,)

    if len(attrs) == 0:
        dct.update(kwargs)
    else:
        attr = attrs[0]
        trait = dct.get(attr, Undefined)
        if trait is Undefined:
            trait = dct[attr] = {}
        dct[attr] = update_subtraits(trait, attrs[1:], **kwargs)
    return obj


def update_nested(original, update, copy=False):
    """Update nested dictionaries

    Parameters
    ----------
    original : dict
        the original (nested) dictionary, which will be updated in-place
    update : dict
        the nested dictionary of updates
    copy : bool, default False
        if True, then copy the original dictionary rather than modifying it

    Returns
    -------
    original : dict
        a reference to the (modified) original dict

    Examples
    --------
    >>> original = {'x': {'b': 2, 'c': 4}}
    >>> update = {'x': {'b': 5, 'd': 6}, 'y': 40}
    >>> update_nested(original, update)  # doctest: +SKIP
    {'x': {'b': 5, 'c': 4, 'd': 6}, 'y': 40}
    >>> original  # doctest: +SKIP
    {'x': {'b': 5, 'c': 4, 'd': 6}, 'y': 40}
    """
    if copy:
        original = deepcopy(original)
    for key, val in update.items():
        if isinstance(val, Mapping):
            orig_val = original.get(key, {})
            if isinstance(orig_val, Mapping):
                original[key] = update_nested(orig_val, val)
            else:
                original[key] = val
        else:
            original[key] = val
    return original


def display_traceback(in_ipython=True):
    exc_info = sys.exc_info()

    if in_ipython:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
    else:
        ip = None

    if ip is not None:
        ip.showtraceback(exc_info)
    else:
        traceback.print_exception(*exc_info)


def infer_encoding_types(args, kwargs, channels):
    """Infer typed keyword arguments for args and kwargs

    Parameters
    ----------
    args : tuple
        List of function args
    kwargs : dict
        Dict of function kwargs
    channels : module
        The module containing all altair encoding channel classes.

    Returns
    -------
    kwargs : dict
        All args and kwargs in a single dict, with keys and types
        based on the channels mapping.
    """
    # Construct a dictionary of channel type to encoding name
    # TODO: cache this somehow?
    channel_objs = (getattr(channels, name) for name in dir(channels))
    channel_objs = (
        c for c in channel_objs if isinstance(c, type) and issubclass(c, SchemaBase)
    )
    channel_to_name = {c: c._encoding_name for c in channel_objs}
    name_to_channel = {}
    for chan, name in channel_to_name.items():
        chans = name_to_channel.setdefault(name, {})
        if chan.__name__.endswith("Datum"):
            key = "datum"
        elif chan.__name__.endswith("Value"):
            key = "value"
        else:
            key = "field"
        chans[key] = chan

    # First use the mapping to convert args to kwargs based on their types.
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) > 0:
            type_ = type(arg[0])
        else:
            type_ = type(arg)

        encoding = channel_to_name.get(type_, None)
        if encoding is None:
            raise NotImplementedError("positional of type {}" "".format(type_))
        if encoding in kwargs:
            raise ValueError("encoding {} specified twice.".format(encoding))
        kwargs[encoding] = arg

    def _wrap_in_channel_class(obj, encoding):
        try:
            condition = obj["condition"]
        except (KeyError, TypeError):
            pass
        else:
            if condition is not Undefined:
                obj = obj.copy()
                obj["condition"] = _wrap_in_channel_class(condition, encoding)

        if isinstance(obj, SchemaBase):
            return obj

        if isinstance(obj, str):
            obj = {"shorthand": obj}

        if isinstance(obj, (list, tuple)):
            return [_wrap_in_channel_class(subobj, encoding) for subobj in obj]

        if encoding not in name_to_channel:
            warnings.warn("Unrecognized encoding channel '{}'".format(encoding))
            return obj

        classes = name_to_channel[encoding]
        cls = classes["value"] if "value" in obj else classes["field"]

        try:
            # Don't force validation here; some objects won't be valid until
            # they're created in the context of a chart.
            return cls.from_dict(obj, validate=False)
        except jsonschema.ValidationError:
            # our attempts at finding the correct class have failed
            return obj

    return {
        encoding: _wrap_in_channel_class(obj, encoding)
        for encoding, obj in kwargs.items()
    }
