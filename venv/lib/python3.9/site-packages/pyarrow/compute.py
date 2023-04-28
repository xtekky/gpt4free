# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from pyarrow._compute import (  # noqa
    Function,
    FunctionOptions,
    FunctionRegistry,
    HashAggregateFunction,
    HashAggregateKernel,
    Kernel,
    ScalarAggregateFunction,
    ScalarAggregateKernel,
    ScalarFunction,
    ScalarKernel,
    VectorFunction,
    VectorKernel,
    # Option classes
    ArraySortOptions,
    AssumeTimezoneOptions,
    CastOptions,
    CountOptions,
    CumulativeSumOptions,
    DayOfWeekOptions,
    DictionaryEncodeOptions,
    ElementWiseAggregateOptions,
    ExtractRegexOptions,
    FilterOptions,
    IndexOptions,
    JoinOptions,
    ListSliceOptions,
    MakeStructOptions,
    MapLookupOptions,
    MatchSubstringOptions,
    ModeOptions,
    NullOptions,
    PadOptions,
    PartitionNthOptions,
    QuantileOptions,
    RandomOptions,
    RankOptions,
    ReplaceSliceOptions,
    ReplaceSubstringOptions,
    RoundOptions,
    RoundTemporalOptions,
    RoundToMultipleOptions,
    ScalarAggregateOptions,
    SelectKOptions,
    SetLookupOptions,
    SliceOptions,
    SortOptions,
    SplitOptions,
    SplitPatternOptions,
    StrftimeOptions,
    StrptimeOptions,
    StructFieldOptions,
    TakeOptions,
    TDigestOptions,
    TrimOptions,
    Utf8NormalizeOptions,
    VarianceOptions,
    WeekOptions,
    # Functions
    call_function,
    function_registry,
    get_function,
    list_functions,
    _group_by,
    # Udf
    register_scalar_function,
    ScalarUdfContext,
    # Expressions
    Expression,
)

from collections import namedtuple
import inspect
from textwrap import dedent
import warnings

import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape


def _get_arg_names(func):
    return func._doc.arg_names


_OptionsClassDoc = namedtuple('_OptionsClassDoc', ('params',))


def _scrape_options_class_doc(options_class):
    if not options_class.__doc__:
        return None
    doc = docscrape.NumpyDocString(options_class.__doc__)
    return _OptionsClassDoc(doc['Parameters'])


def _decorate_compute_function(wrapper, exposed_name, func, options_class):
    # Decorate the given compute function wrapper with useful metadata
    # and documentation.
    cpp_doc = func._doc

    wrapper.__arrow_compute_function__ = dict(
        name=func.name,
        arity=func.arity,
        options_class=cpp_doc.options_class,
        options_required=cpp_doc.options_required)
    wrapper.__name__ = exposed_name
    wrapper.__qualname__ = exposed_name

    doc_pieces = []

    # 1. One-line summary
    summary = cpp_doc.summary
    if not summary:
        arg_str = "arguments" if func.arity > 1 else "argument"
        summary = ("Call compute function {!r} with the given {}"
                   .format(func.name, arg_str))

    doc_pieces.append(f"{summary}.\n\n")

    # 2. Multi-line description
    description = cpp_doc.description
    if description:
        doc_pieces.append(f"{description}\n\n")

    doc_addition = _compute_docstrings.function_doc_additions.get(func.name)

    # 3. Parameter description
    doc_pieces.append(dedent("""\
        Parameters
        ----------
        """))

    # 3a. Compute function parameters
    arg_names = _get_arg_names(func)
    for arg_name in arg_names:
        if func.kind in ('vector', 'scalar_aggregate'):
            arg_type = 'Array-like'
        else:
            arg_type = 'Array-like or scalar-like'
        doc_pieces.append(f"{arg_name} : {arg_type}\n")
        doc_pieces.append("    Argument to compute function.\n")

    # 3b. Compute function option values
    if options_class is not None:
        options_class_doc = _scrape_options_class_doc(options_class)
        if options_class_doc:
            for p in options_class_doc.params:
                doc_pieces.append(f"{p.name} : {p.type}\n")
                for s in p.desc:
                    doc_pieces.append(f"    {s}\n")
        else:
            warnings.warn(f"Options class {options_class.__name__} "
                          f"does not have a docstring", RuntimeWarning)
            options_sig = inspect.signature(options_class)
            for p in options_sig.parameters.values():
                doc_pieces.append(dedent("""\
                {0} : optional
                    Parameter for {1} constructor. Either `options`
                    or `{0}` can be passed, but not both at the same time.
                """.format(p.name, options_class.__name__)))
        doc_pieces.append(dedent(f"""\
            options : pyarrow.compute.{options_class.__name__}, optional
                Alternative way of passing options.
            """))

    doc_pieces.append(dedent("""\
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the default memory pool.
        """))

    # 4. Custom addition (e.g. examples)
    if doc_addition is not None:
        doc_pieces.append("\n{}\n".format(dedent(doc_addition).strip("\n")))

    wrapper.__doc__ = "".join(doc_pieces)
    return wrapper


def _get_options_class(func):
    class_name = func._doc.options_class
    if not class_name:
        return None
    try:
        return globals()[class_name]
    except KeyError:
        warnings.warn("Python binding for {} not exposed"
                      .format(class_name), RuntimeWarning)
        return None


def _handle_options(name, options_class, options, args, kwargs):
    if args or kwargs:
        if options is not None:
            raise TypeError(
                "Function {!r} called with both an 'options' argument "
                "and additional arguments"
                .format(name))
        return options_class(*args, **kwargs)

    if options is not None:
        if isinstance(options, dict):
            return options_class(**options)
        elif isinstance(options, options_class):
            return options
        raise TypeError(
            "Function {!r} expected a {} parameter, got {}"
            .format(name, options_class, type(options)))

    return None


def _make_generic_wrapper(func_name, func, options_class, arity):
    if options_class is None:
        def wrapper(*args, memory_pool=None):
            if arity is not Ellipsis and len(args) != arity:
                raise TypeError(
                    f"{func_name} takes {arity} positional argument(s), "
                    f"but {len(args)} were given"
                )
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args))
            return func.call(args, None, memory_pool)
    else:
        def wrapper(*args, memory_pool=None, options=None, **kwargs):
            if arity is not Ellipsis:
                if len(args) < arity:
                    raise TypeError(
                        f"{func_name} takes {arity} positional argument(s), "
                        f"but {len(args)} were given"
                    )
                option_args = args[arity:]
                args = args[:arity]
            else:
                option_args = ()
            options = _handle_options(func_name, options_class, options,
                                      option_args, kwargs)
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args), options)
            return func.call(args, options, memory_pool)
    return wrapper


def _make_signature(arg_names, var_arg_names, options_class):
    from inspect import Parameter
    params = []
    for name in arg_names:
        params.append(Parameter(name, Parameter.POSITIONAL_ONLY))
    for name in var_arg_names:
        params.append(Parameter(name, Parameter.VAR_POSITIONAL))
    if options_class is not None:
        options_sig = inspect.signature(options_class)
        for p in options_sig.parameters.values():
            assert p.kind in (Parameter.POSITIONAL_OR_KEYWORD,
                              Parameter.KEYWORD_ONLY)
            if var_arg_names:
                # Cannot have a positional argument after a *args
                p = p.replace(kind=Parameter.KEYWORD_ONLY)
            params.append(p)
        params.append(Parameter("options", Parameter.KEYWORD_ONLY,
                                default=None))
    params.append(Parameter("memory_pool", Parameter.KEYWORD_ONLY,
                            default=None))
    return inspect.Signature(params)


def _wrap_function(name, func):
    options_class = _get_options_class(func)
    arg_names = _get_arg_names(func)
    has_vararg = arg_names and arg_names[-1].startswith('*')
    if has_vararg:
        var_arg_names = [arg_names.pop().lstrip('*')]
    else:
        var_arg_names = []

    wrapper = _make_generic_wrapper(
        name, func, options_class, arity=func.arity)
    wrapper.__signature__ = _make_signature(arg_names, var_arg_names,
                                            options_class)
    return _decorate_compute_function(wrapper, name, func, options_class)


def _make_global_functions():
    """
    Make global functions wrapping each compute function.

    Note that some of the automatically-generated wrappers may be overridden
    by custom versions below.
    """
    g = globals()
    reg = function_registry()

    # Avoid clashes with Python keywords
    rewrites = {'and': 'and_',
                'or': 'or_'}

    for cpp_name in reg.list_functions():
        name = rewrites.get(cpp_name, cpp_name)
        func = reg.get_function(cpp_name)
        if func.kind == "hash_aggregate":
            # Hash aggregate functions are not callable,
            # so let's not expose them at module level.
            continue
        assert name not in g, name
        g[cpp_name] = g[name] = _wrap_function(name, func)


_make_global_functions()


def cast(arr, target_type=None, safe=None, options=None):
    """
    Cast array values to another data type. Can also be invoked as an array
    instance method.

    Parameters
    ----------
    arr : Array-like
    target_type : DataType or str
        Type to cast to
    safe : bool, default True
        Check for overflows or other unsafe conversions
    options : CastOptions, default None
        Additional checks pass by CastOptions

    Examples
    --------
    >>> from datetime import datetime
    >>> import pyarrow as pa
    >>> arr = pa.array([datetime(2010, 1, 1), datetime(2015, 1, 1)])
    >>> arr.type
    TimestampType(timestamp[us])

    You can use ``pyarrow.DataType`` objects to specify the target type:

    >>> cast(arr, pa.timestamp('ms'))
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]

    >>> cast(arr, pa.timestamp('ms')).type
    TimestampType(timestamp[ms])

    Alternatively, it is also supported to use the string aliases for these
    types:

    >>> arr.cast('timestamp[ms]')
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]
    >>> arr.cast('timestamp[ms]').type
    TimestampType(timestamp[ms])

    Returns
    -------
    casted : Array
        The cast result as a new Array
    """
    safe_vars_passed = (safe is not None) or (target_type is not None)

    if safe_vars_passed and (options is not None):
        raise ValueError("Must either pass values for 'target_type' and 'safe'"
                         " or pass a value for 'options'")

    if options is None:
        target_type = pa.types.lib.ensure_type(target_type)
        if safe is False:
            options = CastOptions.unsafe(target_type)
        else:
            options = CastOptions.safe(target_type)
    return call_function("cast", [arr], options)


def index(data, value, start=None, end=None, *, memory_pool=None):
    """
    Find the index of the first occurrence of a given value.

    Parameters
    ----------
    data : Array-like
    value : Scalar-like object
        The value to search for.
    start : int, optional
    end : int, optional
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    index : int
        the index, or -1 if not found
    """
    if start is not None:
        if end is not None:
            data = data.slice(start, end - start)
        else:
            data = data.slice(start)
    elif end is not None:
        data = data.slice(0, end)

    if not isinstance(value, pa.Scalar):
        value = pa.scalar(value, type=data.type)
    elif data.type != value.type:
        value = pa.scalar(value.as_py(), type=data.type)
    options = IndexOptions(value=value)
    result = call_function('index', [data], options, memory_pool)
    if start is not None and result.as_py() >= 0:
        result = pa.scalar(result.as_py() + start, type=pa.int64())
    return result


def take(data, indices, *, boundscheck=True, memory_pool=None):
    """
    Select values (or records) from array- or table-like data given integer
    selection indices.

    The result will be of the same type(s) as the input, with elements taken
    from the input array (or record batch / table fields) at the given
    indices. If an index is null then the corresponding value in the output
    will be null.

    Parameters
    ----------
    data : Array, ChunkedArray, RecordBatch, or Table
    indices : Array, ChunkedArray
        Must be of integer type
    boundscheck : boolean, default True
        Whether to boundscheck the indices. If False and there is an out of
        bounds index, will likely cause the process to crash.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : depends on inputs
        Selected values for the given indices

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> indices = pa.array([0, None, 4, 3])
    >>> arr.take(indices)
    <pyarrow.lib.StringArray object at ...>
    [
      "a",
      null,
      "e",
      null
    ]
    """
    options = TakeOptions(boundscheck=boundscheck)
    return call_function('take', [data, indices], options, memory_pool)


def fill_null(values, fill_value):
    """
    Replace each null element in values with fill_value. The fill_value must be
    the same type as values or able to be implicitly casted to the array's
    type.

    This is an alias for :func:`coalesce`.

    Parameters
    ----------
    values : Array, ChunkedArray, or Scalar-like object
        Each null element is replaced with the corresponding value
        from fill_value.
    fill_value : Array, ChunkedArray, or Scalar-like object
        If not same type as data will attempt to cast.

    Returns
    -------
    result : depends on inputs
        Values with all null elements replaced

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([1, 2, None, 3], type=pa.int8())
    >>> fill_value = pa.scalar(5, type=pa.int8())
    >>> arr.fill_null(fill_value)
    <pyarrow.lib.Int8Array object at ...>
    [
      1,
      2,
      5,
      3
    ]
    """
    if not isinstance(fill_value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
        fill_value = pa.scalar(fill_value, type=values.type)
    elif values.type != fill_value.type:
        fill_value = pa.scalar(fill_value.as_py(), type=values.type)

    return call_function("coalesce", [values, fill_value])


def top_k_unstable(values, k, sort_keys=None, *, memory_pool=None):
    """
    Select the indices of the top-k ordered elements from array- or table-like
    data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get top indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array
        Indices of the top-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.top_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      5,
      4,
      2
    ]
    """
    if sort_keys is None:
        sort_keys = []
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        sort_keys.append(("dummy", "descending"))
    else:
        sort_keys = map(lambda key_name: (key_name, "descending"), sort_keys)
    options = SelectKOptions(k, sort_keys)
    return call_function("select_k_unstable", [values], options, memory_pool)


def bottom_k_unstable(values, k, sort_keys=None, *, memory_pool=None):
    """
    Select the indices of the bottom-k ordered elements from
    array- or table-like data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get bottom indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array of indices
        Indices of the bottom-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.bottom_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      0,
      1,
      2
    ]
    """
    if sort_keys is None:
        sort_keys = []
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        sort_keys.append(("dummy", "ascending"))
    else:
        sort_keys = map(lambda key_name: (key_name, "ascending"), sort_keys)
    options = SelectKOptions(k, sort_keys)
    return call_function("select_k_unstable", [values], options, memory_pool)


def random(n, *, initializer='system', options=None, memory_pool=None):
    """
    Generate numbers in the range [0, 1).

    Generated values are uniformly-distributed, double-precision
    in range [0, 1). Algorithm and seed can be changed via RandomOptions.

    Parameters
    ----------
    n : int
        Number of values to generate, must be greater than or equal to 0
    initializer : int or str
        How to initialize the underlying random generator.
        If an integer is given, it is used as a seed.
        If "system" is given, the random generator is initialized with
        a system-specific source of (hopefully true) randomness.
        Other values are invalid.
    options : pyarrow.compute.RandomOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """
    options = RandomOptions(initializer=initializer)
    return call_function("random", [], options, memory_pool, length=n)


def field(*name_or_index):
    """Reference a column of the dataset.

    Stores only the field's name. Type and other information is known only when
    the expression is bound to a dataset having an explicit scheme.

    Nested references are allowed by passing multiple names or a tuple of
    names. For example ``('foo', 'bar')`` references the field named "bar"
    inside the field named "foo".

    Parameters
    ----------
    *name_or_index : string, multiple strings, tuple or int
        The name or index of the (possibly nested) field the expression
        references to.

    Returns
    -------
    field_expr : Expression
        Reference to the given field

    Examples
    --------
    >>> import pyarrow.compute as pc
    >>> pc.field("a")
    <pyarrow.compute.Expression a>
    >>> pc.field(1)
    <pyarrow.compute.Expression FieldPath(1)>
    >>> pc.field(("a", "b"))
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    >>> pc.field("a", "b")
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    """
    n = len(name_or_index)
    if n == 1:
        if isinstance(name_or_index[0], (str, int)):
            return Expression._field(name_or_index[0])
        elif isinstance(name_or_index[0], tuple):
            return Expression._nested_field(name_or_index[0])
        else:
            raise TypeError(
                "field reference should be str, multiple str, tuple or "
                f"integer, got {type(name_or_index[0])}"
            )
    # In case of multiple strings not supplied in a tuple
    else:
        return Expression._nested_field(name_or_index)


def scalar(value):
    """Expression representing a scalar value.

    Parameters
    ----------
    value : bool, int, float or string
        Python value of the scalar. Note that only a subset of types are
        currently supported.

    Returns
    -------
    scalar_expr : Expression
        An Expression representing the scalar value
    """
    return Expression._scalar(value)
