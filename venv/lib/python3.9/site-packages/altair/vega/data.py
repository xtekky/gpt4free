import pandas as pd
from toolz import curried
from ..utils.core import sanitize_dataframe
from ..utils.data import (
    MaxRowsError,
    curry,
    pipe,
    sample,
    to_csv,
    to_json,
    to_values,
    check_data_type,
)


@curried.curry
def limit_rows(data, max_rows=5000):
    """Raise MaxRowsError if the data model has more than max_rows."""
    if not isinstance(data, (list, pd.DataFrame)):
        raise TypeError("Expected dict or DataFrame, got: {}".format(type(data)))
    if len(data) > max_rows:
        raise MaxRowsError(
            "The number of rows in your dataset is greater than the max of {}".format(
                max_rows
            )
        )
    return data


@curried.curry
def default_data_transformer(data):
    return curried.pipe(data, limit_rows, to_values)


__all__ = (
    "MaxRowsError",
    "curry",
    "default_data_transformer",
    "limit_rows",
    "pipe",
    "sanitize_dataframe",
    "sample",
    "to_csv",
    "to_json",
    "to_values",
    "check_data_type",
)
