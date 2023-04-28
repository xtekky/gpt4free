import pytest

from pandas.errors import (
    AbstractMethodError,
    UndefinedVariableError,
)

import pandas as pd


@pytest.mark.parametrize(
    "exc",
    [
        "UnsupportedFunctionCall",
        "UnsortedIndexError",
        "OutOfBoundsDatetime",
        "ParserError",
        "PerformanceWarning",
        "DtypeWarning",
        "EmptyDataError",
        "ParserWarning",
        "MergeError",
        "OptionError",
        "NumbaUtilError",
        "DataError",
        "SpecificationError",
        "SettingWithCopyError",
        "SettingWithCopyWarning",
        "NumExprClobberingError",
        "IndexingError",
        "PyperclipException",
        "CSSWarning",
        "ClosedFileError",
        "PossibleDataLossError",
        "IncompatibilityWarning",
        "AttributeConflictWarning",
        "DatabaseError",
        "PossiblePrecisionLoss",
        "CategoricalConversionWarning",
        "InvalidColumnName",
        "ValueLabelTypeMismatch",
    ],
)
def test_exception_importable(exc):
    from pandas import errors

    err = getattr(errors, exc)
    assert err is not None

    # check that we can raise on them

    msg = "^$"

    with pytest.raises(err, match=msg):
        raise err()


def test_catch_oob():
    from pandas import errors

    msg = "Out of bounds nanosecond timestamp: 1500-01-01 00:00:00"
    with pytest.raises(errors.OutOfBoundsDatetime, match=msg):
        pd.Timestamp("15000101")


@pytest.mark.parametrize(
    "is_local",
    [
        True,
        False,
    ],
)
def test_catch_undefined_variable_error(is_local):
    variable_name = "x"
    if is_local:
        msg = f"local variable '{variable_name}' is not defined"
    else:
        msg = f"name '{variable_name}' is not defined"

    with pytest.raises(UndefinedVariableError, match=msg):
        raise UndefinedVariableError(variable_name, is_local)


class Foo:
    @classmethod
    def classmethod(cls):
        raise AbstractMethodError(cls, methodtype="classmethod")

    @property
    def property(self):
        raise AbstractMethodError(self, methodtype="property")

    def method(self):
        raise AbstractMethodError(self)


def test_AbstractMethodError_classmethod():
    xpr = "This classmethod must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo.classmethod()

    xpr = "This property must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().property

    xpr = "This method must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().method()
