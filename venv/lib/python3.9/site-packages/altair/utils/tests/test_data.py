import os

import pytest
import pandas as pd
from toolz import pipe

from ..data import limit_rows, MaxRowsError, sample, to_values, to_json, to_csv


def _create_dataframe(N):
    data = pd.DataFrame({"x": range(N), "y": range(N)})
    return data


def _create_data_with_values(N):
    data = {"values": [{"x": i, "y": i + 1} for i in range(N)]}
    return data


def test_limit_rows():
    """Test the limit_rows data transformer."""
    data = _create_dataframe(10)
    result = limit_rows(data, max_rows=20)
    assert data is result
    with pytest.raises(MaxRowsError):
        pipe(data, limit_rows(max_rows=5))
    data = _create_data_with_values(10)
    result = pipe(data, limit_rows(max_rows=20))
    assert data is result
    with pytest.raises(MaxRowsError):
        limit_rows(data, max_rows=5)


def test_sample():
    """Test the sample data transformer."""
    data = _create_dataframe(20)
    result = pipe(data, sample(n=10))
    assert len(result) == 10
    assert isinstance(result, pd.DataFrame)
    data = _create_data_with_values(20)
    result = sample(data, n=10)
    assert isinstance(result, dict)
    assert "values" in result
    assert len(result["values"]) == 10
    data = _create_dataframe(20)
    result = pipe(data, sample(frac=0.5))
    assert len(result) == 10
    assert isinstance(result, pd.DataFrame)
    data = _create_data_with_values(20)
    result = sample(data, frac=0.5)
    assert isinstance(result, dict)
    assert "values" in result
    assert len(result["values"]) == 10


def test_to_values():
    """Test the to_values data transformer."""
    data = _create_dataframe(10)
    result = pipe(data, to_values)
    assert result == {"values": data.to_dict(orient="records")}


def test_type_error():
    """Ensure that TypeError is raised for types other than dict/DataFrame."""
    for f in (sample, limit_rows, to_values):
        with pytest.raises(TypeError):
            pipe(0, f)


def test_dataframe_to_json():
    """Test to_json
    - make certain the filename is deterministic
    - make certain the file contents match the data
    """
    data = _create_dataframe(10)
    try:
        result1 = pipe(data, to_json)
        result2 = pipe(data, to_json)
        filename = result1["url"]
        output = pd.read_json(filename)
    finally:
        os.remove(filename)

    assert result1 == result2
    assert output.equals(data)


def test_dict_to_json():
    """Test to_json
    - make certain the filename is deterministic
    - make certain the file contents match the data
    """
    data = _create_data_with_values(10)
    try:
        result1 = pipe(data, to_json)
        result2 = pipe(data, to_json)
        filename = result1["url"]
        output = pd.read_json(filename).to_dict(orient="records")
    finally:
        os.remove(filename)

    assert result1 == result2
    assert data == {"values": output}


def test_dataframe_to_csv():
    """Test to_csv with dataframe input
    - make certain the filename is deterministic
    - make certain the file contents match the data
    """
    data = _create_dataframe(10)
    try:
        result1 = pipe(data, to_csv)
        result2 = pipe(data, to_csv)
        filename = result1["url"]
        output = pd.read_csv(filename)
    finally:
        os.remove(filename)

    assert result1 == result2
    assert output.equals(data)


def test_dict_to_csv():
    """Test to_csv with dict input
    - make certain the filename is deterministic
    - make certain the file contents match the data
    """
    data = _create_data_with_values(10)
    try:
        result1 = pipe(data, to_csv)
        result2 = pipe(data, to_csv)
        filename = result1["url"]
        output = pd.read_csv(filename).to_dict(orient="records")
    finally:
        os.remove(filename)

    assert result1 == result2
    assert data == {"values": output}
