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

import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType

try:
    import pyarrow.dataset as ds
    import pyarrow._exec_plan as ep
except ImportError:
    pass

pytestmark = pytest.mark.dataset


def test_joins_corner_cases():
    t1 = pa.Table.from_pydict({
        "colA": [1, 2, 3, 4, 5, 6],
        "col2": ["a", "b", "c", "d", "e", "f"]
    })

    t2 = pa.Table.from_pydict({
        "colB": [1, 2, 3, 4, 5],
        "col3": ["A", "B", "C", "D", "E"]
    })

    with pytest.raises(pa.ArrowInvalid):
        ep._perform_join("left outer", t1, "", t2, "")

    with pytest.raises(TypeError):
        ep._perform_join("left outer", None, "colA", t2, "colB")

    with pytest.raises(ValueError):
        ep._perform_join("super mario join", t1, "colA", t2, "colB")


@pytest.mark.parametrize("jointype,expected", [
    ("left semi", {
        "colA": [1, 2],
        "col2": ["a", "b"]
    }),
    ("right semi", {
        "colB": [1, 2],
        "col3": ["A", "B"]
    }),
    ("left anti", {
        "colA": [6],
        "col2": ["f"]
    }),
    ("right anti", {
        "colB": [99],
        "col3": ["Z"]
    }),
    ("inner", {
        "colA": [1, 2],
        "col2": ["a", "b"],
        "col3": ["A", "B"]
    }),
    ("left outer", {
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"],
        "col3": ["A", "B", None]
    }),
    ("right outer", {
        "col2": ["a", "b", None],
        "colB": [1, 2, 99],
        "col3": ["A", "B", "Z"]
    }),
    ("full outer", {
        "colA": [1, 2, 6, 99],
        "col2": ["a", "b", "f", None],
        "col3": ["A", "B", None, "Z"]
    })
])
@pytest.mark.parametrize("use_threads", [True, False])
@pytest.mark.parametrize("use_datasets", [False, True])
def test_joins(jointype, expected, use_threads, use_datasets):
    # Allocate table here instead of using parametrize
    # this prevents having arrow allocated memory forever around.
    expected = pa.table(expected)

    t1 = pa.Table.from_pydict({
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"]
    })

    t2 = pa.Table.from_pydict({
        "colB": [99, 2, 1],
        "col3": ["Z", "B", "A"]
    })

    if use_datasets:
        t1 = ds.dataset([t1])
        t2 = ds.dataset([t2])

    r = ep._perform_join(jointype, t1, "colA", t2, "colB",
                         use_threads=use_threads, coalesce_keys=True)
    r = r.combine_chunks()
    if "right" in jointype:
        r = r.sort_by("colB")
    else:
        r = r.sort_by("colA")
    assert r == expected


def test_table_join_collisions():
    t1 = pa.table({
        "colA": [1, 2, 6],
        "colB": [10, 20, 60],
        "colVals": ["a", "b", "f"]
    })

    t2 = pa.table({
        "colB": [99, 20, 10],
        "colVals": ["Z", "B", "A"],
        "colUniq": [100, 200, 300],
        "colA": [99, 2, 1],
    })

    result = ep._perform_join(
        "full outer", t1, ["colA", "colB"], t2, ["colA", "colB"])
    result = result.combine_chunks()
    result = result.sort_by("colUniq")
    assert result == pa.table([
        [None, 2, 1, 6],
        [None, 20, 10, 60],
        [None, "b", "a", "f"],
        [99, 20, 10, None],
        ["Z", "B", "A", None],
        [100, 200, 300, None],
        [99, 2, 1, None],
    ], names=["colA", "colB", "colVals", "colB", "colVals", "colUniq", "colA"])

    result = ep._perform_join("full outer", t1, "colA",
                              t2, "colA", right_suffix="_r",
                              coalesce_keys=False)
    result = result.combine_chunks()
    result = result.sort_by("colA")
    assert result == pa.table({
        "colA": [1, 2, 6, None],
        "colB": [10, 20, 60, None],
        "colVals": ["a", "b", "f", None],
        "colB_r": [10, 20, None, 99],
        "colVals_r": ["A", "B", None, "Z"],
        "colUniq": [300, 200, None, 100],
        "colA_r": [1, 2, None, 99],
    })

    result = ep._perform_join("full outer", t1, "colA",
                              t2, "colA", right_suffix="_r",
                              coalesce_keys=True)
    result = result.combine_chunks()
    result = result.sort_by("colA")
    assert result == pa.table({
        "colA": [1, 2, 6, 99],
        "colB": [10, 20, 60, None],
        "colVals": ["a", "b", "f", None],
        "colB_r": [10, 20, None, 99],
        "colVals_r": ["A", "B", None, "Z"],
        "colUniq": [300, 200, None, 100]
    })


def test_table_join_keys_order():
    t1 = pa.table({
        "colB": [10, 20, 60],
        "colA": [1, 2, 6],
        "colVals": ["a", "b", "f"]
    })

    t2 = pa.table({
        "colVals": ["Z", "B", "A"],
        "colX": [99, 2, 1],
    })

    result = ep._perform_join("full outer", t1, "colA", t2, "colX",
                              left_suffix="_l", right_suffix="_r",
                              coalesce_keys=True)
    result = result.combine_chunks()
    result = result.sort_by("colA")
    assert result == pa.table({
        "colB": [10, 20, 60, None],
        "colA": [1, 2, 6, 99],
        "colVals_l": ["a", "b", "f", None],
        "colVals_r": ["A", "B", None, "Z"],
    })


def test_filter_table_errors():
    t = pa.table({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50]
    })

    with pytest.raises(pa.ArrowTypeError):
        ep._filter_table(
            t, pc.divide(pc.field("a"), pc.scalar(2)),
            output_type=pa.Table
        )

    with pytest.raises(pa.ArrowInvalid):
        ep._filter_table(
            t, (pc.field("Z") <= pc.scalar(2)),
            output_type=pa.Table
        )


@pytest.mark.parametrize("use_datasets", [False, True])
def test_filter_table(use_datasets):
    t = pa.table({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50]
    })
    if use_datasets:
        t = ds.dataset([t])

    result = ep._filter_table(
        t, (pc.field("a") <= pc.scalar(3)) & (pc.field("b") == pc.scalar(20)),
        output_type=pa.Table if not use_datasets else ds.InMemoryDataset
    )
    if use_datasets:
        result = result.to_table()
    assert result == pa.table({
        "a": [2],
        "b": [20]
    })

    result = ep._filter_table(
        t, pc.field("b") > pc.scalar(30),
        output_type=pa.Table if not use_datasets else ds.InMemoryDataset
    )
    if use_datasets:
        result = result.to_table()
    assert result == pa.table({
        "a": [4, 5],
        "b": [40, 50]
    })


def test_filter_table_ordering():
    table1 = pa.table({'a': [1, 2, 3, 4], 'b': ['a'] * 4})
    table2 = pa.table({'a': [1, 2, 3, 4], 'b': ['b'] * 4})
    table = pa.concat_tables([table1, table2])

    for _ in range(20):
        # 20 seems to consistently cause errors when order is not preserved.
        # If the order problem is reintroduced this test will become flaky
        # which is still a signal that the order is not preserved.
        r = ep._filter_table(table, pc.field('a') == 1)
        assert r["b"] == pa.chunked_array([["a"], ["b"]])


def test_complex_filter_table():
    t = pa.table({
        "a": [1, 2, 3, 4, 5, 6, 6],
        "b": [10, 20, 30, 40, 50, 60, 61]
    })

    result = ep._filter_table(
        t, ((pc.bit_wise_and(pc.field("a"), pc.scalar(1)) == pc.scalar(0)) &
            (pc.multiply(pc.field("a"), pc.scalar(10)) == pc.field("b")))
    )

    assert result == pa.table({
        "a": [2, 4, 6],  # second six must be omitted because 6*10 != 61
        "b": [20, 40, 60]
    })


def test_join_extension_array_column():
    storage = pa.array([1, 2, 3], type=pa.int64())
    ty = IntegerType()
    ext_array = pa.ExtensionArray.from_storage(ty, storage)
    dict_array = pa.DictionaryArray.from_arrays(
        pa.array([0, 2, 1]), pa.array(['a', 'b', 'c']))
    t1 = pa.table({
        "colA": [1, 2, 6],
        "colB": ext_array,
        "colVals": ext_array,
    })

    t2 = pa.table({
        "colA": [99, 2, 1],
        "colC": ext_array,
    })

    t3 = pa.table({
        "colA": [99, 2, 1],
        "colC": ext_array,
        "colD": dict_array,
    })

    result = ep._perform_join(
        "left outer", t1, ["colA"], t2, ["colA"])
    assert result["colVals"] == pa.chunked_array(ext_array)

    result = ep._perform_join(
        "left outer", t1, ["colB"], t2, ["colC"])
    assert result["colB"] == pa.chunked_array(ext_array)

    result = ep._perform_join(
        "left outer", t1, ["colA"], t3, ["colA"])
    assert result["colVals"] == pa.chunked_array(ext_array)

    result = ep._perform_join(
        "left outer", t1, ["colB"], t3, ["colC"])
    assert result["colB"] == pa.chunked_array(ext_array)
