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

from __future__ import annotations
from typing import (
    Any,
    Iterable,
    Optional,
    Sequence,
)

import pyarrow as pa

from pyarrow.interchange.column import _PyArrowColumn


class _PyArrowDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    def __init__(
        self, df: pa.Table, nan_as_null: bool = False, allow_copy: bool = True
    ) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `pa.Table.__dataframe__`.
        """
        self._df = df
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with ``NaN`` (or
        # ``NaT``).
        if nan_as_null is True:
            raise RuntimeError(
                "nan_as_null=True currently has no effect, "
                "use the default nan_as_null=False"
            )
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame:
        """
        Construct a new exchange object, potentially changing the parameters.
        ``nan_as_null`` is a keyword intended for the consumer to tell the
        producer to overwrite null values in the data with ``NaN``.
        It is intended for cases where the consumer does not support the bit
        mask or byte mask that is the producer's native representation.
        ``allow_copy`` is a keyword that defines whether or not the library is
        allowed to make a copy of the data. For example, copying data would be
        necessary if a library supports strided buffers, given that this
        protocol specifies contiguous buffers.
        """
        return _PyArrowDataFrame(self._df, nan_as_null, allow_copy)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """
        # The metadata for the data frame, as a dictionary with string keys.
        # Add schema metadata here (pandas metadata or custom metadata)
        if self._df.schema.metadata:
            schema_metadata = {"pyarrow." + k.decode('utf8'): v.decode('utf8')
                               for k, v in self._df.schema.metadata.items()}
            return schema_metadata
        else:
            return {}

    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.
        """
        return self._df.num_columns

    def num_rows(self) -> int:
        """
        Return the number of rows in the DataFrame, if available.
        """
        return self._df.num_rows

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.
        """
        # pyarrow.Table can have columns with different number
        # of chunks so we take the number of chunks that
        # .to_batches() returns as it takes the min chunk size
        # of all the columns (to_batches is a zero copy method)
        batches = self._df.to_batches()
        return len(batches)

    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.
        """
        return self._df.column_names

    def get_column(self, i: int) -> _PyArrowColumn:
        """
        Return the column at the indicated position.
        """
        return _PyArrowColumn(self._df.column(i),
                              allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _PyArrowColumn:
        """
        Return the column whose name is the indicated name.
        """
        return _PyArrowColumn(self._df.column(name),
                              allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_PyArrowColumn]:
        """
        Return an iterator yielding the columns.
        """
        return [
            _PyArrowColumn(col, allow_copy=self._allow_copy)
            for col in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> _PyArrowDataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """
        return _PyArrowDataFrame(
            self._df.select(list(indices)), self._nan_as_null, self._allow_copy
        )

    def select_columns_by_name(
        self, names: Sequence[str]
    ) -> _PyArrowDataFrame:
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """
        return _PyArrowDataFrame(
            self._df.select(list(names)), self._nan_as_null, self._allow_copy
        )

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable[_PyArrowDataFrame]:
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.

        Note that the producer must ensure that all columns are chunked the
        same way.
        """
        if n_chunks and n_chunks > 1:
            chunk_size = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks != 0:
                chunk_size += 1
            batches = self._df.to_batches(max_chunksize=chunk_size)
            # In case when the size of the chunk is such that the resulting
            # list is one less chunk then n_chunks -> append an empty chunk
            if len(batches) == n_chunks - 1:
                batches.append(pa.record_batch([[]], schema=self._df.schema))
        else:
            batches = self._df.to_batches()

        iterator_tables = [_PyArrowDataFrame(
            pa.Table.from_batches([batch]), self._nan_as_null, self._allow_copy
        )
            for batch in batches
        ]
        return iterator_tables
