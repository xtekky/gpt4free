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


import os

from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
                         concat_tables, schema)
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401


class FeatherDataset:
    """
    Encapsulates details of reading a list of Feather files.

    Parameters
    ----------
    path_or_paths : List[str]
        A list of file names
    validate_schema : bool, default True
        Check that individual file schemas are all the same / compatible
    """

    def __init__(self, path_or_paths, validate_schema=True):
        self.paths = path_or_paths
        self.validate_schema = validate_schema

    def read_table(self, columns=None):
        """
        Read multiple feather files as a single pyarrow.Table

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns)
        """
        _fil = read_table(self.paths[0], columns=columns)
        self._tables = [_fil]
        self.schema = _fil.schema

        for path in self.paths[1:]:
            table = read_table(path, columns=columns)
            if self.validate_schema:
                self.validate_schemas(path, table)
            self._tables.append(table)
        return concat_tables(self._tables)

    def validate_schemas(self, piece, table):
        if not self.schema.equals(table.schema):
            raise ValueError('Schema in {!s} was different. \n'
                             '{!s}\n\nvs\n\n{!s}'
                             .format(piece, self.schema,
                                     table.schema))

    def read_pandas(self, columns=None, use_threads=True):
        """
        Read multiple Parquet files as a single pandas DataFrame

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the file
        use_threads : bool, default True
            Use multiple threads when converting to pandas

        Returns
        -------
        pandas.DataFrame
            Content of the file as a pandas DataFrame (of columns)
        """
        return self.read_table(columns=columns).to_pandas(
            use_threads=use_threads)


def check_chunked_overflow(name, col):
    if col.num_chunks == 1:
        return

    if col.type in (ext.binary(), ext.string()):
        raise ValueError("Column '{}' exceeds 2GB maximum capacity of "
                         "a Feather binary column. This restriction may be "
                         "lifted in the future".format(name))
    else:
        # TODO(wesm): Not sure when else this might be reached
        raise ValueError("Column '{}' of type {} was chunked on conversion "
                         "to Arrow and cannot be currently written to "
                         "Feather format".format(name, str(col.type)))


_FEATHER_SUPPORTED_CODECS = {'lz4', 'zstd', 'uncompressed'}


def write_feather(df, dest, compression=None, compression_level=None,
                  chunksize=None, version=2):
    """
    Write a pandas.DataFrame to Feather format.

    Parameters
    ----------
    df : pandas.DataFrame or pyarrow.Table
        Data to write out as Feather format.
    dest : str
        Local destination path.
    compression : string, default None
        Can be one of {"zstd", "lz4", "uncompressed"}. The default of None uses
        LZ4 for V2 files if it is available, otherwise uncompressed.
    compression_level : int, default None
        Use a compression level particular to the chosen compressor. If None
        use the default compression level
    chunksize : int, default None
        For V2 files, the internal maximum size of Arrow RecordBatch chunks
        when writing the Arrow IPC file format. None means use the default,
        which is currently 64K
    version : int, default 2
        Feather file version. Version 2 is the current. Version 1 is the more
        limited legacy format
    """
    if _pandas_api.have_pandas:
        if (_pandas_api.has_sparse and
                isinstance(df, _pandas_api.pd.SparseDataFrame)):
            df = df.to_dense()

    if _pandas_api.is_data_frame(df):
        # Feather v1 creates a new column in the resultant Table to
        # store index information if index type is not RangeIndex

        if version == 1:
            preserve_index = False
        elif version == 2:
            preserve_index = None
        else:
            raise ValueError("Version value should either be 1 or 2")

        table = Table.from_pandas(df, preserve_index=preserve_index)

        if version == 1:
            # Version 1 does not chunking
            for i, name in enumerate(table.schema.names):
                col = table[i]
                check_chunked_overflow(name, col)
    else:
        table = df

    if version == 1:
        if len(table.column_names) > len(set(table.column_names)):
            raise ValueError("cannot serialize duplicate column names")

        if compression is not None:
            raise ValueError("Feather V1 files do not support compression "
                             "option")

        if chunksize is not None:
            raise ValueError("Feather V1 files do not support chunksize "
                             "option")
    else:
        if compression is None and Codec.is_available('lz4_frame'):
            compression = 'lz4'
        elif (compression is not None and
              compression not in _FEATHER_SUPPORTED_CODECS):
            raise ValueError('compression="{}" not supported, must be '
                             'one of {}'.format(compression,
                                                _FEATHER_SUPPORTED_CODECS))

    try:
        _feather.write_feather(table, dest, compression=compression,
                               compression_level=compression_level,
                               chunksize=chunksize, version=version)
    except Exception:
        if isinstance(dest, str):
            try:
                os.remove(dest)
            except os.error:
                pass
        raise


def read_feather(source, columns=None, use_threads=True,
                 memory_map=False, **kwargs):
    """
    Read a pandas.DataFrame from Feather format. To read as pyarrow.Table use
    feather.read_table.

    Parameters
    ----------
    source : str file path, or file-like object
        You can use MemoryMappedFile as source, for explicitly use memory map.
    columns : sequence, optional
        Only read a specific set of columns. If not provided, all columns are
        read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads. If false the
        restriction is used in the conversion to Pandas as well as in the
        reading from Feather format.
    memory_map : boolean, default False
        Use memory mapping when opening file on disk, when source is a str.
    **kwargs
        Additional keyword arguments passed on to `pyarrow.Table.to_pandas`.

    Returns
    -------
    df : pandas.DataFrame
        The contents of the Feather file as a pandas.DataFrame
    """
    return (read_table(
        source, columns=columns, memory_map=memory_map,
        use_threads=use_threads).to_pandas(use_threads=use_threads, **kwargs))


def read_table(source, columns=None, memory_map=False, use_threads=True):
    """
    Read a pyarrow.Table from Feather format

    Parameters
    ----------
    source : str file path, or file-like object
        You can use MemoryMappedFile as source, for explicitly use memory map.
    columns : sequence, optional
        Only read a specific set of columns. If not provided, all columns are
        read.
    memory_map : boolean, default False
        Use memory mapping when opening file on disk, when source is a str
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.

    Returns
    -------
    table : pyarrow.Table
        The contents of the Feather file as a pyarrow.Table
    """
    reader = _feather.FeatherReader(
        source, use_memory_map=memory_map, use_threads=use_threads)

    if columns is None:
        return reader.read()

    column_types = [type(column) for column in columns]
    if all(map(lambda t: t == int, column_types)):
        table = reader.read_indices(columns)
    elif all(map(lambda t: t == str, column_types)):
        table = reader.read_names(columns)
    else:
        column_type_names = [t.__name__ for t in column_types]
        raise TypeError("Columns must be indices or names. "
                        "Got columns {} of types {}"
                        .format(columns, column_type_names))

    # Feather v1 already respects the column selection
    if reader.version < 3:
        return table
    # Feather v2 reads with sorted / deduplicated selection
    elif sorted(set(columns)) == columns:
        return table
    else:
        # follow exact order / selection of names
        return table.select(columns)
