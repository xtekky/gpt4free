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


cdef class StringBuilder(_Weakrefable):
    """
    Builder class for UTF8 strings.

    This class exposes facilities for incrementally adding string values and
    building the null bitmap for a pyarrow.Array (type='string').
    """
    cdef:
        unique_ptr[CStringBuilder] builder

    def __cinit__(self, MemoryPool memory_pool=None):
        cdef CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)
        self.builder.reset(new CStringBuilder(pool))

    def append(self, value):
        """
        Append a single value to the builder.

        The value can either be a string/bytes object or a null value
        (np.nan or None).

        Parameters
        ----------
        value : string/bytes or np.nan/None
            The value to append to the string array builder.
        """
        if value is None or value is np.nan:
            self.builder.get().AppendNull()
        elif isinstance(value, (bytes, str)):
            self.builder.get().Append(tobytes(value))
        else:
            raise TypeError('StringBuilder only accepts string objects')

    def append_values(self, values):
        """
        Append all the values from an iterable.

        Parameters
        ----------
        values : iterable of string/bytes or np.nan/None values
            The values to append to the string array builder.
        """
        for value in values:
            self.append(value)

    def finish(self):
        """
        Return result of builder as an Array object; also resets the builder.

        Returns
        -------
        array : pyarrow.Array
        """
        cdef shared_ptr[CArray] out
        with nogil:
            self.builder.get().Finish(&out)
        return pyarrow_wrap_array(out)

    @property
    def null_count(self):
        return self.builder.get().null_count()

    def __len__(self):
        return self.builder.get().length()
