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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True


cdef class MemoryPool(_Weakrefable):
    """
    Base class for memory allocation.

    Besides tracking its number of allocated bytes, a memory pool also
    takes care of the required 64-byte alignment for Arrow data.
    """

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, "
                        "use pyarrow.*_memory_pool instead."
                        .format(self.__class__.__name__))

    cdef void init(self, CMemoryPool* pool):
        self.pool = pool

    def release_unused(self):
        """
        Attempt to return to the OS any memory being held onto by the pool.

        This function should not be called except potentially for
        benchmarking or debugging as it could be expensive and detrimental to
        performance.

        This is best effort and may not have any effect on some memory pools
        or in some situations (e.g. fragmentation).
        """
        cdef CMemoryPool* pool = c_get_memory_pool()
        with nogil:
            pool.ReleaseUnused()

    def bytes_allocated(self):
        """
        Return the number of bytes that are currently allocated from this
        memory pool.
        """
        return self.pool.bytes_allocated()

    def max_memory(self):
        """
        Return the peak memory allocation in this memory pool.
        This can be an approximate number in multi-threaded applications.

        None is returned if the pool implementation doesn't know how to
        compute this number.
        """
        ret = self.pool.max_memory()
        return ret if ret >= 0 else None

    @property
    def backend_name(self):
        """
        The name of the backend used by this MemoryPool (e.g. "jemalloc").
        """
        return frombytes(self.pool.backend_name())

    def __repr__(self):
        name = f"pyarrow.{self.__class__.__name__}"
        return (f"<{name} "
                f"backend_name={self.backend_name} "
                f"bytes_allocated={self.bytes_allocated()} "
                f"max_memory={self.max_memory()}>")

cdef CMemoryPool* maybe_unbox_memory_pool(MemoryPool memory_pool):
    if memory_pool is None:
        return c_get_memory_pool()
    else:
        return memory_pool.pool


cdef api object box_memory_pool(CMemoryPool *c_pool):
    cdef MemoryPool pool = MemoryPool.__new__(MemoryPool)
    pool.init(c_pool)
    return pool


cdef class LoggingMemoryPool(MemoryPool):
    cdef:
        unique_ptr[CLoggingMemoryPool] logging_pool

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, "
                        "use pyarrow.logging_memory_pool instead."
                        .format(self.__class__.__name__))


cdef class ProxyMemoryPool(MemoryPool):
    """
    Memory pool implementation that tracks the number of bytes and
    maximum memory allocated through its direct calls, while redirecting
    to another memory pool.
    """
    cdef:
        unique_ptr[CProxyMemoryPool] proxy_pool

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, "
                        "use pyarrow.proxy_memory_pool instead."
                        .format(self.__class__.__name__))


def default_memory_pool():
    """
    Return the process-global memory pool.

    Examples
    --------
    >>> default_memory_pool()
    <pyarrow.MemoryPool backend_name=... bytes_allocated=0 max_memory=...>
    """
    cdef:
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    pool.init(c_get_memory_pool())
    return pool


def proxy_memory_pool(MemoryPool parent):
    """
    Create and return a MemoryPool instance that redirects to the
    *parent*, but with separate allocation statistics.

    Parameters
    ----------
    parent : MemoryPool
        The real memory pool that should be used for allocations.
    """
    cdef ProxyMemoryPool out = ProxyMemoryPool.__new__(ProxyMemoryPool)
    out.proxy_pool.reset(new CProxyMemoryPool(parent.pool))
    out.init(out.proxy_pool.get())
    return out


def logging_memory_pool(MemoryPool parent):
    """
    Create and return a MemoryPool instance that redirects to the
    *parent*, but also dumps allocation logs on stderr.

    Parameters
    ----------
    parent : MemoryPool
        The real memory pool that should be used for allocations.
    """
    cdef LoggingMemoryPool out = LoggingMemoryPool.__new__(
        LoggingMemoryPool, parent)
    out.logging_pool.reset(new CLoggingMemoryPool(parent.pool))
    out.init(out.logging_pool.get())
    return out


def system_memory_pool():
    """
    Return a memory pool based on the C malloc heap.
    """
    cdef:
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    pool.init(c_system_memory_pool())
    return pool


def jemalloc_memory_pool():
    """
    Return a memory pool based on the jemalloc heap.

    NotImplementedError is raised if jemalloc support is not enabled.
    """
    cdef:
        CMemoryPool* c_pool
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    check_status(c_jemalloc_memory_pool(&c_pool))
    pool.init(c_pool)
    return pool


def mimalloc_memory_pool():
    """
    Return a memory pool based on the mimalloc heap.

    NotImplementedError is raised if mimalloc support is not enabled.
    """
    cdef:
        CMemoryPool* c_pool
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    check_status(c_mimalloc_memory_pool(&c_pool))
    pool.init(c_pool)
    return pool


def set_memory_pool(MemoryPool pool):
    """
    Set the default memory pool.

    Parameters
    ----------
    pool : MemoryPool
        The memory pool that should be used by default.
    """
    c_set_default_memory_pool(pool.pool)


cdef MemoryPool _default_memory_pool = default_memory_pool()
cdef LoggingMemoryPool _logging_memory_pool = logging_memory_pool(
    _default_memory_pool)


def log_memory_allocations(enable=True):
    """
    Enable or disable memory allocator logging for debugging purposes

    Parameters
    ----------
    enable : bool, default True
        Pass False to disable logging
    """
    if enable:
        set_memory_pool(_logging_memory_pool)
    else:
        set_memory_pool(_default_memory_pool)


def total_allocated_bytes():
    """
    Return the currently allocated bytes from the default memory pool.
    Other memory pools may not be accounted for.
    """
    cdef CMemoryPool* pool = c_get_memory_pool()
    return pool.bytes_allocated()


def jemalloc_set_decay_ms(decay_ms):
    """
    Set arenas.dirty_decay_ms and arenas.muzzy_decay_ms to indicated number of
    milliseconds. A value of 0 (the default) results in dirty / muzzy memory
    pages being released right away to the OS, while a higher value will result
    in a time-based decay. See the jemalloc docs for more information

    It's best to set this at the start of your application.

    Parameters
    ----------
    decay_ms : int
        Number of milliseconds to set for jemalloc decay conf parameters. Note
        that this change will only affect future memory arenas
    """
    check_status(c_jemalloc_set_decay_ms(decay_ms))


def supported_memory_backends():
    """
    Return a list of available memory pool backends
    """
    cdef vector[c_string] backends = c_supported_memory_backends()
    return [backend.decode() for backend in backends]
