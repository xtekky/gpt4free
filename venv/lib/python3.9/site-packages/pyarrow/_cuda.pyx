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


from pyarrow.lib import tobytes
from pyarrow.lib cimport *
from pyarrow.includes.libarrow_cuda cimport *
from pyarrow.lib import py_buffer, allocate_buffer, as_buffer, ArrowTypeError
from pyarrow.util import get_contiguous_span
cimport cpython as cp


cdef class Context(_Weakrefable):
    """
    CUDA driver context.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a CUDA driver context for a particular device.

        If a CUDA context handle is passed, it is wrapped, otherwise
        a default CUDA context for the given device is requested.

        Parameters
        ----------
        device_number : int (default 0)
          Specify the GPU device for which the CUDA driver context is
          requested.
        handle : int, optional
          Specify CUDA handle for a shared context that has been created
          by another library.
        """
        # This method exposed because autodoc doesn't pick __cinit__

    def __cinit__(self, int device_number=0, uintptr_t handle=0):
        cdef CCudaDeviceManager* manager
        manager = GetResultValue(CCudaDeviceManager.Instance())
        cdef int n = manager.num_devices()
        if device_number >= n or device_number < 0:
            self.context.reset()
            raise ValueError('device_number argument must be '
                             'non-negative less than %s' % (n))
        if handle == 0:
            self.context = GetResultValue(manager.GetContext(device_number))
        else:
            self.context = GetResultValue(manager.GetSharedContext(
                device_number, <void*>handle))
        self.device_number = device_number

    @staticmethod
    def from_numba(context=None):
        """
        Create a Context instance from a Numba CUDA context.

        Parameters
        ----------
        context : {numba.cuda.cudadrv.driver.Context, None}
          A Numba CUDA context instance.
          If None, the current Numba context is used.

        Returns
        -------
        shared_context : pyarrow.cuda.Context
          Context instance.
        """
        if context is None:
            import numba.cuda
            context = numba.cuda.current_context()
        return Context(device_number=context.device.id,
                       handle=context.handle.value)

    def to_numba(self):
        """
        Convert Context to a Numba CUDA context.

        Returns
        -------
        context : numba.cuda.cudadrv.driver.Context
          Numba CUDA context instance.
        """
        import ctypes
        import numba.cuda
        device = numba.cuda.gpus[self.device_number]
        handle = ctypes.c_void_p(self.handle)
        context = numba.cuda.cudadrv.driver.Context(device, handle)

        class DummyPendingDeallocs(object):
            # Context is managed by pyarrow
            def add_item(self, *args, **kwargs):
                pass

        context.deallocations = DummyPendingDeallocs()
        return context

    @staticmethod
    def get_num_devices():
        """ Return the number of GPU devices.
        """
        cdef CCudaDeviceManager* manager
        manager = GetResultValue(CCudaDeviceManager.Instance())
        return manager.num_devices()

    @property
    def device_number(self):
        """ Return context device number.
        """
        return self.device_number

    @property
    def handle(self):
        """ Return pointer to context handle.
        """
        return <uintptr_t>self.context.get().handle()

    cdef void init(self, const shared_ptr[CCudaContext]& ctx):
        self.context = ctx

    def synchronize(self):
        """Blocks until the device has completed all preceding requested
        tasks.
        """
        check_status(self.context.get().Synchronize())

    @property
    def bytes_allocated(self):
        """Return the number of allocated bytes.
        """
        return self.context.get().bytes_allocated()

    def get_device_address(self, uintptr_t address):
        """Return the device address that is reachable from kernels running in
        the context

        Parameters
        ----------
        address : int
          Specify memory address value

        Returns
        -------
        device_address : int
          Device address accessible from device context

        Notes
        -----
        The device address is defined as a memory address accessible
        by device. While it is often a device memory address but it
        can be also a host memory address, for instance, when the
        memory is allocated as host memory (using cudaMallocHost or
        cudaHostAlloc) or as managed memory (using cudaMallocManaged)
        or the host memory is page-locked (using cudaHostRegister).
        """
        return GetResultValue(self.context.get().GetDeviceAddress(address))

    def new_buffer(self, int64_t nbytes):
        """Return new device buffer.

        Parameters
        ----------
        nbytes : int
          Specify the number of bytes to be allocated.

        Returns
        -------
        buf : CudaBuffer
          Allocated buffer.
        """
        cdef:
            shared_ptr[CCudaBuffer] cudabuf
        with nogil:
            cudabuf = GetResultValue(self.context.get().Allocate(nbytes))
        return pyarrow_wrap_cudabuffer(cudabuf)

    def foreign_buffer(self, address, size, base=None):
        """
        Create device buffer from address and size as a view.

        The caller is responsible for allocating and freeing the
        memory. When `address==size==0` then a new zero-sized buffer
        is returned.

        Parameters
        ----------
        address : int
          Specify the starting address of the buffer. The address can
          refer to both device or host memory but it must be
          accessible from device after mapping it with
          `get_device_address` method.
        size : int
          Specify the size of device buffer in bytes.
        base : {None, object}
          Specify object that owns the referenced memory.

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of device reachable memory.

        """
        if not address and size == 0:
            return self.new_buffer(0)
        cdef:
            uintptr_t c_addr = self.get_device_address(address)
            int64_t c_size = size
            shared_ptr[CCudaBuffer] cudabuf

        cudabuf = GetResultValue(self.context.get().View(
            <uint8_t*>c_addr, c_size))
        return pyarrow_wrap_cudabuffer_base(cudabuf, base)

    def open_ipc_buffer(self, ipc_handle):
        """ Open existing CUDA IPC memory handle

        Parameters
        ----------
        ipc_handle : IpcMemHandle
          Specify opaque pointer to CUipcMemHandle (driver API).

        Returns
        -------
        buf : CudaBuffer
          referencing device buffer
        """
        handle = pyarrow_unwrap_cudaipcmemhandle(ipc_handle)
        cdef shared_ptr[CCudaBuffer] cudabuf
        with nogil:
            cudabuf = GetResultValue(
                self.context.get().OpenIpcBuffer(handle.get()[0]))
        return pyarrow_wrap_cudabuffer(cudabuf)

    def buffer_from_data(self, object data, int64_t offset=0, int64_t size=-1):
        """Create device buffer and initialize with data.

        Parameters
        ----------
        data : {CudaBuffer, HostBuffer, Buffer, array-like}
          Specify data to be copied to device buffer.
        offset : int
          Specify the offset of input buffer for device data
          buffering. Default: 0.
        size : int
          Specify the size of device buffer in bytes. Default: all
          (starting from input offset)

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer with copied data.
        """
        is_host_data = not pyarrow_is_cudabuffer(data)
        buf = as_buffer(data) if is_host_data else data

        bsize = buf.size
        if offset < 0 or (bsize and offset >= bsize):
            raise ValueError('offset argument is out-of-range')
        if size < 0:
            size = bsize - offset
        elif offset + size > bsize:
            raise ValueError(
                'requested larger slice than available in device buffer')

        if offset != 0 or size != bsize:
            buf = buf.slice(offset, size)

        result = self.new_buffer(size)
        if is_host_data:
            result.copy_from_host(buf, position=0, nbytes=size)
        else:
            result.copy_from_device(buf, position=0, nbytes=size)
        return result

    def buffer_from_object(self, obj):
        """Create device buffer view of arbitrary object that references
        device accessible memory.

        When the object contains a non-contiguous view of device
        accessible memory then the returned device buffer will contain
        contiguous view of the memory, that is, including the
        intermediate data that is otherwise invisible to the input
        object.

        Parameters
        ----------
        obj : {object, Buffer, HostBuffer, CudaBuffer, ...}
          Specify an object that holds (device or host) address that
          can be accessed from device. This includes objects with
          types defined in pyarrow.cuda as well as arbitrary objects
          that implement the CUDA array interface as defined by numba.

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of device accessible memory.

        """
        if isinstance(obj, HostBuffer):
            return self.foreign_buffer(obj.address, obj.size, base=obj)
        elif isinstance(obj, Buffer):
            return CudaBuffer.from_buffer(obj)
        elif isinstance(obj, CudaBuffer):
            return obj
        elif hasattr(obj, '__cuda_array_interface__'):
            desc = obj.__cuda_array_interface__
            addr = desc['data'][0]
            if addr is None:
                return self.new_buffer(0)
            import numpy as np
            start, end = get_contiguous_span(
                desc['shape'], desc.get('strides'),
                np.dtype(desc['typestr']).itemsize)
            return self.foreign_buffer(addr + start, end - start, base=obj)
        raise ArrowTypeError('cannot create device buffer view from'
                             ' `%s` object' % (type(obj)))


cdef class IpcMemHandle(_Weakrefable):
    """A serializable container for a CUDA IPC handle.
    """
    cdef void init(self, shared_ptr[CCudaIpcMemHandle]& h):
        self.handle = h

    @staticmethod
    def from_buffer(Buffer opaque_handle):
        """Create IpcMemHandle from opaque buffer (e.g. from another
        process)

        Parameters
        ----------
        opaque_handle :
          a CUipcMemHandle as a const void*

        Returns
        -------
        ipc_handle : IpcMemHandle
        """
        c_buf = pyarrow_unwrap_buffer(opaque_handle)
        cdef:
            shared_ptr[CCudaIpcMemHandle] handle

        handle = GetResultValue(
            CCudaIpcMemHandle.FromBuffer(c_buf.get().data()))
        return pyarrow_wrap_cudaipcmemhandle(handle)

    def serialize(self, pool=None):
        """Write IpcMemHandle to a Buffer

        Parameters
        ----------
        pool : {MemoryPool, None}
          Specify a pool to allocate memory from

        Returns
        -------
        buf : Buffer
          The serialized buffer.
        """
        cdef CMemoryPool* pool_ = maybe_unbox_memory_pool(pool)
        cdef shared_ptr[CBuffer] buf
        cdef CCudaIpcMemHandle* h = self.handle.get()
        with nogil:
            buf = GetResultValue(h.Serialize(pool_))
        return pyarrow_wrap_buffer(buf)


cdef class CudaBuffer(Buffer):
    """An Arrow buffer with data located in a GPU device.

    To create a CudaBuffer instance, use Context.device_buffer().

    The memory allocated in a CudaBuffer is freed when the buffer object
    is deleted.
    """

    def __init__(self):
        raise TypeError("Do not call CudaBuffer's constructor directly, use "
                        "`<pyarrow.Context instance>.device_buffer`"
                        " method instead.")

    cdef void init_cuda(self,
                        const shared_ptr[CCudaBuffer]& buffer,
                        object base):
        self.cuda_buffer = buffer
        self.init(<shared_ptr[CBuffer]> buffer)
        self.base = base

    @staticmethod
    def from_buffer(buf):
        """ Convert back generic buffer into CudaBuffer

        Parameters
        ----------
        buf : Buffer
          Specify buffer containing CudaBuffer

        Returns
        -------
        dbuf : CudaBuffer
          Resulting device buffer.
        """
        c_buf = pyarrow_unwrap_buffer(buf)
        cuda_buffer = GetResultValue(CCudaBuffer.FromBuffer(c_buf))
        return pyarrow_wrap_cudabuffer(cuda_buffer)

    @staticmethod
    def from_numba(mem):
        """Create a CudaBuffer view from numba MemoryPointer instance.

        Parameters
        ----------
        mem :  numba.cuda.cudadrv.driver.MemoryPointer

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of numba MemoryPointer.
        """
        ctx = Context.from_numba(mem.context)
        if mem.device_pointer.value is None and mem.size==0:
            return ctx.new_buffer(0)
        return ctx.foreign_buffer(mem.device_pointer.value, mem.size, base=mem)

    def to_numba(self):
        """Return numba memory pointer of CudaBuffer instance.
        """
        import ctypes
        from numba.cuda.cudadrv.driver import MemoryPointer
        return MemoryPointer(self.context.to_numba(),
                             pointer=ctypes.c_void_p(self.address),
                             size=self.size)

    cdef getitem(self, int64_t i):
        return self.copy_to_host(position=i, nbytes=1)[0]

    def copy_to_host(self, int64_t position=0, int64_t nbytes=-1,
                     Buffer buf=None,
                     MemoryPool memory_pool=None, c_bool resizable=False):
        """Copy memory from GPU device to CPU host

        Caller is responsible for ensuring that all tasks affecting
        the memory are finished. Use

          `<CudaBuffer instance>.context.synchronize()`

        when needed.

        Parameters
        ----------
        position : int
          Specify the starting position of the source data in GPU
          device buffer. Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          the position until host buffer is full).
        buf : Buffer
          Specify a pre-allocated output buffer in host. Default: None
          (allocate new output buffer).
        memory_pool : MemoryPool
        resizable : bool
          Specify extra arguments to allocate_buffer. Used only when
          buf is None.

        Returns
        -------
        buf : Buffer
          Output buffer in host.

        """
        if position < 0 or (self.size and position > self.size) \
           or (self.size == 0 and position != 0):
            raise ValueError('position argument is out-of-range')
        cdef:
            int64_t c_nbytes
        if buf is None:
            if nbytes < 0:
                # copy all starting from position to new host buffer
                c_nbytes = self.size - position
            else:
                if nbytes > self.size - position:
                    raise ValueError(
                        'requested more to copy than available from '
                        'device buffer')
                # copy nbytes starting from position to new host buffeer
                c_nbytes = nbytes
            buf = allocate_buffer(c_nbytes, memory_pool=memory_pool,
                                  resizable=resizable)
        else:
            if nbytes < 0:
                # copy all from position until given host buffer is full
                c_nbytes = min(self.size - position, buf.size)
            else:
                if nbytes > buf.size:
                    raise ValueError(
                        'requested copy does not fit into host buffer')
                # copy nbytes from position to given host buffer
                c_nbytes = nbytes

        cdef:
            shared_ptr[CBuffer] c_buf = pyarrow_unwrap_buffer(buf)
            int64_t c_position = position
        with nogil:
            check_status(self.cuda_buffer.get()
                         .CopyToHost(c_position, c_nbytes,
                                     c_buf.get().mutable_data()))
        return buf

    def copy_from_host(self, data, int64_t position=0, int64_t nbytes=-1):
        """Copy data from host to device.

        The device buffer must be pre-allocated.

        Parameters
        ----------
        data : {Buffer, array-like}
          Specify data in host. It can be array-like that is valid
          argument to py_buffer
        position : int
          Specify the starting position of the copy in device buffer.
          Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          source until device buffer, starting from position, is full)

        Returns
        -------
        nbytes : int
          Number of bytes copied.
        """
        if position < 0 or position > self.size:
            raise ValueError('position argument is out-of-range')
        cdef:
            int64_t c_nbytes
        buf = as_buffer(data)

        if nbytes < 0:
            # copy from host buffer to device buffer starting from
            # position until device buffer is full
            c_nbytes = min(self.size - position, buf.size)
        else:
            if nbytes > buf.size:
                raise ValueError(
                    'requested more to copy than available from host buffer')
            if nbytes > self.size - position:
                raise ValueError(
                    'requested more to copy than available in device buffer')
            # copy nbytes from host buffer to device buffer starting
            # from position
            c_nbytes = nbytes

        cdef:
            shared_ptr[CBuffer] c_buf = pyarrow_unwrap_buffer(buf)
            int64_t c_position = position
        with nogil:
            check_status(self.cuda_buffer.get().
                         CopyFromHost(c_position, c_buf.get().data(),
                                      c_nbytes))
        return c_nbytes

    def copy_from_device(self, buf, int64_t position=0, int64_t nbytes=-1):
        """Copy data from device to device.

        Parameters
        ----------
        buf : CudaBuffer
          Specify source device buffer.
        position : int
          Specify the starting position of the copy in device buffer.
          Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          source until device buffer, starting from position, is full)

        Returns
        -------
        nbytes : int
          Number of bytes copied.

        """
        if position < 0 or position > self.size:
            raise ValueError('position argument is out-of-range')
        cdef:
            int64_t c_nbytes

        if nbytes < 0:
            # copy from source device buffer to device buffer starting
            # from position until device buffer is full
            c_nbytes = min(self.size - position, buf.size)
        else:
            if nbytes > buf.size:
                raise ValueError(
                    'requested more to copy than available from device buffer')
            if nbytes > self.size - position:
                raise ValueError(
                    'requested more to copy than available in device buffer')
            # copy nbytes from source device buffer to device buffer
            # starting from position
            c_nbytes = nbytes

        cdef:
            shared_ptr[CCudaBuffer] c_buf = pyarrow_unwrap_cudabuffer(buf)
            int64_t c_position = position
            shared_ptr[CCudaContext] c_src_ctx = pyarrow_unwrap_cudacontext(
                buf.context)
            void* c_source_data = <void*>(c_buf.get().address())

        if self.context.handle != buf.context.handle:
            with nogil:
                check_status(self.cuda_buffer.get().
                             CopyFromAnotherDevice(c_src_ctx, c_position,
                                                   c_source_data, c_nbytes))
        else:
            with nogil:
                check_status(self.cuda_buffer.get().
                             CopyFromDevice(c_position, c_source_data,
                                            c_nbytes))
        return c_nbytes

    def export_for_ipc(self):
        """
        Expose this device buffer as IPC memory which can be used in other
        processes.

        After calling this function, this device memory will not be
        freed when the CudaBuffer is destructed.

        Returns
        -------
        ipc_handle : IpcMemHandle
          The exported IPC handle

        """
        cdef shared_ptr[CCudaIpcMemHandle] handle
        with nogil:
            handle = GetResultValue(self.cuda_buffer.get().ExportForIpc())
        return pyarrow_wrap_cudaipcmemhandle(handle)

    @property
    def context(self):
        """Returns the CUDA driver context of this buffer.
        """
        return pyarrow_wrap_cudacontext(self.cuda_buffer.get().context())

    def slice(self, offset=0, length=None):
        """Return slice of device buffer

        Parameters
        ----------
        offset : int, default 0
          Specify offset from the start of device buffer to slice
        length : int, default None
          Specify the length of slice (default is until end of device
          buffer starting from offset). If the length is larger than
          the data available, the returned slice will have a size of
          the available data starting from the offset.

        Returns
        -------
        sliced : CudaBuffer
          Zero-copy slice of device buffer.

        """
        if offset < 0 or (self.size and offset >= self.size):
            raise ValueError('offset argument is out-of-range')
        cdef int64_t offset_ = offset
        cdef int64_t size
        if length is None:
            size = self.size - offset_
        elif offset + length <= self.size:
            size = length
        else:
            size = self.size - offset
        parent = pyarrow_unwrap_cudabuffer(self)
        return pyarrow_wrap_cudabuffer(make_shared[CCudaBuffer](parent,
                                                                offset_, size))

    def to_pybytes(self):
        """Return device buffer content as Python bytes.
        """
        return self.copy_to_host().to_pybytes()

    def __getbuffer__(self, cp.Py_buffer* buffer, int flags):
        # Device buffer contains data pointers on the device. Hence,
        # cannot support buffer protocol PEP-3118 for CudaBuffer.
        raise BufferError('buffer protocol for device buffer not supported')


cdef class HostBuffer(Buffer):
    """Device-accessible CPU memory created using cudaHostAlloc.

    To create a HostBuffer instance, use

      cuda.new_host_buffer(<nbytes>)
    """

    def __init__(self):
        raise TypeError("Do not call HostBuffer's constructor directly,"
                        " use `cuda.new_host_buffer` function instead.")

    cdef void init_host(self, const shared_ptr[CCudaHostBuffer]& buffer):
        self.host_buffer = buffer
        self.init(<shared_ptr[CBuffer]> buffer)

    @property
    def size(self):
        return self.host_buffer.get().size()


cdef class BufferReader(NativeFile):
    """File interface for zero-copy read from CUDA buffers.

    Note: Read methods return pointers to device memory. This means
    you must be careful using this interface with any Arrow code which
    may expect to be able to do anything other than pointer arithmetic
    on the returned buffers.
    """

    def __cinit__(self, CudaBuffer obj):
        self.buffer = obj
        self.reader = new CCudaBufferReader(self.buffer.buffer)
        self.set_random_access_file(
            shared_ptr[CRandomAccessFile](self.reader))
        self.is_readable = True

    def read_buffer(self, nbytes=None):
        """Return a slice view of the underlying device buffer.

        The slice will start at the current reader position and will
        have specified size in bytes.

        Parameters
        ----------
        nbytes : int, default None
          Specify the number of bytes to read. Default: None (read all
          remaining bytes).

        Returns
        -------
        cbuf : CudaBuffer
          New device buffer.

        """
        cdef:
            int64_t c_nbytes
            int64_t bytes_read = 0
            shared_ptr[CCudaBuffer] output

        if nbytes is None:
            c_nbytes = self.size() - self.tell()
        else:
            c_nbytes = nbytes

        with nogil:
            output = static_pointer_cast[CCudaBuffer, CBuffer](
                GetResultValue(self.reader.Read(c_nbytes)))

        return pyarrow_wrap_cudabuffer(output)


cdef class BufferWriter(NativeFile):
    """File interface for writing to CUDA buffers.

    By default writes are unbuffered. Use set_buffer_size to enable
    buffering.
    """

    def __cinit__(self, CudaBuffer buffer):
        self.buffer = buffer
        self.writer = new CCudaBufferWriter(self.buffer.cuda_buffer)
        self.set_output_stream(shared_ptr[COutputStream](self.writer))
        self.is_writable = True

    def writeat(self, int64_t position, object data):
        """Write data to buffer starting from position.

        Parameters
        ----------
        position : int
          Specify device buffer position where the data will be
          written.
        data : array-like
          Specify data, the data instance must implement buffer
          protocol.
        """
        cdef:
            Buffer buf = as_buffer(data)
            const uint8_t* c_data = buf.buffer.get().data()
            int64_t c_size = buf.buffer.get().size()

        with nogil:
            check_status(self.writer.WriteAt(position, c_data, c_size))

    def flush(self):
        """ Flush the buffer stream """
        with nogil:
            check_status(self.writer.Flush())

    def seek(self, int64_t position, int whence=0):
        # TODO: remove this method after NativeFile.seek supports
        # writable files.
        cdef int64_t offset

        with nogil:
            if whence == 0:
                offset = position
            elif whence == 1:
                offset = GetResultValue(self.writer.Tell())
                offset = offset + position
            else:
                with gil:
                    raise ValueError("Invalid value of whence: {0}"
                                     .format(whence))
            check_status(self.writer.Seek(offset))
        return self.tell()

    @property
    def buffer_size(self):
        """Returns size of host (CPU) buffer, 0 for unbuffered
        """
        return self.writer.buffer_size()

    @buffer_size.setter
    def buffer_size(self, int64_t buffer_size):
        """Set CPU buffer size to limit calls to cudaMemcpy

        Parameters
        ----------
        buffer_size : int
          Specify the size of CPU buffer to allocate in bytes.
        """
        with nogil:
            check_status(self.writer.SetBufferSize(buffer_size))

    @property
    def num_bytes_buffered(self):
        """Returns number of bytes buffered on host
        """
        return self.writer.num_bytes_buffered()

# Functions


def new_host_buffer(const int64_t size, int device=0):
    """Return buffer with CUDA-accessible memory on CPU host

    Parameters
    ----------
    size : int
      Specify the number of bytes to be allocated.
    device : int
      Specify GPU device number.

    Returns
    -------
    dbuf : HostBuffer
      Allocated host buffer
    """
    cdef shared_ptr[CCudaHostBuffer] buffer
    with nogil:
        buffer = GetResultValue(AllocateCudaHostBuffer(device, size))
    return pyarrow_wrap_cudahostbuffer(buffer)


def serialize_record_batch(object batch, object ctx):
    """ Write record batch message to GPU device memory

    Parameters
    ----------
    batch : RecordBatch
      Record batch to write
    ctx : Context
      CUDA Context to allocate device memory from

    Returns
    -------
    dbuf : CudaBuffer
      device buffer which contains the record batch message
    """
    cdef shared_ptr[CCudaBuffer] buffer
    cdef CRecordBatch* batch_ = pyarrow_unwrap_batch(batch).get()
    cdef CCudaContext* ctx_ = pyarrow_unwrap_cudacontext(ctx).get()
    with nogil:
        buffer = GetResultValue(CudaSerializeRecordBatch(batch_[0], ctx_))
    return pyarrow_wrap_cudabuffer(buffer)


def read_message(object source, pool=None):
    """ Read Arrow IPC message located on GPU device

    Parameters
    ----------
    source : {CudaBuffer, cuda.BufferReader}
      Device buffer or reader of device buffer.
    pool : MemoryPool (optional)
      Pool to allocate CPU memory for the metadata

    Returns
    -------
    message : Message
      The deserialized message, body still on device
    """
    cdef:
        Message result = Message.__new__(Message)
    cdef CMemoryPool* pool_ = maybe_unbox_memory_pool(pool)
    if not isinstance(source, BufferReader):
        reader = BufferReader(source)
    with nogil:
        result.message = move(
            GetResultValue(ReadMessage(reader.reader, pool_)))
    return result


def read_record_batch(object buffer, object schema, *,
                      DictionaryMemo dictionary_memo=None, pool=None):
    """Construct RecordBatch referencing IPC message located on CUDA device.

    While the metadata is copied to host memory for deserialization,
    the record batch data remains on the device.

    Parameters
    ----------
    buffer :
      Device buffer containing the complete IPC message
    schema : Schema
      The schema for the record batch
    dictionary_memo : DictionaryMemo, optional
        If message contains dictionaries, must pass a populated
        DictionaryMemo
    pool : MemoryPool (optional)
      Pool to allocate metadata from

    Returns
    -------
    batch : RecordBatch
      Reconstructed record batch, with device pointers

    """
    cdef:
        shared_ptr[CSchema] schema_ = pyarrow_unwrap_schema(schema)
        shared_ptr[CCudaBuffer] buffer_ = pyarrow_unwrap_cudabuffer(buffer)
        CDictionaryMemo temp_memo
        CDictionaryMemo* arg_dict_memo
        CMemoryPool* pool_ = maybe_unbox_memory_pool(pool)
        shared_ptr[CRecordBatch] batch

    if dictionary_memo is not None:
        arg_dict_memo = dictionary_memo.memo
    else:
        arg_dict_memo = &temp_memo

    with nogil:
        batch = GetResultValue(CudaReadRecordBatch(
            schema_, arg_dict_memo, buffer_, pool_))
    return pyarrow_wrap_batch(batch)


# Public API


cdef public api bint pyarrow_is_buffer(object buffer):
    return isinstance(buffer, Buffer)

# cudabuffer

cdef public api bint pyarrow_is_cudabuffer(object buffer):
    return isinstance(buffer, CudaBuffer)


cdef public api object \
        pyarrow_wrap_cudabuffer_base(const shared_ptr[CCudaBuffer]& buf, base):
    cdef CudaBuffer result = CudaBuffer.__new__(CudaBuffer)
    result.init_cuda(buf, base)
    return result


cdef public api object \
        pyarrow_wrap_cudabuffer(const shared_ptr[CCudaBuffer]& buf):
    cdef CudaBuffer result = CudaBuffer.__new__(CudaBuffer)
    result.init_cuda(buf, None)
    return result


cdef public api shared_ptr[CCudaBuffer] pyarrow_unwrap_cudabuffer(object obj):
    if pyarrow_is_cudabuffer(obj):
        return (<CudaBuffer>obj).cuda_buffer
    raise TypeError('expected CudaBuffer instance, got %s'
                    % (type(obj).__name__))

# cudahostbuffer

cdef public api bint pyarrow_is_cudahostbuffer(object buffer):
    return isinstance(buffer, HostBuffer)


cdef public api object \
        pyarrow_wrap_cudahostbuffer(const shared_ptr[CCudaHostBuffer]& buf):
    cdef HostBuffer result = HostBuffer.__new__(HostBuffer)
    result.init_host(buf)
    return result


cdef public api shared_ptr[CCudaHostBuffer] \
        pyarrow_unwrap_cudahostbuffer(object obj):
    if pyarrow_is_cudahostbuffer(obj):
        return (<HostBuffer>obj).host_buffer
    raise TypeError('expected HostBuffer instance, got %s'
                    % (type(obj).__name__))

# cudacontext

cdef public api bint pyarrow_is_cudacontext(object ctx):
    return isinstance(ctx, Context)


cdef public api object \
        pyarrow_wrap_cudacontext(const shared_ptr[CCudaContext]& ctx):
    cdef Context result = Context.__new__(Context)
    result.init(ctx)
    return result


cdef public api shared_ptr[CCudaContext] \
        pyarrow_unwrap_cudacontext(object obj):
    if pyarrow_is_cudacontext(obj):
        return (<Context>obj).context
    raise TypeError('expected Context instance, got %s'
                    % (type(obj).__name__))

# cudaipcmemhandle

cdef public api bint pyarrow_is_cudaipcmemhandle(object handle):
    return isinstance(handle, IpcMemHandle)


cdef public api object \
        pyarrow_wrap_cudaipcmemhandle(shared_ptr[CCudaIpcMemHandle]& h):
    cdef IpcMemHandle result = IpcMemHandle.__new__(IpcMemHandle)
    result.init(h)
    return result


cdef public api shared_ptr[CCudaIpcMemHandle] \
        pyarrow_unwrap_cudaipcmemhandle(object obj):
    if pyarrow_is_cudaipcmemhandle(obj):
        return (<IpcMemHandle>obj).handle
    raise TypeError('expected IpcMemHandle instance, got %s'
                    % (type(obj).__name__))
