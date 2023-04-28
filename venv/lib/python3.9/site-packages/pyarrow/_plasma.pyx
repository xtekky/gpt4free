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
# cython: language_level = 3

from libcpp cimport bool as c_bool, nullptr
from libcpp.memory cimport shared_ptr, unique_ptr, make_shared
from libcpp.string cimport string as c_string
from libcpp.vector cimport vector as c_vector
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport int64_t, uint8_t, uintptr_t
from cython.operator cimport dereference as deref, preincrement as inc
from cpython.pycapsule cimport *

from collections.abc import Sequence
import random
import socket
import warnings

import pyarrow
from pyarrow.lib cimport (Buffer, NativeFile, _Weakrefable,
                          check_status, pyarrow_wrap_buffer)
from pyarrow.lib import ArrowException, frombytes
from pyarrow.includes.libarrow cimport (CBuffer, CMutableBuffer,
                                        CFixedSizeBufferWriter, CStatus)
from pyarrow.includes.libplasma cimport *

PLASMA_WAIT_TIMEOUT = 2 ** 30


cdef extern from "plasma/common.h" nogil:
    cdef cppclass CCudaIpcPlaceholder" plasma::internal::CudaIpcPlaceholder":
        pass

    cdef cppclass CUniqueID" plasma::UniqueID":

        @staticmethod
        CUniqueID from_binary(const c_string& binary)

        @staticmethod
        CUniqueID from_random()

        c_bool operator==(const CUniqueID& rhs) const

        c_string hex() const

        c_string binary() const

        @staticmethod
        int64_t size()

    cdef enum CObjectState" plasma::ObjectState":
        PLASMA_CREATED" plasma::ObjectState::PLASMA_CREATED"
        PLASMA_SEALED" plasma::ObjectState::PLASMA_SEALED"

    cdef struct CObjectTableEntry" plasma::ObjectTableEntry":
        int fd
        int device_num
        int64_t map_size
        ptrdiff_t offset
        uint8_t* pointer
        int64_t data_size
        int64_t metadata_size
        int ref_count
        int64_t create_time
        int64_t construct_duration
        CObjectState state
        shared_ptr[CCudaIpcPlaceholder] ipc_handle

    ctypedef unordered_map[CUniqueID, unique_ptr[CObjectTableEntry]] \
        CObjectTable" plasma::ObjectTable"


cdef extern from "plasma/common.h":
    cdef int64_t kDigestSize" plasma::kDigestSize"

cdef extern from "plasma/client.h" nogil:

    cdef cppclass CPlasmaClient" plasma::PlasmaClient":

        CPlasmaClient()

        CStatus Connect(const c_string& store_socket_name,
                        const c_string& manager_socket_name,
                        int release_delay, int num_retries)

        CStatus Create(const CUniqueID& object_id,
                       int64_t data_size, const uint8_t* metadata, int64_t
                       metadata_size, const shared_ptr[CBuffer]* data)

        CStatus CreateAndSeal(const CUniqueID& object_id,
                              const c_string& data, const c_string& metadata)

        CStatus Get(const c_vector[CUniqueID] object_ids, int64_t timeout_ms,
                    c_vector[CObjectBuffer]* object_buffers)

        CStatus Seal(const CUniqueID& object_id)

        CStatus Evict(int64_t num_bytes, int64_t& num_bytes_evicted)

        CStatus Hash(const CUniqueID& object_id, uint8_t* digest)

        CStatus Release(const CUniqueID& object_id)

        CStatus Contains(const CUniqueID& object_id, c_bool* has_object)

        CStatus List(CObjectTable* objects)

        CStatus Subscribe(int* fd)

        CStatus DecodeNotifications(const uint8_t* buffer,
                                    c_vector[CUniqueID]* object_ids,
                                    c_vector[int64_t]* data_sizes,
                                    c_vector[int64_t]* metadata_sizes)

        CStatus GetNotification(int fd, CUniqueID* object_id,
                                int64_t* data_size, int64_t* metadata_size)

        CStatus Disconnect()

        CStatus Delete(const c_vector[CUniqueID] object_ids)

        CStatus SetClientOptions(const c_string& client_name,
                                 int64_t limit_output_memory)

        c_string DebugString()

        int64_t store_capacity()

cdef extern from "plasma/client.h" nogil:

    cdef struct CObjectBuffer" plasma::ObjectBuffer":
        shared_ptr[CBuffer] data
        shared_ptr[CBuffer] metadata


def make_object_id(object_id):
    return ObjectID(object_id)


cdef class ObjectID(_Weakrefable):
    """
    DEPRECATED: An ObjectID represents a string of bytes used to identify Plasma objects.

    .. deprecated:: 10.0.0
       Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.
    """

    cdef:
        CUniqueID data

    def __cinit__(self, object_id):
        if (not isinstance(object_id, bytes) or
                len(object_id) != CUniqueID.size()):
            raise ValueError("Object ID must by 20 bytes,"
                             " is " + str(object_id))
        self.data = CUniqueID.from_binary(object_id)

        warnings.warn(
            "Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.",
            DeprecationWarning, stacklevel=2)

    def __eq__(self, other):
        try:
            return self.data == (<ObjectID?>other).data
        except TypeError:
            return False

    def __hash__(self):
        return hash(self.data.binary())

    def __repr__(self):
        return "ObjectID(" + self.data.hex().decode() + ")"

    def __reduce__(self):
        return (make_object_id, (self.data.binary(),))

    def binary(self):
        """
        Return the binary representation of this ObjectID.

        Returns
        -------
        bytes
            Binary representation of the ObjectID.
        """
        return self.data.binary()

    @staticmethod
    def from_random():
        """
        Returns a randomly generated ObjectID.

        Returns
        -------
        ObjectID
            A randomly generated ObjectID.
        """
        random_id = bytes(bytearray(
            random.getrandbits(8) for _ in range(CUniqueID.size())))
        return ObjectID(random_id)


cdef class ObjectNotAvailable(_Weakrefable):
    """
    Placeholder for an object that was not available within the given timeout.
    """
    pass


cdef class PlasmaBuffer(Buffer):
    """
    DEPRECATED: This is the type returned by calls to get with a PlasmaClient.

    We define our own class instead of directly returning a buffer object so
    that we can add a custom destructor which notifies Plasma that the object
    is no longer being used, so the memory in the Plasma store backing the
    object can potentially be freed.

    .. deprecated:: 10.0.0
       Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.

    Attributes
    ----------
    object_id : ObjectID
        The ID of the object in the buffer.
    client : PlasmaClient
        The PlasmaClient that we use to communicate with the store and manager.
    """

    cdef:
        ObjectID object_id
        PlasmaClient client

    @staticmethod
    cdef PlasmaBuffer create(ObjectID object_id, PlasmaClient client,
                             const shared_ptr[CBuffer]& buffer):
        cdef PlasmaBuffer self = PlasmaBuffer.__new__(PlasmaBuffer)
        self.object_id = object_id
        self.client = client
        self.init(buffer)
        return self

    def __init__(self):
        raise TypeError("Do not call PlasmaBuffer's constructor directly, use "
                        "`PlasmaClient.create` instead.")

    def __dealloc__(self):
        """
        Notify Plasma that the object is no longer needed.

        If the plasma client has been shut down, then don't do anything.
        """
        self.client._release(self.object_id)


class PlasmaObjectNotFound(ArrowException):
    pass


class PlasmaStoreFull(ArrowException):
    pass


class PlasmaObjectExists(ArrowException):
    pass


cdef int plasma_check_status(const CStatus& status) nogil except -1:
    if status.ok():
        return 0

    with gil:
        message = frombytes(status.message())
        if IsPlasmaObjectExists(status):
            raise PlasmaObjectExists(message)
        elif IsPlasmaObjectNotFound(status):
            raise PlasmaObjectNotFound(message)
        elif IsPlasmaStoreFull(status):
            raise PlasmaStoreFull(message)

    return check_status(status)


def get_socket_from_fd(fileno, family, type):
    import socket
    return socket.socket(fileno=fileno, family=family, type=type)


cdef class PlasmaClient(_Weakrefable):
    """
    DEPRECATED: The PlasmaClient is used to interface with a plasma store and manager.

    The PlasmaClient can ask the PlasmaStore to allocate a new buffer, seal a
    buffer, and get a buffer. Buffers are referred to by object IDs, which are
    strings.

    .. deprecated:: 10.0.0
       Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.
    """

    cdef:
        shared_ptr[CPlasmaClient] client
        int notification_fd
        c_string store_socket_name

    def __cinit__(self):
        self.client.reset(new CPlasmaClient())
        self.notification_fd = -1
        self.store_socket_name = b""

        warnings.warn(
            "Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.",
            DeprecationWarning, stacklevel=3)

    cdef _get_object_buffers(self, object_ids, int64_t timeout_ms,
                             c_vector[CObjectBuffer]* result):
        cdef:
            c_vector[CUniqueID] ids
            ObjectID object_id

        for object_id in object_ids:
            ids.push_back(object_id.data)
        with nogil:
            plasma_check_status(self.client.get().Get(ids, timeout_ms, result))

    # XXX C++ API should instead expose some kind of CreateAuto()
    cdef _make_mutable_plasma_buffer(self, ObjectID object_id, uint8_t* data,
                                     int64_t size):
        cdef shared_ptr[CBuffer] buffer
        buffer.reset(new CMutableBuffer(data, size))
        return PlasmaBuffer.create(object_id, self, buffer)

    @property
    def store_socket_name(self):
        return self.store_socket_name.decode()

    def create(self, ObjectID object_id, int64_t data_size,
               c_string metadata=b""):
        """
        Create a new buffer in the PlasmaStore for a particular object ID.

        The returned buffer is mutable until ``seal()`` is called.

        Parameters
        ----------
        object_id : ObjectID
            The object ID used to identify an object.
        data_size : int
            The size in bytes of the created buffer.
        metadata : bytes
            An optional string of bytes encoding whatever metadata the user
            wishes to encode.

        Returns
        -------
        buffer : Buffer
            A mutable buffer where to write the object data.

        Raises
        ------
        PlasmaObjectExists
            This exception is raised if the object could not be created because
            there already is an object with the same ID in the plasma store.

        PlasmaStoreFull
            This exception is raised if the object could
            not be created because the plasma store is unable to evict
            enough objects to create room for it.
        """
        cdef shared_ptr[CBuffer] data
        with nogil:
            plasma_check_status(
                self.client.get().Create(object_id.data, data_size,
                                         <uint8_t*>(metadata.data()),
                                         metadata.size(), &data))
        return self._make_mutable_plasma_buffer(object_id,
                                                data.get().mutable_data(),
                                                data_size)

    def create_and_seal(self, ObjectID object_id, c_string data,
                        c_string metadata=b""):
        """
        Store a new object in the PlasmaStore for a particular object ID.

        Parameters
        ----------
        object_id : ObjectID
            The object ID used to identify an object.
        data : bytes
            The object to store.
        metadata : bytes
            An optional string of bytes encoding whatever metadata the user
            wishes to encode.

        Raises
        ------
        PlasmaObjectExists
            This exception is raised if the object could not be created because
            there already is an object with the same ID in the plasma store.

        PlasmaStoreFull: This exception is raised if the object could
                not be created because the plasma store is unable to evict
                enough objects to create room for it.
        """
        with nogil:
            plasma_check_status(
                self.client.get().CreateAndSeal(object_id.data, data,
                                                metadata))

    def get_buffers(self, object_ids, timeout_ms=-1, with_meta=False):
        """
        Returns data buffer from the PlasmaStore based on object ID.

        If the object has not been sealed yet, this call will block. The
        retrieved buffer is immutable.

        Parameters
        ----------
        object_ids : list
            A list of ObjectIDs used to identify some objects.
        timeout_ms : int
            The number of milliseconds that the get call should block before
            timing out and returning. Pass -1 if the call should block and 0
            if the call should return immediately.
        with_meta : bool

        Returns
        -------
        list
            If with_meta=False, this is a list of PlasmaBuffers for the data
            associated with the object_ids and None if the object was not
            available. If with_meta=True, this is a list of tuples of
            PlasmaBuffer and metadata bytes.
        """
        cdef c_vector[CObjectBuffer] object_buffers
        self._get_object_buffers(object_ids, timeout_ms, &object_buffers)
        result = []
        for i in range(object_buffers.size()):
            if object_buffers[i].data.get() != nullptr:
                data = pyarrow_wrap_buffer(object_buffers[i].data)
            else:
                data = None
            if not with_meta:
                result.append(data)
            else:
                if object_buffers[i].metadata.get() != nullptr:
                    size = object_buffers[i].metadata.get().size()
                    metadata = object_buffers[i].metadata.get().data()[:size]
                else:
                    metadata = None
                result.append((metadata, data))
        return result

    def get_metadata(self, object_ids, timeout_ms=-1):
        """
        Returns metadata buffer from the PlasmaStore based on object ID.

        If the object has not been sealed yet, this call will block. The
        retrieved buffer is immutable.

        Parameters
        ----------
        object_ids : list
            A list of ObjectIDs used to identify some objects.
        timeout_ms : int
            The number of milliseconds that the get call should block before
            timing out and returning. Pass -1 if the call should block and 0
            if the call should return immediately.

        Returns
        -------
        list
            List of PlasmaBuffers for the metadata associated with the
            object_ids and None if the object was not available.
        """
        cdef c_vector[CObjectBuffer] object_buffers
        self._get_object_buffers(object_ids, timeout_ms, &object_buffers)
        result = []
        for i in range(object_buffers.size()):
            if object_buffers[i].metadata.get() != nullptr:
                result.append(pyarrow_wrap_buffer(object_buffers[i].metadata))
            else:
                result.append(None)
        return result

    def put_raw_buffer(self, object value, ObjectID object_id=None,
                       c_string metadata=b"", int memcopy_threads=6):
        """
        Store Python buffer into the object store.

        Parameters
        ----------
        value : Python object that implements the buffer protocol
            A Python buffer object to store.
        object_id : ObjectID, default None
            If this is provided, the specified object ID will be used to refer
            to the object.
        metadata : bytes
            An optional string of bytes encoding whatever metadata the user
            wishes to encode.
        memcopy_threads : int, default 6
            The number of threads to use to write the serialized object into
            the object store for large objects.

        Returns
        -------
        ObjectID
            The object ID associated to the Python buffer object.
        """
        cdef ObjectID target_id = (object_id if object_id
                                   else ObjectID.from_random())
        cdef Buffer arrow_buffer = pyarrow.py_buffer(value)
        write_buffer = self.create(target_id, len(value), metadata)
        stream = pyarrow.FixedSizeBufferWriter(write_buffer)
        stream.set_memcopy_threads(memcopy_threads)
        stream.write(arrow_buffer)
        self.seal(target_id)
        return target_id

    def put(self, object value, ObjectID object_id=None, int memcopy_threads=6,
            serialization_context=None):
        """
        Store a Python value into the object store.

        Parameters
        ----------
        value : object
            A Python object to store.
        object_id : ObjectID, default None
            If this is provided, the specified object ID will be used to refer
            to the object.
        memcopy_threads : int, default 6
            The number of threads to use to write the serialized object into
            the object store for large objects.
        serialization_context : pyarrow.SerializationContext, default None
            Custom serialization and deserialization context.

        Returns
        -------
        ObjectID
            The object ID associated to the Python object.
        """
        cdef ObjectID target_id = (object_id if object_id
                                   else ObjectID.from_random())
        if serialization_context is not None:
            warnings.warn(
                "'serialization_context' is deprecated and will be removed "
                "in a future version.",
                FutureWarning, stacklevel=2
            )
        serialized = pyarrow.lib._serialize(value, serialization_context)
        buffer = self.create(target_id, serialized.total_bytes)
        stream = pyarrow.FixedSizeBufferWriter(buffer)
        stream.set_memcopy_threads(memcopy_threads)
        serialized.write_to(stream)
        self.seal(target_id)
        return target_id

    def get(self, object_ids, int timeout_ms=-1, serialization_context=None):
        """
        Get one or more Python values from the object store.

        Parameters
        ----------
        object_ids : list or ObjectID
            Object ID or list of object IDs associated to the values we get
            from the store.
        timeout_ms : int, default -1
            The number of milliseconds that the get call should block before
            timing out and returning. Pass -1 if the call should block and 0
            if the call should return immediately.
        serialization_context : pyarrow.SerializationContext, default None
            Custom serialization and deserialization context.

        Returns
        -------
        list or object
            Python value or list of Python values for the data associated with
            the object_ids and ObjectNotAvailable if the object was not
            available.
        """
        if serialization_context is not None:
            warnings.warn(
                "'serialization_context' is deprecated and will be removed "
                "in a future version.",
                FutureWarning, stacklevel=2
            )
        if isinstance(object_ids, Sequence):
            results = []
            buffers = self.get_buffers(object_ids, timeout_ms)
            for i in range(len(object_ids)):
                # buffers[i] is None if this object was not available within
                # the timeout
                if buffers[i]:
                    val = pyarrow.lib._deserialize(buffers[i],
                                                   serialization_context)
                    results.append(val)
                else:
                    results.append(ObjectNotAvailable)
            return results
        else:
            return self.get([object_ids], timeout_ms, serialization_context)[0]

    def seal(self, ObjectID object_id):
        """
        Seal the buffer in the PlasmaStore for a particular object ID.

        Once a buffer has been sealed, the buffer is immutable and can only be
        accessed through get.

        Parameters
        ----------
        object_id : ObjectID
            A string used to identify an object.
        """
        with nogil:
            plasma_check_status(self.client.get().Seal(object_id.data))

    def _release(self, ObjectID object_id):
        """
        Notify Plasma that the object is no longer needed.

        Parameters
        ----------
        object_id : ObjectID
            A string used to identify an object.
        """
        with nogil:
            plasma_check_status(self.client.get().Release(object_id.data))

    def contains(self, ObjectID object_id):
        """
        Check if the object is present and sealed in the PlasmaStore.

        Parameters
        ----------
        object_id : ObjectID
            A string used to identify an object.
        """
        cdef c_bool is_contained
        with nogil:
            plasma_check_status(self.client.get().Contains(object_id.data,
                                                           &is_contained))
        return is_contained

    def hash(self, ObjectID object_id):
        """
        Compute the checksum of an object in the object store.

        Parameters
        ----------
        object_id : ObjectID
            A string used to identify an object.

        Returns
        -------
        bytes
            A digest string object's hash. If the object isn't in the object
            store, the string will have length zero.
        """
        cdef c_vector[uint8_t] digest = c_vector[uint8_t](kDigestSize)
        with nogil:
            plasma_check_status(self.client.get().Hash(object_id.data,
                                                       digest.data()))
        return bytes(digest[:])

    def evict(self, int64_t num_bytes):
        """
        Evict some objects until to recover some bytes.

        Recover at least num_bytes bytes if possible.

        Parameters
        ----------
        num_bytes : int
            The number of bytes to attempt to recover.
        """
        cdef int64_t num_bytes_evicted = -1
        with nogil:
            plasma_check_status(
                self.client.get().Evict(num_bytes, num_bytes_evicted))
        return num_bytes_evicted

    def subscribe(self):
        """Subscribe to notifications about sealed objects."""
        with nogil:
            plasma_check_status(
                self.client.get().Subscribe(&self.notification_fd))

    def get_notification_socket(self):
        """
        Get the notification socket.
        """
        return get_socket_from_fd(self.notification_fd,
                                  family=socket.AF_UNIX,
                                  type=socket.SOCK_STREAM)

    def decode_notifications(self, const uint8_t* buf):
        """
        Get the notification from the buffer.

        Returns
        -------
        [ObjectID]
            The list of object IDs in the notification message.
        c_vector[int64_t]
            The data sizes of the objects in the notification message.
        c_vector[int64_t]
            The metadata sizes of the objects in the notification message.
        """
        cdef c_vector[CUniqueID] ids
        cdef c_vector[int64_t] data_sizes
        cdef c_vector[int64_t] metadata_sizes
        with nogil:
            status = self.client.get().DecodeNotifications(buf,
                                                           &ids,
                                                           &data_sizes,
                                                           &metadata_sizes)
            plasma_check_status(status)
        object_ids = []
        for object_id in ids:
            object_ids.append(ObjectID(object_id.binary()))
        return object_ids, data_sizes, metadata_sizes

    def get_next_notification(self):
        """
        Get the next notification from the notification socket.

        Returns
        -------
        ObjectID
            The object ID of the object that was stored.
        int
            The data size of the object that was stored.
        int
            The metadata size of the object that was stored.
        """
        cdef ObjectID object_id = ObjectID(CUniqueID.size() * b"\0")
        cdef int64_t data_size
        cdef int64_t metadata_size
        with nogil:
            status = self.client.get().GetNotification(self.notification_fd,
                                                       &object_id.data,
                                                       &data_size,
                                                       &metadata_size)
            plasma_check_status(status)
        return object_id, data_size, metadata_size

    def to_capsule(self):
        return PyCapsule_New(<void *>self.client.get(), "plasma", NULL)

    def disconnect(self):
        """
        Disconnect this client from the Plasma store.
        """
        with nogil:
            plasma_check_status(self.client.get().Disconnect())

    def delete(self, object_ids):
        """
        Delete the objects with the given IDs from other object store.

        Parameters
        ----------
        object_ids : list
            A list of strings used to identify the objects.
        """
        cdef c_vector[CUniqueID] ids
        cdef ObjectID object_id
        for object_id in object_ids:
            ids.push_back(object_id.data)
        with nogil:
            plasma_check_status(self.client.get().Delete(ids))

    def set_client_options(self, client_name, int64_t limit_output_memory):
        cdef c_string name
        name = client_name.encode()
        with nogil:
            plasma_check_status(
                self.client.get().SetClientOptions(name, limit_output_memory))

    def debug_string(self):
        cdef c_string result
        with nogil:
            result = self.client.get().DebugString()
        return result.decode()

    def list(self):
        """
        Experimental: List the objects in the store.

        Returns
        -------
        dict
            Dictionary from ObjectIDs to an "info" dictionary describing the
            object. The "info" dictionary has the following entries:

            data_size
              size of the object in bytes

            metadata_size
              size of the object metadata in bytes

            ref_count
              Number of clients referencing the object buffer

            create_time
              Unix timestamp of the creation of the object

            construct_duration
              Time the creation of the object took in seconds

            state
              "created" if the object is still being created and
              "sealed" if it is already sealed
        """
        cdef CObjectTable objects
        with nogil:
            plasma_check_status(self.client.get().List(&objects))
        result = dict()
        cdef ObjectID object_id
        cdef CObjectTableEntry entry
        it = objects.begin()
        while it != objects.end():
            object_id = ObjectID(deref(it).first.binary())
            entry = deref(deref(it).second)
            if entry.state == CObjectState.PLASMA_CREATED:
                state = "created"
            else:
                state = "sealed"
            result[object_id] = {
                "data_size": entry.data_size,
                "metadata_size": entry.metadata_size,
                "ref_count": entry.ref_count,
                "create_time": entry.create_time,
                "construct_duration": entry.construct_duration,
                "state": state
            }
            inc(it)
        return result

    def store_capacity(self):
        """
        Get the memory capacity of the store.

        Returns
        -------

        int
            The memory capacity of the store in bytes.
        """
        return self.client.get().store_capacity()


def connect(store_socket_name, int num_retries=-1):
    """
    DEPRECATED: Return a new PlasmaClient that is connected a plasma store and
    optionally a manager.

    .. deprecated:: 10.0.0
       Plasma is deprecated since Arrow 10.0.0. It will be removed in 12.0.0 or so.

    Parameters
    ----------
    store_socket_name : str
        Name of the socket the plasma store is listening at.
    num_retries : int, default -1
        Number of times to try to connect to plasma store. Default value of -1
        uses the default (50)
    """
    cdef PlasmaClient result = PlasmaClient()
    cdef int deprecated_release_delay = 0
    result.store_socket_name = store_socket_name.encode()
    with nogil:
        plasma_check_status(
            result.client.get().Connect(result.store_socket_name, b"",
                                        deprecated_release_delay, num_retries))
    return result
