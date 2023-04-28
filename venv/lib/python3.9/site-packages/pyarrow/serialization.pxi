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

from cpython.ref cimport PyObject

import warnings


def _deprecate_serialization(name):
    msg = (
        "'pyarrow.{}' is deprecated as of 2.0.0 and will be removed in a "
        "future version. Use pickle or the pyarrow IPC functionality instead."
    ).format(name)
    warnings.warn(msg, FutureWarning, stacklevel=3)


def is_named_tuple(cls):
    """
    Return True if cls is a namedtuple and False otherwise.
    """
    b = cls.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(cls, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)


class SerializationCallbackError(ArrowSerializationError):
    def __init__(self, message, example_object):
        ArrowSerializationError.__init__(self, message)
        self.example_object = example_object


class DeserializationCallbackError(ArrowSerializationError):
    def __init__(self, message, type_id):
        ArrowSerializationError.__init__(self, message)
        self.type_id = type_id


cdef class SerializationContext(_Weakrefable):
    cdef:
        object type_to_type_id
        object whitelisted_types
        object types_to_pickle
        object custom_serializers
        object custom_deserializers
        object pickle_serializer
        object pickle_deserializer

    def __init__(self):
        # Types with special serialization handlers
        self.type_to_type_id = dict()
        self.whitelisted_types = dict()
        self.types_to_pickle = set()
        self.custom_serializers = dict()
        self.custom_deserializers = dict()
        self.pickle_serializer = pickle.dumps
        self.pickle_deserializer = pickle.loads

    def set_pickle(self, serializer, deserializer):
        """
        Set the serializer and deserializer to use for objects that are to be
        pickled.

        Parameters
        ----------
        serializer : callable
            The serializer to use (e.g., pickle.dumps or cloudpickle.dumps).
        deserializer : callable
            The deserializer to use (e.g., pickle.dumps or cloudpickle.dumps).
        """
        self.pickle_serializer = serializer
        self.pickle_deserializer = deserializer

    def clone(self):
        """
        Return copy of this SerializationContext.

        Returns
        -------
        clone : SerializationContext
        """
        result = SerializationContext()
        result.type_to_type_id = self.type_to_type_id.copy()
        result.whitelisted_types = self.whitelisted_types.copy()
        result.types_to_pickle = self.types_to_pickle.copy()
        result.custom_serializers = self.custom_serializers.copy()
        result.custom_deserializers = self.custom_deserializers.copy()
        result.pickle_serializer = self.pickle_serializer
        result.pickle_deserializer = self.pickle_deserializer

        return result

    def register_type(self, type_, type_id, pickle=False,
                      custom_serializer=None, custom_deserializer=None):
        r"""
        EXPERIMENTAL: Add type to the list of types we can serialize.

        Parameters
        ----------
        type\_ : type
            The type that we can serialize.
        type_id : string
            A string used to identify the type.
        pickle : bool
            True if the serialization should be done with pickle.
            False if it should be done efficiently with Arrow.
        custom_serializer : callable
            This argument is optional, but can be provided to
            serialize objects of the class in a particular way.
        custom_deserializer : callable
            This argument is optional, but can be provided to
            deserialize objects of the class in a particular way.
        """
        if not isinstance(type_id, str):
            raise TypeError("The type_id argument must be a string. The value "
                            "passed in has type {}.".format(type(type_id)))

        self.type_to_type_id[type_] = type_id
        self.whitelisted_types[type_id] = type_
        if pickle:
            self.types_to_pickle.add(type_id)
        if custom_serializer is not None:
            self.custom_serializers[type_id] = custom_serializer
            self.custom_deserializers[type_id] = custom_deserializer

    def _serialize_callback(self, obj):
        found = False
        for type_ in type(obj).__mro__:
            if type_ in self.type_to_type_id:
                found = True
                break

        if not found:
            raise SerializationCallbackError(
                "pyarrow does not know how to "
                "serialize objects of type {}.".format(type(obj)), obj
            )

        # use the closest match to type(obj)
        type_id = self.type_to_type_id[type_]
        if type_id in self.types_to_pickle:
            serialized_obj = {"data": self.pickle_serializer(obj),
                              "pickle": True}
        elif type_id in self.custom_serializers:
            serialized_obj = {"data": self.custom_serializers[type_id](obj)}
        else:
            if is_named_tuple(type_):
                serialized_obj = {}
                serialized_obj["_pa_getnewargs_"] = obj.__getnewargs__()
            elif hasattr(obj, "__dict__"):
                serialized_obj = obj.__dict__
            else:
                msg = "We do not know how to serialize " \
                      "the object '{}'".format(obj)
                raise SerializationCallbackError(msg, obj)
        return dict(serialized_obj, **{"_pytype_": type_id})

    def _deserialize_callback(self, serialized_obj):
        type_id = serialized_obj["_pytype_"]
        if isinstance(type_id, bytes):
            # ARROW-4675: Python 2 serialized, read in Python 3
            type_id = frombytes(type_id)

        if "pickle" in serialized_obj:
            # The object was pickled, so unpickle it.
            obj = self.pickle_deserializer(serialized_obj["data"])
        else:
            assert type_id not in self.types_to_pickle
            if type_id not in self.whitelisted_types:
                msg = "Type ID " + type_id + " not registered in " \
                      "deserialization callback"
                raise DeserializationCallbackError(msg, type_id)
            type_ = self.whitelisted_types[type_id]
            if type_id in self.custom_deserializers:
                obj = self.custom_deserializers[type_id](
                    serialized_obj["data"])
            else:
                # In this case, serialized_obj should just be
                # the __dict__ field.
                if "_pa_getnewargs_" in serialized_obj:
                    obj = type_.__new__(
                        type_, *serialized_obj["_pa_getnewargs_"])
                else:
                    obj = type_.__new__(type_)
                    serialized_obj.pop("_pytype_")
                    obj.__dict__.update(serialized_obj)
        return obj

    def serialize(self, obj):
        """
        Call pyarrow.serialize and pass this SerializationContext.
        """
        return serialize(obj, context=self)

    def serialize_to(self, object value, sink):
        """
        Call pyarrow.serialize_to and pass this SerializationContext.
        """
        return serialize_to(value, sink, context=self)

    def deserialize(self, what):
        """
        Call pyarrow.deserialize and pass this SerializationContext.
        """
        return deserialize(what, context=self)

    def deserialize_components(self, what):
        """
        Call pyarrow.deserialize_components and pass this SerializationContext.
        """
        return deserialize_components(what, context=self)


_default_serialization_context = SerializationContext()
_default_context_initialized = False


def _get_default_context():
    global _default_context_initialized
    from pyarrow.serialization import _register_default_serialization_handlers
    if not _default_context_initialized:
        _register_default_serialization_handlers(
            _default_serialization_context)
        _default_context_initialized = True
    return _default_serialization_context


cdef class SerializedPyObject(_Weakrefable):
    """
    Arrow-serialized representation of Python object.
    """
    cdef:
        CSerializedPyObject data

    cdef readonly:
        object base

    @property
    def total_bytes(self):
        cdef CMockOutputStream mock_stream
        with nogil:
            check_status(self.data.WriteTo(&mock_stream))

        return mock_stream.GetExtentBytesWritten()

    def write_to(self, sink):
        """
        Write serialized object to a sink.
        """
        cdef shared_ptr[COutputStream] stream
        get_writer(sink, &stream)
        self._write_to(stream.get())

    cdef _write_to(self, COutputStream* stream):
        with nogil:
            check_status(self.data.WriteTo(stream))

    def deserialize(self, SerializationContext context=None):
        """
        Convert back to Python object.
        """
        cdef PyObject* result

        if context is None:
            context = _get_default_context()

        with nogil:
            check_status(DeserializeObject(context, self.data,
                                           <PyObject*> self.base, &result))

        # PyObject_to_object is necessary to avoid a memory leak;
        # also unpack the list the object was wrapped in in serialize
        return PyObject_to_object(result)[0]

    def to_buffer(self, nthreads=1):
        """
        Write serialized data as Buffer.
        """
        cdef Buffer output = allocate_buffer(self.total_bytes)
        sink = FixedSizeBufferWriter(output)
        if nthreads > 1:
            sink.set_memcopy_threads(nthreads)
        self.write_to(sink)
        return output

    @staticmethod
    def from_components(components):
        """
        Reconstruct SerializedPyObject from output of
        SerializedPyObject.to_components.
        """
        cdef:
            int num_tensors = components['num_tensors']
            int num_ndarrays = components['num_ndarrays']
            int num_buffers = components['num_buffers']
            list buffers = components['data']
            SparseTensorCounts num_sparse_tensors = SparseTensorCounts()
            SerializedPyObject result = SerializedPyObject()

        num_sparse_tensors.coo = components['num_sparse_tensors']['coo']
        num_sparse_tensors.csr = components['num_sparse_tensors']['csr']
        num_sparse_tensors.csc = components['num_sparse_tensors']['csc']
        num_sparse_tensors.csf = components['num_sparse_tensors']['csf']
        num_sparse_tensors.ndim_csf = \
            components['num_sparse_tensors']['ndim_csf']

        with nogil:
            check_status(GetSerializedFromComponents(num_tensors,
                                                     num_sparse_tensors,
                                                     num_ndarrays,
                                                     num_buffers,
                                                     buffers, &result.data))

        return result

    def to_components(self, memory_pool=None):
        """
        Return the decomposed dict representation of the serialized object
        containing a collection of Buffer objects which maximize opportunities
        for zero-copy.

        Parameters
        ----------
        memory_pool : MemoryPool default None
            Pool to use for necessary allocations.

        Returns

        """
        cdef PyObject* result
        cdef CMemoryPool* c_pool = maybe_unbox_memory_pool(memory_pool)

        with nogil:
            check_status(self.data.GetComponents(c_pool, &result))

        return PyObject_to_object(result)


def serialize(object value, SerializationContext context=None):
    """
    DEPRECATED: Serialize a general Python sequence for transient storage
    and transport.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    Notes
    -----
    This function produces data that is incompatible with the standard
    Arrow IPC binary protocol, i.e. it cannot be used with ipc.open_stream or
    ipc.open_file. You can use deserialize, deserialize_from, or
    deserialize_components to read it.

    Parameters
    ----------
    value : object
        Python object for the sequence that is to be serialized.
    context : SerializationContext
        Custom serialization and deserialization context, uses a default
        context with some standard type handlers if not specified.

    Returns
    -------
    serialized : SerializedPyObject

    """
    _deprecate_serialization("serialize")
    return _serialize(value, context)


def _serialize(object value, SerializationContext context=None):
    cdef SerializedPyObject serialized = SerializedPyObject()
    wrapped_value = [value]

    if context is None:
        context = _get_default_context()

    with nogil:
        check_status(SerializeObject(context, wrapped_value, &serialized.data))
    return serialized


def serialize_to(object value, sink, SerializationContext context=None):
    """
    DEPRECATED: Serialize a Python sequence to a file.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    Parameters
    ----------
    value : object
        Python object for the sequence that is to be serialized.
    sink : NativeFile or file-like
        File the sequence will be written to.
    context : SerializationContext
        Custom serialization and deserialization context, uses a default
        context with some standard type handlers if not specified.
    """
    _deprecate_serialization("serialize_to")
    serialized = _serialize(value, context)
    serialized.write_to(sink)


def read_serialized(source, base=None):
    """
    DEPRECATED: Read serialized Python sequence from file-like object.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    Parameters
    ----------
    source : NativeFile
        File to read the sequence from.
    base : object
        This object will be the base object of all the numpy arrays
        contained in the sequence.

    Returns
    -------
    serialized : the serialized data
    """
    _deprecate_serialization("read_serialized")
    return _read_serialized(source, base=base)


def _read_serialized(source, base=None):
    cdef shared_ptr[CRandomAccessFile] stream
    get_reader(source, False, &stream)

    cdef SerializedPyObject serialized = SerializedPyObject()
    serialized.base = base
    with nogil:
        check_status(ReadSerializedObject(stream.get(), &serialized.data))

    return serialized


def deserialize_from(source, object base, SerializationContext context=None):
    """
    DEPRECATED: Deserialize a Python sequence from a file.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    This only can interact with data produced by pyarrow.serialize or
    pyarrow.serialize_to.

    Parameters
    ----------
    source : NativeFile
        File to read the sequence from.
    base : object
        This object will be the base object of all the numpy arrays
        contained in the sequence.
    context : SerializationContext
        Custom serialization and deserialization context.

    Returns
    -------
    object
        Python object for the deserialized sequence.
    """
    _deprecate_serialization("deserialize_from")
    serialized = _read_serialized(source, base=base)
    return serialized.deserialize(context)


def deserialize_components(components, SerializationContext context=None):
    """
    DEPRECATED: Reconstruct Python object from output of
    SerializedPyObject.to_components.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    Parameters
    ----------
    components : dict
        Output of SerializedPyObject.to_components
    context : SerializationContext, default None

    Returns
    -------
    object : the Python object that was originally serialized
    """
    _deprecate_serialization("deserialize_components")
    serialized = SerializedPyObject.from_components(components)
    return serialized.deserialize(context)


def deserialize(obj, SerializationContext context=None):
    """
    DEPRECATED: Deserialize Python object from Buffer or other Python
    object supporting the buffer protocol.

    .. deprecated:: 2.0
        The custom serialization functionality is deprecated in pyarrow 2.0,
        and will be removed in a future version. Use the standard library
        ``pickle`` or the IPC functionality of pyarrow (see :ref:`ipc` for
        more).

    This only can interact with data produced by pyarrow.serialize or
    pyarrow.serialize_to.

    Parameters
    ----------
    obj : pyarrow.Buffer or Python object supporting buffer protocol
    context : SerializationContext
        Custom serialization and deserialization context.

    Returns
    -------
    deserialized : object
    """
    _deprecate_serialization("deserialize")
    return _deserialize(obj, context=context)


def _deserialize(obj, SerializationContext context=None):
    source = BufferReader(obj)
    serialized = _read_serialized(source, base=obj)
    return serialized.deserialize(context)
