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

# cython: language_level = 3

import collections
import contextlib
import enum
import re
import socket
import time
import threading
import warnings
import weakref

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libcpp cimport bool as c_bool

from pyarrow.lib cimport *
from pyarrow.lib import (ArrowCancelled, ArrowException, ArrowInvalid,
                         SignalStopHandler)
from pyarrow.lib import as_buffer, frombytes, tobytes
from pyarrow.includes.libarrow_flight cimport *
from pyarrow.ipc import _get_legacy_format_default, _ReadPandasMixin
import pyarrow.lib as lib


cdef CFlightCallOptions DEFAULT_CALL_OPTIONS


cdef int check_flight_status(const CStatus& status) nogil except -1:
    cdef shared_ptr[FlightStatusDetail] detail

    if status.ok():
        return 0

    detail = FlightStatusDetail.UnwrapStatus(status)
    if detail:
        with gil:
            message = frombytes(status.message(), safe=True)
            detail_msg = detail.get().extra_info()
            if detail.get().code() == CFlightStatusInternal:
                raise FlightInternalError(message, detail_msg)
            elif detail.get().code() == CFlightStatusFailed:
                message = _munge_grpc_python_error(message)
                raise FlightServerError(message, detail_msg)
            elif detail.get().code() == CFlightStatusTimedOut:
                raise FlightTimedOutError(message, detail_msg)
            elif detail.get().code() == CFlightStatusCancelled:
                raise FlightCancelledError(message, detail_msg)
            elif detail.get().code() == CFlightStatusUnauthenticated:
                raise FlightUnauthenticatedError(message, detail_msg)
            elif detail.get().code() == CFlightStatusUnauthorized:
                raise FlightUnauthorizedError(message, detail_msg)
            elif detail.get().code() == CFlightStatusUnavailable:
                raise FlightUnavailableError(message, detail_msg)

    size_detail = FlightWriteSizeStatusDetail.UnwrapStatus(status)
    if size_detail:
        with gil:
            message = frombytes(status.message(), safe=True)
            raise FlightWriteSizeExceededError(
                message,
                size_detail.get().limit(), size_detail.get().actual())

    return check_status(status)


_FLIGHT_SERVER_ERROR_REGEX = re.compile(
    r'Flight RPC failed with message: (.*). Detail: '
    r'Python exception: (.*)',
    re.DOTALL
)


def _munge_grpc_python_error(message):
    m = _FLIGHT_SERVER_ERROR_REGEX.match(message)
    if m:
        return ('Flight RPC failed with Python exception \"{}: {}\"'
                .format(m.group(2), m.group(1)))
    else:
        return message


cdef IpcWriteOptions _get_options(options):
    return <IpcWriteOptions> _get_legacy_format_default(
        use_legacy_format=None, options=options)


cdef class FlightCallOptions(_Weakrefable):
    """RPC-layer options for a Flight call."""

    cdef:
        CFlightCallOptions options

    def __init__(self, timeout=None, write_options=None, headers=None,
                 IpcReadOptions read_options=None):
        """Create call options.

        Parameters
        ----------
        timeout : float, None
            A timeout for the call, in seconds. None means that the
            timeout defaults to an implementation-specific value.
        write_options : pyarrow.ipc.IpcWriteOptions, optional
            IPC write options. The default options can be controlled
            by environment variables (see pyarrow.ipc).
        headers : List[Tuple[str, str]], optional
            A list of arbitrary headers as key, value tuples
        read_options : pyarrow.ipc.IpcReadOptions, optional
            Serialization options for reading IPC format.
        """
        cdef IpcWriteOptions c_write_options
        cdef IpcReadOptions c_read_options

        if timeout is not None:
            self.options.timeout = CTimeoutDuration(timeout)
        if write_options is not None:
            c_write_options = _get_options(write_options)
            self.options.write_options = c_write_options.c_options
        if read_options is not None:
            if not isinstance(read_options, IpcReadOptions):
                raise TypeError("expected IpcReadOptions, got {}"
                                .format(type(read_options)))
            self.options.read_options = read_options.c_options
        if headers is not None:
            self.options.headers = headers

    @staticmethod
    cdef CFlightCallOptions* unwrap(obj):
        if not obj:
            return &DEFAULT_CALL_OPTIONS
        elif isinstance(obj, FlightCallOptions):
            return &((<FlightCallOptions> obj).options)
        raise TypeError("Expected a FlightCallOptions object, not "
                        "'{}'".format(type(obj)))


_CertKeyPair = collections.namedtuple('_CertKeyPair', ['cert', 'key'])


class CertKeyPair(_CertKeyPair):
    """A TLS certificate and key for use in Flight."""


cdef class FlightError(Exception):
    """
    The base class for Flight-specific errors.

    A server may raise this class or one of its subclasses to provide
    a more detailed error to clients.

    Parameters
    ----------
    message : str, optional
        The error message.
    extra_info : bytes, optional
        Extra binary error details that were provided by the
        server/will be sent to the client.

    Attributes
    ----------
    extra_info : bytes
        Extra binary error details that were provided by the
        server/will be sent to the client.
  """

    cdef dict __dict__

    def __init__(self, message='', extra_info=b''):
        super().__init__(message)
        self.extra_info = tobytes(extra_info)

    cdef CStatus to_status(self):
        message = tobytes("Flight error: {}".format(str(self)))
        return CStatus_UnknownError(message)


cdef class FlightInternalError(FlightError, ArrowException):
    """An error internal to the Flight server occurred."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusInternal,
                               tobytes(str(self)), self.extra_info)


cdef class FlightTimedOutError(FlightError, ArrowException):
    """The Flight RPC call timed out."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusTimedOut,
                               tobytes(str(self)), self.extra_info)


cdef class FlightCancelledError(FlightError, ArrowCancelled):
    """The operation was cancelled."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusCancelled, tobytes(str(self)),
                               self.extra_info)


cdef class FlightServerError(FlightError, ArrowException):
    """A server error occurred."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusFailed, tobytes(str(self)),
                               self.extra_info)


cdef class FlightUnauthenticatedError(FlightError, ArrowException):
    """The client is not authenticated."""

    cdef CStatus to_status(self):
        return MakeFlightError(
            CFlightStatusUnauthenticated, tobytes(str(self)), self.extra_info)


cdef class FlightUnauthorizedError(FlightError, ArrowException):
    """The client is not authorized to perform the given operation."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusUnauthorized, tobytes(str(self)),
                               self.extra_info)


cdef class FlightUnavailableError(FlightError, ArrowException):
    """The server is not reachable or available."""

    cdef CStatus to_status(self):
        return MakeFlightError(CFlightStatusUnavailable, tobytes(str(self)),
                               self.extra_info)


class FlightWriteSizeExceededError(ArrowInvalid):
    """A write operation exceeded the client-configured limit."""

    def __init__(self, message, limit, actual):
        super().__init__(message)
        self.limit = limit
        self.actual = actual


cdef class Action(_Weakrefable):
    """An action executable on a Flight service."""
    cdef:
        CAction action

    def __init__(self, action_type, buf):
        """Create an action from a type and a buffer.

        Parameters
        ----------
        action_type : bytes or str
        buf : Buffer or bytes-like object
        """
        self.action.type = tobytes(action_type)
        self.action.body = pyarrow_unwrap_buffer(as_buffer(buf))

    @property
    def type(self):
        """The action type."""
        return frombytes(self.action.type)

    @property
    def body(self):
        """The action body (arguments for the action)."""
        return pyarrow_wrap_buffer(self.action.body)

    @staticmethod
    cdef CAction unwrap(action) except *:
        if not isinstance(action, Action):
            raise TypeError("Must provide Action, not '{}'".format(
                type(action)))
        return (<Action> action).action

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.action.SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef Action action = Action.__new__(Action)
        action.action = GetResultValue(
            CAction.Deserialize(tobytes(serialized)))
        return action

    def __eq__(self, Action other):
        return self.action == other.action


_ActionType = collections.namedtuple('_ActionType', ['type', 'description'])


class ActionType(_ActionType):
    """A type of action that is executable on a Flight service."""

    def make_action(self, buf):
        """Create an Action with this type.

        Parameters
        ----------
        buf : obj
            An Arrow buffer or Python bytes or bytes-like object.
        """
        return Action(self.type, buf)


cdef class Result(_Weakrefable):
    """A result from executing an Action."""
    cdef:
        unique_ptr[CFlightResult] result

    def __init__(self, buf):
        """Create a new result.

        Parameters
        ----------
        buf : Buffer or bytes-like object
        """
        self.result.reset(new CFlightResult())
        self.result.get().body = pyarrow_unwrap_buffer(as_buffer(buf))

    @property
    def body(self):
        """Get the Buffer containing the result."""
        return pyarrow_wrap_buffer(self.result.get().body)

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.result.get().SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef Result result = Result.__new__(Result)
        result.result.reset(new CFlightResult(GetResultValue(
            CFlightResult.Deserialize(tobytes(serialized)))))
        return result

    def __eq__(self, Result other):
        return deref(self.result.get()) == deref(other.result.get())


cdef class BasicAuth(_Weakrefable):
    """A container for basic auth."""
    cdef:
        unique_ptr[CBasicAuth] basic_auth

    def __init__(self, username=None, password=None):
        """Create a new basic auth object.

        Parameters
        ----------
        username : string
        password : string
        """
        self.basic_auth.reset(new CBasicAuth())
        if username:
            self.basic_auth.get().username = tobytes(username)
        if password:
            self.basic_auth.get().password = tobytes(password)

    @property
    def username(self):
        """Get the username."""
        return self.basic_auth.get().username

    @property
    def password(self):
        """Get the password."""
        return self.basic_auth.get().password

    @staticmethod
    def deserialize(serialized):
        auth = BasicAuth()
        auth.basic_auth.reset(new CBasicAuth(GetResultValue(
            CBasicAuth.Deserialize(tobytes(serialized)))))
        return auth

    def serialize(self):
        return GetResultValue(self.basic_auth.get().SerializeToString())

    def __eq__(self, BasicAuth other):
        return deref(self.basic_auth.get()) == deref(other.basic_auth.get())


class DescriptorType(enum.Enum):
    """
    The type of a FlightDescriptor.

    Attributes
    ----------

    UNKNOWN
        An unknown descriptor type.

    PATH
        A Flight stream represented by a path.

    CMD
        A Flight stream represented by an application-defined command.

    """

    UNKNOWN = 0
    PATH = 1
    CMD = 2


class FlightMethod(enum.Enum):
    """The implemented methods in Flight."""

    INVALID = 0
    HANDSHAKE = 1
    LIST_FLIGHTS = 2
    GET_FLIGHT_INFO = 3
    GET_SCHEMA = 4
    DO_GET = 5
    DO_PUT = 6
    DO_ACTION = 7
    LIST_ACTIONS = 8
    DO_EXCHANGE = 9


cdef wrap_flight_method(CFlightMethod method):
    if method == CFlightMethodHandshake:
        return FlightMethod.HANDSHAKE
    elif method == CFlightMethodListFlights:
        return FlightMethod.LIST_FLIGHTS
    elif method == CFlightMethodGetFlightInfo:
        return FlightMethod.GET_FLIGHT_INFO
    elif method == CFlightMethodGetSchema:
        return FlightMethod.GET_SCHEMA
    elif method == CFlightMethodDoGet:
        return FlightMethod.DO_GET
    elif method == CFlightMethodDoPut:
        return FlightMethod.DO_PUT
    elif method == CFlightMethodDoAction:
        return FlightMethod.DO_ACTION
    elif method == CFlightMethodListActions:
        return FlightMethod.LIST_ACTIONS
    elif method == CFlightMethodDoExchange:
        return FlightMethod.DO_EXCHANGE
    return FlightMethod.INVALID


cdef class FlightDescriptor(_Weakrefable):
    """A description of a data stream available from a Flight service."""
    cdef:
        CFlightDescriptor descriptor

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "`pyarrow.flight.FlightDescriptor.for_{path,command}` "
                        "function instead."
                        .format(self.__class__.__name__))

    @staticmethod
    def for_path(*path):
        """Create a FlightDescriptor for a resource path."""
        cdef FlightDescriptor result = \
            FlightDescriptor.__new__(FlightDescriptor)
        result.descriptor.type = CDescriptorTypePath
        result.descriptor.path = [tobytes(p) for p in path]
        return result

    @staticmethod
    def for_command(command):
        """Create a FlightDescriptor for an opaque command."""
        cdef FlightDescriptor result = \
            FlightDescriptor.__new__(FlightDescriptor)
        result.descriptor.type = CDescriptorTypeCmd
        result.descriptor.cmd = tobytes(command)
        return result

    @property
    def descriptor_type(self):
        """Get the type of this descriptor."""
        if self.descriptor.type == CDescriptorTypeUnknown:
            return DescriptorType.UNKNOWN
        elif self.descriptor.type == CDescriptorTypePath:
            return DescriptorType.PATH
        elif self.descriptor.type == CDescriptorTypeCmd:
            return DescriptorType.CMD
        raise RuntimeError("Invalid descriptor type!")

    @property
    def command(self):
        """Get the command for this descriptor."""
        if self.descriptor_type != DescriptorType.CMD:
            return None
        return self.descriptor.cmd

    @property
    def path(self):
        """Get the path for this descriptor."""
        if self.descriptor_type != DescriptorType.PATH:
            return None
        return self.descriptor.path

    def __repr__(self):
        if self.descriptor_type == DescriptorType.PATH:
            return "<FlightDescriptor path: {!r}>".format(self.path)
        elif self.descriptor_type == DescriptorType.CMD:
            return "<FlightDescriptor command: {!r}>".format(self.command)
        else:
            return "<FlightDescriptor type: {!r}>".format(self.descriptor_type)

    @staticmethod
    cdef CFlightDescriptor unwrap(descriptor) except *:
        if not isinstance(descriptor, FlightDescriptor):
            raise TypeError("Must provide a FlightDescriptor, not '{}'".format(
                type(descriptor)))
        return (<FlightDescriptor> descriptor).descriptor

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.descriptor.SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef FlightDescriptor descriptor = \
            FlightDescriptor.__new__(FlightDescriptor)
        descriptor.descriptor = GetResultValue(
            CFlightDescriptor.Deserialize(tobytes(serialized)))
        return descriptor

    def __eq__(self, FlightDescriptor other):
        return self.descriptor == other.descriptor


cdef class Ticket(_Weakrefable):
    """A ticket for requesting a Flight stream."""

    cdef:
        CTicket ticket

    def __init__(self, ticket):
        self.ticket.ticket = tobytes(ticket)

    @property
    def ticket(self):
        return self.ticket.ticket

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.ticket.SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef Ticket ticket = Ticket.__new__(Ticket)
        ticket.ticket = GetResultValue(
            CTicket.Deserialize(tobytes(serialized)))
        return ticket

    def __eq__(self, Ticket other):
        return self.ticket == other.ticket

    def __repr__(self):
        return '<Ticket {}>'.format(self.ticket.ticket)


cdef class Location(_Weakrefable):
    """The location of a Flight service."""
    cdef:
        CLocation location

    def __init__(self, uri):
        check_flight_status(CLocation.Parse(tobytes(uri)).Value(&self.location))

    def __repr__(self):
        return '<Location {}>'.format(self.location.ToString())

    @property
    def uri(self):
        return self.location.ToString()

    def equals(self, Location other):
        return self == other

    def __eq__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return self.location.Equals((<Location> other).location)

    @staticmethod
    def for_grpc_tcp(host, port):
        """Create a Location for a TCP-based gRPC service."""
        cdef:
            c_string c_host = tobytes(host)
            int c_port = port
            Location result = Location.__new__(Location)
        check_flight_status(
            CLocation.ForGrpcTcp(c_host, c_port).Value(&result.location))
        return result

    @staticmethod
    def for_grpc_tls(host, port):
        """Create a Location for a TLS-based gRPC service."""
        cdef:
            c_string c_host = tobytes(host)
            int c_port = port
            Location result = Location.__new__(Location)
        check_flight_status(
            CLocation.ForGrpcTls(c_host, c_port).Value(&result.location))
        return result

    @staticmethod
    def for_grpc_unix(path):
        """Create a Location for a domain socket-based gRPC service."""
        cdef:
            c_string c_path = tobytes(path)
            Location result = Location.__new__(Location)
        check_flight_status(CLocation.ForGrpcUnix(c_path).Value(&result.location))
        return result

    @staticmethod
    cdef Location wrap(CLocation location):
        cdef Location result = Location.__new__(Location)
        result.location = location
        return result

    @staticmethod
    cdef CLocation unwrap(object location) except *:
        cdef CLocation c_location
        if isinstance(location, str):
            check_flight_status(
                CLocation.Parse(tobytes(location)).Value(&c_location))
            return c_location
        elif not isinstance(location, Location):
            raise TypeError("Must provide a Location, not '{}'".format(
                type(location)))
        return (<Location> location).location


cdef class FlightEndpoint(_Weakrefable):
    """A Flight stream, along with the ticket and locations to access it."""
    cdef:
        CFlightEndpoint endpoint

    def __init__(self, ticket, locations):
        """Create a FlightEndpoint from a ticket and list of locations.

        Parameters
        ----------
        ticket : Ticket or bytes
            the ticket needed to access this flight
        locations : list of string URIs
            locations where this flight is available

        Raises
        ------
        ArrowException
            If one of the location URIs is not a valid URI.
        """
        cdef:
            CLocation c_location

        if isinstance(ticket, Ticket):
            self.endpoint.ticket.ticket = tobytes(ticket.ticket)
        else:
            self.endpoint.ticket.ticket = tobytes(ticket)

        for location in locations:
            if isinstance(location, Location):
                c_location = (<Location> location).location
            else:
                c_location = CLocation()
                check_flight_status(
                    CLocation.Parse(tobytes(location)).Value(&c_location))
            self.endpoint.locations.push_back(c_location)

    @property
    def ticket(self):
        """Get the ticket in this endpoint."""
        return Ticket(self.endpoint.ticket.ticket)

    @property
    def locations(self):
        return [Location.wrap(location)
                for location in self.endpoint.locations]

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.endpoint.SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef FlightEndpoint endpoint = FlightEndpoint.__new__(FlightEndpoint)
        endpoint.endpoint = GetResultValue(
            CFlightEndpoint.Deserialize(tobytes(serialized)))
        return endpoint

    def __repr__(self):
        return "<FlightEndpoint ticket: {!r} locations: {!r}>".format(
            self.ticket, self.locations)

    def __eq__(self, FlightEndpoint other):
        return self.endpoint == other.endpoint


cdef class SchemaResult(_Weakrefable):
    """A result from a getschema request. Holding a schema"""
    cdef:
        unique_ptr[CSchemaResult] result

    def __init__(self, Schema schema):
        """Create a SchemaResult from a schema.

        Parameters
        ----------
        schema: Schema
            the schema of the data in this flight.
        """
        cdef:
            shared_ptr[CSchema] c_schema = pyarrow_unwrap_schema(schema)
        check_flight_status(CreateSchemaResult(c_schema, &self.result))

    @property
    def schema(self):
        """The schema of the data in this flight."""
        cdef:
            shared_ptr[CSchema] schema
            CDictionaryMemo dummy_memo

        check_flight_status(self.result.get().GetSchema(&dummy_memo).Value(&schema))
        return pyarrow_wrap_schema(schema)

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.result.get().SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef SchemaResult result = SchemaResult.__new__(SchemaResult)
        result.result.reset(new CSchemaResult(GetResultValue(
            CSchemaResult.Deserialize(tobytes(serialized)))))
        return result

    def __eq__(self, SchemaResult other):
        return deref(self.result.get()) == deref(other.result.get())


cdef class FlightInfo(_Weakrefable):
    """A description of a Flight stream."""
    cdef:
        unique_ptr[CFlightInfo] info

    def __init__(self, Schema schema, FlightDescriptor descriptor, endpoints,
                 total_records, total_bytes):
        """Create a FlightInfo object from a schema, descriptor, and endpoints.

        Parameters
        ----------
        schema : Schema
            the schema of the data in this flight.
        descriptor : FlightDescriptor
            the descriptor for this flight.
        endpoints : list of FlightEndpoint
            a list of endpoints where this flight is available.
        total_records : int
            the total records in this flight, or -1 if unknown
        total_bytes : int
            the total bytes in this flight, or -1 if unknown
        """
        cdef:
            shared_ptr[CSchema] c_schema = pyarrow_unwrap_schema(schema)
            vector[CFlightEndpoint] c_endpoints

        for endpoint in endpoints:
            if isinstance(endpoint, FlightEndpoint):
                c_endpoints.push_back((<FlightEndpoint> endpoint).endpoint)
            else:
                raise TypeError('Endpoint {} is not instance of'
                                ' FlightEndpoint'.format(endpoint))

        check_flight_status(CreateFlightInfo(c_schema,
                                             descriptor.descriptor,
                                             c_endpoints,
                                             total_records,
                                             total_bytes, &self.info))

    @property
    def total_records(self):
        """The total record count of this flight, or -1 if unknown."""
        return self.info.get().total_records()

    @property
    def total_bytes(self):
        """The size in bytes of the data in this flight, or -1 if unknown."""
        return self.info.get().total_bytes()

    @property
    def schema(self):
        """The schema of the data in this flight."""
        cdef:
            shared_ptr[CSchema] schema
            CDictionaryMemo dummy_memo

        check_flight_status(self.info.get().GetSchema(&dummy_memo).Value(&schema))
        return pyarrow_wrap_schema(schema)

    @property
    def descriptor(self):
        """The descriptor of the data in this flight."""
        cdef FlightDescriptor result = \
            FlightDescriptor.__new__(FlightDescriptor)
        result.descriptor = self.info.get().descriptor()
        return result

    @property
    def endpoints(self):
        """The endpoints where this flight is available."""
        # TODO: get Cython to iterate over reference directly
        cdef:
            vector[CFlightEndpoint] endpoints = self.info.get().endpoints()
            FlightEndpoint py_endpoint

        result = []
        for endpoint in endpoints:
            py_endpoint = FlightEndpoint.__new__(FlightEndpoint)
            py_endpoint.endpoint = endpoint
            result.append(py_endpoint)
        return result

    def serialize(self):
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        return GetResultValue(self.info.get().SerializeToString())

    @classmethod
    def deserialize(cls, serialized):
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
        cdef FlightInfo info = FlightInfo.__new__(FlightInfo)
        info.info = move(GetResultValue(
            CFlightInfo.Deserialize(tobytes(serialized))))
        return info


cdef class FlightStreamChunk(_Weakrefable):
    """A RecordBatch with application metadata on the side."""
    cdef:
        CFlightStreamChunk chunk

    @property
    def data(self):
        if self.chunk.data == NULL:
            return None
        return pyarrow_wrap_batch(self.chunk.data)

    @property
    def app_metadata(self):
        if self.chunk.app_metadata == NULL:
            return None
        return pyarrow_wrap_buffer(self.chunk.app_metadata)

    def __iter__(self):
        return iter((self.data, self.app_metadata))

    def __repr__(self):
        return "<FlightStreamChunk with data: {} with metadata: {}>".format(
            self.chunk.data != NULL, self.chunk.app_metadata != NULL)


cdef class _MetadataRecordBatchReader(_Weakrefable, _ReadPandasMixin):
    """A reader for Flight streams."""

    # Needs to be separate class so the "real" class can subclass the
    # pure-Python mixin class

    cdef dict __dict__
    cdef shared_ptr[CMetadataRecordBatchReader] reader

    def __iter__(self):
        while True:
            yield self.read_chunk()

    @property
    def schema(self):
        """Get the schema for this reader."""
        cdef shared_ptr[CSchema] c_schema
        with nogil:
            check_flight_status(self.reader.get().GetSchema().Value(&c_schema))
        return pyarrow_wrap_schema(c_schema)

    def read_all(self):
        """Read the entire contents of the stream as a Table."""
        cdef:
            shared_ptr[CTable] c_table
        with nogil:
            check_flight_status(self.reader.get().ToTable().Value(&c_table))
        return pyarrow_wrap_table(c_table)

    def read_chunk(self):
        """Read the next RecordBatch along with any metadata.

        Returns
        -------
        data : RecordBatch
            The next RecordBatch in the stream.
        app_metadata : Buffer or None
            Application-specific metadata for the batch as defined by
            Flight.

        Raises
        ------
        StopIteration
            when the stream is finished
        """
        cdef:
            FlightStreamChunk chunk = FlightStreamChunk()

        with nogil:
            check_flight_status(self.reader.get().Next().Value(&chunk.chunk))

        if chunk.chunk.data == NULL and chunk.chunk.app_metadata == NULL:
            raise StopIteration

        return chunk

    def to_reader(self):
        """Convert this reader into a regular RecordBatchReader.

        This may fail if the schema cannot be read from the remote end.

        Returns
        -------
        RecordBatchReader
        """
        cdef RecordBatchReader reader
        reader = RecordBatchReader.__new__(RecordBatchReader)
        reader.reader = GetResultValue(MakeRecordBatchReader(self.reader))
        return reader


cdef class MetadataRecordBatchReader(_MetadataRecordBatchReader):
    """The base class for readers for Flight streams.

    See Also
    --------
    FlightStreamReader
    """


cdef class FlightStreamReader(MetadataRecordBatchReader):
    """A reader that can also be canceled."""

    def cancel(self):
        """Cancel the read operation."""
        with nogil:
            (<CFlightStreamReader*> self.reader.get()).Cancel()

    def read_all(self):
        """Read the entire contents of the stream as a Table."""
        cdef:
            shared_ptr[CTable] c_table
            CStopToken stop_token
        with SignalStopHandler() as stop_handler:
            stop_token = (<StopToken> stop_handler.stop_token).stop_token
            with nogil:
                check_flight_status(
                    (<CFlightStreamReader*> self.reader.get())
                    .ToTableWithStopToken(stop_token).Value(&c_table))
        return pyarrow_wrap_table(c_table)


cdef class MetadataRecordBatchWriter(_CRecordBatchWriter):
    """A RecordBatchWriter that also allows writing application metadata.

    This class is a context manager; on exit, close() will be called.
    """

    cdef CMetadataRecordBatchWriter* _writer(self) nogil:
        return <CMetadataRecordBatchWriter*> self.writer.get()

    def begin(self, schema: Schema, options=None):
        """Prepare to write data to this stream with the given schema."""
        cdef:
            shared_ptr[CSchema] c_schema = pyarrow_unwrap_schema(schema)
            CIpcWriteOptions c_options = _get_options(options).c_options
        with nogil:
            check_flight_status(self._writer().Begin(c_schema, c_options))

    def write_metadata(self, buf):
        """Write Flight metadata by itself."""
        cdef shared_ptr[CBuffer] c_buf = pyarrow_unwrap_buffer(as_buffer(buf))
        with nogil:
            check_flight_status(
                self._writer().WriteMetadata(c_buf))

    def write_batch(self, RecordBatch batch):
        """
        Write RecordBatch to stream.

        Parameters
        ----------
        batch : RecordBatch
        """
        cdef:
            shared_ptr[const CKeyValueMetadata] custom_metadata

        # Override superclass method to use check_flight_status so we
        # can generate FlightWriteSizeExceededError. We don't do this
        # for write_table as callers who intend to handle the error
        # and retry with a smaller batch should be working with
        # individual batches to have control.

        with nogil:
            check_flight_status(
                self._writer().WriteRecordBatch(deref(batch.batch), custom_metadata))

    def write_table(self, Table table, max_chunksize=None, **kwargs):
        """
        Write Table to stream in (contiguous) RecordBatch objects.

        Parameters
        ----------
        table : Table
        max_chunksize : int, default None
            Maximum size for RecordBatch chunks. Individual chunks may be
            smaller depending on the chunk layout of individual columns.
        """
        cdef:
            # max_chunksize must be > 0 to have any impact
            int64_t c_max_chunksize = -1

        if 'chunksize' in kwargs:
            max_chunksize = kwargs['chunksize']
            msg = ('The parameter chunksize is deprecated for the write_table '
                   'methods as of 0.15, please use parameter '
                   'max_chunksize instead')
            warnings.warn(msg, FutureWarning)

        if max_chunksize is not None:
            c_max_chunksize = max_chunksize

        with nogil:
            check_flight_status(
                self._writer().WriteTable(table.table[0], c_max_chunksize))

    def close(self):
        """
        Close stream and write end-of-stream 0 marker.
        """
        with nogil:
            check_flight_status(self._writer().Close())

    def write_with_metadata(self, RecordBatch batch, buf):
        """Write a RecordBatch along with Flight metadata.

        Parameters
        ----------
        batch : RecordBatch
            The next RecordBatch in the stream.
        buf : Buffer
            Application-specific metadata for the batch as defined by
            Flight.
        """
        cdef shared_ptr[CBuffer] c_buf = pyarrow_unwrap_buffer(as_buffer(buf))
        with nogil:
            check_flight_status(
                self._writer().WriteWithMetadata(deref(batch.batch), c_buf))


cdef class FlightStreamWriter(MetadataRecordBatchWriter):
    """A writer that also allows closing the write side of a stream."""

    def done_writing(self):
        """Indicate that the client is done writing, but not done reading."""
        with nogil:
            check_flight_status(
                (<CFlightStreamWriter*> self.writer.get()).DoneWriting())


cdef class FlightMetadataReader(_Weakrefable):
    """A reader for Flight metadata messages sent during a DoPut."""

    cdef:
        unique_ptr[CFlightMetadataReader] reader

    def read(self):
        """Read the next metadata message."""
        cdef shared_ptr[CBuffer] buf
        with nogil:
            check_flight_status(self.reader.get().ReadMetadata(&buf))
        if buf == NULL:
            return None
        return pyarrow_wrap_buffer(buf)


cdef class FlightMetadataWriter(_Weakrefable):
    """A sender for Flight metadata messages during a DoPut."""

    cdef:
        unique_ptr[CFlightMetadataWriter] writer

    def write(self, message):
        """Write the next metadata message.

        Parameters
        ----------
        message : Buffer
        """
        cdef shared_ptr[CBuffer] buf = \
            pyarrow_unwrap_buffer(as_buffer(message))
        with nogil:
            check_flight_status(self.writer.get().WriteMetadata(deref(buf)))


cdef class FlightClient(_Weakrefable):
    """A client to a Flight service.

    Connect to a Flight service on the given host and port.

    Parameters
    ----------
    location : str, tuple or Location
        Location to connect to. Either a gRPC URI like `grpc://localhost:port`,
        a tuple of (host, port) pair, or a Location instance.
    tls_root_certs : bytes or None
        PEM-encoded
    cert_chain: bytes or None
        Client certificate if using mutual TLS
    private_key: bytes or None
        Client private key for cert_chain is using mutual TLS
    override_hostname : str or None
        Override the hostname checked by TLS. Insecure, use with caution.
    middleware : list optional, default None
        A list of ClientMiddlewareFactory instances.
    write_size_limit_bytes : int optional, default None
        A soft limit on the size of a data payload sent to the
        server. Enabled if positive. If enabled, writing a record
        batch that (when serialized) exceeds this limit will raise an
        exception; the client can retry the write with a smaller
        batch.
    disable_server_verification : boolean optional, default False
        A flag that indicates that, if the client is connecting
        with TLS, that it skips server verification. If this is
        enabled, all other TLS settings are overridden.
    generic_options : list optional, default None
        A list of generic (string, int or string) option tuples passed
        to the underlying transport. Effect is implementation
        dependent.
    """
    cdef:
        unique_ptr[CFlightClient] client

    def __init__(self, location, *, tls_root_certs=None, cert_chain=None,
                 private_key=None, override_hostname=None, middleware=None,
                 write_size_limit_bytes=None,
                 disable_server_verification=None, generic_options=None):
        if isinstance(location, (bytes, str)):
            location = Location(location)
        elif isinstance(location, tuple):
            host, port = location
            if tls_root_certs or disable_server_verification is not None:
                location = Location.for_grpc_tls(host, port)
            else:
                location = Location.for_grpc_tcp(host, port)
        elif not isinstance(location, Location):
            raise TypeError('`location` argument must be a string, tuple or a '
                            'Location instance')
        self.init(location, tls_root_certs, cert_chain, private_key,
                  override_hostname, middleware, write_size_limit_bytes,
                  disable_server_verification, generic_options)

    cdef init(self, Location location, tls_root_certs, cert_chain,
              private_key, override_hostname, middleware,
              write_size_limit_bytes, disable_server_verification,
              generic_options):
        cdef:
            int c_port = 0
            CLocation c_location = Location.unwrap(location)
            CFlightClientOptions c_options = CFlightClientOptions.Defaults()
            function[cb_client_middleware_start_call] start_call = \
                &_client_middleware_start_call
            CIntStringVariant variant

        if tls_root_certs:
            c_options.tls_root_certs = tobytes(tls_root_certs)
        if cert_chain:
            c_options.cert_chain = tobytes(cert_chain)
        if private_key:
            c_options.private_key = tobytes(private_key)
        if override_hostname:
            c_options.override_hostname = tobytes(override_hostname)
        if disable_server_verification is not None:
            c_options.disable_server_verification = disable_server_verification
        if middleware:
            for factory in middleware:
                c_options.middleware.push_back(
                    <shared_ptr[CClientMiddlewareFactory]>
                    make_shared[CPyClientMiddlewareFactory](
                        <PyObject*> factory, start_call))
        if write_size_limit_bytes is not None:
            c_options.write_size_limit_bytes = write_size_limit_bytes
        else:
            c_options.write_size_limit_bytes = 0
        if generic_options:
            for key, value in generic_options:
                if isinstance(value, (str, bytes)):
                    variant = CIntStringVariant(<c_string> tobytes(value))
                else:
                    variant = CIntStringVariant(<int> value)
                c_options.generic_options.push_back(
                    pair[c_string, CIntStringVariant](tobytes(key), variant))

        with nogil:
            check_flight_status(CFlightClient.Connect(c_location, c_options
                                                      ).Value(&self.client))

    def wait_for_available(self, timeout=5):
        """Block until the server can be contacted.

        Parameters
        ----------
        timeout : int, default 5
            The maximum seconds to wait.
        """
        deadline = time.time() + timeout
        while True:
            try:
                list(self.list_flights())
            except FlightUnavailableError:
                if time.time() < deadline:
                    time.sleep(0.025)
                    continue
                else:
                    raise
            except NotImplementedError:
                # allow if list_flights is not implemented, because
                # the server can be contacted nonetheless
                break
            else:
                break

    @classmethod
    def connect(cls, location, tls_root_certs=None, cert_chain=None,
                private_key=None, override_hostname=None,
                disable_server_verification=None):
        """Connect to a Flight server.

        .. deprecated:: 0.15.0
            Use the ``FlightClient`` constructor or ``pyarrow.flight.connect`` function instead.
        """
        warnings.warn("The 'FlightClient.connect' method is deprecated, use "
                      "FlightClient constructor or pyarrow.flight.connect "
                      "function instead")
        return FlightClient(
            location, tls_root_certs=tls_root_certs,
            cert_chain=cert_chain, private_key=private_key,
            override_hostname=override_hostname,
            disable_server_verification=disable_server_verification
        )

    def authenticate(self, auth_handler, options: FlightCallOptions = None):
        """Authenticate to the server.

        Parameters
        ----------
        auth_handler : ClientAuthHandler
            The authentication mechanism to use.
        options : FlightCallOptions
            Options for this call.
        """
        cdef:
            unique_ptr[CClientAuthHandler] handler
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)

        if not isinstance(auth_handler, ClientAuthHandler):
            raise TypeError(
                "FlightClient.authenticate takes a ClientAuthHandler, "
                "not '{}'".format(type(auth_handler)))
        handler.reset((<ClientAuthHandler> auth_handler).to_handler())
        with nogil:
            check_flight_status(
                self.client.get().Authenticate(deref(c_options),
                                               move(handler)))

    def authenticate_basic_token(self, username, password,
                                 options: FlightCallOptions = None):
        """Authenticate to the server with HTTP basic authentication.

        Parameters
        ----------
        username : string
            Username to authenticate with
        password : string
            Password to authenticate with
        options  : FlightCallOptions
            Options for this call

        Returns
        -------
        tuple : Tuple[str, str]
            A tuple representing the FlightCallOptions authorization
            header entry of a bearer token.
        """
        cdef:
            CResult[pair[c_string, c_string]] result
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            c_string user = tobytes(username)
            c_string pw = tobytes(password)

        with nogil:
            result = self.client.get().AuthenticateBasicToken(deref(c_options),
                                                              user, pw)
            check_flight_status(result.status())

        return GetResultValue(result)

    def list_actions(self, options: FlightCallOptions = None):
        """List the actions available on a service."""
        cdef:
            vector[CActionType] results
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)

        with SignalStopHandler() as stop_handler:
            c_options.stop_token = \
                (<StopToken> stop_handler.stop_token).stop_token
            with nogil:
                check_flight_status(
                    self.client.get().ListActions(deref(c_options)).Value(&results))

            result = []
            for action_type in results:
                py_action = ActionType(frombytes(action_type.type),
                                       frombytes(action_type.description))
                result.append(py_action)

            return result

    def do_action(self, action, options: FlightCallOptions = None):
        """
        Execute an action on a service.

        Parameters
        ----------
        action : str, tuple, or Action
            Can be action type name (no body), type and body, or any Action
            object
        options : FlightCallOptions
            RPC options

        Returns
        -------
        results : iterator of Result values
        """
        cdef:
            unique_ptr[CResultStream] results
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)

        if isinstance(action, (str, bytes)):
            action = Action(action, b'')
        elif isinstance(action, tuple):
            action = Action(*action)
        elif not isinstance(action, Action):
            raise TypeError("Action must be Action instance, string, or tuple")

        cdef CAction c_action = Action.unwrap(<Action> action)
        with nogil:
            check_flight_status(
                self.client.get().DoAction(
                    deref(c_options), c_action).Value(&results))

        def _do_action_response():
            cdef:
                Result result
            while True:
                result = Result.__new__(Result)
                with nogil:
                    check_flight_status(results.get().Next().Value(&result.result))
                    if result.result == NULL:
                        break
                yield result
        return _do_action_response()

    def list_flights(self, criteria: bytes = None,
                     options: FlightCallOptions = None):
        """List the flights available on a service."""
        cdef:
            unique_ptr[CFlightListing] listing
            FlightInfo result
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            CCriteria c_criteria

        if criteria:
            c_criteria.expression = tobytes(criteria)

        with SignalStopHandler() as stop_handler:
            c_options.stop_token = \
                (<StopToken> stop_handler.stop_token).stop_token
            with nogil:
                check_flight_status(
                    self.client.get().ListFlights(deref(c_options),
                                                  c_criteria).Value(&listing))

            while True:
                result = FlightInfo.__new__(FlightInfo)
                with nogil:
                    check_flight_status(listing.get().Next().Value(&result.info))
                    if result.info == NULL:
                        break
                yield result

    def get_flight_info(self, descriptor: FlightDescriptor,
                        options: FlightCallOptions = None):
        """Request information about an available flight."""
        cdef:
            FlightInfo result = FlightInfo.__new__(FlightInfo)
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            CFlightDescriptor c_descriptor = \
                FlightDescriptor.unwrap(descriptor)

        with nogil:
            check_flight_status(self.client.get().GetFlightInfo(
                deref(c_options), c_descriptor).Value(&result.info))

        return result

    def get_schema(self, descriptor: FlightDescriptor,
                   options: FlightCallOptions = None):
        """Request schema for an available flight."""
        cdef:
            SchemaResult result = SchemaResult.__new__(SchemaResult)
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            CFlightDescriptor c_descriptor = \
                FlightDescriptor.unwrap(descriptor)
        with nogil:
            check_status(
                self.client.get()
                    .GetSchema(deref(c_options), c_descriptor).Value(&result.result)
            )

        return result

    def do_get(self, ticket: Ticket, options: FlightCallOptions = None):
        """Request the data for a flight.

        Returns
        -------
        reader : FlightStreamReader
        """
        cdef:
            unique_ptr[CFlightStreamReader] reader
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)

        with nogil:
            check_flight_status(
                self.client.get().DoGet(
                    deref(c_options), ticket.ticket).Value(&reader))
        result = FlightStreamReader()
        result.reader.reset(reader.release())
        return result

    def do_put(self, descriptor: FlightDescriptor, Schema schema not None,
               options: FlightCallOptions = None):
        """Upload data to a flight.

        Returns
        -------
        writer : FlightStreamWriter
        reader : FlightMetadataReader
        """
        cdef:
            shared_ptr[CSchema] c_schema = pyarrow_unwrap_schema(schema)
            CDoPutResult c_do_put_result
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            CFlightDescriptor c_descriptor = \
                FlightDescriptor.unwrap(descriptor)

        with nogil:
            check_flight_status(self.client.get().DoPut(
                deref(c_options),
                c_descriptor,
                c_schema).Value(&c_do_put_result))
        py_writer = FlightStreamWriter()
        py_writer.writer.reset(c_do_put_result.writer.release())
        py_reader = FlightMetadataReader()
        py_reader.reader.reset(c_do_put_result.reader.release())
        return py_writer, py_reader

    def do_exchange(self, descriptor: FlightDescriptor,
                    options: FlightCallOptions = None):
        """Start a bidirectional data exchange with a server.

        Parameters
        ----------
        descriptor : FlightDescriptor
            A descriptor for the flight.
        options : FlightCallOptions
            RPC options.

        Returns
        -------
        writer : FlightStreamWriter
        reader : FlightStreamReader
        """
        cdef:
            CDoExchangeResult c_do_exchange_result
            CFlightCallOptions* c_options = FlightCallOptions.unwrap(options)
            CFlightDescriptor c_descriptor = \
                FlightDescriptor.unwrap(descriptor)

        with nogil:
            check_flight_status(self.client.get().DoExchange(
                deref(c_options),
                c_descriptor).Value(&c_do_exchange_result))
        py_writer = FlightStreamWriter()
        py_writer.writer.reset(c_do_exchange_result.writer.release())
        py_reader = FlightStreamReader()
        py_reader.reader.reset(c_do_exchange_result.reader.release())
        return py_writer, py_reader

    def close(self):
        """Close the client and disconnect."""
        check_flight_status(self.client.get().Close())

    def __del__(self):
        # Not ideal, but close() wasn't originally present so
        # applications may not be calling it
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


cdef class FlightDataStream(_Weakrefable):
    """
    Abstract base class for Flight data streams.

    See Also
    --------
    RecordBatchStream
    GeneratorStream
    """

    cdef CFlightDataStream* to_stream(self) except *:
        """Create the C++ data stream for the backing Python object.

        We don't expose the C++ object to Python, so we can manage its
        lifetime from the Cython/C++ side.
        """
        raise NotImplementedError


cdef class RecordBatchStream(FlightDataStream):
    """A Flight data stream backed by RecordBatches.

    The remainder of this DoGet request will be handled in C++,
    without having to acquire the GIL.

    """
    cdef:
        object data_source
        CIpcWriteOptions write_options

    def __init__(self, data_source, options=None):
        """Create a RecordBatchStream from a data source.

        Parameters
        ----------
        data_source : RecordBatchReader or Table
            The data to stream to the client.
        options : pyarrow.ipc.IpcWriteOptions, optional
            Optional IPC options to control how to write the data.
        """
        if (not isinstance(data_source, RecordBatchReader) and
                not isinstance(data_source, lib.Table)):
            raise TypeError("Expected RecordBatchReader or Table, "
                            "but got: {}".format(type(data_source)))
        self.data_source = data_source
        self.write_options = _get_options(options).c_options

    cdef CFlightDataStream* to_stream(self) except *:
        cdef:
            shared_ptr[CRecordBatchReader] reader
        if isinstance(self.data_source, RecordBatchReader):
            reader = (<RecordBatchReader> self.data_source).reader
        elif isinstance(self.data_source, lib.Table):
            table = (<Table> self.data_source).table
            reader.reset(new TableBatchReader(deref(table)))
        else:
            raise RuntimeError("Can't construct RecordBatchStream "
                               "from type {}".format(type(self.data_source)))
        return new CRecordBatchStream(reader, self.write_options)


cdef class GeneratorStream(FlightDataStream):
    """A Flight data stream backed by a Python generator."""
    cdef:
        shared_ptr[CSchema] schema
        object generator
        # A substream currently being consumed by the client, if
        # present. Produced by the generator.
        unique_ptr[CFlightDataStream] current_stream
        CIpcWriteOptions c_options

    def __init__(self, schema, generator, options=None):
        """Create a GeneratorStream from a Python generator.

        Parameters
        ----------
        schema : Schema
            The schema for the data to be returned.

        generator : iterator or iterable
            The generator should yield other FlightDataStream objects,
            Tables, RecordBatches, or RecordBatchReaders.

        options : pyarrow.ipc.IpcWriteOptions, optional
        """
        self.schema = pyarrow_unwrap_schema(schema)
        self.generator = iter(generator)
        self.c_options = _get_options(options).c_options

    cdef CFlightDataStream* to_stream(self) except *:
        cdef:
            function[cb_data_stream_next] callback = &_data_stream_next
        return new CPyGeneratorFlightDataStream(self, self.schema, callback,
                                                self.c_options)


cdef class ServerCallContext(_Weakrefable):
    """Per-call state/context."""
    cdef:
        const CServerCallContext* context

    def peer_identity(self):
        """Get the identity of the authenticated peer.

        May be the empty string.
        """
        return tobytes(self.context.peer_identity())

    def peer(self):
        """Get the address of the peer."""
        # Set safe=True as gRPC on Windows sometimes gives garbage bytes
        return frombytes(self.context.peer(), safe=True)

    def is_cancelled(self):
        """Check if the current RPC call has been canceled by the client."""
        return self.context.is_cancelled()

    def get_middleware(self, key):
        """
        Get a middleware instance by key.

        Returns None if the middleware was not found.
        """
        cdef:
            CServerMiddleware* c_middleware = \
                self.context.GetMiddleware(CPyServerMiddlewareName)
            CPyServerMiddleware* middleware
            vector[CTracingServerMiddlewareTraceKey] c_trace_context
        if c_middleware == NULL:
            c_middleware = self.context.GetMiddleware(tobytes(key))

        if c_middleware == NULL:
            return None
        elif c_middleware.name() == CPyServerMiddlewareName:
            middleware = <CPyServerMiddleware*> c_middleware
            py_middleware = <_ServerMiddlewareWrapper> middleware.py_object()
            return py_middleware.middleware.get(key)
        elif c_middleware.name() == CTracingServerMiddlewareName:
            c_trace_context = (<CTracingServerMiddleware*> c_middleware
                               ).GetTraceContext()
            trace_context = {pair.key: pair.value for pair in c_trace_context}
            return TracingServerMiddleware(trace_context)
        return None

    @staticmethod
    cdef ServerCallContext wrap(const CServerCallContext& context):
        cdef ServerCallContext result = \
            ServerCallContext.__new__(ServerCallContext)
        result.context = &context
        return result


cdef class ServerAuthReader(_Weakrefable):
    """A reader for messages from the client during an auth handshake."""
    cdef:
        CServerAuthReader* reader

    def read(self):
        cdef c_string token
        if not self.reader:
            raise ValueError("Cannot use ServerAuthReader outside "
                             "ServerAuthHandler.authenticate")
        with nogil:
            check_flight_status(self.reader.Read(&token))
        return token

    cdef void poison(self):
        """Prevent further usage of this object.

        This object is constructed by taking a pointer to a reference,
        so we want to make sure Python users do not access this after
        the reference goes away.
        """
        self.reader = NULL

    @staticmethod
    cdef ServerAuthReader wrap(CServerAuthReader* reader):
        cdef ServerAuthReader result = \
            ServerAuthReader.__new__(ServerAuthReader)
        result.reader = reader
        return result


cdef class ServerAuthSender(_Weakrefable):
    """A writer for messages to the client during an auth handshake."""
    cdef:
        CServerAuthSender* sender

    def write(self, message):
        cdef c_string c_message = tobytes(message)
        if not self.sender:
            raise ValueError("Cannot use ServerAuthSender outside "
                             "ServerAuthHandler.authenticate")
        with nogil:
            check_flight_status(self.sender.Write(c_message))

    cdef void poison(self):
        """Prevent further usage of this object.

        This object is constructed by taking a pointer to a reference,
        so we want to make sure Python users do not access this after
        the reference goes away.
        """
        self.sender = NULL

    @staticmethod
    cdef ServerAuthSender wrap(CServerAuthSender* sender):
        cdef ServerAuthSender result = \
            ServerAuthSender.__new__(ServerAuthSender)
        result.sender = sender
        return result


cdef class ClientAuthReader(_Weakrefable):
    """A reader for messages from the server during an auth handshake."""
    cdef:
        CClientAuthReader* reader

    def read(self):
        cdef c_string token
        if not self.reader:
            raise ValueError("Cannot use ClientAuthReader outside "
                             "ClientAuthHandler.authenticate")
        with nogil:
            check_flight_status(self.reader.Read(&token))
        return token

    cdef void poison(self):
        """Prevent further usage of this object.

        This object is constructed by taking a pointer to a reference,
        so we want to make sure Python users do not access this after
        the reference goes away.
        """
        self.reader = NULL

    @staticmethod
    cdef ClientAuthReader wrap(CClientAuthReader* reader):
        cdef ClientAuthReader result = \
            ClientAuthReader.__new__(ClientAuthReader)
        result.reader = reader
        return result


cdef class ClientAuthSender(_Weakrefable):
    """A writer for messages to the server during an auth handshake."""
    cdef:
        CClientAuthSender* sender

    def write(self, message):
        cdef c_string c_message = tobytes(message)
        if not self.sender:
            raise ValueError("Cannot use ClientAuthSender outside "
                             "ClientAuthHandler.authenticate")
        with nogil:
            check_flight_status(self.sender.Write(c_message))

    cdef void poison(self):
        """Prevent further usage of this object.

        This object is constructed by taking a pointer to a reference,
        so we want to make sure Python users do not access this after
        the reference goes away.
        """
        self.sender = NULL

    @staticmethod
    cdef ClientAuthSender wrap(CClientAuthSender* sender):
        cdef ClientAuthSender result = \
            ClientAuthSender.__new__(ClientAuthSender)
        result.sender = sender
        return result


cdef CStatus _data_stream_next(void* self, CFlightPayload* payload) except *:
    """Callback for implementing FlightDataStream in Python."""
    cdef:
        unique_ptr[CFlightDataStream] data_stream

    py_stream = <object> self
    if not isinstance(py_stream, GeneratorStream):
        raise RuntimeError("self object in callback is not GeneratorStream")
    stream = <GeneratorStream> py_stream

    # The generator is allowed to yield a reader or table which we
    # yield from; if that sub-generator is empty, we need to reset and
    # try again. However, limit the number of attempts so that we
    # don't just spin forever.
    max_attempts = 128
    for _ in range(max_attempts):
        if stream.current_stream != nullptr:
            check_flight_status(
                stream.current_stream.get().Next().Value(payload))
            # If the stream ended, see if there's another stream from the
            # generator
            if payload.ipc_message.metadata != nullptr:
                return CStatus_OK()
            stream.current_stream.reset(nullptr)

        try:
            result = next(stream.generator)
        except StopIteration:
            payload.ipc_message.metadata.reset(<CBuffer*> nullptr)
            return CStatus_OK()
        except FlightError as flight_error:
            return (<FlightError> flight_error).to_status()

        if isinstance(result, (list, tuple)):
            result, metadata = result
        else:
            result, metadata = result, None

        if isinstance(result, (Table, RecordBatchReader)):
            if metadata:
                raise ValueError("Can only return metadata alongside a "
                                 "RecordBatch.")
            result = RecordBatchStream(result)

        stream_schema = pyarrow_wrap_schema(stream.schema)
        if isinstance(result, FlightDataStream):
            if metadata:
                raise ValueError("Can only return metadata alongside a "
                                 "RecordBatch.")
            data_stream = unique_ptr[CFlightDataStream](
                (<FlightDataStream> result).to_stream())
            substream_schema = pyarrow_wrap_schema(data_stream.get().schema())
            if substream_schema != stream_schema:
                raise ValueError("Got a FlightDataStream whose schema "
                                 "does not match the declared schema of this "
                                 "GeneratorStream. "
                                 "Got: {}\nExpected: {}".format(
                                     substream_schema, stream_schema))
            stream.current_stream.reset(
                new CPyFlightDataStream(result, move(data_stream)))
            # Loop around and try again
            continue
        elif isinstance(result, RecordBatch):
            batch = <RecordBatch> result
            if batch.schema != stream_schema:
                raise ValueError("Got a RecordBatch whose schema does not "
                                 "match the declared schema of this "
                                 "GeneratorStream. "
                                 "Got: {}\nExpected: {}".format(batch.schema,
                                                                stream_schema))
            check_flight_status(GetRecordBatchPayload(
                deref(batch.batch),
                stream.c_options,
                &payload.ipc_message))
            if metadata:
                payload.app_metadata = pyarrow_unwrap_buffer(
                    as_buffer(metadata))
        else:
            raise TypeError("GeneratorStream must be initialized with "
                            "an iterator of FlightDataStream, Table, "
                            "RecordBatch, or RecordBatchStreamReader objects, "
                            "not {}.".format(type(result)))
        # Don't loop around
        return CStatus_OK()
    # Ran out of attempts (the RPC handler kept yielding empty tables/readers)
    raise RuntimeError("While getting next payload, ran out of attempts to "
                       "get something to send "
                       "(application server implementation error)")


cdef CStatus _list_flights(void* self, const CServerCallContext& context,
                           const CCriteria* c_criteria,
                           unique_ptr[CFlightListing]* listing) except *:
    """Callback for implementing ListFlights in Python."""
    cdef:
        vector[CFlightInfo] flights

    try:
        result = (<object> self).list_flights(ServerCallContext.wrap(context),
                                              c_criteria.expression)
        for info in result:
            if not isinstance(info, FlightInfo):
                raise TypeError("FlightServerBase.list_flights must return "
                                "FlightInfo instances, but got {}".format(
                                    type(info)))
            flights.push_back(deref((<FlightInfo> info).info.get()))
        listing.reset(new CSimpleFlightListing(flights))
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _get_flight_info(void* self, const CServerCallContext& context,
                              CFlightDescriptor c_descriptor,
                              unique_ptr[CFlightInfo]* info) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        FlightDescriptor py_descriptor = \
            FlightDescriptor.__new__(FlightDescriptor)
    py_descriptor.descriptor = c_descriptor
    try:
        result = (<object> self).get_flight_info(
            ServerCallContext.wrap(context),
            py_descriptor)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    if not isinstance(result, FlightInfo):
        raise TypeError("FlightServerBase.get_flight_info must return "
                        "a FlightInfo instance, but got {}".format(
                            type(result)))
    info.reset(new CFlightInfo(deref((<FlightInfo> result).info.get())))
    return CStatus_OK()

cdef CStatus _get_schema(void* self, const CServerCallContext& context,
                         CFlightDescriptor c_descriptor,
                         unique_ptr[CSchemaResult]* info) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        FlightDescriptor py_descriptor = \
            FlightDescriptor.__new__(FlightDescriptor)
    py_descriptor.descriptor = c_descriptor
    result = (<object> self).get_schema(ServerCallContext.wrap(context),
                                        py_descriptor)
    if not isinstance(result, SchemaResult):
        raise TypeError("FlightServerBase.get_schema_info must return "
                        "a SchemaResult instance, but got {}".format(
                            type(result)))
    info.reset(new CSchemaResult(deref((<SchemaResult> result).result.get())))
    return CStatus_OK()

cdef CStatus _do_put(void* self, const CServerCallContext& context,
                     unique_ptr[CFlightMessageReader] reader,
                     unique_ptr[CFlightMetadataWriter] writer) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        MetadataRecordBatchReader py_reader = MetadataRecordBatchReader()
        FlightMetadataWriter py_writer = FlightMetadataWriter()
        FlightDescriptor descriptor = \
            FlightDescriptor.__new__(FlightDescriptor)

    descriptor.descriptor = reader.get().descriptor()
    py_reader.reader.reset(reader.release())
    py_writer.writer.reset(writer.release())
    try:
        (<object> self).do_put(ServerCallContext.wrap(context), descriptor,
                               py_reader, py_writer)
        return CStatus_OK()
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()


cdef CStatus _do_get(void* self, const CServerCallContext& context,
                     CTicket ticket,
                     unique_ptr[CFlightDataStream]* stream) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        unique_ptr[CFlightDataStream] data_stream

    py_ticket = Ticket(ticket.ticket)
    try:
        result = (<object> self).do_get(ServerCallContext.wrap(context),
                                        py_ticket)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    if not isinstance(result, FlightDataStream):
        raise TypeError("FlightServerBase.do_get must return "
                        "a FlightDataStream")
    data_stream = unique_ptr[CFlightDataStream](
        (<FlightDataStream> result).to_stream())
    stream[0] = unique_ptr[CFlightDataStream](
        new CPyFlightDataStream(result, move(data_stream)))
    return CStatus_OK()


cdef CStatus _do_exchange(void* self, const CServerCallContext& context,
                          unique_ptr[CFlightMessageReader] reader,
                          unique_ptr[CFlightMessageWriter] writer) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        MetadataRecordBatchReader py_reader = MetadataRecordBatchReader()
        MetadataRecordBatchWriter py_writer = MetadataRecordBatchWriter()
        FlightDescriptor descriptor = \
            FlightDescriptor.__new__(FlightDescriptor)

    descriptor.descriptor = reader.get().descriptor()
    py_reader.reader.reset(reader.release())
    py_writer.writer.reset(writer.release())
    try:
        (<object> self).do_exchange(ServerCallContext.wrap(context),
                                    descriptor, py_reader, py_writer)
        return CStatus_OK()
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()


cdef CStatus _do_action_result_next(
    void* self,
    unique_ptr[CFlightResult]* result
) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        CFlightResult* c_result

    try:
        action_result = next(<object> self)
        if not isinstance(action_result, Result):
            action_result = Result(action_result)
        c_result = (<Result> action_result).result.get()
        result.reset(new CFlightResult(deref(c_result)))
    except StopIteration:
        result.reset(nullptr)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _do_action(void* self, const CServerCallContext& context,
                        const CAction& action,
                        unique_ptr[CResultStream]* result) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        function[cb_result_next] ptr = &_do_action_result_next
    py_action = Action(action.type, pyarrow_wrap_buffer(action.body))
    try:
        responses = (<object> self).do_action(ServerCallContext.wrap(context),
                                              py_action)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    # Let the application return an iterator or anything convertible
    # into one
    if responses is None:
        # Server didn't return anything
        responses = []
    result.reset(new CPyFlightResultStream(iter(responses), ptr))
    return CStatus_OK()


cdef CStatus _list_actions(void* self, const CServerCallContext& context,
                           vector[CActionType]* actions) except *:
    """Callback for implementing Flight servers in Python."""
    cdef:
        CActionType action_type
    # Method should return a list of ActionTypes or similar tuple
    try:
        result = (<object> self).list_actions(ServerCallContext.wrap(context))
        for action in result:
            if not isinstance(action, tuple):
                raise TypeError(
                    "Results of list_actions must be ActionType or tuple")
            action_type.type = tobytes(action[0])
            action_type.description = tobytes(action[1])
            actions.push_back(action_type)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _server_authenticate(void* self, CServerAuthSender* outgoing,
                                  CServerAuthReader* incoming) except *:
    """Callback for implementing authentication in Python."""
    sender = ServerAuthSender.wrap(outgoing)
    reader = ServerAuthReader.wrap(incoming)
    try:
        (<object> self).authenticate(sender, reader)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    finally:
        sender.poison()
        reader.poison()
    return CStatus_OK()

cdef CStatus _is_valid(void* self, const c_string& token,
                       c_string* peer_identity) except *:
    """Callback for implementing authentication in Python."""
    cdef c_string c_result
    try:
        c_result = tobytes((<object> self).is_valid(token))
        peer_identity[0] = c_result
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _client_authenticate(void* self, CClientAuthSender* outgoing,
                                  CClientAuthReader* incoming) except *:
    """Callback for implementing authentication in Python."""
    sender = ClientAuthSender.wrap(outgoing)
    reader = ClientAuthReader.wrap(incoming)
    try:
        (<object> self).authenticate(sender, reader)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    finally:
        sender.poison()
        reader.poison()
    return CStatus_OK()


cdef CStatus _get_token(void* self, c_string* token) except *:
    """Callback for implementing authentication in Python."""
    cdef c_string c_result
    try:
        c_result = tobytes((<object> self).get_token())
        token[0] = c_result
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _middleware_sending_headers(
        void* self, CAddCallHeaders* add_headers) except *:
    """Callback for implementing middleware."""
    try:
        headers = (<object> self).sending_headers()
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()

    if headers:
        for header, values in headers.items():
            if isinstance(values, (str, bytes)):
                values = (values,)
            # Headers in gRPC (and HTTP/1, HTTP/2) are required to be
            # valid, lowercase ASCII.
            header = header.lower()
            if isinstance(header, str):
                header = header.encode("ascii")
            for value in values:
                if isinstance(value, str):
                    value = value.encode("ascii")
                # Allow bytes values to pass through.
                add_headers.AddHeader(header, value)

    return CStatus_OK()


cdef CStatus _middleware_call_completed(
        void* self,
        const CStatus& call_status) except *:
    """Callback for implementing middleware."""
    try:
        try:
            check_flight_status(call_status)
        except Exception as e:
            (<object> self).call_completed(e)
        else:
            (<object> self).call_completed(None)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef CStatus _middleware_received_headers(
        void* self,
        const CCallHeaders& c_headers) except *:
    """Callback for implementing middleware."""
    try:
        headers = convert_headers(c_headers)
        (<object> self).received_headers(headers)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()
    return CStatus_OK()


cdef dict convert_headers(const CCallHeaders& c_headers):
    cdef:
        CCallHeaders.const_iterator header_iter = c_headers.cbegin()
    headers = {}
    while header_iter != c_headers.cend():
        header = c_string(deref(header_iter).first).decode("ascii")
        value = c_string(deref(header_iter).second)
        if not header.endswith("-bin"):
            # Text header values in gRPC (and HTTP/1, HTTP/2) are
            # required to be valid ASCII. Binary header values are
            # exposed as bytes.
            value = value.decode("ascii")
        headers.setdefault(header, []).append(value)
        postincrement(header_iter)
    return headers


cdef CStatus _server_middleware_start_call(
        void* self,
        const CCallInfo& c_info,
        const CCallHeaders& c_headers,
        shared_ptr[CServerMiddleware]* c_instance) except *:
    """Callback for implementing server middleware."""
    instance = None
    try:
        call_info = wrap_call_info(c_info)
        headers = convert_headers(c_headers)
        instance = (<object> self).start_call(call_info, headers)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()

    if instance:
        ServerMiddleware.wrap(instance, c_instance)

    return CStatus_OK()


cdef CStatus _client_middleware_start_call(
        void* self,
        const CCallInfo& c_info,
        unique_ptr[CClientMiddleware]* c_instance) except *:
    """Callback for implementing client middleware."""
    instance = None
    try:
        call_info = wrap_call_info(c_info)
        instance = (<object> self).start_call(call_info)
    except FlightError as flight_error:
        return (<FlightError> flight_error).to_status()

    if instance:
        ClientMiddleware.wrap(instance, c_instance)

    return CStatus_OK()


cdef class ServerAuthHandler(_Weakrefable):
    """Authentication middleware for a server.

    To implement an authentication mechanism, subclass this class and
    override its methods.

    """

    def authenticate(self, outgoing, incoming):
        """Conduct the handshake with the client.

        May raise an error if the client cannot authenticate.

        Parameters
        ----------
        outgoing : ServerAuthSender
            A channel to send messages to the client.
        incoming : ServerAuthReader
            A channel to read messages from the client.
        """
        raise NotImplementedError

    def is_valid(self, token):
        """Validate a client token, returning their identity.

        May return an empty string (if the auth mechanism does not
        name the peer) or raise an exception (if the token is
        invalid).

        Parameters
        ----------
        token : bytes
            The authentication token from the client.

        """
        raise NotImplementedError

    cdef PyServerAuthHandler* to_handler(self):
        cdef PyServerAuthHandlerVtable vtable
        vtable.authenticate = _server_authenticate
        vtable.is_valid = _is_valid
        return new PyServerAuthHandler(self, vtable)


cdef class ClientAuthHandler(_Weakrefable):
    """Authentication plugin for a client."""

    def authenticate(self, outgoing, incoming):
        """Conduct the handshake with the server.

        Parameters
        ----------
        outgoing : ClientAuthSender
            A channel to send messages to the server.
        incoming : ClientAuthReader
            A channel to read messages from the server.
        """
        raise NotImplementedError

    def get_token(self):
        """Get the auth token for a call."""
        raise NotImplementedError

    cdef PyClientAuthHandler* to_handler(self):
        cdef PyClientAuthHandlerVtable vtable
        vtable.authenticate = _client_authenticate
        vtable.get_token = _get_token
        return new PyClientAuthHandler(self, vtable)


_CallInfo = collections.namedtuple("_CallInfo", ["method"])


class CallInfo(_CallInfo):
    """Information about a particular RPC for Flight middleware."""


cdef wrap_call_info(const CCallInfo& c_info):
    method = wrap_flight_method(c_info.method)
    return CallInfo(method=method)


cdef class ClientMiddlewareFactory(_Weakrefable):
    """A factory for new middleware instances.

    All middleware methods will be called from the same thread as the
    RPC method implementation. That is, thread-locals set in the
    client are accessible from the middleware itself.

    """

    def start_call(self, info):
        """Called at the start of an RPC.

        This must be thread-safe and must not raise exceptions.

        Parameters
        ----------
        info : CallInfo
            Information about the call.

        Returns
        -------
        instance : ClientMiddleware
            An instance of ClientMiddleware (the instance to use for
            the call), or None if this call is not intercepted.

        """


cdef class ClientMiddleware(_Weakrefable):
    """Client-side middleware for a call, instantiated per RPC.

    Methods here should be fast and must be infallible: they should
    not raise exceptions or stall indefinitely.

    """

    def sending_headers(self):
        """A callback before headers are sent.

        Returns
        -------
        headers : dict
            A dictionary of header values to add to the request, or
            None if no headers are to be added. The dictionary should
            have string keys and string or list-of-string values.

            Bytes values are allowed, but the underlying transport may
            not support them or may restrict them. For gRPC, binary
            values are only allowed on headers ending in "-bin".

            Header names must be lowercase ASCII.

        """

    def received_headers(self, headers):
        """A callback when headers are received.

        The default implementation does nothing.

        Parameters
        ----------
        headers : dict
            A dictionary of headers from the server. Keys are strings
            and values are lists of strings (for text headers) or
            bytes (for binary headers).

        """

    def call_completed(self, exception):
        """A callback when the call finishes.

        The default implementation does nothing.

        Parameters
        ----------
        exception : ArrowException
            If the call errored, this is the equivalent
            exception. Will be None if the call succeeded.

        """

    @staticmethod
    cdef void wrap(object py_middleware,
                   unique_ptr[CClientMiddleware]* c_instance):
        cdef PyClientMiddlewareVtable vtable
        vtable.sending_headers = _middleware_sending_headers
        vtable.received_headers = _middleware_received_headers
        vtable.call_completed = _middleware_call_completed
        c_instance[0].reset(new CPyClientMiddleware(py_middleware, vtable))


cdef class ServerMiddlewareFactory(_Weakrefable):
    """A factory for new middleware instances.

    All middleware methods will be called from the same thread as the
    RPC method implementation. That is, thread-locals set in the
    middleware are accessible from the method itself.

    """

    def start_call(self, info, headers):
        """Called at the start of an RPC.

        This must be thread-safe.

        Parameters
        ----------
        info : CallInfo
            Information about the call.
        headers : dict
            A dictionary of headers from the client. Keys are strings
            and values are lists of strings (for text headers) or
            bytes (for binary headers).

        Returns
        -------
        instance : ServerMiddleware
            An instance of ServerMiddleware (the instance to use for
            the call), or None if this call is not intercepted.

        Raises
        ------
        exception : pyarrow.ArrowException
            If an exception is raised, the call will be rejected with
            the given error.

        """


cdef class TracingServerMiddlewareFactory(ServerMiddlewareFactory):
    """A factory for tracing middleware instances.

    This enables OpenTelemetry support in Arrow (if Arrow was compiled
    with OpenTelemetry support enabled). A new span will be started on
    each RPC call. The TracingServerMiddleware instance can then be
    retrieved within an RPC handler to get the propagated context,
    which can be used to start a new span on the Python side.

    Because the Python/C++ OpenTelemetry libraries do not
    interoperate, spans on the C++ side are not directly visible to
    the Python side and vice versa.

    """


cdef class ServerMiddleware(_Weakrefable):
    """Server-side middleware for a call, instantiated per RPC.

    Methods here should be fast and must be infalliable: they should
    not raise exceptions or stall indefinitely.

    """

    def sending_headers(self):
        """A callback before headers are sent.

        Returns
        -------
        headers : dict
            A dictionary of header values to add to the response, or
            None if no headers are to be added. The dictionary should
            have string keys and string or list-of-string values.

            Bytes values are allowed, but the underlying transport may
            not support them or may restrict them. For gRPC, binary
            values are only allowed on headers ending in "-bin".

            Header names must be lowercase ASCII.

        """

    def call_completed(self, exception):
        """A callback when the call finishes.

        Parameters
        ----------
        exception : pyarrow.ArrowException
            If the call errored, this is the equivalent
            exception. Will be None if the call succeeded.

        """

    @staticmethod
    cdef void wrap(object py_middleware,
                   shared_ptr[CServerMiddleware]* c_instance):
        cdef PyServerMiddlewareVtable vtable
        vtable.sending_headers = _middleware_sending_headers
        vtable.call_completed = _middleware_call_completed
        c_instance[0].reset(new CPyServerMiddleware(py_middleware, vtable))


class TracingServerMiddleware(ServerMiddleware):
    __slots__ = ["trace_context"]

    def __init__(self, trace_context):
        self.trace_context = trace_context


cdef class _ServerMiddlewareFactoryWrapper(ServerMiddlewareFactory):
    """Wrapper to bundle server middleware into a single C++ one."""

    cdef:
        dict factories

    def __init__(self, dict factories):
        self.factories = factories

    def start_call(self, info, headers):
        instances = {}
        for key, factory in self.factories.items():
            instance = factory.start_call(info, headers)
            if instance:
                # TODO: prevent duplicate keys
                instances[key] = instance
        if instances:
            wrapper = _ServerMiddlewareWrapper(instances)
            return wrapper
        return None


cdef class _ServerMiddlewareWrapper(ServerMiddleware):
    cdef:
        dict middleware

    def __init__(self, dict middleware):
        self.middleware = middleware

    def sending_headers(self):
        headers = collections.defaultdict(list)
        for instance in self.middleware.values():
            more_headers = instance.sending_headers()
            if not more_headers:
                continue
            # Manually merge with existing headers (since headers are
            # multi-valued)
            for key, values in more_headers.items():
                # ARROW-16606 gRPC aborts given non-lowercase headers
                key = key.lower()
                if isinstance(values, (bytes, str)):
                    values = (values,)
                headers[key].extend(values)
        return headers

    def call_completed(self, exception):
        for instance in self.middleware.values():
            instance.call_completed(exception)


cdef class _FlightServerFinalizer(_Weakrefable):
    """
    A finalizer that shuts down the server on destruction.

    See ARROW-16597. If the server is still active at interpreter
    exit, the process may segfault.
    """

    cdef:
        shared_ptr[PyFlightServer] server

    def finalize(self):
        cdef:
            PyFlightServer* server = self.server.get()
            CStatus status
        if server == NULL:
            return
        try:
            with nogil:
                status = server.Shutdown()
                if status.ok():
                    status = server.Wait()
            check_flight_status(status)
        finally:
            self.server.reset()


cdef class FlightServerBase(_Weakrefable):
    """A Flight service definition.

    To start the server, create an instance of this class with an
    appropriate location. The server will be running as soon as the
    instance is created; it is not required to call :meth:`serve`.

    Override methods to define your Flight service.

    Parameters
    ----------
    location : str, tuple or Location optional, default None
        Location to serve on. Either a gRPC URI like `grpc://localhost:port`,
        a tuple of (host, port) pair, or a Location instance.
        If None is passed then the server will be started on localhost with a
        system provided random port.
    auth_handler : ServerAuthHandler optional, default None
        An authentication mechanism to use. May be None.
    tls_certificates : list optional, default None
        A list of (certificate, key) pairs.
    verify_client : boolean optional, default False
        If True, then enable mutual TLS: require the client to present
        a client certificate, and validate the certificate.
    root_certificates : bytes optional, default None
        If enabling mutual TLS, this specifies the PEM-encoded root
        certificate used to validate client certificates.
    middleware : dict optional, default None
        A dictionary of :class:`ServerMiddlewareFactory` instances. The
        string keys can be used to retrieve the middleware instance within
        RPC handlers (see :meth:`ServerCallContext.get_middleware`).

    """

    cdef:
        shared_ptr[PyFlightServer] server
        object finalizer

    def __init__(self, location=None, auth_handler=None,
                 tls_certificates=None, verify_client=None,
                 root_certificates=None, middleware=None):
        self.finalizer = None
        if isinstance(location, (bytes, str)):
            location = Location(location)
        elif isinstance(location, (tuple, type(None))):
            if location is None:
                location = ('localhost', 0)
            host, port = location
            if tls_certificates:
                location = Location.for_grpc_tls(host, port)
            else:
                location = Location.for_grpc_tcp(host, port)
        elif not isinstance(location, Location):
            raise TypeError('`location` argument must be a string, tuple or a '
                            'Location instance')
        self.init(location, auth_handler, tls_certificates, verify_client,
                  tobytes(root_certificates or b""), middleware)

    cdef init(self, Location location, ServerAuthHandler auth_handler,
              list tls_certificates, c_bool verify_client,
              bytes root_certificates, dict middleware):
        cdef:
            PyFlightServerVtable vtable = PyFlightServerVtable()
            PyFlightServer* c_server
            unique_ptr[CFlightServerOptions] c_options
            CCertKeyPair c_cert
            function[cb_server_middleware_start_call] start_call = \
                &_server_middleware_start_call
            pair[c_string, shared_ptr[CServerMiddlewareFactory]] c_middleware

        c_options.reset(new CFlightServerOptions(Location.unwrap(location)))
        # mTLS configuration
        c_options.get().verify_client = verify_client
        c_options.get().root_certificates = root_certificates

        if auth_handler:
            if not isinstance(auth_handler, ServerAuthHandler):
                raise TypeError("auth_handler must be a ServerAuthHandler, "
                                "not a '{}'".format(type(auth_handler)))
            c_options.get().auth_handler.reset(
                (<ServerAuthHandler> auth_handler).to_handler())

        if tls_certificates:
            for cert, key in tls_certificates:
                c_cert.pem_cert = tobytes(cert)
                c_cert.pem_key = tobytes(key)
                c_options.get().tls_certificates.push_back(c_cert)

        if middleware:
            non_tracing_middleware = {}
            enable_tracing = None
            for key, factory in middleware.items():
                if isinstance(factory, TracingServerMiddlewareFactory):
                    if enable_tracing is not None:
                        raise ValueError(
                            "Can only provide "
                            "TracingServerMiddlewareFactory once")
                    if tobytes(key) == CPyServerMiddlewareName:
                        raise ValueError(f"Middleware key cannot be {key}")
                    enable_tracing = key
                else:
                    non_tracing_middleware[key] = factory

            if enable_tracing:
                c_middleware.first = tobytes(enable_tracing)
                c_middleware.second = MakeTracingServerMiddlewareFactory()
                c_options.get().middleware.push_back(c_middleware)

            py_middleware = _ServerMiddlewareFactoryWrapper(
                non_tracing_middleware)
            c_middleware.first = CPyServerMiddlewareName
            c_middleware.second.reset(new CPyServerMiddlewareFactory(
                py_middleware,
                start_call))
            c_options.get().middleware.push_back(c_middleware)

        vtable.list_flights = &_list_flights
        vtable.get_flight_info = &_get_flight_info
        vtable.get_schema = &_get_schema
        vtable.do_put = &_do_put
        vtable.do_get = &_do_get
        vtable.do_exchange = &_do_exchange
        vtable.list_actions = &_list_actions
        vtable.do_action = &_do_action

        c_server = new PyFlightServer(self, vtable)
        self.server.reset(c_server)
        with nogil:
            check_flight_status(c_server.Init(deref(c_options)))
        cdef _FlightServerFinalizer finalizer = _FlightServerFinalizer()
        finalizer.server = self.server
        self.finalizer = weakref.finalize(self, finalizer.finalize)

    @property
    def port(self):
        """
        Get the port that this server is listening on.

        Returns a non-positive value if the operation is invalid
        (e.g. init() was not called or server is listening on a domain
        socket).
        """
        return self.server.get().port()

    def list_flights(self, context, criteria):
        """List flights available on this service.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        criteria : bytes
            Filter criteria provided by the client.

        Returns
        -------
        iterator of FlightInfo

        """
        raise NotImplementedError

    def get_flight_info(self, context, descriptor):
        """Get information about a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.

        Returns
        -------
        FlightInfo

        """
        raise NotImplementedError

    def get_schema(self, context, descriptor):
        """Get the schema of a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.

        Returns
        -------
        Schema

        """
        raise NotImplementedError

    def do_put(self, context, descriptor, reader: MetadataRecordBatchReader,
               writer: FlightMetadataWriter):
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.
        reader : MetadataRecordBatchReader
            A reader for data uploaded by the client.
        writer : FlightMetadataWriter
            A writer to send responses to the client.

        """
        raise NotImplementedError

    def do_get(self, context, ticket):
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        ticket : Ticket
            The ticket for the flight.

        Returns
        -------
        FlightDataStream
            A stream of data to send back to the client.

        """
        raise NotImplementedError

    def do_exchange(self, context, descriptor, reader, writer):
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.
        reader : MetadataRecordBatchReader
            A reader for data uploaded by the client.
        writer : MetadataRecordBatchWriter
            A writer to send responses to the client.

        """
        raise NotImplementedError

    def list_actions(self, context):
        """List custom actions available on this server.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.

        Returns
        -------
        iterator of ActionType or tuple

        """
        raise NotImplementedError

    def do_action(self, context, action):
        """Execute a custom action.

        This method should return an iterator, or it should be a
        generator. Applications should override this method to
        implement their own behavior. The default method raises a
        NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        action : Action
            The action to execute.

        Returns
        -------
        iterator of bytes

        """
        raise NotImplementedError

    def serve(self):
        """Block until the server shuts down.

        This method only returns if shutdown() is called or a signal a
        received.
        """
        if self.server.get() == nullptr:
            raise ValueError("run() on uninitialized FlightServerBase")
        with nogil:
            check_flight_status(self.server.get().ServeWithSignals())

    def run(self):
        """Block until the server shuts down.

        .. deprecated:: 0.15.0
            Use the ``FlightServer.serve`` method instead
        """
        warnings.warn("The 'FlightServer.run' method is deprecated, use "
                      "FlightServer.serve method instead")
        self.serve()

    def shutdown(self):
        """Shut down the server, blocking until current requests finish.

        Do not call this directly from the implementation of a Flight
        method, as then the server will block forever waiting for that
        request to finish. Instead, call this method from a background
        thread.
        """
        # Must not hold the GIL: shutdown waits for pending RPCs to
        # complete. Holding the GIL means Python-implemented Flight
        # methods will never get to run, so this will hang
        # indefinitely.
        if self.server.get() == nullptr:
            raise ValueError("shutdown() on uninitialized FlightServerBase")
        with nogil:
            check_flight_status(self.server.get().Shutdown())

    def wait(self):
        """Block until server is terminated with shutdown."""
        with nogil:
            self.server.get().Wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.finalizer:
            self.finalizer()


def connect(location, **kwargs):
    """
    Connect to a Flight server.

    Parameters
    ----------
    location : str, tuple, or Location
        Location to connect to. Either a URI like "grpc://localhost:port",
        a tuple of (host, port), or a Location instance.
    tls_root_certs : bytes or None
        PEM-encoded.
    cert_chain: str or None
        If provided, enables TLS mutual authentication.
    private_key: str or None
        If provided, enables TLS mutual authentication.
    override_hostname : str or None
        Override the hostname checked by TLS. Insecure, use with caution.
    middleware : list or None
        A list of ClientMiddlewareFactory instances to apply.
    write_size_limit_bytes : int or None
        A soft limit on the size of a data payload sent to the
        server. Enabled if positive. If enabled, writing a record
        batch that (when serialized) exceeds this limit will raise an
        exception; the client can retry the write with a smaller
        batch.
    disable_server_verification : boolean or None
        Disable verifying the server when using TLS.
        Insecure, use with caution.
    generic_options : list or None
        A list of generic (string, int or string) options to pass to
        the underlying transport.

    Returns
    -------
    client : FlightClient
    """
    return FlightClient(location, **kwargs)
