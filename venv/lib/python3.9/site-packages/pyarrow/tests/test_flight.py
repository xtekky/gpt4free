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

import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json

import numpy as np
import pytest
import pyarrow as pa

from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util

try:
    from pyarrow import flight
    from pyarrow.flight import (
        FlightClient, FlightServerBase,
        ServerAuthHandler, ClientAuthHandler,
        ServerMiddleware, ServerMiddlewareFactory,
        ClientMiddleware, ClientMiddlewareFactory,
    )
except ImportError:
    flight = None
    FlightClient, FlightServerBase = object, object
    ServerAuthHandler, ClientAuthHandler = object, object
    ServerMiddleware, ServerMiddlewareFactory = object, object
    ClientMiddleware, ClientMiddlewareFactory = object, object

# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not flight'
pytestmark = pytest.mark.flight


def test_import():
    # So we see the ImportError somewhere
    import pyarrow.flight  # noqa


def resource_root():
    """Get the path to the test resources directory."""
    if not os.environ.get("ARROW_TEST_DATA"):
        raise RuntimeError("Test resources not found; set "
                           "ARROW_TEST_DATA to <repo root>/testing/data")
    return pathlib.Path(os.environ["ARROW_TEST_DATA"]) / "flight"


def read_flight_resource(path):
    """Get the contents of a test resource file."""
    root = resource_root()
    if not root:
        return None
    try:
        with (root / path).open("rb") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(
            "Test resource {} not found; did you initialize the "
            "test resource submodule?\n{}".format(root / path,
                                                  traceback.format_exc()))


def example_tls_certs():
    """Get the paths to test TLS certificates."""
    return {
        "root_cert": read_flight_resource("root-ca.pem"),
        "certificates": [
            flight.CertKeyPair(
                cert=read_flight_resource("cert0.pem"),
                key=read_flight_resource("cert0.key"),
            ),
            flight.CertKeyPair(
                cert=read_flight_resource("cert1.pem"),
                key=read_flight_resource("cert1.key"),
            ),
        ]
    }


def simple_ints_table():
    data = [
        pa.array([-10, -5, 0, 5, 10])
    ]
    return pa.Table.from_arrays(data, names=['some_ints'])


def simple_dicts_table():
    dict_values = pa.array(["foo", "baz", "quux"], type=pa.utf8())
    data = [
        pa.chunked_array([
            pa.DictionaryArray.from_arrays([1, 0, None], dict_values),
            pa.DictionaryArray.from_arrays([2, 1], dict_values)
        ])
    ]
    return pa.Table.from_arrays(data, names=['some_dicts'])


def multiple_column_table():
    return pa.Table.from_arrays([pa.array(['foo', 'bar', 'baz', 'qux']),
                                 pa.array([1, 2, 3, 4])],
                                names=['a', 'b'])


class ConstantFlightServer(FlightServerBase):
    """A Flight server that always returns the same data.

    See ARROW-4796: this server implementation will segfault if Flight
    does not properly hold a reference to the Table object.
    """

    CRITERIA = b"the expected criteria"

    def __init__(self, location=None, options=None, **kwargs):
        super().__init__(location, **kwargs)
        # Ticket -> Table
        self.table_factories = {
            b'ints': simple_ints_table,
            b'dicts': simple_dicts_table,
            b'multi': multiple_column_table,
        }
        self.options = options

    def list_flights(self, context, criteria):
        if criteria == self.CRITERIA:
            yield flight.FlightInfo(
                pa.schema([]),
                flight.FlightDescriptor.for_path('/foo'),
                [],
                -1, -1
            )

    def do_get(self, context, ticket):
        # Return a fresh table, so that Flight is the only one keeping a
        # reference.
        table = self.table_factories[ticket.ticket]()
        return flight.RecordBatchStream(table, options=self.options)


class MetadataFlightServer(FlightServerBase):
    """A Flight server that numbers incoming/outgoing data."""

    def __init__(self, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options

    def do_get(self, context, ticket):
        data = [
            pa.array([-10, -5, 0, 5, 10])
        ]
        table = pa.Table.from_arrays(data, names=['a'])
        return flight.GeneratorStream(
            table.schema,
            self.number_batches(table),
            options=self.options)

    def do_put(self, context, descriptor, reader, writer):
        counter = 0
        expected_data = [-10, -5, 0, 5, 10]
        while True:
            try:
                batch, buf = reader.read_chunk()
                assert batch.equals(pa.RecordBatch.from_arrays(
                    [pa.array([expected_data[counter]])],
                    ['a']
                ))
                assert buf is not None
                client_counter, = struct.unpack('<i', buf.to_pybytes())
                assert counter == client_counter
                writer.write(struct.pack('<i', counter))
                counter += 1
            except StopIteration:
                return

    @staticmethod
    def number_batches(table):
        for idx, batch in enumerate(table.to_batches()):
            buf = struct.pack('<i', idx)
            yield batch, buf


class EchoFlightServer(FlightServerBase):
    """A Flight server that returns the last data uploaded."""

    def __init__(self, location=None, expected_schema=None, **kwargs):
        super().__init__(location, **kwargs)
        self.last_message = None
        self.expected_schema = expected_schema

    def do_get(self, context, ticket):
        return flight.RecordBatchStream(self.last_message)

    def do_put(self, context, descriptor, reader, writer):
        if self.expected_schema:
            assert self.expected_schema == reader.schema
        self.last_message = reader.read_all()

    def do_exchange(self, context, descriptor, reader, writer):
        for chunk in reader:
            pass


class EchoStreamFlightServer(EchoFlightServer):
    """An echo server that streams individual record batches."""

    def do_get(self, context, ticket):
        return flight.GeneratorStream(
            self.last_message.schema,
            self.last_message.to_batches(max_chunksize=1024))

    def list_actions(self, context):
        return []

    def do_action(self, context, action):
        if action.type == "who-am-i":
            return [context.peer_identity(), context.peer().encode("utf-8")]
        raise NotImplementedError


class GetInfoFlightServer(FlightServerBase):
    """A Flight server that tests GetFlightInfo."""

    def get_flight_info(self, context, descriptor):
        return flight.FlightInfo(
            pa.schema([('a', pa.int32())]),
            descriptor,
            [
                flight.FlightEndpoint(b'', ['grpc://test']),
                flight.FlightEndpoint(
                    b'',
                    [flight.Location.for_grpc_tcp('localhost', 5005)],
                ),
            ],
            -1,
            -1,
        )

    def get_schema(self, context, descriptor):
        info = self.get_flight_info(context, descriptor)
        return flight.SchemaResult(info.schema)


class ListActionsFlightServer(FlightServerBase):
    """A Flight server that tests ListActions."""

    @classmethod
    def expected_actions(cls):
        return [
            ("action-1", "description"),
            ("action-2", ""),
            flight.ActionType("action-3", "more detail"),
        ]

    def list_actions(self, context):
        yield from self.expected_actions()


class ListActionsErrorFlightServer(FlightServerBase):
    """A Flight server that tests ListActions."""

    def list_actions(self, context):
        yield ("action-1", "")
        yield "foo"


class CheckTicketFlightServer(FlightServerBase):
    """A Flight server that compares the given ticket to an expected value."""

    def __init__(self, expected_ticket, location=None, **kwargs):
        super().__init__(location, **kwargs)
        self.expected_ticket = expected_ticket

    def do_get(self, context, ticket):
        assert self.expected_ticket == ticket.ticket
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        table = pa.Table.from_arrays(data1, names=['a'])
        return flight.RecordBatchStream(table)

    def do_put(self, context, descriptor, reader):
        self.last_message = reader.read_all()


class InvalidStreamFlightServer(FlightServerBase):
    """A Flight server that tries to return messages with differing schemas."""

    schema = pa.schema([('a', pa.int32())])

    def do_get(self, context, ticket):
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        data2 = [pa.array([-10.0, -5.0, 0.0, 5.0, 10.0], type=pa.float64())]
        assert data1.type != data2.type
        table1 = pa.Table.from_arrays(data1, names=['a'])
        table2 = pa.Table.from_arrays(data2, names=['a'])
        assert table1.schema == self.schema

        return flight.GeneratorStream(self.schema, [table1, table2])


class NeverSendsDataFlightServer(FlightServerBase):
    """A Flight server that never actually yields data."""

    schema = pa.schema([('a', pa.int32())])

    def do_get(self, context, ticket):
        if ticket.ticket == b'yield_data':
            # Check that the server handler will ignore empty tables
            # up to a certain extent
            data = [
                self.schema.empty_table(),
                self.schema.empty_table(),
                pa.RecordBatch.from_arrays([range(5)], schema=self.schema),
            ]
            return flight.GeneratorStream(self.schema, data)
        return flight.GeneratorStream(
            self.schema, itertools.repeat(self.schema.empty_table()))


class SlowFlightServer(FlightServerBase):
    """A Flight server that delays its responses to test timeouts."""

    def do_get(self, context, ticket):
        return flight.GeneratorStream(pa.schema([('a', pa.int32())]),
                                      self.slow_stream())

    def do_action(self, context, action):
        time.sleep(0.5)
        return []

    @staticmethod
    def slow_stream():
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        yield pa.Table.from_arrays(data1, names=['a'])
        # The second message should never get sent; the client should
        # cancel before we send this
        time.sleep(10)
        yield pa.Table.from_arrays(data1, names=['a'])


class ErrorFlightServer(FlightServerBase):
    """A Flight server that uses all the Flight-specific errors."""

    @staticmethod
    def error_cases():
        return {
            "internal": flight.FlightInternalError,
            "timedout": flight.FlightTimedOutError,
            "cancel": flight.FlightCancelledError,
            "unauthenticated": flight.FlightUnauthenticatedError,
            "unauthorized": flight.FlightUnauthorizedError,
            "notimplemented": NotImplementedError,
            "invalid": pa.ArrowInvalid,
            "key": KeyError,
        }

    def do_action(self, context, action):
        error_cases = ErrorFlightServer.error_cases()
        if action.type in error_cases:
            raise error_cases[action.type]("foo")
        elif action.type == "protobuf":
            err_msg = b'this is an error message'
            raise flight.FlightUnauthorizedError("foo", err_msg)
        raise NotImplementedError

    def list_flights(self, context, criteria):
        yield flight.FlightInfo(
            pa.schema([]),
            flight.FlightDescriptor.for_path('/foo'),
            [],
            -1, -1
        )
        raise flight.FlightInternalError("foo")

    def do_put(self, context, descriptor, reader, writer):
        if descriptor.command == b"internal":
            raise flight.FlightInternalError("foo")
        elif descriptor.command == b"timedout":
            raise flight.FlightTimedOutError("foo")
        elif descriptor.command == b"cancel":
            raise flight.FlightCancelledError("foo")
        elif descriptor.command == b"unauthenticated":
            raise flight.FlightUnauthenticatedError("foo")
        elif descriptor.command == b"unauthorized":
            raise flight.FlightUnauthorizedError("foo")
        elif descriptor.command == b"protobuf":
            err_msg = b'this is an error message'
            raise flight.FlightUnauthorizedError("foo", err_msg)


class ExchangeFlightServer(FlightServerBase):
    """A server for testing DoExchange."""

    def __init__(self, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options

    def do_exchange(self, context, descriptor, reader, writer):
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Must provide a command descriptor")
        elif descriptor.command == b"echo":
            return self.exchange_echo(context, reader, writer)
        elif descriptor.command == b"get":
            return self.exchange_do_get(context, reader, writer)
        elif descriptor.command == b"put":
            return self.exchange_do_put(context, reader, writer)
        elif descriptor.command == b"transform":
            return self.exchange_transform(context, reader, writer)
        else:
            raise pa.ArrowInvalid(
                "Unknown command: {}".format(descriptor.command))

    def exchange_do_get(self, context, reader, writer):
        """Emulate DoGet with DoExchange."""
        data = pa.Table.from_arrays([
            pa.array(range(0, 10 * 1024))
        ], names=["a"])
        writer.begin(data.schema)
        writer.write_table(data)

    def exchange_do_put(self, context, reader, writer):
        """Emulate DoPut with DoExchange."""
        num_batches = 0
        for chunk in reader:
            if not chunk.data:
                raise pa.ArrowInvalid("All chunks must have data.")
            num_batches += 1
        writer.write_metadata(str(num_batches).encode("utf-8"))

    def exchange_echo(self, context, reader, writer):
        """Run a simple echo server."""
        started = False
        for chunk in reader:
            if not started and chunk.data:
                writer.begin(chunk.data.schema, options=self.options)
                started = True
            if chunk.app_metadata and chunk.data:
                writer.write_with_metadata(chunk.data, chunk.app_metadata)
            elif chunk.app_metadata:
                writer.write_metadata(chunk.app_metadata)
            elif chunk.data:
                writer.write_batch(chunk.data)
            else:
                assert False, "Should not happen"

    def exchange_transform(self, context, reader, writer):
        """Sum rows in an uploaded table."""
        for field in reader.schema:
            if not pa.types.is_integer(field.type):
                raise pa.ArrowInvalid("Invalid field: " + repr(field))
        table = reader.read_all()
        sums = [0] * table.num_rows
        for column in table:
            for row, value in enumerate(column):
                sums[row] += value.as_py()
        result = pa.Table.from_arrays([pa.array(sums)], names=["sum"])
        writer.begin(result.schema)
        writer.write_table(result)


class HttpBasicServerAuthHandler(ServerAuthHandler):
    """An example implementation of HTTP basic authentication."""

    def __init__(self, creds):
        super().__init__()
        self.creds = creds

    def authenticate(self, outgoing, incoming):
        buf = incoming.read()
        auth = flight.BasicAuth.deserialize(buf)
        if auth.username not in self.creds:
            raise flight.FlightUnauthenticatedError("unknown user")
        if self.creds[auth.username] != auth.password:
            raise flight.FlightUnauthenticatedError("wrong password")
        outgoing.write(tobytes(auth.username))

    def is_valid(self, token):
        if not token:
            raise flight.FlightUnauthenticatedError("token not provided")
        if token not in self.creds:
            raise flight.FlightUnauthenticatedError("unknown user")
        return token


class HttpBasicClientAuthHandler(ClientAuthHandler):
    """An example implementation of HTTP basic authentication."""

    def __init__(self, username, password):
        super().__init__()
        self.basic_auth = flight.BasicAuth(username, password)
        self.token = None

    def authenticate(self, outgoing, incoming):
        auth = self.basic_auth.serialize()
        outgoing.write(auth)
        self.token = incoming.read()

    def get_token(self):
        return self.token


class TokenServerAuthHandler(ServerAuthHandler):
    """An example implementation of authentication via handshake."""

    def __init__(self, creds):
        super().__init__()
        self.creds = creds

    def authenticate(self, outgoing, incoming):
        username = incoming.read()
        password = incoming.read()
        if username in self.creds and self.creds[username] == password:
            outgoing.write(base64.b64encode(b'secret:' + username))
        else:
            raise flight.FlightUnauthenticatedError(
                "invalid username/password")

    def is_valid(self, token):
        token = base64.b64decode(token)
        if not token.startswith(b'secret:'):
            raise flight.FlightUnauthenticatedError("invalid token")
        return token[7:]


class TokenClientAuthHandler(ClientAuthHandler):
    """An example implementation of authentication via handshake."""

    def __init__(self, username, password):
        super().__init__()
        self.username = username
        self.password = password
        self.token = b''

    def authenticate(self, outgoing, incoming):
        outgoing.write(self.username)
        outgoing.write(self.password)
        self.token = incoming.read()

    def get_token(self):
        return self.token


class NoopAuthHandler(ServerAuthHandler):
    """A no-op auth handler."""

    def authenticate(self, outgoing, incoming):
        """Do nothing."""

    def is_valid(self, token):
        """
        Returning an empty string.
        Returning None causes Type error.
        """
        return ""


def case_insensitive_header_lookup(headers, lookup_key):
    """Lookup the value of given key in the given headers.
       The key lookup is case insensitive.
    """
    for key in headers:
        if key.lower() == lookup_key.lower():
            return headers.get(key)


class ClientHeaderAuthMiddlewareFactory(ClientMiddlewareFactory):
    """ClientMiddlewareFactory that creates ClientAuthHeaderMiddleware."""

    def __init__(self):
        self.call_credential = []

    def start_call(self, info):
        return ClientHeaderAuthMiddleware(self)

    def set_call_credential(self, call_credential):
        self.call_credential = call_credential


class ClientHeaderAuthMiddleware(ClientMiddleware):
    """
    ClientMiddleware that extracts the authorization header
    from the server.

    This is an example of a ClientMiddleware that can extract
    the bearer token authorization header from a HTTP header
    authentication enabled server.

    Parameters
    ----------
    factory : ClientHeaderAuthMiddlewareFactory
        This factory is used to set call credentials if an
        authorization header is found in the headers from the server.
    """

    def __init__(self, factory):
        self.factory = factory

    def received_headers(self, headers):
        auth_header = case_insensitive_header_lookup(headers, 'Authorization')
        self.factory.set_call_credential([
            b'authorization',
            auth_header[0].encode("utf-8")])


class HeaderAuthServerMiddlewareFactory(ServerMiddlewareFactory):
    """Validates incoming username and password."""

    def start_call(self, info, headers):
        auth_header = case_insensitive_header_lookup(
            headers,
            'Authorization'
        )
        values = auth_header[0].split(' ')
        token = ''
        error_message = 'Invalid credentials'

        if values[0] == 'Basic':
            decoded = base64.b64decode(values[1])
            pair = decoded.decode("utf-8").split(':')
            if not (pair[0] == 'test' and pair[1] == 'password'):
                raise flight.FlightUnauthenticatedError(error_message)
            token = 'token1234'
        elif values[0] == 'Bearer':
            token = values[1]
            if not token == 'token1234':
                raise flight.FlightUnauthenticatedError(error_message)
        else:
            raise flight.FlightUnauthenticatedError(error_message)

        return HeaderAuthServerMiddleware(token)


class HeaderAuthServerMiddleware(ServerMiddleware):
    """A ServerMiddleware that transports incoming username and password."""

    def __init__(self, token):
        self.token = token

    def sending_headers(self):
        return {'authorization': 'Bearer ' + self.token}


class HeaderAuthFlightServer(FlightServerBase):
    """A Flight server that tests with basic token authentication. """

    def do_action(self, context, action):
        middleware = context.get_middleware("auth")
        if middleware:
            auth_header = case_insensitive_header_lookup(
                middleware.sending_headers(), 'Authorization')
            values = auth_header.split(' ')
            return [values[1].encode("utf-8")]
        raise flight.FlightUnauthenticatedError(
            'No token auth middleware found.')


class ArbitraryHeadersServerMiddlewareFactory(ServerMiddlewareFactory):
    """A ServerMiddlewareFactory that transports arbitrary headers."""

    def start_call(self, info, headers):
        return ArbitraryHeadersServerMiddleware(headers)


class ArbitraryHeadersServerMiddleware(ServerMiddleware):
    """A ServerMiddleware that transports arbitrary headers."""

    def __init__(self, incoming):
        self.incoming = incoming

    def sending_headers(self):
        return self.incoming


class ArbitraryHeadersFlightServer(FlightServerBase):
    """A Flight server that tests multiple arbitrary headers."""

    def do_action(self, context, action):
        middleware = context.get_middleware("arbitrary-headers")
        if middleware:
            headers = middleware.sending_headers()
            header_1 = case_insensitive_header_lookup(
                headers,
                'test-header-1'
            )
            header_2 = case_insensitive_header_lookup(
                headers,
                'test-header-2'
            )
            value1 = header_1[0].encode("utf-8")
            value2 = header_2[0].encode("utf-8")
            return [value1, value2]
        raise flight.FlightServerError("No headers middleware found")


class HeaderServerMiddleware(ServerMiddleware):
    """Expose a per-call value to the RPC method body."""

    def __init__(self, special_value):
        self.special_value = special_value


class HeaderServerMiddlewareFactory(ServerMiddlewareFactory):
    """Expose a per-call hard-coded value to the RPC method body."""

    def start_call(self, info, headers):
        return HeaderServerMiddleware("right value")


class HeaderFlightServer(FlightServerBase):
    """Echo back the per-call hard-coded value."""

    def do_action(self, context, action):
        middleware = context.get_middleware("test")
        if middleware:
            return [middleware.special_value.encode()]
        return [b""]


class MultiHeaderFlightServer(FlightServerBase):
    """Test sending/receiving multiple (binary-valued) headers."""

    def do_action(self, context, action):
        middleware = context.get_middleware("test")
        headers = repr(middleware.client_headers).encode("utf-8")
        return [headers]


class SelectiveAuthServerMiddlewareFactory(ServerMiddlewareFactory):
    """Deny access to certain methods based on a header."""

    def start_call(self, info, headers):
        if info.method == flight.FlightMethod.LIST_ACTIONS:
            # No auth needed
            return

        token = headers.get("x-auth-token")
        if not token:
            raise flight.FlightUnauthenticatedError("No token")

        token = token[0]
        if token != "password":
            raise flight.FlightUnauthenticatedError("Invalid token")

        return HeaderServerMiddleware(token)


class SelectiveAuthClientMiddlewareFactory(ClientMiddlewareFactory):
    def start_call(self, info):
        return SelectiveAuthClientMiddleware()


class SelectiveAuthClientMiddleware(ClientMiddleware):
    def sending_headers(self):
        return {
            "x-auth-token": "password",
        }


class RecordingServerMiddlewareFactory(ServerMiddlewareFactory):
    """Record what methods were called."""

    def __init__(self):
        super().__init__()
        self.methods = []

    def start_call(self, info, headers):
        self.methods.append(info.method)
        return None


class RecordingClientMiddlewareFactory(ClientMiddlewareFactory):
    """Record what methods were called."""

    def __init__(self):
        super().__init__()
        self.methods = []

    def start_call(self, info):
        self.methods.append(info.method)
        return None


class MultiHeaderClientMiddlewareFactory(ClientMiddlewareFactory):
    """Test sending/receiving multiple (binary-valued) headers."""

    def __init__(self):
        # Read in test_middleware_multi_header below.
        # The middleware instance will update this value.
        self.last_headers = {}

    def start_call(self, info):
        return MultiHeaderClientMiddleware(self)


class MultiHeaderClientMiddleware(ClientMiddleware):
    """Test sending/receiving multiple (binary-valued) headers."""

    EXPECTED = {
        "x-text": ["foo", "bar"],
        "x-binary-bin": [b"\x00", b"\x01"],
        # ARROW-16606: ensure mixed-case headers are accepted
        "x-MIXED-case": ["baz"],
        b"x-other-MIXED-case": ["baz"],
    }

    def __init__(self, factory):
        self.factory = factory

    def sending_headers(self):
        return self.EXPECTED

    def received_headers(self, headers):
        # Let the test code know what the last set of headers we
        # received were.
        self.factory.last_headers = headers


class MultiHeaderServerMiddlewareFactory(ServerMiddlewareFactory):
    """Test sending/receiving multiple (binary-valued) headers."""

    def start_call(self, info, headers):
        return MultiHeaderServerMiddleware(headers)


class MultiHeaderServerMiddleware(ServerMiddleware):
    """Test sending/receiving multiple (binary-valued) headers."""

    def __init__(self, client_headers):
        self.client_headers = client_headers

    def sending_headers(self):
        return MultiHeaderClientMiddleware.EXPECTED


class LargeMetadataFlightServer(FlightServerBase):
    """Regression test for ARROW-13253."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = b' ' * (2 ** 31 + 1)

    def do_get(self, context, ticket):
        schema = pa.schema([('a', pa.int64())])
        return flight.GeneratorStream(schema, [
            (pa.record_batch([[1]], schema=schema), self._metadata),
        ])

    def do_exchange(self, context, descriptor, reader, writer):
        writer.write_metadata(self._metadata)


def test_flight_server_location_argument():
    locations = [
        None,
        'grpc://localhost:0',
        ('localhost', find_free_port()),
    ]
    for location in locations:
        with FlightServerBase(location) as server:
            assert isinstance(server, FlightServerBase)


def test_server_exit_reraises_exception():
    with pytest.raises(ValueError):
        with FlightServerBase():
            raise ValueError()


@pytest.mark.slow
def test_client_wait_for_available():
    location = ('localhost', find_free_port())
    server = None

    def serve():
        global server
        time.sleep(0.5)
        server = FlightServerBase(location)
        server.serve()

    with FlightClient(location) as client:
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        started = time.time()
        client.wait_for_available(timeout=5)
        elapsed = time.time() - started
        assert elapsed >= 0.5


def test_flight_list_flights():
    """Try a simple list_flights call."""
    with ConstantFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        assert list(client.list_flights()) == []
        flights = client.list_flights(ConstantFlightServer.CRITERIA)
        assert len(list(flights)) == 1


def test_flight_client_close():
    with ConstantFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        assert list(client.list_flights()) == []
        client.close()
        client.close()  # Idempotent
        with pytest.raises(pa.ArrowInvalid):
            list(client.list_flights())


def test_flight_do_get_ints():
    """Try a simple do_get call."""
    table = simple_ints_table()

    with ConstantFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)

    options = pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V4)
    with ConstantFlightServer(options=options) as server, \
            flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)

        # Also test via RecordBatchReader interface
        data = client.do_get(flight.Ticket(b'ints')).to_reader().read_all()
        assert data.equals(table)

    with pytest.raises(flight.FlightServerError,
                       match="expected IpcWriteOptions, got <class 'int'>"):
        with ConstantFlightServer(options=42) as server, \
                flight.connect(('localhost', server.port)) as client:
            data = client.do_get(flight.Ticket(b'ints')).read_all()


@pytest.mark.pandas
def test_do_get_ints_pandas():
    """Try a simple do_get call."""
    table = simple_ints_table()

    with ConstantFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_pandas()
        assert list(data['some_ints']) == table.column(0).to_pylist()


def test_flight_do_get_dicts():
    table = simple_dicts_table()

    with ConstantFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'dicts')).read_all()
        assert data.equals(table)


def test_flight_do_get_ticket():
    """Make sure Tickets get passed to the server."""
    data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
    table = pa.Table.from_arrays(data1, names=['a'])
    with CheckTicketFlightServer(expected_ticket=b'the-ticket') as server, \
            flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'the-ticket')).read_all()
        assert data.equals(table)


def test_flight_get_info():
    """Make sure FlightEndpoint accepts string and object URIs."""
    with GetInfoFlightServer() as server:
        client = FlightClient(('localhost', server.port))
        info = client.get_flight_info(flight.FlightDescriptor.for_command(b''))
        assert info.total_records == -1
        assert info.total_bytes == -1
        assert info.schema == pa.schema([('a', pa.int32())])
        assert len(info.endpoints) == 2
        assert len(info.endpoints[0].locations) == 1
        assert info.endpoints[0].locations[0] == flight.Location('grpc://test')
        assert info.endpoints[1].locations[0] == \
            flight.Location.for_grpc_tcp('localhost', 5005)


def test_flight_get_schema():
    """Make sure GetSchema returns correct schema."""
    with GetInfoFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        info = client.get_schema(flight.FlightDescriptor.for_command(b''))
        assert info.schema == pa.schema([('a', pa.int32())])


def test_list_actions():
    """Make sure the return type of ListActions is validated."""
    # ARROW-6392
    with ListActionsErrorFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        with pytest.raises(
                flight.FlightServerError,
                match=("Results of list_actions must be "
                       "ActionType or tuple")
        ):
            list(client.list_actions())

    with ListActionsFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        assert list(client.list_actions()) == \
            ListActionsFlightServer.expected_actions()


class ConvenienceServer(FlightServerBase):
    """
    Server for testing various implementation conveniences (auto-boxing, etc.)
    """

    @property
    def simple_action_results(self):
        return [b'foo', b'bar', b'baz']

    def do_action(self, context, action):
        if action.type == 'simple-action':
            return self.simple_action_results
        elif action.type == 'echo':
            return [action.body]
        elif action.type == 'bad-action':
            return ['foo']
        elif action.type == 'arrow-exception':
            raise pa.ArrowMemoryError()
        elif action.type == 'forever':
            def gen():
                while not context.is_cancelled():
                    yield b'foo'
            return gen()


def test_do_action_result_convenience():
    with ConvenienceServer() as server, \
            FlightClient(('localhost', server.port)) as client:

        # do_action as action type without body
        results = [x.body for x in client.do_action('simple-action')]
        assert results == server.simple_action_results

        # do_action with tuple of type and body
        body = b'the-body'
        results = [x.body for x in client.do_action(('echo', body))]
        assert results == [body]


def test_nicer_server_exceptions():
    with ConvenienceServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError,
                           match="a bytes-like object is required"):
            list(client.do_action('bad-action'))
        # While Flight/C++ sends across the original status code, it
        # doesn't get mapped to the equivalent code here, since we
        # want to be able to distinguish between client- and server-
        # side errors.
        with pytest.raises(flight.FlightServerError,
                           match="ArrowMemoryError"):
            list(client.do_action('arrow-exception'))


def test_get_port():
    """Make sure port() works."""
    server = GetInfoFlightServer("grpc://localhost:0")
    try:
        assert server.port > 0
    finally:
        server.shutdown()


@pytest.mark.skipif(os.name == 'nt',
                    reason="Unix sockets can't be tested on Windows")
def test_flight_domain_socket():
    """Try a simple do_get call over a Unix domain socket."""
    with tempfile.NamedTemporaryFile() as sock:
        sock.close()
        location = flight.Location.for_grpc_unix(sock.name)
        with ConstantFlightServer(location=location), \
                FlightClient(location) as client:

            reader = client.do_get(flight.Ticket(b'ints'))
            table = simple_ints_table()
            assert reader.schema.equals(table.schema)
            data = reader.read_all()
            assert data.equals(table)

            reader = client.do_get(flight.Ticket(b'dicts'))
            table = simple_dicts_table()
            assert reader.schema.equals(table.schema)
            data = reader.read_all()
            assert data.equals(table)


@pytest.mark.slow
def test_flight_large_message():
    """Try sending/receiving a large message via Flight.

    See ARROW-4421: by default, gRPC won't allow us to send messages >
    4MiB in size.
    """
    data = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024 * 1024))
    ], names=['a'])

    with EchoFlightServer(expected_schema=data.schema) as server, \
            FlightClient(('localhost', server.port)) as client:
        writer, _ = client.do_put(flight.FlightDescriptor.for_path('test'),
                                  data.schema)
        # Write a single giant chunk
        writer.write_table(data, 10 * 1024 * 1024)
        writer.close()
        result = client.do_get(flight.Ticket(b'')).read_all()
        assert result.equals(data)


def test_flight_generator_stream():
    """Try downloading a flight of RecordBatches in a GeneratorStream."""
    data = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024))
    ], names=['a'])

    with EchoStreamFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        writer, _ = client.do_put(flight.FlightDescriptor.for_path('test'),
                                  data.schema)
        writer.write_table(data)
        writer.close()
        result = client.do_get(flight.Ticket(b'')).read_all()
        assert result.equals(data)


def test_flight_invalid_generator_stream():
    """Try streaming data with mismatched schemas."""
    with InvalidStreamFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        with pytest.raises(pa.ArrowException):
            client.do_get(flight.Ticket(b'')).read_all()


def test_timeout_fires():
    """Make sure timeouts fire on slow requests."""
    # Do this in a separate thread so that if it fails, we don't hang
    # the entire test process
    with SlowFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        action = flight.Action("", b"")
        options = flight.FlightCallOptions(timeout=0.2)
        # gRPC error messages change based on version, so don't look
        # for a particular error
        with pytest.raises(flight.FlightTimedOutError):
            list(client.do_action(action, options=options))


def test_timeout_passes():
    """Make sure timeouts do not fire on fast requests."""
    with ConstantFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        options = flight.FlightCallOptions(timeout=5.0)
        client.do_get(flight.Ticket(b'ints'), options=options).read_all()


def test_read_options():
    """Make sure ReadOptions can be used."""
    expected = pa.Table.from_arrays([pa.array([1, 2, 3, 4])], names=["b"])
    with ConstantFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        options = flight.FlightCallOptions(
            read_options=IpcReadOptions(included_fields=[1]))
        response1 = client.do_get(flight.Ticket(
            b'multi'), options=options).read_all()
        response2 = client.do_get(flight.Ticket(b'multi')).read_all()

        assert response2.num_columns == 2
        assert response1.num_columns == 1
        assert response1 == expected
        assert response2 == multiple_column_table()


basic_auth_handler = HttpBasicServerAuthHandler(creds={
    b"test": b"p4ssw0rd",
})

token_auth_handler = TokenServerAuthHandler(creds={
    b"test": b"p4ssw0rd",
})


@pytest.mark.slow
def test_http_basic_unauth():
    """Test that auth fails when not authenticated."""
    with EchoStreamFlightServer(auth_handler=basic_auth_handler) as server, \
            FlightClient(('localhost', server.port)) as client:
        action = flight.Action("who-am-i", b"")
        with pytest.raises(flight.FlightUnauthenticatedError,
                           match=".*unauthenticated.*"):
            list(client.do_action(action))


@pytest.mark.skipif(os.name == 'nt',
                    reason="ARROW-10013: gRPC on Windows corrupts peer()")
def test_http_basic_auth():
    """Test a Python implementation of HTTP basic authentication."""
    with EchoStreamFlightServer(auth_handler=basic_auth_handler) as server, \
            FlightClient(('localhost', server.port)) as client:
        action = flight.Action("who-am-i", b"")
        client.authenticate(HttpBasicClientAuthHandler('test', 'p4ssw0rd'))
        results = client.do_action(action)
        identity = next(results)
        assert identity.body.to_pybytes() == b'test'
        peer_address = next(results)
        assert peer_address.body.to_pybytes() != b''


def test_http_basic_auth_invalid_password():
    """Test that auth fails with the wrong password."""
    with EchoStreamFlightServer(auth_handler=basic_auth_handler) as server, \
            FlightClient(('localhost', server.port)) as client:
        action = flight.Action("who-am-i", b"")
        with pytest.raises(flight.FlightUnauthenticatedError,
                           match=".*wrong password.*"):
            client.authenticate(HttpBasicClientAuthHandler('test', 'wrong'))
            next(client.do_action(action))


def test_token_auth():
    """Test an auth mechanism that uses a handshake."""
    with EchoStreamFlightServer(auth_handler=token_auth_handler) as server, \
            FlightClient(('localhost', server.port)) as client:
        action = flight.Action("who-am-i", b"")
        client.authenticate(TokenClientAuthHandler('test', 'p4ssw0rd'))
        identity = next(client.do_action(action))
        assert identity.body.to_pybytes() == b'test'


def test_token_auth_invalid():
    """Test an auth mechanism that uses a handshake."""
    with EchoStreamFlightServer(auth_handler=token_auth_handler) as server, \
            FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            client.authenticate(TokenClientAuthHandler('test', 'wrong'))


header_auth_server_middleware_factory = HeaderAuthServerMiddlewareFactory()
no_op_auth_handler = NoopAuthHandler()


def test_authenticate_basic_token():
    """Test authenticate_basic_token with bearer token and auth headers."""
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={
        "auth": HeaderAuthServerMiddlewareFactory()
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        token_pair = client.authenticate_basic_token(b'test', b'password')
        assert token_pair[0] == b'authorization'
        assert token_pair[1] == b'Bearer token1234'


def test_authenticate_basic_token_invalid_password():
    """Test authenticate_basic_token with an invalid password."""
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={
        "auth": HeaderAuthServerMiddlewareFactory()
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            client.authenticate_basic_token(b'test', b'badpassword')


def test_authenticate_basic_token_and_action():
    """Test authenticate_basic_token and doAction after authentication."""
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={
        "auth": HeaderAuthServerMiddlewareFactory()
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        token_pair = client.authenticate_basic_token(b'test', b'password')
        assert token_pair[0] == b'authorization'
        assert token_pair[1] == b'Bearer token1234'
        options = flight.FlightCallOptions(headers=[token_pair])
        result = list(client.do_action(
            action=flight.Action('test-action', b''), options=options))
        assert result[0].body.to_pybytes() == b'token1234'


def test_authenticate_basic_token_with_client_middleware():
    """Test authenticate_basic_token with client middleware
       to intercept authorization header returned by the
       HTTP header auth enabled server.
    """
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={
        "auth": HeaderAuthServerMiddlewareFactory()
    }) as server:
        client_auth_middleware = ClientHeaderAuthMiddlewareFactory()
        client = FlightClient(
            ('localhost', server.port),
            middleware=[client_auth_middleware]
        )
        encoded_credentials = base64.b64encode(b'test:password')
        options = flight.FlightCallOptions(headers=[
            (b'authorization', b'Basic ' + encoded_credentials)
        ])
        result = list(client.do_action(
            action=flight.Action('test-action', b''), options=options))
        assert result[0].body.to_pybytes() == b'token1234'
        assert client_auth_middleware.call_credential[0] == b'authorization'
        assert client_auth_middleware.call_credential[1] == \
            b'Bearer ' + b'token1234'
        result2 = list(client.do_action(
            action=flight.Action('test-action', b''), options=options))
        assert result2[0].body.to_pybytes() == b'token1234'
        assert client_auth_middleware.call_credential[0] == b'authorization'
        assert client_auth_middleware.call_credential[1] == \
            b'Bearer ' + b'token1234'
        client.close()


def test_arbitrary_headers_in_flight_call_options():
    """Test passing multiple arbitrary headers to the middleware."""
    with ArbitraryHeadersFlightServer(
        auth_handler=no_op_auth_handler,
        middleware={
            "auth": HeaderAuthServerMiddlewareFactory(),
            "arbitrary-headers": ArbitraryHeadersServerMiddlewareFactory()
        }) as server, \
            FlightClient(('localhost', server.port)) as client:
        token_pair = client.authenticate_basic_token(b'test', b'password')
        assert token_pair[0] == b'authorization'
        assert token_pair[1] == b'Bearer token1234'
        options = flight.FlightCallOptions(headers=[
            token_pair,
            (b'test-header-1', b'value1'),
            (b'test-header-2', b'value2')
        ])
        result = list(client.do_action(flight.Action(
            "test-action", b""), options=options))
        assert result[0].body.to_pybytes() == b'value1'
        assert result[1].body.to_pybytes() == b'value2'


def test_location_invalid():
    """Test constructing invalid URIs."""
    with pytest.raises(pa.ArrowInvalid, match=".*Cannot parse URI:.*"):
        flight.connect("%")

    with pytest.raises(pa.ArrowInvalid, match=".*Cannot parse URI:.*"):
        ConstantFlightServer("%")


def test_location_unknown_scheme():
    """Test creating locations for unknown schemes."""
    assert flight.Location("s3://foo").uri == b"s3://foo"
    assert flight.Location("https://example.com/bar.parquet").uri == \
        b"https://example.com/bar.parquet"


@pytest.mark.slow
@pytest.mark.requires_testing_data
def test_tls_fails():
    """Make sure clients cannot connect when cert verification fails."""
    certs = example_tls_certs()

    # Ensure client doesn't connect when certificate verification
    # fails (this is a slow test since gRPC does retry a few times)
    with ConstantFlightServer(tls_certificates=certs["certificates"]) as s, \
            FlightClient("grpc+tls://localhost:" + str(s.port)) as client:
        # gRPC error messages change based on version, so don't look
        # for a particular error
        with pytest.raises(flight.FlightUnavailableError):
            client.do_get(flight.Ticket(b'ints')).read_all()


@pytest.mark.requires_testing_data
def test_tls_do_get():
    """Try a simple do_get call over TLS."""
    table = simple_ints_table()
    certs = example_tls_certs()

    with ConstantFlightServer(tls_certificates=certs["certificates"]) as s, \
        FlightClient(('localhost', s.port),
                     tls_root_certs=certs["root_cert"]) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)


@pytest.mark.requires_testing_data
def test_tls_disable_server_verification():
    """Try a simple do_get call over TLS with server verification disabled."""
    table = simple_ints_table()
    certs = example_tls_certs()

    with ConstantFlightServer(tls_certificates=certs["certificates"]) as s:
        try:
            client = FlightClient(('localhost', s.port),
                                  disable_server_verification=True)
        except NotImplementedError:
            pytest.skip('disable_server_verification feature is not available')
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)
        client.close()


@pytest.mark.requires_testing_data
def test_tls_override_hostname():
    """Check that incorrectly overriding the hostname fails."""
    certs = example_tls_certs()

    with ConstantFlightServer(tls_certificates=certs["certificates"]) as s,\
        flight.connect(('localhost', s.port),
                       tls_root_certs=certs["root_cert"],
                       override_hostname="fakehostname") as client:
        with pytest.raises(flight.FlightUnavailableError):
            client.do_get(flight.Ticket(b'ints'))


def test_flight_do_get_metadata():
    """Try a simple do_get call with metadata."""
    data = [
        pa.array([-10, -5, 0, 5, 10])
    ]
    table = pa.Table.from_arrays(data, names=['a'])

    batches = []
    with MetadataFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b''))
        idx = 0
        while True:
            try:
                batch, metadata = reader.read_chunk()
                batches.append(batch)
                server_idx, = struct.unpack('<i', metadata.to_pybytes())
                assert idx == server_idx
                idx += 1
            except StopIteration:
                break
        data = pa.Table.from_batches(batches)
        assert data.equals(table)


def test_flight_do_get_metadata_v4():
    """Try a simple do_get call with V4 metadata version."""
    table = pa.Table.from_arrays(
        [pa.array([-10, -5, 0, 5, 10])], names=['a'])
    options = pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V4)
    with MetadataFlightServer(options=options) as server, \
            FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b''))
        data = reader.read_all()
        assert data.equals(table)


def test_flight_do_put_metadata():
    """Try a simple do_put call with metadata."""
    data = [
        pa.array([-10, -5, 0, 5, 10])
    ]
    table = pa.Table.from_arrays(data, names=['a'])

    with MetadataFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        writer, metadata_reader = client.do_put(
            flight.FlightDescriptor.for_path(''),
            table.schema)
        with writer:
            for idx, batch in enumerate(table.to_batches(max_chunksize=1)):
                metadata = struct.pack('<i', idx)
                writer.write_with_metadata(batch, metadata)
                buf = metadata_reader.read()
                assert buf is not None
                server_idx, = struct.unpack('<i', buf.to_pybytes())
                assert idx == server_idx


def test_flight_do_put_limit():
    """Try a simple do_put call with a size limit."""
    large_batch = pa.RecordBatch.from_arrays([
        pa.array(np.ones(768, dtype=np.int64())),
    ], names=['a'])

    with EchoFlightServer() as server, \
        FlightClient(('localhost', server.port),
                     write_size_limit_bytes=4096) as client:
        writer, metadata_reader = client.do_put(
            flight.FlightDescriptor.for_path(''),
            large_batch.schema)
        with writer:
            with pytest.raises(flight.FlightWriteSizeExceededError,
                               match="exceeded soft limit") as excinfo:
                writer.write_batch(large_batch)
            assert excinfo.value.limit == 4096
            smaller_batches = [
                large_batch.slice(0, 384),
                large_batch.slice(384),
            ]
            for batch in smaller_batches:
                writer.write_batch(batch)
        expected = pa.Table.from_batches([large_batch])
        actual = client.do_get(flight.Ticket(b'')).read_all()
        assert expected == actual


@pytest.mark.slow
def test_cancel_do_get():
    """Test canceling a DoGet operation on the client side."""
    with ConstantFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b'ints'))
        reader.cancel()
        with pytest.raises(flight.FlightCancelledError,
                           match="(?i).*cancel.*"):
            reader.read_chunk()


@pytest.mark.slow
def test_cancel_do_get_threaded():
    """Test canceling a DoGet operation from another thread."""
    with SlowFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b'ints'))

        read_first_message = threading.Event()
        stream_canceled = threading.Event()
        result_lock = threading.Lock()
        raised_proper_exception = threading.Event()

        def block_read():
            reader.read_chunk()
            read_first_message.set()
            stream_canceled.wait(timeout=5)
            try:
                reader.read_chunk()
            except flight.FlightCancelledError:
                with result_lock:
                    raised_proper_exception.set()

        thread = threading.Thread(target=block_read, daemon=True)
        thread.start()
        read_first_message.wait(timeout=5)
        reader.cancel()
        stream_canceled.set()
        thread.join(timeout=1)

        with result_lock:
            assert raised_proper_exception.is_set()


def test_streaming_do_action():
    with ConvenienceServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        results = client.do_action(flight.Action('forever', b''))
        assert next(results).body == b'foo'
        # Implicit cancel when destructed
        del results


def test_roundtrip_types():
    """Make sure serializable types round-trip."""
    action = flight.Action("action1", b"action1-body")
    assert action == flight.Action.deserialize(action.serialize())

    ticket = flight.Ticket("foo")
    assert ticket == flight.Ticket.deserialize(ticket.serialize())

    result = flight.Result(b"result1")
    assert result == flight.Result.deserialize(result.serialize())

    basic_auth = flight.BasicAuth("username1", "password1")
    assert basic_auth == flight.BasicAuth.deserialize(basic_auth.serialize())

    schema_result = flight.SchemaResult(pa.schema([('a', pa.int32())]))
    assert schema_result == flight.SchemaResult.deserialize(
        schema_result.serialize())

    desc = flight.FlightDescriptor.for_command("test")
    assert desc == flight.FlightDescriptor.deserialize(desc.serialize())

    desc = flight.FlightDescriptor.for_path("a", "b", "test.arrow")
    assert desc == flight.FlightDescriptor.deserialize(desc.serialize())

    info = flight.FlightInfo(
        pa.schema([('a', pa.int32())]),
        desc,
        [
            flight.FlightEndpoint(b'', ['grpc://test']),
            flight.FlightEndpoint(
                b'',
                [flight.Location.for_grpc_tcp('localhost', 5005)],
            ),
        ],
        -1,
        -1,
    )
    info2 = flight.FlightInfo.deserialize(info.serialize())
    assert info.schema == info2.schema
    assert info.descriptor == info2.descriptor
    assert info.total_bytes == info2.total_bytes
    assert info.total_records == info2.total_records
    assert info.endpoints == info2.endpoints

    endpoint = flight.FlightEndpoint(
        ticket,
        ['grpc://test', flight.Location.for_grpc_tcp('localhost', 5005)]
    )
    assert endpoint == flight.FlightEndpoint.deserialize(endpoint.serialize())


def test_roundtrip_errors():
    """Ensure that Flight errors propagate from server to client."""
    with ErrorFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:

        for arg, exc_type in ErrorFlightServer.error_cases().items():
            with pytest.raises(exc_type, match=".*foo.*"):
                list(client.do_action(flight.Action(arg, b"")))
        with pytest.raises(flight.FlightInternalError, match=".*foo.*"):
            list(client.list_flights())

        data = [pa.array([-10, -5, 0, 5, 10])]
        table = pa.Table.from_arrays(data, names=['a'])

        exceptions = {
            'internal': flight.FlightInternalError,
            'timedout': flight.FlightTimedOutError,
            'cancel': flight.FlightCancelledError,
            'unauthenticated': flight.FlightUnauthenticatedError,
            'unauthorized': flight.FlightUnauthorizedError,
        }

        for command, exception in exceptions.items():

            with pytest.raises(exception, match=".*foo.*"):
                writer, reader = client.do_put(
                    flight.FlightDescriptor.for_command(command),
                    table.schema)
                writer.write_table(table)
                writer.close()

            with pytest.raises(exception, match=".*foo.*"):
                writer, reader = client.do_put(
                    flight.FlightDescriptor.for_command(command),
                    table.schema)
                writer.close()


def test_do_put_independent_read_write():
    """Ensure that separate threads can read/write on a DoPut."""
    # ARROW-6063: previously this would cause gRPC to abort when the
    # writer was closed (due to simultaneous reads), or would hang
    # forever.
    data = [
        pa.array([-10, -5, 0, 5, 10])
    ]
    table = pa.Table.from_arrays(data, names=['a'])

    with MetadataFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        writer, metadata_reader = client.do_put(
            flight.FlightDescriptor.for_path(''),
            table.schema)

        count = [0]

        def _reader_thread():
            while metadata_reader.read() is not None:
                count[0] += 1

        thread = threading.Thread(target=_reader_thread)
        thread.start()

        batches = table.to_batches(max_chunksize=1)
        with writer:
            for idx, batch in enumerate(batches):
                metadata = struct.pack('<i', idx)
                writer.write_with_metadata(batch, metadata)
            # Causes the server to stop writing and end the call
            writer.done_writing()
            # Thus reader thread will break out of loop
            thread.join()
        # writer.close() won't segfault since reader thread has
        # stopped
        assert count[0] == len(batches)


def test_server_middleware_same_thread():
    """Ensure that server middleware run on the same thread as the RPC."""
    with HeaderFlightServer(middleware={
        "test": HeaderServerMiddlewareFactory(),
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        results = list(client.do_action(flight.Action(b"test", b"")))
        assert len(results) == 1
        value = results[0].body.to_pybytes()
        assert b"right value" == value


def test_middleware_reject():
    """Test rejecting an RPC with server middleware."""
    with HeaderFlightServer(middleware={
        "test": SelectiveAuthServerMiddlewareFactory(),
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        # The middleware allows this through without auth.
        with pytest.raises(pa.ArrowNotImplementedError):
            list(client.list_actions())

        # But not anything else.
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(client.do_action(flight.Action(b"", b"")))

        client = FlightClient(
            ('localhost', server.port),
            middleware=[SelectiveAuthClientMiddlewareFactory()]
        )
        response = next(client.do_action(flight.Action(b"", b"")))
        assert b"password" == response.body.to_pybytes()


def test_middleware_mapping():
    """Test that middleware records methods correctly."""
    server_middleware = RecordingServerMiddlewareFactory()
    client_middleware = RecordingClientMiddlewareFactory()
    with FlightServerBase(middleware={"test": server_middleware}) as server, \
        FlightClient(
            ('localhost', server.port),
            middleware=[client_middleware]
    ) as client:

        descriptor = flight.FlightDescriptor.for_command(b"")
        with pytest.raises(NotImplementedError):
            list(client.list_flights())
        with pytest.raises(NotImplementedError):
            client.get_flight_info(descriptor)
        with pytest.raises(NotImplementedError):
            client.get_schema(descriptor)
        with pytest.raises(NotImplementedError):
            client.do_get(flight.Ticket(b""))
        with pytest.raises(NotImplementedError):
            writer, _ = client.do_put(descriptor, pa.schema([]))
            writer.close()
        with pytest.raises(NotImplementedError):
            list(client.do_action(flight.Action(b"", b"")))
        with pytest.raises(NotImplementedError):
            list(client.list_actions())
        with pytest.raises(NotImplementedError):
            writer, _ = client.do_exchange(descriptor)
            writer.close()

        expected = [
            flight.FlightMethod.LIST_FLIGHTS,
            flight.FlightMethod.GET_FLIGHT_INFO,
            flight.FlightMethod.GET_SCHEMA,
            flight.FlightMethod.DO_GET,
            flight.FlightMethod.DO_PUT,
            flight.FlightMethod.DO_ACTION,
            flight.FlightMethod.LIST_ACTIONS,
            flight.FlightMethod.DO_EXCHANGE,
        ]
        assert server_middleware.methods == expected
        assert client_middleware.methods == expected


def test_extra_info():
    with ErrorFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        try:
            list(client.do_action(flight.Action("protobuf", b"")))
            assert False
        except flight.FlightUnauthorizedError as e:
            assert e.extra_info is not None
            ei = e.extra_info
            assert ei == b'this is an error message'


@pytest.mark.requires_testing_data
def test_mtls():
    """Test mutual TLS (mTLS) with gRPC."""
    certs = example_tls_certs()
    table = simple_ints_table()

    with ConstantFlightServer(
            tls_certificates=[certs["certificates"][0]],
            verify_client=True,
            root_certificates=certs["root_cert"]) as s, \
        FlightClient(
            ('localhost', s.port),
            tls_root_certs=certs["root_cert"],
            cert_chain=certs["certificates"][0].cert,
            private_key=certs["certificates"][0].key) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)


def test_doexchange_get():
    """Emulate DoGet with DoExchange."""
    expected = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024))
    ], names=["a"])

    with ExchangeFlightServer() as server, \
            FlightClient(("localhost", server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b"get")
        writer, reader = client.do_exchange(descriptor)
        with writer:
            table = reader.read_all()
        assert expected == table


def test_doexchange_put():
    """Emulate DoPut with DoExchange."""
    data = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024))
    ], names=["a"])
    batches = data.to_batches(max_chunksize=512)

    with ExchangeFlightServer() as server, \
            FlightClient(("localhost", server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b"put")
        writer, reader = client.do_exchange(descriptor)
        with writer:
            writer.begin(data.schema)
            for batch in batches:
                writer.write_batch(batch)
            writer.done_writing()
            chunk = reader.read_chunk()
            assert chunk.data is None
            expected_buf = str(len(batches)).encode("utf-8")
            assert chunk.app_metadata == expected_buf


def test_doexchange_echo():
    """Try a DoExchange echo server."""
    data = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024))
    ], names=["a"])
    batches = data.to_batches(max_chunksize=512)

    with ExchangeFlightServer() as server, \
            FlightClient(("localhost", server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b"echo")
        writer, reader = client.do_exchange(descriptor)
        with writer:
            # Read/write metadata before starting data.
            for i in range(10):
                buf = str(i).encode("utf-8")
                writer.write_metadata(buf)
                chunk = reader.read_chunk()
                assert chunk.data is None
                assert chunk.app_metadata == buf

            # Now write data without metadata.
            writer.begin(data.schema)
            for batch in batches:
                writer.write_batch(batch)
                assert reader.schema == data.schema
                chunk = reader.read_chunk()
                assert chunk.data == batch
                assert chunk.app_metadata is None

            # And write data with metadata.
            for i, batch in enumerate(batches):
                buf = str(i).encode("utf-8")
                writer.write_with_metadata(batch, buf)
                chunk = reader.read_chunk()
                assert chunk.data == batch
                assert chunk.app_metadata == buf


def test_doexchange_echo_v4():
    """Try a DoExchange echo server using the V4 metadata version."""
    data = pa.Table.from_arrays([
        pa.array(range(0, 10 * 1024))
    ], names=["a"])
    batches = data.to_batches(max_chunksize=512)

    options = pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V4)
    with ExchangeFlightServer(options=options) as server, \
            FlightClient(("localhost", server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b"echo")
        writer, reader = client.do_exchange(descriptor)
        with writer:
            # Now write data without metadata.
            writer.begin(data.schema, options=options)
            for batch in batches:
                writer.write_batch(batch)
                assert reader.schema == data.schema
                chunk = reader.read_chunk()
                assert chunk.data == batch
                assert chunk.app_metadata is None


def test_doexchange_transform():
    """Transform a table with a service."""
    data = pa.Table.from_arrays([
        pa.array(range(0, 1024)),
        pa.array(range(1, 1025)),
        pa.array(range(2, 1026)),
    ], names=["a", "b", "c"])
    expected = pa.Table.from_arrays([
        pa.array(range(3, 1024 * 3 + 3, 3)),
    ], names=["sum"])

    with ExchangeFlightServer() as server, \
            FlightClient(("localhost", server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b"transform")
        writer, reader = client.do_exchange(descriptor)
        with writer:
            writer.begin(data.schema)
            writer.write_table(data)
            writer.done_writing()
            table = reader.read_all()
        assert expected == table


def test_middleware_multi_header():
    """Test sending/receiving multiple (binary-valued) headers."""
    with MultiHeaderFlightServer(middleware={
        "test": MultiHeaderServerMiddlewareFactory(),
    }) as server:
        headers = MultiHeaderClientMiddlewareFactory()
        with FlightClient(
                ('localhost', server.port),
                middleware=[headers]) as client:
            response = next(client.do_action(flight.Action(b"", b"")))
            # The server echoes the headers it got back to us.
            raw_headers = response.body.to_pybytes().decode("utf-8")
            client_headers = ast.literal_eval(raw_headers)
            # Don't directly compare; gRPC may add headers like User-Agent.
            for header, values in MultiHeaderClientMiddleware.EXPECTED.items():
                header = header.lower()
                if isinstance(header, bytes):
                    header = header.decode("ascii")
                assert client_headers.get(header) == values
                assert headers.last_headers.get(header) == values


@pytest.mark.requires_testing_data
def test_generic_options():
    """Test setting generic client options."""
    certs = example_tls_certs()

    with ConstantFlightServer(tls_certificates=certs["certificates"]) as s:
        # Try setting a string argument that will make requests fail
        options = [("grpc.ssl_target_name_override", "fakehostname")]
        client = flight.connect(('localhost', s.port),
                                tls_root_certs=certs["root_cert"],
                                generic_options=options)
        with pytest.raises(flight.FlightUnavailableError):
            client.do_get(flight.Ticket(b'ints'))
        client.close()
        # Try setting an int argument that will make requests fail
        options = [("grpc.max_receive_message_length", 32)]
        client = flight.connect(('localhost', s.port),
                                tls_root_certs=certs["root_cert"],
                                generic_options=options)
        with pytest.raises(pa.ArrowInvalid):
            client.do_get(flight.Ticket(b'ints'))
        client.close()


class CancelFlightServer(FlightServerBase):
    """A server for testing StopToken."""

    def do_get(self, context, ticket):
        schema = pa.schema([])
        rb = pa.RecordBatch.from_arrays([], schema=schema)
        return flight.GeneratorStream(schema, itertools.repeat(rb))

    def do_exchange(self, context, descriptor, reader, writer):
        schema = pa.schema([])
        rb = pa.RecordBatch.from_arrays([], schema=schema)
        writer.begin(schema)
        while not context.is_cancelled():
            writer.write_batch(rb)
            time.sleep(0.5)


def test_interrupt():
    if threading.current_thread().ident != threading.main_thread().ident:
        pytest.skip("test only works from main Python thread")
    # Skips test if not available
    raise_signal = util.get_raise_signal()

    def signal_from_thread():
        time.sleep(0.5)
        raise_signal(signal.SIGINT)

    exc_types = (KeyboardInterrupt, pa.ArrowCancelled)

    def test(read_all):
        try:
            try:
                t = threading.Thread(target=signal_from_thread)
                with pytest.raises(exc_types) as exc_info:
                    t.start()
                    read_all()
            finally:
                t.join()
        except KeyboardInterrupt:
            # In case KeyboardInterrupt didn't interrupt read_all
            # above, at least prevent it from stopping the test suite
            pytest.fail("KeyboardInterrupt didn't interrupt Flight read_all")
        # __context__ is sometimes None
        e = exc_info.value
        assert isinstance(e, (pa.ArrowCancelled, KeyboardInterrupt)) or \
            isinstance(e.__context__, (pa.ArrowCancelled, KeyboardInterrupt))

    with CancelFlightServer() as server, \
            FlightClient(("localhost", server.port)) as client:

        reader = client.do_get(flight.Ticket(b""))
        test(reader.read_all)

        descriptor = flight.FlightDescriptor.for_command(b"echo")
        writer, reader = client.do_exchange(descriptor)
        test(reader.read_all)
        try:
            writer.close()
        except (KeyboardInterrupt, flight.FlightCancelledError):
            # Silence the Cancelled/Interrupt exception
            pass


def test_never_sends_data():
    # Regression test for ARROW-12779
    match = "application server implementation error"
    with NeverSendsDataFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError, match=match):
            client.do_get(flight.Ticket(b'')).read_all()

        # Check that the server handler will ignore empty tables
        # up to a certain extent
        table = client.do_get(flight.Ticket(b'yield_data')).read_all()
        assert table.num_rows == 5


@pytest.mark.large_memory
@pytest.mark.slow
def test_large_descriptor():
    # Regression test for ARROW-13253. Placed here with appropriate marks
    # since some CI pipelines can't run the C++ equivalent
    large_descriptor = flight.FlightDescriptor.for_command(
        b' ' * (2 ** 31 + 1))
    with FlightServerBase() as server, \
            flight.connect(('localhost', server.port)) as client:
        with pytest.raises(OSError,
                           match="Failed to serialize Flight descriptor"):
            writer, _ = client.do_put(large_descriptor, pa.schema([]))
            writer.close()
        with pytest.raises(pa.ArrowException,
                           match="Failed to serialize Flight descriptor"):
            client.do_exchange(large_descriptor)


@pytest.mark.large_memory
@pytest.mark.slow
def test_large_metadata_client():
    # Regression test for ARROW-13253
    descriptor = flight.FlightDescriptor.for_command(b'')
    metadata = b' ' * (2 ** 31 + 1)
    with EchoFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        with pytest.raises(pa.ArrowCapacityError,
                           match="app_metadata size overflow"):
            writer, _ = client.do_put(descriptor, pa.schema([]))
            with writer:
                writer.write_metadata(metadata)
                writer.close()
        with pytest.raises(pa.ArrowCapacityError,
                           match="app_metadata size overflow"):
            writer, reader = client.do_exchange(descriptor)
            with writer:
                writer.write_metadata(metadata)

    del metadata
    with LargeMetadataFlightServer() as server, \
            flight.connect(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError,
                           match="app_metadata size overflow"):
            reader = client.do_get(flight.Ticket(b''))
            reader.read_all()
        with pytest.raises(pa.ArrowException,
                           match="app_metadata size overflow"):
            writer, reader = client.do_exchange(descriptor)
            with writer:
                reader.read_all()


class ActionNoneFlightServer(EchoFlightServer):
    """A server that implements a side effect to a non iterable action."""
    VALUES = []

    def do_action(self, context, action):
        if action.type == "get_value":
            return [json.dumps(self.VALUES).encode('utf-8')]
        elif action.type == "append":
            self.VALUES.append(True)
            return None
        raise NotImplementedError


def test_none_action_side_effect():
    """Ensure that actions are executed even when we don't consume iterator.

    See https://issues.apache.org/jira/browse/ARROW-14255
    """

    with ActionNoneFlightServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        client.do_action(flight.Action("append", b""))
        r = client.do_action(flight.Action("get_value", b""))
        assert json.loads(next(r).body.to_pybytes()) == [True]


@pytest.mark.slow  # Takes a while for gRPC to "realize" writes fail
def test_write_error_propagation():
    """
    Ensure that exceptions during writing preserve error context.

    See https://issues.apache.org/jira/browse/ARROW-16592.
    """
    expected_message = "foo"
    expected_info = b"bar"
    exc = flight.FlightCancelledError(
        expected_message, extra_info=expected_info)
    descriptor = flight.FlightDescriptor.for_command(b"")
    schema = pa.schema([("int64", pa.int64())])

    class FailServer(flight.FlightServerBase):
        def do_put(self, context, descriptor, reader, writer):
            raise exc

        def do_exchange(self, context, descriptor, reader, writer):
            raise exc

    with FailServer() as server, \
            FlightClient(('localhost', server.port)) as client:
        # DoPut
        writer, reader = client.do_put(descriptor, schema)

        # Set a concurrent reader - ensure this doesn't block the
        # writer side from calling Close()
        def _reader():
            try:
                while True:
                    reader.read()
            except flight.FlightError:
                return

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        with pytest.raises(flight.FlightCancelledError) as exc_info:
            while True:
                writer.write_batch(pa.record_batch([[1]], schema=schema))
        assert exc_info.value.extra_info == expected_info

        with pytest.raises(flight.FlightCancelledError) as exc_info:
            writer.close()
        assert exc_info.value.extra_info == expected_info
        thread.join()

        # DoExchange
        writer, reader = client.do_exchange(descriptor)

        def _reader():
            try:
                while True:
                    reader.read_chunk()
            except flight.FlightError:
                return

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        with pytest.raises(flight.FlightCancelledError) as exc_info:
            while True:
                writer.write_metadata(b" ")
        assert exc_info.value.extra_info == expected_info

        with pytest.raises(flight.FlightCancelledError) as exc_info:
            writer.close()
        assert exc_info.value.extra_info == expected_info
        thread.join()


def test_interpreter_shutdown():
    """
    Ensure that the gRPC server is stopped at interpreter shutdown.

    See https://issues.apache.org/jira/browse/ARROW-16597.
    """
    util.invoke_script("arrow_16597.py")


class TracingFlightServer(FlightServerBase):
    """A server that echoes back trace context values."""

    def do_action(self, context, action):
        trace_context = context.get_middleware("tracing").trace_context
        # Don't turn this method into a generator since then
        # trace_context will be evaluated after we've exited the scope
        # of the OTel span (and so the value we want won't be present)
        return ((f"{key}: {value}").encode("utf-8")
                for (key, value) in trace_context.items())


def test_tracing():
    with TracingFlightServer(middleware={
            "tracing": flight.TracingServerMiddlewareFactory(),
    }) as server, \
            FlightClient(('localhost', server.port)) as client:
        # We can't tell if Arrow was built with OpenTelemetry support,
        # so we can't count on any particular values being there; we
        # can only ensure things don't blow up either way.
        options = flight.FlightCallOptions(headers=[
            # Pretend we have an OTel implementation
            (b"traceparent", b"00-000ff00f00f0ff000f0f00ff0f00fff0-"
                             b"000f0000f0f00000-00"),
            (b"tracestate", b""),
        ])
        for value in client.do_action((b"", b""), options=options):
            pass


def test_do_put_does_not_crash_when_schema_is_none():
    client = FlightClient('grpc+tls://localhost:9643',
                          disable_server_verification=True)
    msg = ("Argument 'schema' has incorrect type "
           r"\(expected pyarrow.lib.Schema, got NoneType\)")
    with pytest.raises(TypeError, match=msg):
        client.do_put(flight.FlightDescriptor.for_command('foo'),
                      schema=None)
