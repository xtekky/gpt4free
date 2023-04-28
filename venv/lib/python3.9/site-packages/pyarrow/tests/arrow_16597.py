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

# This file is called from a test in test_flight.py.
import time

import pyarrow as pa
import pyarrow.flight as flight


class Server(flight.FlightServerBase):
    def do_put(self, context, descriptor, reader, writer):
        time.sleep(1)
        raise flight.FlightCancelledError("")


if __name__ == "__main__":
    server = Server("grpc://localhost:0")
    client = flight.connect(f"grpc://localhost:{server.port}")
    schema = pa.schema([])
    writer, reader = client.do_put(
        flight.FlightDescriptor.for_command(b""), schema)
    writer.done_writing()
