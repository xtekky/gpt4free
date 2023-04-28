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

from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys

import pytest

import pyarrow as pa


pytestmark = pytest.mark.gdb

here = os.path.dirname(os.path.abspath(__file__))

# The GDB script may be found in the source tree (if available)
# or in another location given by the ARROW_GDB_SCRIPT environment variable.
gdb_script = (os.environ.get('ARROW_GDB_SCRIPT') or
              os.path.join(here, "../../../cpp/gdb_arrow.py"))

gdb_command = ["gdb", "--nx"]


def environment_for_gdb():
    env = {}
    for var in ['PATH', 'LD_LIBRARY_PATH']:
        try:
            env[var] = os.environ[var]
        except KeyError:
            pass
    return env


@lru_cache()
def is_gdb_available():
    try:
        # Try to use the same arguments as in GdbSession so that the
        # same error return gets propagated.
        proc = subprocess.run(gdb_command + ["--version"],
                              env=environment_for_gdb(), bufsize=0,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return False
    return proc.returncode == 0


@lru_cache()
def python_executable():
    path = shutil.which("python3")
    assert path is not None, "Couldn't find python3 executable"
    return path


def skip_if_gdb_unavailable():
    if not is_gdb_available():
        pytest.skip("gdb command unavailable")


def skip_if_gdb_script_unavailable():
    if not os.path.exists(gdb_script):
        pytest.skip("gdb script not found")


class GdbSession:
    proc = None
    verbose = True

    def __init__(self, *args, **env):
        # Let stderr through to let pytest display it separately on errors
        gdb_env = environment_for_gdb()
        gdb_env.update(env)
        self.proc = subprocess.Popen(gdb_command + list(args),
                                     env=gdb_env, bufsize=0,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.last_stdout = []
        self.last_stdout_line = b""

    def wait_until_ready(self):
        """
        Record output until the gdb prompt displays.  Return recorded output.
        """
        # TODO: add timeout?
        while (not self.last_stdout_line.startswith(b"(gdb) ") and
               self.proc.poll() is None):
            block = self.proc.stdout.read(4096)
            if self.verbose:
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            block, sep, last_line = block.rpartition(b"\n")
            if sep:
                self.last_stdout.append(self.last_stdout_line)
                self.last_stdout.append(block + sep)
                self.last_stdout_line = last_line
            else:
                assert block == b""
                self.last_stdout_line += last_line

        if self.proc.poll() is not None:
            raise IOError("gdb session terminated unexpectedly")

        out = b"".join(self.last_stdout).decode('utf-8')
        self.last_stdout = []
        self.last_stdout_line = b""
        return out

    def issue_command(self, line):
        line = line.encode('utf-8') + b"\n"
        if self.verbose:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def run_command(self, line):
        self.issue_command(line)
        return self.wait_until_ready()

    def print_value(self, expr):
        """
        Ask gdb to print the value of an expression and return the result.
        """
        out = self.run_command(f"p {expr}")
        out, n = re.subn(r"^\$\d+ = ", "", out)
        assert n == 1, out
        # gdb may add whitespace depending on result width, remove it
        return out.strip()

    def select_frame(self, func_name):
        """
        Select the innermost frame with the given function name.
        """
        # Ideally, we would use the "frame function" command,
        # but it's not available on old GDB versions (such as 8.1.1),
        # so instead parse the stack trace for a matching frame number.
        out = self.run_command("info stack")
        pat = r"(?mi)^#(\d+)\s+.* in " + re.escape(func_name) + r"\b"
        m = re.search(pat, out)
        if m is None:
            pytest.fail(f"Could not select frame for function {func_name}")

        frame_num = int(m[1])
        out = self.run_command(f"frame {frame_num}")
        assert f"in {func_name}" in out

    def join(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.stdout.close()  # avoid ResourceWarning
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def __del__(self):
        self.join()


@pytest.fixture(scope='session')
def gdb():
    skip_if_gdb_unavailable()
    gdb = GdbSession("-q", python_executable())
    try:
        gdb.wait_until_ready()
        gdb.run_command("set confirm off")
        gdb.run_command("set print array-indexes on")
        # Make sure gdb formatting is not terminal-dependent
        gdb.run_command("set width unlimited")
        gdb.run_command("set charset UTF-8")
        yield gdb
    finally:
        gdb.join()


@pytest.fixture(scope='session')
def gdb_arrow(gdb):
    if 'deb' not in pa.cpp_build_info.build_type:
        pytest.skip("Arrow C++ debug symbols not available")

    skip_if_gdb_script_unavailable()
    gdb.run_command(f"source {gdb_script}")

    lib_path_var = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
    lib_path = os.environ.get(lib_path_var)
    if lib_path:
        # GDB starts the inferior process in a pristine shell, need
        # to propagate the library search path to find the Arrow DLL
        gdb.run_command(f"set env {lib_path_var} {lib_path}")

    code = "from pyarrow.lib import _gdb_test_session; _gdb_test_session()"
    out = gdb.run_command(f"run -c '{code}'")
    assert ("Trace/breakpoint trap" in out or
            "received signal" in out), out
    gdb.select_frame("arrow::gdb::TestSession")
    return gdb


def test_gdb_session(gdb):
    out = gdb.run_command("show version")
    assert out.startswith("GNU gdb ("), out


def test_gdb_arrow(gdb_arrow):
    s = gdb_arrow.print_value("42 + 1")
    assert s == "43"


def check_stack_repr(gdb, expr, expected):
    """
    Check printing a stack-located value.
    """
    s = gdb.print_value(expr)
    if isinstance(expected, re.Pattern):
        assert expected.match(s), s
    else:
        assert s == expected


def check_heap_repr(gdb, expr, expected):
    """
    Check printing a heap-located value, given its address.
    """
    s = gdb.print_value(f"*{expr}")
    # GDB may prefix the value with an address or type specification
    if s != expected:
        assert s.endswith(f" {expected}")


def test_status(gdb_arrow):
    check_stack_repr(gdb_arrow, "ok_status", "arrow::Status::OK()")
    check_stack_repr(gdb_arrow, "error_status",
                     'arrow::Status::IOError("This is an error")')
    check_stack_repr(
        gdb_arrow, "error_detail_status",
        'arrow::Status::IOError("This is an error", '
        'detail=[custom-detail-id] "This is a detail")')

    check_stack_repr(gdb_arrow, "ok_result", "arrow::Result<int>(42)")
    check_stack_repr(
        gdb_arrow, "error_result",
        'arrow::Result<int>(arrow::Status::IOError("This is an error"))')
    check_stack_repr(
        gdb_arrow, "error_detail_result",
        'arrow::Result<int>(arrow::Status::IOError("This is an error", '
        'detail=[custom-detail-id] "This is a detail"))')


def test_buffer_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, "buffer_null",
                     "arrow::Buffer of size 0, read-only")
    check_stack_repr(gdb_arrow, "buffer_abc",
                     'arrow::Buffer of size 3, read-only, "abc"')
    check_stack_repr(
        gdb_arrow, "buffer_special_chars",
        r'arrow::Buffer of size 12, read-only, "foo\"bar\000\r\n\t\037"')
    check_stack_repr(gdb_arrow, "buffer_mutable",
                     'arrow::MutableBuffer of size 3, mutable, "abc"')


def test_buffer_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, "heap_buffer",
                    'arrow::Buffer of size 3, read-only, "abc"')
    check_heap_repr(gdb_arrow, "heap_buffer_mutable.get()",
                    'arrow::Buffer of size 3, mutable, "abc"')


def test_decimals(gdb_arrow):
    v128 = "98765432109876543210987654321098765432"
    check_stack_repr(gdb_arrow, "decimal128_zero", "arrow::Decimal128(0)")
    check_stack_repr(gdb_arrow, "decimal128_pos",
                     f"arrow::Decimal128({v128})")
    check_stack_repr(gdb_arrow, "decimal128_neg",
                     f"arrow::Decimal128(-{v128})")
    check_stack_repr(gdb_arrow, "basic_decimal128_zero",
                     "arrow::BasicDecimal128(0)")
    check_stack_repr(gdb_arrow, "basic_decimal128_pos",
                     f"arrow::BasicDecimal128({v128})")
    check_stack_repr(gdb_arrow, "basic_decimal128_neg",
                     f"arrow::BasicDecimal128(-{v128})")

    v256 = ("9876543210987654321098765432109876543210"
            "987654321098765432109876543210987654")
    check_stack_repr(gdb_arrow, "decimal256_zero", "arrow::Decimal256(0)")
    check_stack_repr(gdb_arrow, "decimal256_pos",
                     f"arrow::Decimal256({v256})")
    check_stack_repr(gdb_arrow, "decimal256_neg",
                     f"arrow::Decimal256(-{v256})")
    check_stack_repr(gdb_arrow, "basic_decimal256_zero",
                     "arrow::BasicDecimal256(0)")
    check_stack_repr(gdb_arrow, "basic_decimal256_pos",
                     f"arrow::BasicDecimal256({v256})")
    check_stack_repr(gdb_arrow, "basic_decimal256_neg",
                     f"arrow::BasicDecimal256(-{v256})")


def test_metadata(gdb_arrow):
    check_heap_repr(gdb_arrow, "empty_metadata.get()",
                    "arrow::KeyValueMetadata of size 0")
    check_heap_repr(
        gdb_arrow, "metadata.get()",
        ('arrow::KeyValueMetadata of size 2 = {'
         '["key_text"] = "some value", ["key_binary"] = "z\\000\\037\\377"}'))


def test_types_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, "null_type", "arrow::null()")
    check_stack_repr(gdb_arrow, "bool_type", "arrow::boolean()")

    check_stack_repr(gdb_arrow, "date32_type", "arrow::date32()")
    check_stack_repr(gdb_arrow, "date64_type", "arrow::date64()")
    check_stack_repr(gdb_arrow, "time_type_s",
                     "arrow::time32(arrow::TimeUnit::SECOND)")
    check_stack_repr(gdb_arrow, "time_type_ms",
                     "arrow::time32(arrow::TimeUnit::MILLI)")
    check_stack_repr(gdb_arrow, "time_type_us",
                     "arrow::time64(arrow::TimeUnit::MICRO)")
    check_stack_repr(gdb_arrow, "time_type_ns",
                     "arrow::time64(arrow::TimeUnit::NANO)")
    check_stack_repr(gdb_arrow, "timestamp_type_s",
                     "arrow::timestamp(arrow::TimeUnit::SECOND)")
    check_stack_repr(
        gdb_arrow, "timestamp_type_ms_timezone",
        'arrow::timestamp(arrow::TimeUnit::MILLI, "Europe/Paris")')
    check_stack_repr(gdb_arrow, "timestamp_type_us",
                     "arrow::timestamp(arrow::TimeUnit::MICRO)")
    check_stack_repr(
        gdb_arrow, "timestamp_type_ns_timezone",
        'arrow::timestamp(arrow::TimeUnit::NANO, "Europe/Paris")')

    check_stack_repr(gdb_arrow, "day_time_interval_type",
                     "arrow::day_time_interval()")
    check_stack_repr(gdb_arrow, "month_interval_type",
                     "arrow::month_interval()")
    check_stack_repr(gdb_arrow, "month_day_nano_interval_type",
                     "arrow::month_day_nano_interval()")
    check_stack_repr(gdb_arrow, "duration_type_s",
                     "arrow::duration(arrow::TimeUnit::SECOND)")
    check_stack_repr(gdb_arrow, "duration_type_ns",
                     "arrow::duration(arrow::TimeUnit::NANO)")

    check_stack_repr(gdb_arrow, "decimal128_type",
                     "arrow::decimal128(16, 5)")
    check_stack_repr(gdb_arrow, "decimal256_type",
                     "arrow::decimal256(42, 12)")

    check_stack_repr(gdb_arrow, "binary_type", "arrow::binary()")
    check_stack_repr(gdb_arrow, "string_type", "arrow::utf8()")
    check_stack_repr(gdb_arrow, "large_binary_type", "arrow::large_binary()")
    check_stack_repr(gdb_arrow, "large_string_type", "arrow::large_utf8()")
    check_stack_repr(gdb_arrow, "fixed_size_binary_type",
                     "arrow::fixed_size_binary(10)")

    check_stack_repr(gdb_arrow, "list_type",
                     "arrow::list(arrow::uint8())")
    check_stack_repr(gdb_arrow, "large_list_type",
                     "arrow::large_list(arrow::large_utf8())")
    check_stack_repr(gdb_arrow, "fixed_size_list_type",
                     "arrow::fixed_size_list(arrow::float64(), 3)")
    check_stack_repr(
        gdb_arrow, "map_type_unsorted",
        "arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=false)")
    check_stack_repr(
        gdb_arrow, "map_type_sorted",
        "arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=true)")

    check_stack_repr(gdb_arrow, "struct_type_empty",
                     "arrow::struct_({})")
    check_stack_repr(
        gdb_arrow, "struct_type",
        ('arrow::struct_({arrow::field("ints", arrow::int8()), '
         'arrow::field("strs", arrow::utf8(), nullable=false)})'))

    check_stack_repr(
        gdb_arrow, "sparse_union_type",
        ('arrow::sparse_union(fields={arrow::field("ints", arrow::int8()), '
         'arrow::field("strs", arrow::utf8(), nullable=false)}, '
         'type_codes={7, 42})'))
    check_stack_repr(
        gdb_arrow, "dense_union_type",
        ('arrow::dense_union(fields={arrow::field("ints", arrow::int8()), '
         'arrow::field("strs", arrow::utf8(), nullable=false)}, '
         'type_codes={7, 42})'))

    check_stack_repr(
        gdb_arrow, "dict_type_unordered",
        "arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=false)")
    check_stack_repr(
        gdb_arrow, "dict_type_ordered",
        "arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=true)")

    check_stack_repr(
        gdb_arrow, "uuid_type",
        ('arrow::ExtensionType "extension<uuid>" '
         'with storage type arrow::fixed_size_binary(16)'))


def test_types_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, "heap_null_type", "arrow::null()")
    check_heap_repr(gdb_arrow, "heap_bool_type", "arrow::boolean()")

    check_heap_repr(gdb_arrow, "heap_time_type_ns",
                    "arrow::time64(arrow::TimeUnit::NANO)")
    check_heap_repr(
        gdb_arrow, "heap_timestamp_type_ns_timezone",
        'arrow::timestamp(arrow::TimeUnit::NANO, "Europe/Paris")')

    check_heap_repr(gdb_arrow, "heap_decimal128_type",
                    "arrow::decimal128(16, 5)")

    check_heap_repr(gdb_arrow, "heap_list_type",
                    "arrow::list(arrow::uint8())")
    check_heap_repr(gdb_arrow, "heap_large_list_type",
                    "arrow::large_list(arrow::large_utf8())")
    check_heap_repr(gdb_arrow, "heap_fixed_size_list_type",
                    "arrow::fixed_size_list(arrow::float64(), 3)")
    check_heap_repr(
        gdb_arrow, "heap_map_type",
        "arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=false)")

    check_heap_repr(
        gdb_arrow, "heap_struct_type",
        ('arrow::struct_({arrow::field("ints", arrow::int8()), '
         'arrow::field("strs", arrow::utf8(), nullable=false)})'))

    check_heap_repr(
        gdb_arrow, "heap_dict_type",
        "arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=false)")

    check_heap_repr(
        gdb_arrow, "heap_uuid_type",
        ('arrow::ExtensionType "extension<uuid>" '
         'with storage type arrow::fixed_size_binary(16)'))


def test_fields_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, "int_field",
                     'arrow::field("ints", arrow::int64())')
    check_stack_repr(
        gdb_arrow, "float_field",
        'arrow::field("floats", arrow::float32(), nullable=false)')


def test_fields_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, "heap_int_field",
                    'arrow::field("ints", arrow::int64())')


def test_scalars_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, "null_scalar", "arrow::NullScalar")
    check_stack_repr(gdb_arrow, "bool_scalar",
                     "arrow::BooleanScalar of value true")
    check_stack_repr(gdb_arrow, "bool_scalar_null",
                     "arrow::BooleanScalar of null value")
    check_stack_repr(gdb_arrow, "int8_scalar",
                     "arrow::Int8Scalar of value -42")
    check_stack_repr(gdb_arrow, "uint8_scalar",
                     "arrow::UInt8Scalar of value 234")
    check_stack_repr(gdb_arrow, "int64_scalar",
                     "arrow::Int64Scalar of value -9223372036854775808")
    check_stack_repr(gdb_arrow, "uint64_scalar",
                     "arrow::UInt64Scalar of value 18446744073709551615")
    check_stack_repr(gdb_arrow, "half_float_scalar",
                     "arrow::HalfFloatScalar of value -1.5 [48640]")
    check_stack_repr(gdb_arrow, "float_scalar",
                     "arrow::FloatScalar of value 1.25")
    check_stack_repr(gdb_arrow, "double_scalar",
                     "arrow::DoubleScalar of value 2.5")

    check_stack_repr(gdb_arrow, "time_scalar_s",
                     "arrow::Time32Scalar of value 100s")
    check_stack_repr(gdb_arrow, "time_scalar_ms",
                     "arrow::Time32Scalar of value 1000ms")
    check_stack_repr(gdb_arrow, "time_scalar_us",
                     "arrow::Time64Scalar of value 10000us")
    check_stack_repr(gdb_arrow, "time_scalar_ns",
                     "arrow::Time64Scalar of value 100000ns")
    check_stack_repr(gdb_arrow, "time_scalar_null",
                     "arrow::Time64Scalar of null value [ns]")

    check_stack_repr(gdb_arrow, "duration_scalar_s",
                     "arrow::DurationScalar of value -100s")
    check_stack_repr(gdb_arrow, "duration_scalar_ms",
                     "arrow::DurationScalar of value -1000ms")
    check_stack_repr(gdb_arrow, "duration_scalar_us",
                     "arrow::DurationScalar of value -10000us")
    check_stack_repr(gdb_arrow, "duration_scalar_ns",
                     "arrow::DurationScalar of value -100000ns")
    check_stack_repr(gdb_arrow, "duration_scalar_null",
                     "arrow::DurationScalar of null value [ns]")

    check_stack_repr(
        gdb_arrow, "timestamp_scalar_s",
        "arrow::TimestampScalar of value 12345s [no timezone]")
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_ms",
        "arrow::TimestampScalar of value -123456ms [no timezone]")
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_us",
        "arrow::TimestampScalar of value 1234567us [no timezone]")
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_ns",
        "arrow::TimestampScalar of value -12345678ns [no timezone]")
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_null",
        "arrow::TimestampScalar of null value [ns, no timezone]")

    check_stack_repr(
        gdb_arrow, "timestamp_scalar_s_tz",
        'arrow::TimestampScalar of value 12345s ["Europe/Paris"]')
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_ms_tz",
        'arrow::TimestampScalar of value -123456ms ["Europe/Paris"]')
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_us_tz",
        'arrow::TimestampScalar of value 1234567us ["Europe/Paris"]')
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_ns_tz",
        'arrow::TimestampScalar of value -12345678ns ["Europe/Paris"]')
    check_stack_repr(
        gdb_arrow, "timestamp_scalar_null_tz",
        'arrow::TimestampScalar of null value [ns, "Europe/Paris"]')

    check_stack_repr(gdb_arrow, "month_interval_scalar",
                     "arrow::MonthIntervalScalar of value 23M")
    check_stack_repr(gdb_arrow, "month_interval_scalar_null",
                     "arrow::MonthIntervalScalar of null value")
    check_stack_repr(gdb_arrow, "day_time_interval_scalar",
                     "arrow::DayTimeIntervalScalar of value 23d-456ms")
    check_stack_repr(gdb_arrow, "day_time_interval_scalar_null",
                     "arrow::DayTimeIntervalScalar of null value")
    check_stack_repr(
        gdb_arrow, "month_day_nano_interval_scalar",
        "arrow::MonthDayNanoIntervalScalar of value 1M23d-456ns")
    check_stack_repr(
        gdb_arrow, "month_day_nano_interval_scalar_null",
        "arrow::MonthDayNanoIntervalScalar of null value")

    check_stack_repr(gdb_arrow, "date32_scalar",
                     "arrow::Date32Scalar of value 23d [1970-01-24]")
    check_stack_repr(gdb_arrow, "date32_scalar_null",
                     "arrow::Date32Scalar of null value")
    check_stack_repr(gdb_arrow, "date64_scalar",
                     "arrow::Date64Scalar of value 3888000000ms [1970-02-15]")
    check_stack_repr(gdb_arrow, "date64_scalar_null",
                     "arrow::Date64Scalar of null value")

    check_stack_repr(
        gdb_arrow, "decimal128_scalar_null",
        "arrow::Decimal128Scalar of null value [precision=10, scale=4]")
    check_stack_repr(
        gdb_arrow, "decimal128_scalar_pos_scale_pos",
        "arrow::Decimal128Scalar of value 123.4567 [precision=10, scale=4]")
    check_stack_repr(
        gdb_arrow, "decimal128_scalar_pos_scale_neg",
        "arrow::Decimal128Scalar of value -123.4567 [precision=10, scale=4]")
    check_stack_repr(
        gdb_arrow, "decimal128_scalar_neg_scale_pos",
        ("arrow::Decimal128Scalar of value 1.234567e+10 "
         "[precision=10, scale=-4]"))
    check_stack_repr(
        gdb_arrow, "decimal128_scalar_neg_scale_neg",
        ("arrow::Decimal128Scalar of value -1.234567e+10 "
         "[precision=10, scale=-4]"))

    check_stack_repr(
        gdb_arrow, "decimal256_scalar_null",
        "arrow::Decimal256Scalar of null value [precision=50, scale=4]")
    check_stack_repr(
        gdb_arrow, "decimal256_scalar_pos_scale_pos",
        ("arrow::Decimal256Scalar of value "
         "123456789012345678901234567890123456789012.3456 "
         "[precision=50, scale=4]"))
    check_stack_repr(
        gdb_arrow, "decimal256_scalar_pos_scale_neg",
        ("arrow::Decimal256Scalar of value "
         "-123456789012345678901234567890123456789012.3456 "
         "[precision=50, scale=4]"))
    check_stack_repr(
        gdb_arrow, "decimal256_scalar_neg_scale_pos",
        ("arrow::Decimal256Scalar of value "
         "1.234567890123456789012345678901234567890123456e+49 "
         "[precision=50, scale=-4]"))
    check_stack_repr(
        gdb_arrow, "decimal256_scalar_neg_scale_neg",
        ("arrow::Decimal256Scalar of value "
         "-1.234567890123456789012345678901234567890123456e+49 "
         "[precision=50, scale=-4]"))

    check_stack_repr(
        gdb_arrow, "binary_scalar_null",
        "arrow::BinaryScalar of null value")
    check_stack_repr(
        gdb_arrow, "binary_scalar_unallocated",
        "arrow::BinaryScalar of value <unallocated>")
    check_stack_repr(
        gdb_arrow, "binary_scalar_empty",
        'arrow::BinaryScalar of size 0, value ""')
    check_stack_repr(
        gdb_arrow, "binary_scalar_abc",
        'arrow::BinaryScalar of size 3, value "abc"')
    check_stack_repr(
        gdb_arrow, "binary_scalar_bytes",
        r'arrow::BinaryScalar of size 3, value "\000\037\377"')
    check_stack_repr(
        gdb_arrow, "large_binary_scalar_abc",
        'arrow::LargeBinaryScalar of size 3, value "abc"')

    check_stack_repr(
        gdb_arrow, "string_scalar_null",
        "arrow::StringScalar of null value")
    check_stack_repr(
        gdb_arrow, "string_scalar_unallocated",
        "arrow::StringScalar of value <unallocated>")
    check_stack_repr(
        gdb_arrow, "string_scalar_empty",
        'arrow::StringScalar of size 0, value ""')
    check_stack_repr(
        gdb_arrow, "string_scalar_hehe",
        'arrow::StringScalar of size 6, value "héhé"')
    # FIXME: excessive escaping ('\\xff' vs. '\x00')
    check_stack_repr(
        gdb_arrow, "string_scalar_invalid_chars",
        r'arrow::StringScalar of size 11, value "abc\x00def\\xffghi"')
    check_stack_repr(
        gdb_arrow, "large_string_scalar_hehe",
        'arrow::LargeStringScalar of size 6, value "héhé"')

    check_stack_repr(
        gdb_arrow, "fixed_size_binary_scalar",
        'arrow::FixedSizeBinaryScalar of size 3, value "abc"')
    check_stack_repr(
        gdb_arrow, "fixed_size_binary_scalar_null",
        'arrow::FixedSizeBinaryScalar of size 3, null with value "   "')

    check_stack_repr(
        gdb_arrow, "dict_scalar",
        re.compile(
            (r'^arrow::DictionaryScalar of index '
             r'arrow::Int8Scalar of value 42, '
             r'dictionary arrow::StringArray ')))
    check_stack_repr(
        gdb_arrow, "dict_scalar_null",
        ('arrow::DictionaryScalar of type '
         'arrow::dictionary(arrow::int8(), arrow::utf8(), ordered=false), '
         'null value'))

    check_stack_repr(
        gdb_arrow, "list_scalar",
        ('arrow::ListScalar of value arrow::Int32Array of '
         'length 3, offset 0, null count 0 = {[0] = 4, [1] = 5, [2] = 6}'))
    check_stack_repr(
        gdb_arrow, "list_scalar_null",
        'arrow::ListScalar of type arrow::list(arrow::int32()), null value')
    check_stack_repr(
        gdb_arrow, "large_list_scalar",
        ('arrow::LargeListScalar of value arrow::Int32Array of '
         'length 3, offset 0, null count 0 = {[0] = 4, [1] = 5, [2] = 6}'))
    check_stack_repr(
        gdb_arrow, "large_list_scalar_null",
        ('arrow::LargeListScalar of type arrow::large_list(arrow::int32()), '
         'null value'))
    check_stack_repr(
        gdb_arrow, "fixed_size_list_scalar",
        ('arrow::FixedSizeListScalar of value arrow::Int32Array of '
         'length 3, offset 0, null count 0 = {[0] = 4, [1] = 5, [2] = 6}'))
    check_stack_repr(
        gdb_arrow, "fixed_size_list_scalar_null",
        ('arrow::FixedSizeListScalar of type '
         'arrow::fixed_size_list(arrow::int32(), 3), null value'))

    check_stack_repr(
        gdb_arrow, "struct_scalar",
        ('arrow::StructScalar = {["ints"] = arrow::Int32Scalar of value 42, '
         '["strs"] = arrow::StringScalar of size 9, value "some text"}'))
    check_stack_repr(
        gdb_arrow, "struct_scalar_null",
        ('arrow::StructScalar of type arrow::struct_('
         '{arrow::field("ints", arrow::int32()), '
         'arrow::field("strs", arrow::utf8())}), null value'))

    check_stack_repr(
        gdb_arrow, "sparse_union_scalar",
        ('arrow::SparseUnionScalar of type code 7, '
         'value arrow::Int32Scalar of value 43'))
    check_stack_repr(
        gdb_arrow, "sparse_union_scalar_null", re.compile(
            r'^arrow::SparseUnionScalar of type arrow::sparse_union\(.*\), '
            r'type code 7, null value$'))
    check_stack_repr(
        gdb_arrow, "dense_union_scalar",
        ('arrow::DenseUnionScalar of type code 7, '
         'value arrow::Int32Scalar of value 43'))
    check_stack_repr(
        gdb_arrow, "dense_union_scalar_null", re.compile(
            r'^arrow::DenseUnionScalar of type arrow::dense_union\(.*\), '
            r'type code 7, null value$'))

    check_stack_repr(
        gdb_arrow, "extension_scalar",
        ('arrow::ExtensionScalar of type "extension<uuid>", '
         'value arrow::FixedSizeBinaryScalar of size 16, '
         'value "0123456789abcdef"'))
    check_stack_repr(
        gdb_arrow, "extension_scalar_null",
        'arrow::ExtensionScalar of type "extension<uuid>", null value')


def test_scalars_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, "heap_null_scalar", "arrow::NullScalar")
    check_heap_repr(gdb_arrow, "heap_bool_scalar",
                    "arrow::BooleanScalar of value true")
    check_heap_repr(
        gdb_arrow, "heap_decimal128_scalar",
        "arrow::Decimal128Scalar of value 123.4567 [precision=10, scale=4]")
    check_heap_repr(
        gdb_arrow, "heap_decimal256_scalar",
        ("arrow::Decimal256Scalar of value "
         "123456789012345678901234567890123456789012.3456 "
         "[precision=50, scale=4]"))

    check_heap_repr(
        gdb_arrow, "heap_map_scalar",
        ('arrow::MapScalar of type arrow::map(arrow::utf8(), arrow::int32(), '
         'keys_sorted=false), value length 2, offset 0, null count 0'))
    check_heap_repr(
        gdb_arrow, "heap_map_scalar_null",
        ('arrow::MapScalar of type arrow::map(arrow::utf8(), arrow::int32(), '
         'keys_sorted=false), null value'))


def test_array_data(gdb_arrow):
    check_stack_repr(
        gdb_arrow, "int32_array_data",
        ("arrow::ArrayData of type arrow::int32(), length 4, offset 0, "
         "null count 1 = {[0] = -5, [1] = 6, [2] = null, [3] = 42}"))


def test_arrays_stack(gdb_arrow):
    check_stack_repr(
        gdb_arrow, "int32_array",
        ("arrow::Int32Array of length 4, offset 0, null count 1 = "
         "{[0] = -5, [1] = 6, [2] = null, [3] = 42}"))
    check_stack_repr(
        gdb_arrow, "list_array",
        ("arrow::ListArray of type arrow::list(arrow::int64()), "
         "length 3, offset 0, null count 1"))


def test_arrays_heap(gdb_arrow):
    # Null
    check_heap_repr(
        gdb_arrow, "heap_null_array",
        "arrow::NullArray of length 2, offset 0, null count 2")

    # Primitive
    check_heap_repr(
        gdb_arrow, "heap_int32_array",
        ("arrow::Int32Array of length 4, offset 0, null count 1 = {"
         "[0] = -5, [1] = 6, [2] = null, [3] = 42}"))
    check_heap_repr(
        gdb_arrow, "heap_int32_array_no_nulls",
        ("arrow::Int32Array of length 4, offset 0, null count 0 = {"
         "[0] = -5, [1] = 6, [2] = 3, [3] = 42}"))
    check_heap_repr(
        gdb_arrow, "heap_int32_array_sliced_1_9",
        ("arrow::Int32Array of length 9, offset 1, unknown null count = {"
         "[0] = 2, [1] = -3, [2] = 4, [3] = null, [4] = -5, [5] = 6, "
         "[6] = -7, [7] = 8, [8] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_int32_array_sliced_2_6",
        ("arrow::Int32Array of length 6, offset 2, unknown null count = {"
         "[0] = -3, [1] = 4, [2] = null, [3] = -5, [4] = 6, [5] = -7}"))
    check_heap_repr(
        gdb_arrow, "heap_int32_array_sliced_8_4",
        ("arrow::Int32Array of length 4, offset 8, unknown null count = {"
         "[0] = 8, [1] = null, [2] = -9, [3] = -10}"))
    check_heap_repr(
        gdb_arrow, "heap_int32_array_sliced_empty",
        "arrow::Int32Array of length 0, offset 6, unknown null count")

    check_heap_repr(
        gdb_arrow, "heap_double_array",
        ("arrow::DoubleArray of length 2, offset 0, null count 1 = {"
         "[0] = -1.5, [1] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_float16_array",
        ("arrow::HalfFloatArray of length 2, offset 0, null count 0 = {"
         "[0] = 0.0, [1] = -1.5}"))

    # Boolean
    check_heap_repr(
        gdb_arrow, "heap_bool_array",
        ("arrow::BooleanArray of length 18, offset 0, null count 6 = {"
         "[0] = false, [1] = false, [2] = true, [3] = true, [4] = null, "
         "[5] = null, [6] = false, [7] = false, [8] = true, [9] = true, "
         "[10] = null, [11] = null, [12] = false, [13] = false, "
         "[14] = true, [15] = true, [16] = null, [17] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_bool_array_sliced_1_9",
        ("arrow::BooleanArray of length 9, offset 1, unknown null count = {"
         "[0] = false, [1] = true, [2] = true, [3] = null, [4] = null, "
         "[5] = false, [6] = false, [7] = true, [8] = true}"))
    check_heap_repr(
        gdb_arrow, "heap_bool_array_sliced_2_6",
        ("arrow::BooleanArray of length 6, offset 2, unknown null count = {"
         "[0] = true, [1] = true, [2] = null, [3] = null, [4] = false, "
         "[5] = false}"))
    check_heap_repr(
        gdb_arrow, "heap_bool_array_sliced_empty",
        "arrow::BooleanArray of length 0, offset 6, unknown null count")

    # Temporal
    check_heap_repr(
        gdb_arrow, "heap_date32_array",
        ("arrow::Date32Array of length 6, offset 0, null count 1 = {"
         "[0] = 0d [1970-01-01], [1] = null, [2] = 18336d [2020-03-15], "
         "[3] = -9004d [1945-05-08], [4] = -719162d [0001-01-01], "
         "[5] = -719163d [year <= 0]}"))
    check_heap_repr(
        gdb_arrow, "heap_date64_array",
        ("arrow::Date64Array of length 5, offset 0, null count 0 = {"
         "[0] = 1584230400000ms [2020-03-15], "
         "[1] = -777945600000ms [1945-05-08], "
         "[2] = -62135596800000ms [0001-01-01], "
         "[3] = -62135683200000ms [year <= 0], "
         "[4] = 123ms [non-multiple of 86400000]}"))
    check_heap_repr(
        gdb_arrow, "heap_time32_array_s",
        ("arrow::Time32Array of type arrow::time32(arrow::TimeUnit::SECOND), "
         "length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -123s, [2] = 456s}"))
    check_heap_repr(
        gdb_arrow, "heap_time32_array_ms",
        ("arrow::Time32Array of type arrow::time32(arrow::TimeUnit::MILLI), "
         "length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -123ms, [2] = 456ms}"))
    check_heap_repr(
        gdb_arrow, "heap_time64_array_us",
        ("arrow::Time64Array of type arrow::time64(arrow::TimeUnit::MICRO), "
         "length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -123us, [2] = 456us}"))
    check_heap_repr(
        gdb_arrow, "heap_time64_array_ns",
        ("arrow::Time64Array of type arrow::time64(arrow::TimeUnit::NANO), "
         "length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -123ns, [2] = 456ns}"))
    check_heap_repr(
        gdb_arrow, "heap_month_interval_array",
        ("arrow::MonthIntervalArray of length 3, offset 0, null count 1 = {"
         "[0] = 123M, [1] = -456M, [2] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_day_time_interval_array",
        ("arrow::DayTimeIntervalArray of length 2, offset 0, null count 1 = {"
         "[0] = 1d-600ms, [1] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_month_day_nano_interval_array",
        ("arrow::MonthDayNanoIntervalArray of length 2, offset 0, "
         "null count 1 = {[0] = 1M-600d5000ns, [1] = null}"))
    check_heap_repr(
        gdb_arrow, "heap_duration_array_s",
        ("arrow::DurationArray of type arrow::duration"
         "(arrow::TimeUnit::SECOND), length 2, offset 0, null count 1 = {"
         "[0] = null, [1] = -1234567890123456789s}"))
    check_heap_repr(
        gdb_arrow, "heap_duration_array_ns",
        ("arrow::DurationArray of type arrow::duration"
         "(arrow::TimeUnit::NANO), length 2, offset 0, null count 1 = {"
         "[0] = null, [1] = -1234567890123456789ns}"))
    check_heap_repr(
        gdb_arrow, "heap_timestamp_array_s",
        ("arrow::TimestampArray of type arrow::timestamp"
         "(arrow::TimeUnit::SECOND), length 4, offset 0, null count 1 = {"
         "[0] = null, [1] = 0s [1970-01-01 00:00:00], "
         "[2] = -2203932304s [1900-02-28 12:34:56], "
         "[3] = 63730281600s [3989-07-14 00:00:00]}"))
    check_heap_repr(
        gdb_arrow, "heap_timestamp_array_ms",
        ("arrow::TimestampArray of type arrow::timestamp"
         "(arrow::TimeUnit::MILLI), length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -2203932303877ms [1900-02-28 12:34:56.123], "
         "[2] = 63730281600789ms [3989-07-14 00:00:00.789]}"))
    check_heap_repr(
        gdb_arrow, "heap_timestamp_array_us",
        ("arrow::TimestampArray of type arrow::timestamp"
         "(arrow::TimeUnit::MICRO), length 3, offset 0, null count 1 = {"
         "[0] = null, "
         "[1] = -2203932303345679us [1900-02-28 12:34:56.654321], "
         "[2] = 63730281600456789us [3989-07-14 00:00:00.456789]}"))
    check_heap_repr(
        gdb_arrow, "heap_timestamp_array_ns",
        ("arrow::TimestampArray of type arrow::timestamp"
         "(arrow::TimeUnit::NANO), length 2, offset 0, null count 1 = {"
         "[0] = null, "
         "[1] = -2203932303012345679ns [1900-02-28 12:34:56.987654321]}"))

    # Decimal
    check_heap_repr(
        gdb_arrow, "heap_decimal128_array",
        ("arrow::Decimal128Array of type arrow::decimal128(30, 6), "
         "length 3, offset 0, null count 1 = {"
         "[0] = null, [1] = -1234567890123456789.012345, "
         "[2] = 1234567890123456789.012345}"))
    check_heap_repr(
        gdb_arrow, "heap_decimal256_array",
        ("arrow::Decimal256Array of type arrow::decimal256(50, 6), "
         "length 2, offset 0, null count 1 = {"
         "[0] = null, "
         "[1] = -123456789012345678901234567890123456789.012345}"))
    check_heap_repr(
        gdb_arrow, "heap_decimal128_array_sliced",
        ("arrow::Decimal128Array of type arrow::decimal128(30, 6), "
         "length 1, offset 1, unknown null count = {"
         "[0] = -1234567890123456789.012345}"))

    # Binary-like
    check_heap_repr(
        gdb_arrow, "heap_fixed_size_binary_array",
        (r'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(3), '
         r'length 3, offset 0, null count 1 = {'
         r'[0] = null, [1] = "abc", [2] = "\000\037\377"}'))
    check_heap_repr(
        gdb_arrow, "heap_fixed_size_binary_array_zero_width",
        (r'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(0), '
         r'length 2, offset 0, null count 1 = {[0] = null, [1] = ""}'))
    check_heap_repr(
        gdb_arrow, "heap_fixed_size_binary_array_sliced",
        (r'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(3), '
         r'length 1, offset 1, unknown null count = {[0] = "abc"}'))
    check_heap_repr(
        gdb_arrow, "heap_binary_array",
        (r'arrow::BinaryArray of length 3, offset 0, null count 1 = {'
         r'[0] = null, [1] = "abcd", [2] = "\000\037\377"}'))
    check_heap_repr(
        gdb_arrow, "heap_large_binary_array",
        (r'arrow::LargeBinaryArray of length 3, offset 0, null count 1 = {'
         r'[0] = null, [1] = "abcd", [2] = "\000\037\377"}'))
    check_heap_repr(
        gdb_arrow, "heap_string_array",
        (r'arrow::StringArray of length 3, offset 0, null count 1 = {'
         r'[0] = null, [1] = "héhé", [2] = "invalid \\xff char"}'))
    check_heap_repr(
        gdb_arrow, "heap_large_string_array",
        (r'arrow::LargeStringArray of length 3, offset 0, null count 1 = {'
         r'[0] = null, [1] = "héhé", [2] = "invalid \\xff char"}'))
    check_heap_repr(
        gdb_arrow, "heap_binary_array_sliced",
        (r'arrow::BinaryArray of length 1, offset 1, unknown null count = '
         r'{[0] = "abcd"}'))

    # Nested
    check_heap_repr(
        gdb_arrow, "heap_list_array",
        ("arrow::ListArray of type arrow::list(arrow::int64()), "
         "length 3, offset 0, null count 1"))


def test_schema(gdb_arrow):
    check_heap_repr(gdb_arrow, "schema_empty",
                    "arrow::Schema with 0 fields")
    check_heap_repr(
        gdb_arrow, "schema_non_empty",
        ('arrow::Schema with 2 fields = {["ints"] = arrow::int8(), '
         '["strs"] = arrow::utf8()}'))
    check_heap_repr(
        gdb_arrow, "schema_with_metadata",
        ('arrow::Schema with 2 fields and 2 metadata items = '
         '{["ints"] = arrow::int8(), ["strs"] = arrow::utf8()}'))


def test_chunked_array(gdb_arrow):
    check_stack_repr(
        gdb_arrow, "chunked_array",
        ("arrow::ChunkedArray of type arrow::int32(), length 5, null count 1 "
         "with 2 chunks = {[0] = length 2, offset 0, null count 0, "
         "[1] = length 3, offset 0, null count 1}"))


def test_record_batch(gdb_arrow):
    expected_prefix = 'arrow::RecordBatch with 2 columns, 3 rows'
    expected_suffix = (
        '{["ints"] = arrow::ArrayData of type arrow::int32(), '
        'length 3, offset 0, null count 0 = '
        '{[0] = 1, [1] = 2, [2] = 3}, '
        '["strs"] = arrow::ArrayData of type arrow::utf8(), '
        'length 3, offset 0, null count 1 = '
        '{[0] = "abc", [1] = null, [2] = "def"}}')

    expected = f"{expected_prefix} = {expected_suffix}"
    # Representations may differ between those two because of
    # RecordBatch (base class) vs. SimpleRecordBatch (concrete class).
    check_heap_repr(gdb_arrow, "batch", expected)
    check_heap_repr(gdb_arrow, "batch.get()", expected)

    expected = f"{expected_prefix}, 3 metadata items = {expected_suffix}"
    check_heap_repr(gdb_arrow, "batch_with_metadata", expected)


def test_table(gdb_arrow):
    expected_table = (
        'arrow::Table with 2 columns, 5 rows = {'
        '["ints"] = arrow::ChunkedArray of type arrow::int32(), '
        'length 5, null count 0 with 2 chunks = '
        '{[0] = length 3, offset 0, null count 0, '
        '[1] = length 2, offset 0, null count 0}, '
        '["strs"] = arrow::ChunkedArray of type arrow::utf8(), '
        'length 5, null count 1 with 3 chunks = '
        '{[0] = length 2, offset 0, null count 1, '
        '[1] = length 1, offset 0, null count 0, '
        '[2] = length 2, offset 0, null count 0}}')

    # Same as RecordBatch above (Table vs. SimpleTable)
    check_heap_repr(gdb_arrow, "table", expected_table)
    check_heap_repr(gdb_arrow, "table.get()", expected_table)


def test_datum(gdb_arrow):
    check_stack_repr(gdb_arrow, "empty_datum", "arrow::Datum (empty)")
    check_stack_repr(
        gdb_arrow, "scalar_datum",
        "arrow::Datum of value arrow::BooleanScalar of null value")
    check_stack_repr(
        gdb_arrow, "array_datum",
        re.compile(r"^arrow::Datum of value arrow::ArrayData of type "))
    check_stack_repr(
        gdb_arrow, "chunked_array_datum",
        re.compile(r"^arrow::Datum of value arrow::ChunkedArray of type "))
    check_stack_repr(
        gdb_arrow, "batch_datum",
        re.compile(r"^arrow::Datum of value arrow::RecordBatch "
                   r"with 2 columns, 3 rows "))
    check_stack_repr(
        gdb_arrow, "table_datum",
        re.compile(r"^arrow::Datum of value arrow::Table "
                   r"with 2 columns, 5 rows "))
