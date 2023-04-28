// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <cstdlib>
#include <memory>
#include <utility>

#include "arrow/array.h"
#include "arrow/chunked_array.h"
#include "arrow/datum.h"
#include "arrow/extension_type.h"
#include "arrow/ipc/json_simple.h"
#include "arrow/record_batch.h"
#include "arrow/scalar.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/util/debug.h"
#include "arrow/util/decimal.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/python/gdb.h"

namespace arrow {

using ipc::internal::json::ArrayFromJSON;
using ipc::internal::json::ChunkedArrayFromJSON;
using ipc::internal::json::ScalarFromJSON;

namespace gdb {

// Add a nested `arrow` namespace to exercise type lookup from GDB (ARROW-15652)
namespace arrow {
void DummyFunction() {}
}  // namespace arrow

namespace {

class CustomStatusDetail : public StatusDetail {
 public:
  const char* type_id() const override { return "custom-detail-id"; }
  std::string ToString() const override { return "This is a detail"; }
};

class UuidType : public ExtensionType {
 public:
  UuidType() : ExtensionType(fixed_size_binary(16)) {}

  std::string extension_name() const override { return "uuid"; }

  bool ExtensionEquals(const ExtensionType& other) const override {
    return (other.extension_name() == this->extension_name());
  }

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override {
    return std::make_shared<ExtensionArray>(data);
  }

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override {
    return Status::NotImplemented("");
  }

  std::string Serialize() const override { return "uuid-serialized"; }
};

std::shared_ptr<Array> SliceArrayFromJSON(const std::shared_ptr<DataType>& ty,
                                          std::string_view json, int64_t offset = 0,
                                          int64_t length = -1) {
  auto array = *ArrayFromJSON(ty, json);
  if (length != -1) {
    return array->Slice(offset, length);
  } else {
    return array->Slice(offset);
  }
}

}  // namespace

void TestSession() {
  // We define local variables for all types for which we want to test
  // pretty-printing.
  // Then, at the end of this function, we trap to the debugger, so that
  // test instrumentation can print values from this frame by interacting
  // with the debugger.
  // The test instrumentation is in pyarrow/tests/test_gdb.py

#ifdef __clang__
  _Pragma("clang diagnostic push");
  _Pragma("clang diagnostic ignored \"-Wunused-variable\"");
#elif defined(__GNUC__)
  _Pragma("GCC diagnostic push");
  _Pragma("GCC diagnostic ignored \"-Wunused-variable\"");
#endif

  arrow::DummyFunction();

  // Status & Result
  auto ok_status = Status::OK();
  auto error_status = Status::IOError("This is an error");
  auto error_detail_status =
      error_status.WithDetail(std::make_shared<CustomStatusDetail>());
  auto ok_result = Result<int>(42);
  auto error_result = Result<int>(error_status);
  auto error_detail_result = Result<int>(error_detail_status);

  // String views
  std::string_view string_view_abc{"abc"};
  std::string special_chars = std::string("foo\"bar") + '\x00' + "\r\n\t\x1f";
  std::string_view string_view_special_chars(special_chars);

  // Buffers
  Buffer buffer_null{nullptr, 0};
  Buffer buffer_abc{string_view_abc};
  Buffer buffer_special_chars{string_view_special_chars};
  char mutable_array[3] = {'a', 'b', 'c'};
  MutableBuffer buffer_mutable{reinterpret_cast<uint8_t*>(mutable_array), 3};
  auto heap_buffer = std::make_shared<Buffer>(string_view_abc);
  auto heap_buffer_mutable = *AllocateBuffer(buffer_abc.size());
  memcpy(heap_buffer_mutable->mutable_data(), buffer_abc.data(), buffer_abc.size());

  // KeyValueMetadata
  auto empty_metadata = key_value_metadata({}, {});
  auto metadata = key_value_metadata(
      {"key_text", "key_binary"}, {"some value", std::string("z") + '\x00' + "\x1f\xff"});

  // Decimals
  Decimal128 decimal128_zero{};
  Decimal128 decimal128_pos{"98765432109876543210987654321098765432"};
  Decimal128 decimal128_neg{"-98765432109876543210987654321098765432"};
  BasicDecimal128 basic_decimal128_zero{};
  BasicDecimal128 basic_decimal128_pos{decimal128_pos.native_endian_array()};
  BasicDecimal128 basic_decimal128_neg{decimal128_neg.native_endian_array()};
  Decimal256 decimal256_zero{};
  Decimal256 decimal256_pos{
      "9876543210987654321098765432109876543210987654321098765432109876543210987654"};
  Decimal256 decimal256_neg{
      "-9876543210987654321098765432109876543210987654321098765432109876543210987654"};
  BasicDecimal256 basic_decimal256_zero{};
  BasicDecimal256 basic_decimal256_pos{decimal256_pos.native_endian_array()};
  BasicDecimal256 basic_decimal256_neg{decimal256_neg.native_endian_array()};

  // Data types
  NullType null_type;
  auto heap_null_type = null();
  BooleanType bool_type;
  auto heap_bool_type = boolean();

  Date32Type date32_type;
  Date64Type date64_type;
  Time32Type time_type_s(TimeUnit::SECOND);
  Time32Type time_type_ms(TimeUnit::MILLI);
  Time64Type time_type_us(TimeUnit::MICRO);
  Time64Type time_type_ns(TimeUnit::NANO);
  auto heap_time_type_ns = time64(TimeUnit::NANO);

  TimestampType timestamp_type_s(TimeUnit::SECOND);
  TimestampType timestamp_type_ms_timezone(TimeUnit::MILLI, "Europe/Paris");
  TimestampType timestamp_type_us(TimeUnit::MICRO);
  TimestampType timestamp_type_ns_timezone(TimeUnit::NANO, "Europe/Paris");
  auto heap_timestamp_type_ns_timezone = timestamp(TimeUnit::NANO, "Europe/Paris");

  DayTimeIntervalType day_time_interval_type;
  MonthIntervalType month_interval_type;
  MonthDayNanoIntervalType month_day_nano_interval_type;

  DurationType duration_type_s(TimeUnit::SECOND);
  DurationType duration_type_ns(TimeUnit::NANO);

  BinaryType binary_type;
  StringType string_type;
  LargeBinaryType large_binary_type;
  LargeStringType large_string_type;
  FixedSizeBinaryType fixed_size_binary_type(10);
  auto heap_fixed_size_binary_type = fixed_size_binary(10);

  Decimal128Type decimal128_type(16, 5);
  Decimal256Type decimal256_type(42, 12);
  auto heap_decimal128_type = decimal128(16, 5);

  ListType list_type(uint8());
  LargeListType large_list_type(large_utf8());
  auto heap_list_type = list(uint8());
  auto heap_large_list_type = large_list(large_utf8());

  FixedSizeListType fixed_size_list_type(float64(), 3);
  auto heap_fixed_size_list_type = fixed_size_list(float64(), 3);

  DictionaryType dict_type_unordered(int16(), utf8());
  DictionaryType dict_type_ordered(int16(), utf8(), /*ordered=*/true);
  auto heap_dict_type = dictionary(int16(), utf8());

  MapType map_type_unsorted(utf8(), binary());
  MapType map_type_sorted(utf8(), binary(), /*keys_sorted=*/true);
  auto heap_map_type = map(utf8(), binary());

  StructType struct_type_empty({});
  StructType struct_type(
      {field("ints", int8()), field("strs", utf8(), /*nullable=*/false)});
  auto heap_struct_type =
      struct_({field("ints", int8()), field("strs", utf8(), /*nullable=*/false)});

  std::vector<int8_t> union_type_codes({7, 42});
  FieldVector union_fields(
      {field("ints", int8()), field("strs", utf8(), /*nullable=*/false)});
  SparseUnionType sparse_union_type(union_fields, union_type_codes);
  DenseUnionType dense_union_type(union_fields, union_type_codes);

  UuidType uuid_type{};
  std::shared_ptr<DataType> heap_uuid_type = std::make_shared<UuidType>();

  // Schema
  auto schema_empty = schema({});
  auto schema_non_empty = schema({field("ints", int8()), field("strs", utf8())});
  auto schema_with_metadata = schema_non_empty->WithMetadata(
      key_value_metadata({"key1", "key2"}, {"value1", "value2"}));

  // Fields
  Field int_field("ints", int64());
  Field float_field("floats", float32(), /*nullable=*/false);
  auto heap_int_field = field("ints", int64());

  // Scalars
  NullScalar null_scalar;
  auto heap_null_scalar = MakeNullScalar(null());

  BooleanScalar bool_scalar_null{};
  BooleanScalar bool_scalar{true};
  auto heap_bool_scalar = *MakeScalar(boolean(), true);

  Int8Scalar int8_scalar_null{};
  UInt8Scalar uint8_scalar_null{};
  Int64Scalar int64_scalar_null{};
  UInt64Scalar uint64_scalar_null{};
  Int8Scalar int8_scalar{-42};
  UInt8Scalar uint8_scalar{234};
  Int64Scalar int64_scalar{-9223372036854775807LL - 1};
  UInt64Scalar uint64_scalar{18446744073709551615ULL};
  HalfFloatScalar half_float_scalar{48640};  // -1.5
  FloatScalar float_scalar{1.25f};
  DoubleScalar double_scalar{2.5};

  Time32Scalar time_scalar_s{100, TimeUnit::SECOND};
  Time32Scalar time_scalar_ms{1000, TimeUnit::MILLI};
  Time64Scalar time_scalar_us{10000, TimeUnit::MICRO};
  Time64Scalar time_scalar_ns{100000, TimeUnit::NANO};
  Time64Scalar time_scalar_null{time64(TimeUnit::NANO)};

  DurationScalar duration_scalar_s{-100, TimeUnit::SECOND};
  DurationScalar duration_scalar_ms{-1000, TimeUnit::MILLI};
  DurationScalar duration_scalar_us{-10000, TimeUnit::MICRO};
  DurationScalar duration_scalar_ns{-100000, TimeUnit::NANO};
  DurationScalar duration_scalar_null{duration(TimeUnit::NANO)};

  TimestampScalar timestamp_scalar_s{12345, timestamp(TimeUnit::SECOND)};
  TimestampScalar timestamp_scalar_ms{-123456, timestamp(TimeUnit::MILLI)};
  TimestampScalar timestamp_scalar_us{1234567, timestamp(TimeUnit::MICRO)};
  TimestampScalar timestamp_scalar_ns{-12345678, timestamp(TimeUnit::NANO)};
  TimestampScalar timestamp_scalar_null{timestamp(TimeUnit::NANO)};

  TimestampScalar timestamp_scalar_s_tz{12345,
                                        timestamp(TimeUnit::SECOND, "Europe/Paris")};
  TimestampScalar timestamp_scalar_ms_tz{-123456,
                                         timestamp(TimeUnit::MILLI, "Europe/Paris")};
  TimestampScalar timestamp_scalar_us_tz{1234567,
                                         timestamp(TimeUnit::MICRO, "Europe/Paris")};
  TimestampScalar timestamp_scalar_ns_tz{-12345678,
                                         timestamp(TimeUnit::NANO, "Europe/Paris")};
  TimestampScalar timestamp_scalar_null_tz{timestamp(TimeUnit::NANO, "Europe/Paris")};

  MonthIntervalScalar month_interval_scalar{23};
  MonthIntervalScalar month_interval_scalar_null{};
  DayTimeIntervalScalar day_time_interval_scalar{{23, -456}};
  DayTimeIntervalScalar day_time_interval_scalar_null{};
  MonthDayNanoIntervalScalar month_day_nano_interval_scalar{{1, 23, -456}};
  MonthDayNanoIntervalScalar month_day_nano_interval_scalar_null{};

  Date32Scalar date32_scalar{23};
  Date32Scalar date32_scalar_null{};
  Date64Scalar date64_scalar{45 * 86400000LL};
  Date64Scalar date64_scalar_null{};

  Decimal128Scalar decimal128_scalar_pos_scale_pos{Decimal128("1234567"),
                                                   decimal128(10, 4)};
  Decimal128Scalar decimal128_scalar_pos_scale_neg{Decimal128("-1234567"),
                                                   decimal128(10, 4)};
  Decimal128Scalar decimal128_scalar_neg_scale_pos{Decimal128("1234567"),
                                                   decimal128(10, -4)};
  Decimal128Scalar decimal128_scalar_neg_scale_neg{Decimal128("-1234567"),
                                                   decimal128(10, -4)};
  Decimal128Scalar decimal128_scalar_null{decimal128(10, 4)};
  auto heap_decimal128_scalar = *MakeScalar(decimal128(10, 4), Decimal128("1234567"));

  Decimal256Scalar decimal256_scalar_pos_scale_pos{
      Decimal256("1234567890123456789012345678901234567890123456"), decimal256(50, 4)};
  Decimal256Scalar decimal256_scalar_pos_scale_neg{
      Decimal256("-1234567890123456789012345678901234567890123456"), decimal256(50, 4)};
  Decimal256Scalar decimal256_scalar_neg_scale_pos{
      Decimal256("1234567890123456789012345678901234567890123456"), decimal256(50, -4)};
  Decimal256Scalar decimal256_scalar_neg_scale_neg{
      Decimal256("-1234567890123456789012345678901234567890123456"), decimal256(50, -4)};
  Decimal256Scalar decimal256_scalar_null{decimal256(50, 4)};
  auto heap_decimal256_scalar = *MakeScalar(
      decimal256(50, 4), Decimal256("1234567890123456789012345678901234567890123456"));

  BinaryScalar binary_scalar_null{};
  BinaryScalar binary_scalar_unallocated{std::shared_ptr<Buffer>{nullptr}};
  BinaryScalar binary_scalar_empty{Buffer::FromString("")};
  BinaryScalar binary_scalar_abc{Buffer::FromString("abc")};
  BinaryScalar binary_scalar_bytes{
      Buffer::FromString(std::string() + '\x00' + "\x1f\xff")};

  StringScalar string_scalar_null{};
  StringScalar string_scalar_unallocated{std::shared_ptr<Buffer>{nullptr}};
  StringScalar string_scalar_empty{Buffer::FromString("")};
  StringScalar string_scalar_hehe{Buffer::FromString("héhé")};
  StringScalar string_scalar_invalid_chars{
      Buffer::FromString(std::string("abc") + '\x00' + "def\xffghi")};

  LargeBinaryScalar large_binary_scalar_abc{Buffer::FromString("abc")};
  LargeStringScalar large_string_scalar_hehe{Buffer::FromString("héhé")};

  FixedSizeBinaryScalar fixed_size_binary_scalar{Buffer::FromString("abc"),
                                                 fixed_size_binary(3)};
  FixedSizeBinaryScalar fixed_size_binary_scalar_null{
      Buffer::FromString("   "), fixed_size_binary(3), /*is_valid=*/false};

  std::shared_ptr<Array> dict_array;
  dict_array = *ArrayFromJSON(utf8(), R"(["foo", "bar", "quux"])");
  DictionaryScalar dict_scalar{{std::make_shared<Int8Scalar>(42), dict_array},
                               dictionary(int8(), utf8())};
  DictionaryScalar dict_scalar_null{dictionary(int8(), utf8())};

  std::shared_ptr<Array> list_value_array = *ArrayFromJSON(int32(), R"([4, 5, 6])");
  std::shared_ptr<Array> list_zero_length = *ArrayFromJSON(int32(), R"([])");
  ListScalar list_scalar{list_value_array};
  ListScalar list_scalar_null{list_zero_length, list(int32()), /*is_valid=*/false};
  LargeListScalar large_list_scalar{list_value_array};
  LargeListScalar large_list_scalar_null{list_zero_length, large_list(int32()),
                                         /*is_valid=*/false};
  FixedSizeListScalar fixed_size_list_scalar{list_value_array};
  FixedSizeListScalar fixed_size_list_scalar_null{
      list_value_array, fixed_size_list(int32(), 3), /*is_valid=*/false};

  auto struct_scalar_type = struct_({field("ints", int32()), field("strs", utf8())});
  StructScalar struct_scalar{
      ScalarVector{MakeScalar(int32_t(42)), MakeScalar("some text")}, struct_scalar_type};
  StructScalar struct_scalar_null{struct_scalar.value, struct_scalar_type,
                                  /*is_valid=*/false};

  auto sparse_union_scalar_type =
      sparse_union(FieldVector{field("ints", int32()), field("strs", utf8())}, {7, 42});
  auto dense_union_scalar_type =
      dense_union(FieldVector{field("ints", int32()), field("strs", utf8())}, {7, 42});
  std::vector<std::shared_ptr<Scalar>> union_values = {MakeScalar(int32_t(43)),
                                                       MakeNullScalar(utf8())};
  SparseUnionScalar sparse_union_scalar{union_values, 7, sparse_union_scalar_type};
  DenseUnionScalar dense_union_scalar{union_values[0], 7, dense_union_scalar_type};

  union_values[0] = MakeNullScalar(int32());
  SparseUnionScalar sparse_union_scalar_null{union_values, 7, sparse_union_scalar_type};
  DenseUnionScalar dense_union_scalar_null{union_values[0], 7, dense_union_scalar_type};

  auto extension_scalar_type = std::make_shared<UuidType>();
  ExtensionScalar extension_scalar{
      std::make_shared<FixedSizeBinaryScalar>(Buffer::FromString("0123456789abcdef"),
                                              extension_scalar_type->storage_type()),
      extension_scalar_type};
  ExtensionScalar extension_scalar_null{extension_scalar.value, extension_scalar_type,
                                        /*is_valid=*/false};

  std::shared_ptr<Scalar> heap_map_scalar;
  ARROW_CHECK_OK(
      ScalarFromJSON(map(utf8(), int32()), R"([["a", 5], ["b", 6]])", &heap_map_scalar));
  auto heap_map_scalar_null = MakeNullScalar(heap_map_scalar->type);

  // Array and ArrayData
  auto heap_null_array = SliceArrayFromJSON(null(), "[null, null]");

  auto heap_int32_array = SliceArrayFromJSON(int32(), "[-5, 6, null, 42]");
  ArrayData int32_array_data{*heap_int32_array->data()};
  Int32Array int32_array{heap_int32_array->data()->Copy()};

  auto heap_int32_array_no_nulls = SliceArrayFromJSON(int32(), "[-5, 6, 3, 42]");

  const char* json_int32_array = "[-1, 2, -3, 4, null, -5, 6, -7, 8, null, -9, -10]";
  auto heap_int32_array_sliced_1_9 = SliceArrayFromJSON(int32(), json_int32_array, 1, 9);
  auto heap_int32_array_sliced_2_6 = SliceArrayFromJSON(int32(), json_int32_array, 2, 6);
  auto heap_int32_array_sliced_8_4 = SliceArrayFromJSON(int32(), json_int32_array, 8, 4);
  auto heap_int32_array_sliced_empty =
      SliceArrayFromJSON(int32(), json_int32_array, 6, 0);

  const char* json_bool_array =
      "[false, false, true, true, null, null, false, false, true, true, "
      "null, null, false, false, true, true, null, null]";
  auto heap_bool_array = SliceArrayFromJSON(boolean(), json_bool_array);
  auto heap_bool_array_sliced_1_9 = SliceArrayFromJSON(boolean(), json_bool_array, 1, 9);
  auto heap_bool_array_sliced_2_6 = SliceArrayFromJSON(boolean(), json_bool_array, 2, 6);
  auto heap_bool_array_sliced_empty =
      SliceArrayFromJSON(boolean(), json_bool_array, 6, 0);

  auto heap_list_array = SliceArrayFromJSON(list(int64()), "[[1, 2], null, []]");
  ListArray list_array{heap_list_array->data()};

  const char* json_double_array = "[-1.5, null]";
  auto heap_double_array = SliceArrayFromJSON(float64(), json_double_array);

  const char* json_float16_array = "[0, 48640]";
  auto heap_float16_array =
      *SliceArrayFromJSON(uint16(), json_float16_array)->View(float16());

  auto heap_date32_array =
      SliceArrayFromJSON(date32(), "[0, null, 18336, -9004, -719162, -719163]");
  auto heap_date64_array = SliceArrayFromJSON(
      date64(), "[1584230400000, -777945600000, -62135596800000, -62135683200000, 123]");

  const char* json_time_array = "[null, -123, 456]";
  auto heap_time32_array_s =
      SliceArrayFromJSON(time32(TimeUnit::SECOND), json_time_array);
  auto heap_time32_array_ms =
      SliceArrayFromJSON(time32(TimeUnit::MILLI), json_time_array);
  auto heap_time64_array_us =
      SliceArrayFromJSON(time64(TimeUnit::MICRO), json_time_array);
  auto heap_time64_array_ns = SliceArrayFromJSON(time64(TimeUnit::NANO), json_time_array);

  auto heap_month_interval_array =
      SliceArrayFromJSON(month_interval(), "[123, -456, null]");
  auto heap_day_time_interval_array =
      SliceArrayFromJSON(day_time_interval(), "[[1, -600], null]");
  auto heap_month_day_nano_interval_array =
      SliceArrayFromJSON(month_day_nano_interval(), "[[1, -600, 5000], null]");

  const char* json_duration_array = "[null, -1234567890123456789]";
  auto heap_duration_array_s =
      SliceArrayFromJSON(duration(TimeUnit::SECOND), json_duration_array);
  auto heap_duration_array_ns =
      SliceArrayFromJSON(duration(TimeUnit::NANO), json_duration_array);

  auto heap_timestamp_array_s = SliceArrayFromJSON(
      timestamp(TimeUnit::SECOND),
      R"([null, "1970-01-01 00:00:00", "1900-02-28 12:34:56", "3989-07-14 00:00:00"])");
  auto heap_timestamp_array_ms = SliceArrayFromJSON(
      timestamp(TimeUnit::MILLI),
      R"([null, "1900-02-28 12:34:56.123", "3989-07-14 00:00:00.789"])");
  auto heap_timestamp_array_us = SliceArrayFromJSON(
      timestamp(TimeUnit::MICRO),
      R"([null, "1900-02-28 12:34:56.654321", "3989-07-14 00:00:00.456789"])");
  auto heap_timestamp_array_ns = SliceArrayFromJSON(
      timestamp(TimeUnit::NANO), R"([null, "1900-02-28 12:34:56.987654321"])");

  auto heap_decimal128_array = SliceArrayFromJSON(
      decimal128(30, 6),
      R"([null, "-1234567890123456789.012345", "1234567890123456789.012345"])");
  auto heap_decimal256_array = SliceArrayFromJSON(
      decimal256(50, 6), R"([null, "-123456789012345678901234567890123456789.012345"])");
  auto heap_decimal128_array_sliced = heap_decimal128_array->Slice(1, 1);

  auto heap_fixed_size_binary_array =
      SliceArrayFromJSON(fixed_size_binary(3), "[null, \"abc\", \"\\u0000\\u001f\xff\"]");
  auto heap_fixed_size_binary_array_zero_width =
      SliceArrayFromJSON(fixed_size_binary(0), R"([null, ""])");
  auto heap_fixed_size_binary_array_sliced = heap_fixed_size_binary_array->Slice(1, 1);

  const char* json_binary_array = "[null, \"abcd\", \"\\u0000\\u001f\xff\"]";
  auto heap_binary_array = SliceArrayFromJSON(binary(), json_binary_array);
  auto heap_large_binary_array = SliceArrayFromJSON(large_binary(), json_binary_array);
  const char* json_string_array = "[null, \"héhé\", \"invalid \xff char\"]";
  auto heap_string_array = SliceArrayFromJSON(utf8(), json_string_array);
  auto heap_large_string_array = SliceArrayFromJSON(large_utf8(), json_string_array);
  auto heap_binary_array_sliced = heap_binary_array->Slice(1, 1);

  // ChunkedArray
  ArrayVector array_chunks(2);
  array_chunks[0] = *ArrayFromJSON(int32(), "[1, 2]");
  array_chunks[1] = *ArrayFromJSON(int32(), "[3, null, 4]");
  ChunkedArray chunked_array{array_chunks};

  // RecordBatch
  auto batch_schema = schema({field("ints", int32()), field("strs", utf8())});
  ArrayVector batch_columns{2};
  batch_columns[0] = *ArrayFromJSON(int32(), "[1, 2, 3]");
  batch_columns[1] = *ArrayFromJSON(utf8(), R"(["abc", null, "def"])");
  auto batch = RecordBatch::Make(batch_schema, /*num_rows=*/3, batch_columns);
  auto batch_with_metadata = batch->ReplaceSchemaMetadata(
      key_value_metadata({"key1", "key2", "key3"}, {"value1", "value2", "value3"}));

  // Table
  ChunkedArrayVector table_columns{2};
  ARROW_CHECK_OK(
      ChunkedArrayFromJSON(int32(), {"[1, 2, 3]", "[4, 5]"}, &table_columns[0]));
  ARROW_CHECK_OK(ChunkedArrayFromJSON(
      utf8(), {R"(["abc", null])", R"(["def"])", R"(["ghi", "jkl"])"},
      &table_columns[1]));
  auto table = Table::Make(batch_schema, table_columns);

  // Datum
  Datum empty_datum{};
  Datum scalar_datum{MakeNullScalar(boolean())};
  Datum array_datum{heap_int32_array};
  Datum chunked_array_datum{chunked_array};
  Datum batch_datum{batch};
  Datum table_datum{table};

#ifdef __clang__
  _Pragma("clang diagnostic pop");
#elif defined(__GNUC__)
  _Pragma("GCC diagnostic pop");
#endif

  // Hook into debugger
  ::arrow::internal::DebugTrap();
}

}  // namespace gdb
}  // namespace arrow
