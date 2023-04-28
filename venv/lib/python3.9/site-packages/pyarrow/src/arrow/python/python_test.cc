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

#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "platform.h"

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/table.h"
#include "arrow/util/decimal.h"
#include "arrow/util/logging.h"

#include "arrow/python/arrow_to_pandas.h"
#include "arrow/python/decimal.h"
#include "arrow/python/helpers.h"
#include "arrow/python/numpy_convert.h"
#include "arrow/python/numpy_interop.h"
#include "arrow/python/python_test.h"
#include "arrow/python/python_to_arrow.h"

#define ASSERT_EQ(x, y) { \
  auto&& _left = (x); \
  auto&& _right = (y); \
  if (_left != _right) { \
    return Status::Invalid("Expected equality between `", #x, "` and `", #y, \
                           "`, but ", arrow::py::testing::ToString(_left), \
                           " != ", arrow::py::testing::ToString(_right)); \
  } \
}

#define ASSERT_NE(x, y) { \
  auto&& _left = (x); \
  auto&& _right = (y); \
  if (_left == _right) { \
    return Status::Invalid("Expected inequality between `", #x, "` and `", #y, \
                           "`, but ", arrow::py::testing::ToString(_left), \
                           " == ", arrow::py::testing::ToString(_right)); \
  } \
}

#define ASSERT_FALSE(v) { \
  auto&& _v = (v); \
  if (!!_v) { \
    return Status::Invalid("Expected `", #v, "` to evaluate to false, but got ", \
                           arrow::py::testing::ToString(_v)); \
  } \
}

#define ASSERT_TRUE(v){ \
  auto&& _v = (v); \
  if (!_v) { \
    return Status::Invalid("Expected `", #v, "` to evaluate to true, but got ", \
                           arrow::py::testing::ToString(_v)); \
  } \
}

#define ASSERT_FALSE_MSG(v, msg) { \
  auto&& _v = (v); \
  if (!!_v) { \
    return Status::Invalid("Expected `", #v, "` to evaluate to false, but got ", \
                           arrow::py::testing::ToString(_v), ": ", msg); \
  } \
}

#define ASSERT_TRUE_MSG(v, msg) { \
  auto&& _v = (v); \
  if (!_v) { \
    return Status::Invalid("Expected `", #v, "` to evaluate to true, but got ", \
                           arrow::py::testing::ToString(_v), ": ", msg); \
  } \
}

#define ASSERT_OK(expr) { \
  for (::arrow::Status _st = ::arrow::internal::GenericToStatus((expr)); !_st.ok();) \
  return Status::Invalid("`", #expr, "` failed with ", _st.ToString()); \
}

#define ASSERT_RAISES(code, expr) { \
  for (::arrow::Status _st_expr = ::arrow::internal::GenericToStatus((expr)); \
       !_st_expr.Is##code();) \
  return Status::Invalid("Expected `", #expr, "` to fail with ", \
                         #code, ", but got ", _st_expr.ToString()); \
}

namespace arrow {

using internal::checked_cast;

namespace py {
namespace testing {

// ARROW-17938: Some standard libraries have ambiguous operator<<(nullptr_t),
// work around it using a custom printer function.

template <typename T>
std::string ToString(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

template <>
std::string ToString(const std::nullptr_t&) {
  return "nullptr";
}

namespace {

Status TestOwnedRefMoves() {
  std::vector<OwnedRef> vec;
  PyObject *u, *v;
  u = PyList_New(0);
  v = PyList_New(0);

  {
    OwnedRef ref(u);
    vec.push_back(std::move(ref));
    ASSERT_EQ(ref.obj(), nullptr);
  }
  vec.emplace_back(v);
  ASSERT_EQ(Py_REFCNT(u), 1);
  ASSERT_EQ(Py_REFCNT(v), 1);
  return Status::OK();
}

Status TestOwnedRefNoGILMoves() {
  PyAcquireGIL lock;
  lock.release();

  {
    std::vector<OwnedRef> vec;
    PyObject *u, *v;
    {
      lock.acquire();
      u = PyList_New(0);
      v = PyList_New(0);
      lock.release();
    }
    {
      OwnedRefNoGIL ref(u);
      vec.push_back(std::move(ref));
      ASSERT_EQ(ref.obj(), nullptr);
    }
    vec.emplace_back(v);
    ASSERT_EQ(Py_REFCNT(u), 1);
    ASSERT_EQ(Py_REFCNT(v), 1);
    return Status::OK();
  }
}

std::string FormatPythonException(const std::string& exc_class_name) {
  std::stringstream ss;
  ss << "Python exception: ";
  ss << exc_class_name;
  return ss.str();
}

Status TestCheckPyErrorStatus() {
  Status st;
  std::string expected_detail = "";

  auto check_error = [](Status& st, const char* expected_message = "some error",
                        std::string expected_detail = "") {
    st = CheckPyError();
    ASSERT_EQ(st.message(), expected_message);
    ASSERT_FALSE(PyErr_Occurred());
    if (expected_detail.size() > 0) {
      auto detail = st.detail();
      ASSERT_NE(detail, nullptr);
      ASSERT_EQ(detail->ToString(), expected_detail);
    }
    return Status::OK();
  };

  for (PyObject* exc_type : {PyExc_Exception, PyExc_SyntaxError}) {
    PyErr_SetString(exc_type, "some error");
    ASSERT_OK(check_error(st));
    ASSERT_TRUE(st.IsUnknownError());
  }

  PyErr_SetString(PyExc_TypeError, "some error");
  ASSERT_OK(check_error(st, "some error", FormatPythonException("TypeError")));
  ASSERT_TRUE(st.IsTypeError());

  PyErr_SetString(PyExc_ValueError, "some error");
  ASSERT_OK(check_error(st));
  ASSERT_TRUE(st.IsInvalid());

  PyErr_SetString(PyExc_KeyError, "some error");
  ASSERT_OK(check_error(st, "'some error'"));
  ASSERT_TRUE(st.IsKeyError());

  for (PyObject* exc_type : {PyExc_OSError, PyExc_IOError}) {
    PyErr_SetString(exc_type, "some error");
    ASSERT_OK(check_error(st));
    ASSERT_TRUE(st.IsIOError());
  }

  PyErr_SetString(PyExc_NotImplementedError, "some error");
  ASSERT_OK(check_error(st, "some error", FormatPythonException("NotImplementedError")));
  ASSERT_TRUE(st.IsNotImplemented());

  // No override if a specific status code is given
  PyErr_SetString(PyExc_TypeError, "some error");
  st = CheckPyError(StatusCode::SerializationError);
  ASSERT_TRUE(st.IsSerializationError());
  ASSERT_EQ(st.message(), "some error");
  ASSERT_FALSE(PyErr_Occurred());

  return Status::OK();
}

Status TestCheckPyErrorStatusNoGIL() {
  PyAcquireGIL lock;
  {
    Status st;
    PyErr_SetString(PyExc_ZeroDivisionError, "zzzt");
    st = ConvertPyError();
    ASSERT_FALSE(PyErr_Occurred());
    lock.release();
    ASSERT_TRUE(st.IsUnknownError());
    ASSERT_EQ(st.message(), "zzzt");
    ASSERT_EQ(st.detail()->ToString(), FormatPythonException("ZeroDivisionError"));
    return Status::OK();
  }
}

Status TestRestorePyErrorBasics() {
  PyErr_SetString(PyExc_ZeroDivisionError, "zzzt");
  auto st = ConvertPyError();
  ASSERT_FALSE(PyErr_Occurred());
  ASSERT_TRUE(st.IsUnknownError());
  ASSERT_EQ(st.message(), "zzzt");
  ASSERT_EQ(st.detail()->ToString(), FormatPythonException("ZeroDivisionError"));

  RestorePyError(st);
  ASSERT_TRUE(PyErr_Occurred());
  PyObject* exc_type;
  PyObject* exc_value;
  PyObject* exc_traceback;
  PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
  ASSERT_TRUE(PyErr_GivenExceptionMatches(exc_type, PyExc_ZeroDivisionError));
  std::string py_message;
  ASSERT_OK(internal::PyObject_StdStringStr(exc_value, &py_message));
  ASSERT_EQ(py_message, "zzzt");

  return Status::OK();
}

Status TestPyBufferInvalidInputObject() {
  std::shared_ptr<Buffer> res;
  PyObject* input = Py_None;
  auto old_refcnt = Py_REFCNT(input);
  {
    Status st = PyBuffer::FromPyObject(input).status();
    ASSERT_TRUE_MSG(IsPyError(st), st.ToString());
    ASSERT_FALSE(PyErr_Occurred());
  }
  ASSERT_EQ(old_refcnt, Py_REFCNT(input));
  return Status::OK();
}

// Because of how it is declared, the Numpy C API instance initialized
// within libarrow_python.dll may not be visible in this test under Windows
// ("unresolved external symbol arrow_ARRAY_API referenced").
#ifndef _WIN32
Status TestPyBufferNumpyArray() {
  npy_intp dims[1] = {10};

  OwnedRef arr_ref(PyArray_SimpleNew(1, dims, NPY_FLOAT));
  PyObject* arr = arr_ref.obj();
  ASSERT_NE(arr, nullptr);
  auto old_refcnt = Py_REFCNT(arr);
  auto buf = std::move(PyBuffer::FromPyObject(arr)).ValueOrDie();

  ASSERT_TRUE(buf->is_cpu());
  ASSERT_EQ(buf->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
  ASSERT_TRUE(buf->is_mutable());
  ASSERT_EQ(buf->mutable_data(), buf->data());
  ASSERT_EQ(old_refcnt + 1, Py_REFCNT(arr));
  buf.reset();
  ASSERT_EQ(old_refcnt, Py_REFCNT(arr));

  // Read-only
  PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(arr), NPY_ARRAY_WRITEABLE);
  buf = std::move(PyBuffer::FromPyObject(arr)).ValueOrDie();
  ASSERT_TRUE(buf->is_cpu());
  ASSERT_EQ(buf->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
  ASSERT_FALSE(buf->is_mutable());
  ASSERT_EQ(old_refcnt + 1, Py_REFCNT(arr));
  buf.reset();
  ASSERT_EQ(old_refcnt, Py_REFCNT(arr));

  return Status::OK();
}

Status TestNumPyBufferNumpyArray() {
  npy_intp dims[1] = {10};

  OwnedRef arr_ref(PyArray_SimpleNew(1, dims, NPY_FLOAT));
  PyObject* arr = arr_ref.obj();
  ASSERT_NE(arr, nullptr);
  auto old_refcnt = Py_REFCNT(arr);

  auto buf = std::make_shared<NumPyBuffer>(arr);
  ASSERT_TRUE(buf->is_cpu());
  ASSERT_EQ(buf->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
  ASSERT_TRUE(buf->is_mutable());
  ASSERT_EQ(buf->mutable_data(), buf->data());
  ASSERT_EQ(old_refcnt + 1, Py_REFCNT(arr));
  buf.reset();
  ASSERT_EQ(old_refcnt, Py_REFCNT(arr));

  // Read-only
  PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(arr), NPY_ARRAY_WRITEABLE);
  buf = std::make_shared<NumPyBuffer>(arr);
  ASSERT_TRUE(buf->is_cpu());
  ASSERT_EQ(buf->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
  ASSERT_FALSE(buf->is_mutable());
  ASSERT_EQ(old_refcnt + 1, Py_REFCNT(arr));
  buf.reset();
  ASSERT_EQ(old_refcnt, Py_REFCNT(arr));

  return Status::OK();
}
#endif

Status TestPythonDecimalToString(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("-39402950693754869342983");
  PyObject* python_object = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  ASSERT_NE(python_object, nullptr);

  std::string string_result;
  ASSERT_OK(internal::PythonDecimalToString(python_object, &string_result));

  return Status::OK();
}

Status TestInferPrecisionAndScale(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("-394029506937548693.42983");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));

  const auto expected_precision =
      static_cast<int32_t>(decimal_string.size() - 2);  // 1 for -, 1 for .
  const int32_t expected_scale = 5;

  ASSERT_EQ(expected_precision, metadata.precision());
  ASSERT_EQ(expected_scale, metadata.scale());

  return Status::OK();
}

Status TestInferPrecisionAndNegativeScale(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("-3.94042983E+10");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));

  const auto expected_precision = 11;
  const int32_t expected_scale = 0;

  ASSERT_EQ(expected_precision, metadata.precision());
  ASSERT_EQ(expected_scale, metadata.scale());

  return Status::OK();
}

Status TestInferAllLeadingZeros(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("0.001");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));
  ASSERT_EQ(3, metadata.precision());
  ASSERT_EQ(3, metadata.scale());

  return Status::OK();
}

Status TestInferAllLeadingZerosExponentialNotationPositive(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("0.01E5");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));
  ASSERT_EQ(4, metadata.precision());
  ASSERT_EQ(0, metadata.scale());

  return Status::OK();
}

Status TestInferAllLeadingZerosExponentialNotationNegative(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("0.01E3");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));
  ASSERT_EQ(2, metadata.precision());
  ASSERT_EQ(0, metadata.scale());

  return Status::OK();
}

Status TestObjectBlockWriteFails(){
  StringBuilder builder;
  const char value[] = {'\xf1', '\0'};

  for (int i = 0; i < 1000; ++i) {
    ASSERT_OK(builder.Append(value, static_cast<int32_t>(strlen(value))));
  }

  std::shared_ptr<Array> arr;
  ASSERT_OK(builder.Finish(&arr));

  auto f1 = field("f1", utf8());
  auto f2 = field("f2", utf8());
  auto f3 = field("f3", utf8());
  std::vector<std::shared_ptr<Field>> fields = {f1, f2, f3};
  std::vector<std::shared_ptr<Array>> cols = {arr, arr, arr};

  auto schema = ::arrow::schema(fields);
  auto table = Table::Make(schema, cols);

  Status st;
  Py_BEGIN_ALLOW_THREADS;
  PyObject* out;
  PandasOptions options;
  options.use_threads = true;
  st = ConvertTableToPandas(options, table, &out);
  Py_END_ALLOW_THREADS;
  ASSERT_RAISES(UnknownError, st);

  return Status::OK();
}

Status TestMixedTypeFails(){
  OwnedRef list_ref(PyList_New(3));
  PyObject* list = list_ref.obj();

  ASSERT_NE(list, nullptr);

  PyObject* str = PyUnicode_FromString("abc");
  ASSERT_NE(str, nullptr);

  PyObject* integer = PyLong_FromLong(1234L);
  ASSERT_NE(integer, nullptr);

  PyObject* doub = PyFloat_FromDouble(123.0234);
  ASSERT_NE(doub, nullptr);

  // This steals a reference to each object, so we don't need to decref them later
  // just the list
  ASSERT_EQ(PyList_SetItem(list, 0, str), 0);
  ASSERT_EQ(PyList_SetItem(list, 1, integer), 0);
  ASSERT_EQ(PyList_SetItem(list, 2, doub), 0);

  ASSERT_RAISES(TypeError, ConvertPySequence(list, nullptr, {}));

  return Status::OK();
}

template <typename DecimalValue>
Status DecimalTestFromPythonDecimalRescale(std::shared_ptr<DataType> type,
                                         PyObject* python_decimal,
                                         std::optional<int> expected) {
  DecimalValue value;
  const auto& decimal_type = checked_cast<const DecimalType&>(*type);

  if (expected.has_value()) {
    ASSERT_OK(
        internal::DecimalFromPythonDecimal(python_decimal, decimal_type, &value));
    ASSERT_EQ(expected.value(), value);

    ASSERT_OK(internal::DecimalFromPyObject(python_decimal, decimal_type, &value));
    ASSERT_EQ(expected.value(), value);
  } else {
    ASSERT_RAISES(Invalid,
                  internal::DecimalFromPythonDecimal(python_decimal,
                                                     decimal_type, &value));
    ASSERT_RAISES(Invalid,
                  internal::DecimalFromPyObject(python_decimal,
                                                decimal_type, &value));
  }
  return Status::OK();
}

Status TestFromPythonDecimalRescaleNotTruncateable(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("1.001");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  // We fail when truncating values that would lose data if cast to a decimal type with
  // lower scale
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal128>(::arrow::decimal128(10, 2),
                                                            python_decimal, {}));
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal256>(::arrow::decimal256(10, 2),
                                                            python_decimal, {}));

  return Status::OK();
}

Status TestFromPythonDecimalRescaleTruncateable(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("1.000");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  // We allow truncation of values that do not lose precision when dividing by 10 * the
  // difference between the scales, e.g., 1.000 -> 1.00
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal128>(
      ::arrow::decimal128(10, 2), python_decimal, 100));
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal256>(
      ::arrow::decimal256(10, 2), python_decimal, 100));

  return Status::OK();
}

Status TestFromPythonNegativeDecimalRescale(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("-1.000");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal128>(
      ::arrow::decimal128(10, 9), python_decimal, -1000000000));
  ASSERT_OK(DecimalTestFromPythonDecimalRescale<Decimal256>(
      ::arrow::decimal256(10, 9), python_decimal, -1000000000));

  return Status::OK();
}

Status TestDecimal128FromPythonInteger(){
  Decimal128 value;
  OwnedRef python_long(PyLong_FromLong(42));
  auto type = ::arrow::decimal128(10, 2);
  const auto& decimal_type = checked_cast<const DecimalType&>(*type);
  ASSERT_OK(internal::DecimalFromPyObject(python_long.obj(), decimal_type, &value));
  ASSERT_EQ(4200, value);
  return Status::OK();
}

Status TestDecimal256FromPythonInteger(){
  Decimal256 value;
  OwnedRef python_long(PyLong_FromLong(42));
  auto type = ::arrow::decimal256(10, 2);
  const auto& decimal_type = checked_cast<const DecimalType&>(*type);
  ASSERT_OK(internal::DecimalFromPyObject(python_long.obj(), decimal_type, &value));
  ASSERT_EQ(4200, value);
  return Status::OK();
}

Status TestDecimal128OverflowFails(){
  Decimal128 value;
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("9999999999999999999999999999999999999.9");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));
  ASSERT_EQ(38, metadata.precision());
  ASSERT_EQ(1, metadata.scale());

  auto type = ::arrow::decimal(38, 38);
  const auto& decimal_type = checked_cast<const DecimalType&>(*type);
  ASSERT_RAISES(Invalid,
                internal::DecimalFromPythonDecimal(python_decimal,
                                                   decimal_type, &value));
  return Status::OK();
}

Status TestDecimal256OverflowFails(){
  Decimal256 value;
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("999999999999999999999999999999999999999999999999999999999999999999999999999.9");
  PyObject* python_decimal = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(python_decimal));
  ASSERT_EQ(76, metadata.precision());
  ASSERT_EQ(1, metadata.scale());

  auto type = ::arrow::decimal(76, 76);
  const auto& decimal_type = checked_cast<const DecimalType&>(*type);
  ASSERT_RAISES(Invalid,
                internal::DecimalFromPythonDecimal(python_decimal,
                                                   decimal_type, &value));
  return Status::OK();
}

Status TestNoneAndNaN(){
  OwnedRef list_ref(PyList_New(4));
  PyObject* list = list_ref.obj();

  ASSERT_NE(list, nullptr);

  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;
  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));
  PyObject* constructor = decimal_constructor_.obj();
  PyObject* decimal_value = internal::DecimalFromString(constructor, "1.234");
  ASSERT_NE(decimal_value, nullptr);

  Py_INCREF(Py_None);
  PyObject* missing_value1 = Py_None;
  ASSERT_NE(missing_value1, nullptr);

  PyObject* missing_value2 = PyFloat_FromDouble(NPY_NAN);
  ASSERT_NE(missing_value2, nullptr);

  PyObject* missing_value3 = internal::DecimalFromString(constructor, "nan");
  ASSERT_NE(missing_value3, nullptr);

  // This steals a reference to each object, so we don't need to decref them later,
  // just the list
  ASSERT_EQ(0, PyList_SetItem(list, 0, decimal_value));
  ASSERT_EQ(0, PyList_SetItem(list, 1, missing_value1));
  ASSERT_EQ(0, PyList_SetItem(list, 2, missing_value2));
  ASSERT_EQ(0, PyList_SetItem(list, 3, missing_value3));

  PyConversionOptions options;
  ASSERT_RAISES(TypeError,
                ConvertPySequence(list, nullptr, options));

  options.from_pandas = true;
  auto chunked = std::move(ConvertPySequence(list, nullptr, options)).ValueOrDie();
  ASSERT_EQ(chunked->num_chunks(), 1);

  auto arr = chunked->chunk(0);
  ASSERT_TRUE(arr->IsValid(0));
  ASSERT_TRUE(arr->IsNull(1));
  ASSERT_TRUE(arr->IsNull(2));
  ASSERT_TRUE(arr->IsNull(3));

  return Status::OK();
}

Status TestMixedPrecisionAndScale(){
  std::vector<std::string> strings{{"0.001", "1.01E5", "1.01E5"}};

  OwnedRef list_ref(PyList_New(static_cast<Py_ssize_t>(strings.size())));
  PyObject* list = list_ref.obj();

  ASSERT_NE(list, nullptr);

  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;
  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));
  // PyList_SetItem steals a reference to the item so we don't decref it later
  PyObject* decimal_constructor = decimal_constructor_.obj();
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(strings.size()); ++i) {
    const int result = PyList_SetItem(
        list, i, internal::DecimalFromString(decimal_constructor, strings.at(i)));
    ASSERT_EQ(0, result);
  }

  auto arr = std::move(ConvertPySequence(list, nullptr, {})).ValueOrDie();
  const auto& type = checked_cast<const DecimalType&>(*arr->type());

  int32_t expected_precision = 9;
  int32_t expected_scale = 3;
  ASSERT_EQ(expected_precision, type.precision());
  ASSERT_EQ(expected_scale, type.scale());

  return Status::OK();
}

Status TestMixedPrecisionAndScaleSequenceConvert(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string_1("0.01");
  PyObject* value1 = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string_1);
  ASSERT_NE(value1, nullptr);

  std::string decimal_string_2("0.001");
  PyObject* value2 = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string_2);
  ASSERT_NE(value2, nullptr);

  OwnedRef list_ref(PyList_New(2));
  PyObject* list = list_ref.obj();

  // This steals a reference to each object, so we don't need to decref them later
  // just the list
  ASSERT_EQ(PyList_SetItem(list, 0, value1), 0);
  ASSERT_EQ(PyList_SetItem(list, 1, value2), 0);

  auto arr = std::move(ConvertPySequence(list, nullptr, {})).ValueOrDie();
  const auto& type = checked_cast<const Decimal128Type&>(*arr->type());
  ASSERT_EQ(3, type.precision());
  ASSERT_EQ(3, type.scale());

  return Status::OK();
}

Status TestSimpleInference(){
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;

  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));

  std::string decimal_string("0.01");
  PyObject* value = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);
  ASSERT_NE(value, nullptr);
  internal::DecimalMetadata metadata;
  ASSERT_OK(metadata.Update(value));
  ASSERT_EQ(2, metadata.precision());
  ASSERT_EQ(2, metadata.scale());

  return Status::OK();
}

Status TestUpdateWithNaN(){
  internal::DecimalMetadata metadata;
  OwnedRef decimal_constructor_;
  OwnedRef decimal_module;
  RETURN_NOT_OK(internal::ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(internal::ImportFromModule(decimal_module.obj(), "Decimal",
                                           &decimal_constructor_));
  std::string decimal_string("nan");
  PyObject* nan_value = internal::DecimalFromString(decimal_constructor_.obj(),
                                                        decimal_string);

  ASSERT_OK(metadata.Update(nan_value));
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), metadata.precision());
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), metadata.scale());

  return Status::OK();
}

}  // namespace

std::vector<TestCase> GetCppTestCases() {
  return {
    {"test_owned_ref_moves", TestOwnedRefMoves},
    {"test_owned_ref_nogil_moves", TestOwnedRefNoGILMoves},
    {"test_check_pyerror_status", TestCheckPyErrorStatus},
    {"test_check_pyerror_status_nogil", TestCheckPyErrorStatusNoGIL},
    {"test_restore_pyerror_basics", TestRestorePyErrorBasics},
    {"test_pybuffer_invalid_input_object", TestPyBufferInvalidInputObject},
#ifndef _WIN32
    {"test_pybuffer_numpy_array", TestPyBufferNumpyArray},
    {"test_numpybuffer_numpy_array", TestNumPyBufferNumpyArray},
#endif
    {"test_python_decimal_to_string", TestPythonDecimalToString},
    {"test_infer_precision_and_scale", TestInferPrecisionAndScale},
    {"test_infer_precision_and_negative_scale", TestInferPrecisionAndNegativeScale},
    {"test_infer_all_leading_zeros", TestInferAllLeadingZeros},
    {"test_infer_all_leading_zeros_exponential_notation_positive",
     TestInferAllLeadingZerosExponentialNotationPositive},
    {"test_infer_all_leading_zeros_exponential_notation_negative",
     TestInferAllLeadingZerosExponentialNotationNegative},
    {"test_object_block_write_fails", TestObjectBlockWriteFails},
    {"test_mixed_type_fails", TestMixedTypeFails},
    {"test_from_python_decimal_rescale_not_truncateable",
     TestFromPythonDecimalRescaleNotTruncateable},
    {"test_from_python_decimal_rescale_truncateable",
     TestFromPythonDecimalRescaleTruncateable},
    {"test_from_python_negative_decimal_rescale", TestFromPythonNegativeDecimalRescale},
    {"test_decimal128_from_python_integer", TestDecimal128FromPythonInteger},
    {"test_decimal256_from_python_integer", TestDecimal256FromPythonInteger},
    {"test_decimal128_overflow_fails", TestDecimal128OverflowFails},
    {"test_decimal256_overflow_fails", TestDecimal256OverflowFails},
    {"test_none_and_nan", TestNoneAndNaN},
    {"test_mixed_precision_and_scale", TestMixedPrecisionAndScale},
    {"test_mixed_precision_and_scale_sequence_convert",
     TestMixedPrecisionAndScaleSequenceConvert},
    {"test_simple_inference", TestSimpleInference},
    {"test_update_with_nan", TestUpdateWithNaN},
  };
}

}  // namespace testing
}  // namespace py
}  // namespace arrow
