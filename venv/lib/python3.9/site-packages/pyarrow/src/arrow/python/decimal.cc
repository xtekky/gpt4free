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

#include <algorithm>
#include <limits>

#include "arrow/type_fwd.h"
#include "arrow/util/decimal.h"
#include "arrow/util/logging.h"
#include "arrow/python/common.h"
#include "arrow/python/decimal.h"
#include "arrow/python/helpers.h"

namespace arrow {
namespace py {
namespace internal {

Status ImportDecimalType(OwnedRef* decimal_type) {
  OwnedRef decimal_module;
  RETURN_NOT_OK(ImportModule("decimal", &decimal_module));
  RETURN_NOT_OK(ImportFromModule(decimal_module.obj(), "Decimal", decimal_type));
  return Status::OK();
}

Status PythonDecimalToString(PyObject* python_decimal, std::string* out) {
  // Call Python's str(decimal_object)
  return PyObject_StdStringStr(python_decimal, out);
}

// \brief Infer the precision and scale of a Python decimal.Decimal instance
// \param python_decimal[in] An instance of decimal.Decimal
// \param precision[out] The value of the inferred precision
// \param scale[out] The value of the inferred scale
// \return The status of the operation
static Status InferDecimalPrecisionAndScale(PyObject* python_decimal, int32_t* precision,
                                            int32_t* scale) {
  DCHECK_NE(python_decimal, NULLPTR);
  DCHECK_NE(precision, NULLPTR);
  DCHECK_NE(scale, NULLPTR);

  // TODO(phillipc): Make sure we perform PyDecimal_Check(python_decimal) as a DCHECK
  OwnedRef as_tuple(PyObject_CallMethod(python_decimal, const_cast<char*>("as_tuple"),
                                        const_cast<char*>("")));
  RETURN_IF_PYERROR();
  DCHECK(PyTuple_Check(as_tuple.obj()));

  OwnedRef digits(PyObject_GetAttrString(as_tuple.obj(), "digits"));
  RETURN_IF_PYERROR();
  DCHECK(PyTuple_Check(digits.obj()));

  const auto num_digits = static_cast<int32_t>(PyTuple_Size(digits.obj()));
  RETURN_IF_PYERROR();

  OwnedRef py_exponent(PyObject_GetAttrString(as_tuple.obj(), "exponent"));
  RETURN_IF_PYERROR();
  DCHECK(IsPyInteger(py_exponent.obj()));

  const auto exponent = static_cast<int32_t>(PyLong_AsLong(py_exponent.obj()));
  RETURN_IF_PYERROR();

  if (exponent < 0) {
    // If exponent > num_digits, we have a number with leading zeros
    // such as 0.01234.  Ensure we have enough precision for leading zeros
    // (which are not included in num_digits).
    *precision = std::max(num_digits, -exponent);
    *scale = -exponent;
  } else {
    // Trailing zeros are not included in num_digits, need to add to precision.
    // Note we don't generate negative scales as they are poorly supported
    // in non-Arrow systems.
    *precision = num_digits + exponent;
    *scale = 0;
  }
  return Status::OK();
}

PyObject* DecimalFromString(PyObject* decimal_constructor,
                            const std::string& decimal_string) {
  DCHECK_NE(decimal_constructor, nullptr);

  auto string_size = decimal_string.size();
  DCHECK_GT(string_size, 0);

  auto string_bytes = decimal_string.c_str();
  DCHECK_NE(string_bytes, nullptr);

  return PyObject_CallFunction(decimal_constructor, const_cast<char*>("s#"), string_bytes,
                               static_cast<Py_ssize_t>(string_size));
}

namespace {

template <typename ArrowDecimal>
Status DecimalFromStdString(const std::string& decimal_string,
                            const DecimalType& arrow_type, ArrowDecimal* out) {
  int32_t inferred_precision;
  int32_t inferred_scale;

  RETURN_NOT_OK(ArrowDecimal::FromString(decimal_string, out, &inferred_precision,
                                         &inferred_scale));

  const int32_t precision = arrow_type.precision();
  const int32_t scale = arrow_type.scale();

  if (scale != inferred_scale) {
    DCHECK_NE(out, NULLPTR);
    ARROW_ASSIGN_OR_RAISE(*out, out->Rescale(inferred_scale, scale));
  }

  auto inferred_scale_delta = inferred_scale - scale;
  if (ARROW_PREDICT_FALSE((inferred_precision - inferred_scale_delta) > precision)) {
    return Status::Invalid(
        "Decimal type with precision ", inferred_precision,
        " does not fit into precision inferred from first array element: ", precision);
  }

  return Status::OK();
}

template <typename ArrowDecimal>
Status InternalDecimalFromPythonDecimal(PyObject* python_decimal,
                                        const DecimalType& arrow_type,
                                        ArrowDecimal* out) {
  DCHECK_NE(python_decimal, NULLPTR);
  DCHECK_NE(out, NULLPTR);

  std::string string;
  RETURN_NOT_OK(PythonDecimalToString(python_decimal, &string));
  return DecimalFromStdString(string, arrow_type, out);
}

template <typename ArrowDecimal>
Status InternalDecimalFromPyObject(PyObject* obj, const DecimalType& arrow_type,
                                   ArrowDecimal* out) {
  DCHECK_NE(obj, NULLPTR);
  DCHECK_NE(out, NULLPTR);

  if (IsPyInteger(obj)) {
    // TODO: add a fast path for small-ish ints
    std::string string;
    RETURN_NOT_OK(PyObject_StdStringStr(obj, &string));
    return DecimalFromStdString(string, arrow_type, out);
  } else if (PyDecimal_Check(obj)) {
    return InternalDecimalFromPythonDecimal<ArrowDecimal>(obj, arrow_type, out);
  } else {
    return Status::TypeError("int or Decimal object expected, got ",
                             Py_TYPE(obj)->tp_name);
  }
}

}  // namespace

Status DecimalFromPythonDecimal(PyObject* python_decimal, const DecimalType& arrow_type,
                                Decimal128* out) {
  return InternalDecimalFromPythonDecimal(python_decimal, arrow_type, out);
}

Status DecimalFromPyObject(PyObject* obj, const DecimalType& arrow_type,
                           Decimal128* out) {
  return InternalDecimalFromPyObject(obj, arrow_type, out);
}

Status DecimalFromPythonDecimal(PyObject* python_decimal, const DecimalType& arrow_type,
                                Decimal256* out) {
  return InternalDecimalFromPythonDecimal(python_decimal, arrow_type, out);
}

Status DecimalFromPyObject(PyObject* obj, const DecimalType& arrow_type,
                           Decimal256* out) {
  return InternalDecimalFromPyObject(obj, arrow_type, out);
}

bool PyDecimal_Check(PyObject* obj) {
  static OwnedRef decimal_type;
  if (!decimal_type.obj()) {
    ARROW_CHECK_OK(ImportDecimalType(&decimal_type));
    DCHECK(PyType_Check(decimal_type.obj()));
  }
  // PyObject_IsInstance() is slower as it has to check for virtual subclasses
  const int result =
      PyType_IsSubtype(Py_TYPE(obj), reinterpret_cast<PyTypeObject*>(decimal_type.obj()));
  ARROW_CHECK_NE(result, -1) << " error during PyType_IsSubtype check";
  return result == 1;
}

bool PyDecimal_ISNAN(PyObject* obj) {
  DCHECK(PyDecimal_Check(obj)) << "obj is not an instance of decimal.Decimal";
  OwnedRef is_nan(
      PyObject_CallMethod(obj, const_cast<char*>("is_nan"), const_cast<char*>("")));
  return PyObject_IsTrue(is_nan.obj()) == 1;
}

DecimalMetadata::DecimalMetadata()
    : DecimalMetadata(std::numeric_limits<int32_t>::min(),
                      std::numeric_limits<int32_t>::min()) {}

DecimalMetadata::DecimalMetadata(int32_t precision, int32_t scale)
    : precision_(precision), scale_(scale) {}

Status DecimalMetadata::Update(int32_t suggested_precision, int32_t suggested_scale) {
  const int32_t current_scale = scale_;
  scale_ = std::max(current_scale, suggested_scale);

  const int32_t current_precision = precision_;

  if (current_precision == std::numeric_limits<int32_t>::min()) {
    precision_ = suggested_precision;
  } else {
    auto num_digits = std::max(current_precision - current_scale,
                               suggested_precision - suggested_scale);
    precision_ = std::max(num_digits + scale_, current_precision);
  }

  return Status::OK();
}

Status DecimalMetadata::Update(PyObject* object) {
  bool is_decimal = PyDecimal_Check(object);

  if (ARROW_PREDICT_FALSE(!is_decimal || PyDecimal_ISNAN(object))) {
    return Status::OK();
  }

  int32_t precision = 0;
  int32_t scale = 0;
  RETURN_NOT_OK(InferDecimalPrecisionAndScale(object, &precision, &scale));
  return Update(precision, scale);
}

}  // namespace internal
}  // namespace py
}  // namespace arrow
