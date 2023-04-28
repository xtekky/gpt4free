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

#pragma once

#include <string>

#include "arrow/python/visibility.h"
#include "arrow/type.h"

namespace arrow {

class Decimal128;
class Decimal256;

namespace py {

class OwnedRef;

//
// Python Decimal support
//

namespace internal {

// \brief Import the Python Decimal type
ARROW_PYTHON_EXPORT
Status ImportDecimalType(OwnedRef* decimal_type);

// \brief Convert a Python Decimal object to a C++ string
// \param[in] python_decimal A Python decimal.Decimal instance
// \param[out] The string representation of the Python Decimal instance
// \return The status of the operation
ARROW_PYTHON_EXPORT
Status PythonDecimalToString(PyObject* python_decimal, std::string* out);

// \brief Convert a C++ std::string to a Python Decimal instance
// \param[in] decimal_constructor The decimal type object
// \param[in] decimal_string A decimal string
// \return An instance of decimal.Decimal
ARROW_PYTHON_EXPORT
PyObject* DecimalFromString(PyObject* decimal_constructor,
                            const std::string& decimal_string);

// \brief Convert a Python decimal to an Arrow Decimal128 object
// \param[in] python_decimal A Python decimal.Decimal instance
// \param[in] arrow_type An instance of arrow::DecimalType
// \param[out] out A pointer to a Decimal128
// \return The status of the operation
ARROW_PYTHON_EXPORT
Status DecimalFromPythonDecimal(PyObject* python_decimal, const DecimalType& arrow_type,
                                Decimal128* out);

// \brief Convert a Python object to an Arrow Decimal128 object
// \param[in] python_decimal A Python int or decimal.Decimal instance
// \param[in] arrow_type An instance of arrow::DecimalType
// \param[out] out A pointer to a Decimal128
// \return The status of the operation
ARROW_PYTHON_EXPORT
Status DecimalFromPyObject(PyObject* obj, const DecimalType& arrow_type, Decimal128* out);

// \brief Convert a Python decimal to an Arrow Decimal256 object
// \param[in] python_decimal A Python decimal.Decimal instance
// \param[in] arrow_type An instance of arrow::DecimalType
// \param[out] out A pointer to a Decimal256
// \return The status of the operation
ARROW_PYTHON_EXPORT
Status DecimalFromPythonDecimal(PyObject* python_decimal, const DecimalType& arrow_type,
                                Decimal256* out);

// \brief Convert a Python object to an Arrow Decimal256 object
// \param[in] python_decimal A Python int or decimal.Decimal instance
// \param[in] arrow_type An instance of arrow::DecimalType
// \param[out] out A pointer to a Decimal256
// \return The status of the operation
ARROW_PYTHON_EXPORT
Status DecimalFromPyObject(PyObject* obj, const DecimalType& arrow_type, Decimal256* out);

// \brief Check whether obj is an instance of Decimal
ARROW_PYTHON_EXPORT
bool PyDecimal_Check(PyObject* obj);

// \brief Check whether obj is nan. This function will abort the program if the argument
// is not a Decimal instance
ARROW_PYTHON_EXPORT
bool PyDecimal_ISNAN(PyObject* obj);

// \brief Helper class to track and update the precision and scale of a decimal
class ARROW_PYTHON_EXPORT DecimalMetadata {
 public:
  DecimalMetadata();
  DecimalMetadata(int32_t precision, int32_t scale);

  // \brief Adjust the precision and scale of a decimal type given a new precision and a
  // new scale \param[in] suggested_precision A candidate precision \param[in]
  // suggested_scale A candidate scale \return The status of the operation
  Status Update(int32_t suggested_precision, int32_t suggested_scale);

  // \brief A convenient interface for updating the precision and scale based on a Python
  // Decimal object \param object A Python Decimal object \return The status of the
  // operation
  Status Update(PyObject* object);

  int32_t precision() const { return precision_; }
  int32_t scale() const { return scale_; }

 private:
  int32_t precision_;
  int32_t scale_;
};

}  // namespace internal
}  // namespace py
}  // namespace arrow
