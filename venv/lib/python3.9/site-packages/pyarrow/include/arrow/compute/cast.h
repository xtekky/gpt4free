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

#include <memory>
#include <string>
#include <vector>

#include "arrow/compute/function.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;

namespace compute {

class ExecContext;

/// \addtogroup compute-concrete-options
/// @{

class ARROW_EXPORT CastOptions : public FunctionOptions {
 public:
  explicit CastOptions(bool safe = true);

  static constexpr char const kTypeName[] = "CastOptions";
  static CastOptions Safe(TypeHolder to_type = {}) {
    CastOptions safe(true);
    safe.to_type = std::move(to_type);
    return safe;
  }

  static CastOptions Unsafe(TypeHolder to_type = {}) {
    CastOptions unsafe(false);
    unsafe.to_type = std::move(to_type);
    return unsafe;
  }

  // Type being casted to. May be passed separate to eager function
  // compute::Cast
  TypeHolder to_type;

  bool allow_int_overflow;
  bool allow_time_truncate;
  bool allow_time_overflow;
  bool allow_decimal_truncate;
  bool allow_float_truncate;
  // Indicate if conversions from Binary/FixedSizeBinary to string must
  // validate the utf8 payload.
  bool allow_invalid_utf8;
};

/// @}

/// \brief Return true if a cast function is defined
ARROW_EXPORT
bool CanCast(const DataType& from_type, const DataType& to_type);

// ----------------------------------------------------------------------
// Convenience invocation APIs for a number of kernels

/// \brief Cast from one array type to another
/// \param[in] value array to cast
/// \param[in] to_type type to cast to
/// \param[in] options casting options
/// \param[in] ctx the function execution context, optional
/// \return the resulting array
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<std::shared_ptr<Array>> Cast(const Array& value, const TypeHolder& to_type,
                                    const CastOptions& options = CastOptions::Safe(),
                                    ExecContext* ctx = NULLPTR);

/// \brief Cast from one array type to another
/// \param[in] value array to cast
/// \param[in] options casting options. The "to_type" field must be populated
/// \param[in] ctx the function execution context, optional
/// \return the resulting array
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Cast(const Datum& value, const CastOptions& options,
                   ExecContext* ctx = NULLPTR);

/// \brief Cast from one value to another
/// \param[in] value datum to cast
/// \param[in] to_type type to cast to
/// \param[in] options casting options
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Cast(const Datum& value, const TypeHolder& to_type,
                   const CastOptions& options = CastOptions::Safe(),
                   ExecContext* ctx = NULLPTR);

}  // namespace compute
}  // namespace arrow
