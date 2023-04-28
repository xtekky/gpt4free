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

// Private header, not to be exported

#pragma once

#include <utility>

#include "arrow/scalar.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/visitor_generate.h"

namespace arrow {

#define SCALAR_VISIT_INLINE(TYPE_CLASS)                                              \
  case TYPE_CLASS##Type::type_id:                                                    \
    return visitor->Visit(internal::checked_cast<const TYPE_CLASS##Scalar&>(scalar), \
                          std::forward<ARGS>(args)...);

/// \brief Apply the visitors Visit() method specialized to the scalar type
///
/// \tparam VISITOR Visitor type that implements Visit() for all scalar types.
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `scalar` argument
/// \return Status
///
/// A visitor is a type that implements specialized logic for each Arrow type.
/// Example usage:
///
/// ```
/// class ExampleVisitor {
///   arrow::Status Visit(arrow::Int32Scalar scalar) { ... }
///   arrow::Status Visit(arrow::Int64Scalar scalar) { ... }
///   ...
/// }
/// ExampleVisitor visitor;
/// VisitScalarInline(some_scalar, &visitor);
/// ```
template <typename VISITOR, typename... ARGS>
inline Status VisitScalarInline(const Scalar& scalar, VISITOR* visitor, ARGS&&... args) {
  switch (scalar.type->id()) {
    ARROW_GENERATE_FOR_ALL_TYPES(SCALAR_VISIT_INLINE);
    default:
      break;
  }
  return Status::NotImplemented("Scalar visitor for type not implemented ",
                                scalar.type->ToString());
}

#undef SCALAR_VISIT_INLINE

}  // namespace arrow
