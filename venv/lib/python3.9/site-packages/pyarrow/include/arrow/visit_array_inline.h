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

#include "arrow/array.h"
#include "arrow/extension_type.h"
#include "arrow/visitor_generate.h"

namespace arrow {

#define ARRAY_VISIT_INLINE(TYPE_CLASS)                                                   \
  case TYPE_CLASS##Type::type_id:                                                        \
    return visitor->Visit(                                                               \
        internal::checked_cast<const typename TypeTraits<TYPE_CLASS##Type>::ArrayType&>( \
            array),                                                                      \
        std::forward<ARGS>(args)...);

/// \brief Apply the visitors Visit() method specialized to the array type
///
/// \tparam VISITOR Visitor type that implements Visit() for all array types.
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `arr` argument
/// \return Status
///
/// A visitor is a type that implements specialized logic for each Arrow type.
/// Example usage:
///
/// ```
/// class ExampleVisitor {
///   arrow::Status Visit(arrow::NumericArray<Int32Type> arr) { ... }
///   arrow::Status Visit(arrow::NumericArray<Int64Type> arr) { ... }
///   ...
/// }
/// ExampleVisitor visitor;
/// VisitArrayInline(some_array, &visitor);
/// ```
template <typename VISITOR, typename... ARGS>
inline Status VisitArrayInline(const Array& array, VISITOR* visitor, ARGS&&... args) {
  switch (array.type_id()) {
    ARROW_GENERATE_FOR_ALL_TYPES(ARRAY_VISIT_INLINE);
    default:
      break;
  }
  return Status::NotImplemented("Type not implemented");
}

#undef ARRAY_VISIT_INLINE

}  // namespace arrow
