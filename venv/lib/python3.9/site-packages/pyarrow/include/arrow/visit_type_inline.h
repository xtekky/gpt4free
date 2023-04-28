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

#include "arrow/extension_type.h"
#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/visitor_generate.h"

namespace arrow {

#define TYPE_VISIT_INLINE(TYPE_CLASS)                                            \
  case TYPE_CLASS##Type::type_id:                                                \
    return visitor->Visit(internal::checked_cast<const TYPE_CLASS##Type&>(type), \
                          std::forward<ARGS>(args)...);

/// \brief Calls `visitor` with the corresponding concrete type class
///
/// \tparam VISITOR Visitor type that implements Visit() for all Arrow types.
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `type` argument
/// \return Status
///
/// A visitor is a type that implements specialized logic for each Arrow type.
/// Example usage:
///
/// ```
/// class ExampleVisitor {
///   arrow::Status Visit(const arrow::Int32Type& type) { ... }
///   arrow::Status Visit(const arrow::Int64Type& type) { ... }
///   ...
/// }
/// ExampleVisitor visitor;
/// VisitTypeInline(some_type, &visitor);
/// ```
template <typename VISITOR, typename... ARGS>
inline Status VisitTypeInline(const DataType& type, VISITOR* visitor, ARGS&&... args) {
  switch (type.id()) {
    ARROW_GENERATE_FOR_ALL_TYPES(TYPE_VISIT_INLINE);
    default:
      break;
  }
  return Status::NotImplemented("Type not implemented");
}

#undef TYPE_VISIT_INLINE

#define TYPE_VISIT_INLINE(TYPE_CLASS)                          \
  case TYPE_CLASS##Type::type_id:                              \
    return std::forward<VISITOR>(visitor)(                     \
        internal::checked_cast<const TYPE_CLASS##Type&>(type), \
        std::forward<ARGS>(args)...);

/// \brief Call `visitor` with the corresponding concrete type class
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `type` argument
///
/// Unlike VisitTypeInline which calls `visitor.Visit`, here `visitor`
/// itself is called.
/// `visitor` must support a `const DataType&` argument as a fallback,
/// in addition to concrete type classes.
///
/// The intent is for this to be called on a generic lambda
/// that may internally use `if constexpr` or similar constructs.
template <typename VISITOR, typename... ARGS>
inline auto VisitType(const DataType& type, VISITOR&& visitor, ARGS&&... args)
    -> decltype(std::forward<VISITOR>(visitor)(type, args...)) {
  switch (type.id()) {
    ARROW_GENERATE_FOR_ALL_TYPES(TYPE_VISIT_INLINE);
    default:
      break;
  }
  return std::forward<VISITOR>(visitor)(type, std::forward<ARGS>(args)...);
}

#undef TYPE_VISIT_INLINE

#define TYPE_ID_VISIT_INLINE(TYPE_CLASS)                              \
  case TYPE_CLASS##Type::type_id: {                                   \
    const TYPE_CLASS##Type* concrete_ptr = NULLPTR;                   \
    return visitor->Visit(concrete_ptr, std::forward<ARGS>(args)...); \
  }

/// \brief Calls `visitor` with a nullptr of the corresponding concrete type class
///
/// \tparam VISITOR Visitor type that implements Visit() for all Arrow types.
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `type` argument
/// \return Status
template <typename VISITOR, typename... ARGS>
inline Status VisitTypeIdInline(Type::type id, VISITOR* visitor, ARGS&&... args) {
  switch (id) {
    ARROW_GENERATE_FOR_ALL_TYPES(TYPE_ID_VISIT_INLINE);
    default:
      break;
  }
  return Status::NotImplemented("Type not implemented");
}

#undef TYPE_ID_VISIT_INLINE

}  // namespace arrow
