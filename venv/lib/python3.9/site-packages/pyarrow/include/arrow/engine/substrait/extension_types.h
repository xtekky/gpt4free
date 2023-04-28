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

// This API is EXPERIMENTAL.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "arrow/engine/substrait/visibility.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace engine {

// arrow::ExtensionTypes are provided to wrap uuid, fixed_char, varchar, interval_year,
// and interval_day which are first-class types in substrait but do not appear in
// the arrow type system.
//
// Note that these are not automatically registered with arrow::RegisterExtensionType(),
// which means among other things that serialization of these types to IPC would fail.

/// fixed_size_binary(16) for storing Universally Unique IDentifiers
ARROW_ENGINE_EXPORT
std::shared_ptr<DataType> uuid();

/// fixed_size_binary(length) constrained to contain only valid UTF-8
ARROW_ENGINE_EXPORT
std::shared_ptr<DataType> fixed_char(int32_t length);

/// utf8() constrained to be shorter than `length`
ARROW_ENGINE_EXPORT
std::shared_ptr<DataType> varchar(int32_t length);

/// fixed_size_list(int32(), 2) storing a number of [years, months]
ARROW_ENGINE_EXPORT
std::shared_ptr<DataType> interval_year();

/// fixed_size_list(int32(), 2) storing a number of [days, seconds]
ARROW_ENGINE_EXPORT
std::shared_ptr<DataType> interval_day();

/// Return true if t is Uuid, otherwise false
ARROW_ENGINE_EXPORT
bool UnwrapUuid(const DataType&);

/// Return FixedChar length if t is FixedChar, otherwise nullopt
ARROW_ENGINE_EXPORT
std::optional<int32_t> UnwrapFixedChar(const DataType&);

/// Return Varchar (max) length if t is VarChar, otherwise nullopt
ARROW_ENGINE_EXPORT
std::optional<int32_t> UnwrapVarChar(const DataType& t);

/// Return true if t is IntervalYear, otherwise false
ARROW_ENGINE_EXPORT
bool UnwrapIntervalYear(const DataType&);

/// Return true if t is IntervalDay, otherwise false
ARROW_ENGINE_EXPORT
bool UnwrapIntervalDay(const DataType&);

}  // namespace engine
}  // namespace arrow
