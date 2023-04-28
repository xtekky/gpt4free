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

// NOTE: API is EXPERIMENTAL and will change without going through a
// deprecation cycle

#pragma once

/// \defgroup compute-concrete-options Concrete option classes for compute functions
/// @{
/// @}

#include "arrow/compute/api_aggregate.h"  // IWYU pragma: export
#include "arrow/compute/api_scalar.h"     // IWYU pragma: export
#include "arrow/compute/api_vector.h"     // IWYU pragma: export
#include "arrow/compute/cast.h"           // IWYU pragma: export
#include "arrow/compute/function.h"       // IWYU pragma: export
#include "arrow/compute/kernel.h"         // IWYU pragma: export
#include "arrow/compute/registry.h"       // IWYU pragma: export
#include "arrow/datum.h"                  // IWYU pragma: export

/// \defgroup execnode-expressions Utilities for creating expressions to
/// use in execution plans
/// @{
/// @}

#include "arrow/compute/exec/expression.h"  // IWYU pragma: export

/// \defgroup execnode-options Concrete option classes for ExecNode options
/// @{
/// @}

#include "arrow/compute/exec/options.h"  // IWYU pragma: export

/// \defgroup execnode-row Utilities for working with data in a row-major format
/// @{
/// @}

#include "arrow/compute/row/grouper.h"  // IWYU pragma: export

/// \defgroup execnode-components Components associated with ExecNode
/// @{
/// @}

#include "arrow/compute/exec.h"            // IWYU pragma: export
#include "arrow/compute/exec/exec_plan.h"  // IWYU pragma: export
