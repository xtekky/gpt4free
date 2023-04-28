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

#include <functional>
#include <string>
#include <vector>

#include "arrow/compute/type_fwd.h"
#include "arrow/engine/substrait/type_fwd.h"
#include "arrow/engine/substrait/visibility.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace engine {

/// How strictly to adhere to the input structure when converting between Substrait and
/// Acero representations of a plan. This allows the user to trade conversion accuracy
/// for performance and lenience.
enum class ARROW_ENGINE_EXPORT ConversionStrictness {
  /// When a primitive is used at the input that doesn't have an exact match at the
  /// output, reject the conversion. This effectively asserts that there is no (known)
  /// information loss in the conversion, and that plans should either round-trip back and
  /// forth exactly or not at all. This option is primarily intended for testing and
  /// debugging.
  EXACT_ROUNDTRIP,

  /// When a primitive is used at the input that doesn't have an exact match at the
  /// output, attempt to model it with some collection of primitives at the output. This
  /// means that even if the incoming plan is completely optimal by some metric, the
  /// returned plan is fairly likely to not be optimal anymore, and round-trips back and
  /// forth may make the plan increasingly suboptimal. However, every primitive at the
  /// output can be (manually) traced back to exactly one primitive at the input, which
  /// may be useful when debugging.
  PRESERVE_STRUCTURE,

  /// Behaves like PRESERVE_STRUCTURE, but prefers performance over structural accuracy.
  /// Basic optimizations *may* be applied, in order to attempt to not regress in terms of
  /// plan performance: if the incoming plan was already aggressively optimized, the goal
  /// is for the output plan to not be less performant. In practical use cases, this is
  /// probably the option you want.
  ///
  /// Note that no guarantees are made on top of PRESERVE_STRUCTURE. Past and future
  /// versions of Arrow may even ignore this option entirely and treat it exactly like
  /// PRESERVE_STRUCTURE.
  BEST_EFFORT,
};

using NamedTableProvider =
    std::function<Result<compute::Declaration>(const std::vector<std::string>&)>;
static NamedTableProvider kDefaultNamedTableProvider;

class ExtensionProvider;

ARROW_ENGINE_EXPORT std::shared_ptr<ExtensionProvider> default_extension_provider();

/// Options that control the conversion between Substrait and Acero representations of a
/// plan.
struct ARROW_ENGINE_EXPORT ConversionOptions {
  /// \brief How strictly the converter should adhere to the structure of the input.
  ConversionStrictness strictness = ConversionStrictness::BEST_EFFORT;
  /// \brief A custom strategy to be used for providing named tables
  ///
  /// The default behavior will return an invalid status if the plan has any
  /// named table relations.
  NamedTableProvider named_table_provider = kDefaultNamedTableProvider;
  std::shared_ptr<ExtensionProvider> extension_provider = default_extension_provider();
};

}  // namespace engine
}  // namespace arrow
