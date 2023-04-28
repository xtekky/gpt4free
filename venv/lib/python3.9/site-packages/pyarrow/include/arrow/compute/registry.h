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

#include <memory>
#include <string>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

class Function;
class FunctionOptionsType;

/// \brief A mutable central function registry for built-in functions as well
/// as user-defined functions. Functions are implementations of
/// arrow::compute::Function.
///
/// Generally, each function contains kernels which are implementations of a
/// function for a specific argument signature. After looking up a function in
/// the registry, one can either execute it eagerly with Function::Execute or
/// use one of the function's dispatch methods to pick a suitable kernel for
/// lower-level function execution.
class ARROW_EXPORT FunctionRegistry {
 public:
  ~FunctionRegistry();

  /// \brief Construct a new registry.
  ///
  /// Most users only need to use the global registry.
  static std::unique_ptr<FunctionRegistry> Make();

  /// \brief Construct a new nested registry with the given parent.
  ///
  /// Most users only need to use the global registry. The returned registry never changes
  /// its parent, even when an operation allows overwritting.
  static std::unique_ptr<FunctionRegistry> Make(FunctionRegistry* parent);

  /// \brief Check whether a new function can be added to the registry.
  ///
  /// \returns Status::KeyError if a function with the same name is already registered.
  Status CanAddFunction(std::shared_ptr<Function> function, bool allow_overwrite = false);

  /// \brief Add a new function to the registry.
  ///
  /// \returns Status::KeyError if a function with the same name is already registered.
  Status AddFunction(std::shared_ptr<Function> function, bool allow_overwrite = false);

  /// \brief Check whether an alias can be added for the given function name.
  ///
  /// \returns Status::KeyError if the function with the given name is not registered.
  Status CanAddAlias(const std::string& target_name, const std::string& source_name);

  /// \brief Add alias for the given function name.
  ///
  /// \returns Status::KeyError if the function with the given name is not registered.
  Status AddAlias(const std::string& target_name, const std::string& source_name);

  /// \brief Check whether a new function options type can be added to the registry.
  ///
  /// \return Status::KeyError if a function options type with the same name is already
  /// registered.
  Status CanAddFunctionOptionsType(const FunctionOptionsType* options_type,
                                   bool allow_overwrite = false);

  /// \brief Add a new function options type to the registry.
  ///
  /// \returns Status::KeyError if a function options type with the same name is already
  /// registered.
  Status AddFunctionOptionsType(const FunctionOptionsType* options_type,
                                bool allow_overwrite = false);

  /// \brief Retrieve a function by name from the registry.
  Result<std::shared_ptr<Function>> GetFunction(const std::string& name) const;

  /// \brief Return vector of all entry names in the registry.
  ///
  /// Helpful for displaying a manifest of available functions.
  std::vector<std::string> GetFunctionNames() const;

  /// \brief Retrieve a function options type by name from the registry.
  Result<const FunctionOptionsType*> GetFunctionOptionsType(
      const std::string& name) const;

  /// \brief The number of currently registered functions.
  int num_functions() const;

 private:
  FunctionRegistry();

  // Use PIMPL pattern to not have std::unordered_map here
  class FunctionRegistryImpl;
  std::unique_ptr<FunctionRegistryImpl> impl_;

  explicit FunctionRegistry(FunctionRegistryImpl* impl);
};

/// \brief Return the process-global function registry.
ARROW_EXPORT FunctionRegistry* GetFunctionRegistry();

}  // namespace compute
}  // namespace arrow
