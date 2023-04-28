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
// deprecation cycle.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "arrow/compute/kernel.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/compare.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

/// \defgroup compute-functions Abstract compute function API
///
/// @{

/// \brief Extension point for defining options outside libarrow (but
/// still within this project).
class ARROW_EXPORT FunctionOptionsType {
 public:
  virtual ~FunctionOptionsType() = default;

  virtual const char* type_name() const = 0;
  virtual std::string Stringify(const FunctionOptions&) const = 0;
  virtual bool Compare(const FunctionOptions&, const FunctionOptions&) const = 0;
  virtual Result<std::shared_ptr<Buffer>> Serialize(const FunctionOptions&) const;
  virtual Result<std::unique_ptr<FunctionOptions>> Deserialize(
      const Buffer& buffer) const;
  virtual std::unique_ptr<FunctionOptions> Copy(const FunctionOptions&) const = 0;
};

/// \brief Base class for specifying options configuring a function's behavior,
/// such as error handling.
class ARROW_EXPORT FunctionOptions : public util::EqualityComparable<FunctionOptions> {
 public:
  virtual ~FunctionOptions() = default;

  const FunctionOptionsType* options_type() const { return options_type_; }
  const char* type_name() const { return options_type()->type_name(); }

  bool Equals(const FunctionOptions& other) const;
  using util::EqualityComparable<FunctionOptions>::Equals;
  using util::EqualityComparable<FunctionOptions>::operator==;
  using util::EqualityComparable<FunctionOptions>::operator!=;
  std::string ToString() const;
  std::unique_ptr<FunctionOptions> Copy() const;
  /// \brief Serialize an options struct to a buffer.
  Result<std::shared_ptr<Buffer>> Serialize() const;
  /// \brief Deserialize an options struct from a buffer.
  /// Note: this will only look for `type_name` in the default FunctionRegistry;
  /// to use a custom FunctionRegistry, look up the FunctionOptionsType, then
  /// call FunctionOptionsType::Deserialize().
  static Result<std::unique_ptr<FunctionOptions>> Deserialize(
      const std::string& type_name, const Buffer& buffer);

 protected:
  explicit FunctionOptions(const FunctionOptionsType* type) : options_type_(type) {}
  const FunctionOptionsType* options_type_;
};

ARROW_EXPORT void PrintTo(const FunctionOptions&, std::ostream*);

/// \brief Contains the number of required arguments for the function.
///
/// Naming conventions taken from https://en.wikipedia.org/wiki/Arity.
struct ARROW_EXPORT Arity {
  /// \brief A function taking no arguments
  static Arity Nullary() { return Arity(0, false); }

  /// \brief A function taking 1 argument
  static Arity Unary() { return Arity(1, false); }

  /// \brief A function taking 2 arguments
  static Arity Binary() { return Arity(2, false); }

  /// \brief A function taking 3 arguments
  static Arity Ternary() { return Arity(3, false); }

  /// \brief A function taking a variable number of arguments
  ///
  /// \param[in] min_args the minimum number of arguments required when
  /// invoking the function
  static Arity VarArgs(int min_args = 0) { return Arity(min_args, true); }

  // NOTE: the 0-argument form (default constructor) is required for Cython
  explicit Arity(int num_args = 0, bool is_varargs = false)
      : num_args(num_args), is_varargs(is_varargs) {}

  /// The number of required arguments (or the minimum number for varargs
  /// functions).
  int num_args;

  /// If true, then the num_args is the minimum number of required arguments.
  bool is_varargs = false;
};

struct ARROW_EXPORT FunctionDoc {
  /// \brief A one-line summary of the function, using a verb.
  ///
  /// For example, "Add two numeric arrays or scalars".
  std::string summary;

  /// \brief A detailed description of the function, meant to follow the summary.
  std::string description;

  /// \brief Symbolic names (identifiers) for the function arguments.
  ///
  /// Some bindings may use this to generate nicer function signatures.
  std::vector<std::string> arg_names;

  // TODO add argument descriptions?

  /// \brief Name of the options class, if any.
  std::string options_class;

  /// \brief Whether options are required for function execution
  ///
  /// If false, then either the function does not have an options class
  /// or there is a usable default options value.
  bool options_required;

  FunctionDoc() = default;

  FunctionDoc(std::string summary, std::string description,
              std::vector<std::string> arg_names, std::string options_class = "",
              bool options_required = false)
      : summary(std::move(summary)),
        description(std::move(description)),
        arg_names(std::move(arg_names)),
        options_class(std::move(options_class)),
        options_required(options_required) {}

  static const FunctionDoc& Empty();
};

/// \brief An executor of a function with a preconfigured kernel
class ARROW_EXPORT FunctionExecutor {
 public:
  virtual ~FunctionExecutor() = default;
  /// \brief Initialize or re-initialize the preconfigured kernel
  ///
  /// This method may be called zero or more times. Depending on how
  /// the FunctionExecutor was obtained, it may already have been initialized.
  virtual Status Init(const FunctionOptions* options = NULLPTR,
                      ExecContext* exec_ctx = NULLPTR) = 0;
  /// \brief Execute the preconfigured kernel with arguments that must fit it
  ///
  /// The method requires the arguments be castable to the preconfigured types.
  ///
  /// \param[in] args Arguments to execute the function on
  /// \param[in] length Length of arguments batch or -1 to default it. If the
  /// function has no parameters, this determines the batch length, defaulting
  /// to 0. Otherwise, if the function is scalar, this must equal the argument
  /// batch's inferred length or be -1 to default to it. This is ignored for
  /// vector functions.
  virtual Result<Datum> Execute(const std::vector<Datum>& args, int64_t length = -1) = 0;
};

/// \brief Base class for compute functions. Function implementations contain a
/// collection of "kernels" which are implementations of the function for
/// specific argument types. Selecting a viable kernel for executing a function
/// is referred to as "dispatching".
class ARROW_EXPORT Function {
 public:
  /// \brief The kind of function, which indicates in what contexts it is
  /// valid for use.
  enum Kind {
    /// A function that performs scalar data operations on whole arrays of
    /// data. Can generally process Array or Scalar values. The size of the
    /// output will be the same as the size (or broadcasted size, in the case
    /// of mixing Array and Scalar inputs) of the input.
    SCALAR,

    /// A function with array input and output whose behavior depends on the
    /// values of the entire arrays passed, rather than the value of each scalar
    /// value.
    VECTOR,

    /// A function that computes scalar summary statistics from array input.
    SCALAR_AGGREGATE,

    /// A function that computes grouped summary statistics from array input
    /// and an array of group identifiers.
    HASH_AGGREGATE,

    /// A function that dispatches to other functions and does not contain its
    /// own kernels.
    META
  };

  virtual ~Function() = default;

  /// \brief The name of the kernel. The registry enforces uniqueness of names.
  const std::string& name() const { return name_; }

  /// \brief The kind of kernel, which indicates in what contexts it is valid
  /// for use.
  Function::Kind kind() const { return kind_; }

  /// \brief Contains the number of arguments the function requires, or if the
  /// function accepts variable numbers of arguments.
  const Arity& arity() const { return arity_; }

  /// \brief Return the function documentation
  const FunctionDoc& doc() const { return doc_; }

  /// \brief Returns the number of registered kernels for this function.
  virtual int num_kernels() const = 0;

  /// \brief Return a kernel that can execute the function given the exact
  /// argument types (without implicit type casts).
  ///
  /// NB: This function is overridden in CastFunction.
  virtual Result<const Kernel*> DispatchExact(const std::vector<TypeHolder>& types) const;

  /// \brief Return a best-match kernel that can execute the function given the argument
  /// types, after implicit casts are applied.
  ///
  /// \param[in,out] values Argument types. An element may be modified to
  /// indicate that the returned kernel only approximately matches the input
  /// value descriptors; callers are responsible for casting inputs to the type
  /// required by the kernel.
  virtual Result<const Kernel*> DispatchBest(std::vector<TypeHolder>* values) const;

  /// \brief Get a function executor with a best-matching kernel
  ///
  /// The returned executor will by default work with the default FunctionOptions
  /// and KernelContext. If you want to change that, call `FunctionExecutor::Init`.
  virtual Result<std::shared_ptr<FunctionExecutor>> GetBestExecutor(
      std::vector<TypeHolder> inputs) const;

  /// \brief Execute the function eagerly with the passed input arguments with
  /// kernel dispatch, batch iteration, and memory allocation details taken
  /// care of.
  ///
  /// If the `options` pointer is null, then `default_options()` will be used.
  ///
  /// This function can be overridden in subclasses.
  virtual Result<Datum> Execute(const std::vector<Datum>& args,
                                const FunctionOptions* options, ExecContext* ctx) const;

  virtual Result<Datum> Execute(const ExecBatch& batch, const FunctionOptions* options,
                                ExecContext* ctx) const;

  /// \brief Returns the default options for this function.
  ///
  /// Whatever option semantics a Function has, implementations must guarantee
  /// that default_options() is valid to pass to Execute as options.
  const FunctionOptions* default_options() const { return default_options_; }

  virtual Status Validate() const;

 protected:
  Function(std::string name, Function::Kind kind, const Arity& arity, FunctionDoc doc,
           const FunctionOptions* default_options)
      : name_(std::move(name)),
        kind_(kind),
        arity_(arity),
        doc_(std::move(doc)),
        default_options_(default_options) {}

  Status CheckArity(size_t num_args) const;

  std::string name_;
  Function::Kind kind_;
  Arity arity_;
  const FunctionDoc doc_;
  const FunctionOptions* default_options_ = NULLPTR;
};

namespace detail {

template <typename KernelType>
class FunctionImpl : public Function {
 public:
  /// \brief Return pointers to current-available kernels for inspection
  std::vector<const KernelType*> kernels() const {
    std::vector<const KernelType*> result;
    for (const auto& kernel : kernels_) {
      result.push_back(&kernel);
    }
    return result;
  }

  int num_kernels() const override { return static_cast<int>(kernels_.size()); }

 protected:
  FunctionImpl(std::string name, Function::Kind kind, const Arity& arity, FunctionDoc doc,
               const FunctionOptions* default_options)
      : Function(std::move(name), kind, arity, std::move(doc), default_options) {}

  std::vector<KernelType> kernels_;
};

/// \brief Look up a kernel in a function. If no Kernel is found, nullptr is returned.
ARROW_EXPORT
const Kernel* DispatchExactImpl(const Function* func, const std::vector<TypeHolder>&);

/// \brief Return an error message if no Kernel is found.
ARROW_EXPORT
Status NoMatchingKernel(const Function* func, const std::vector<TypeHolder>&);

}  // namespace detail

/// \brief A function that executes elementwise operations on arrays or
/// scalars, and therefore whose results generally do not depend on the order
/// of the values in the arguments. Accepts and returns arrays that are all of
/// the same size. These functions roughly correspond to the functions used in
/// SQL expressions.
class ARROW_EXPORT ScalarFunction : public detail::FunctionImpl<ScalarKernel> {
 public:
  using KernelType = ScalarKernel;

  ScalarFunction(std::string name, const Arity& arity, FunctionDoc doc,
                 const FunctionOptions* default_options = NULLPTR)
      : detail::FunctionImpl<ScalarKernel>(std::move(name), Function::SCALAR, arity,
                                           std::move(doc), default_options) {}

  /// \brief Add a kernel with given input/output types, no required state
  /// initialization, preallocation for fixed-width types, and default null
  /// handling (intersect validity bitmaps of inputs).
  Status AddKernel(std::vector<InputType> in_types, OutputType out_type,
                   ArrayKernelExec exec, KernelInit init = NULLPTR);

  /// \brief Add a kernel (function implementation). Returns error if the
  /// kernel's signature does not match the function's arity.
  Status AddKernel(ScalarKernel kernel);
};

/// \brief A function that executes general array operations that may yield
/// outputs of different sizes or have results that depend on the whole array
/// contents. These functions roughly correspond to the functions found in
/// non-SQL array languages like APL and its derivatives.
class ARROW_EXPORT VectorFunction : public detail::FunctionImpl<VectorKernel> {
 public:
  using KernelType = VectorKernel;

  VectorFunction(std::string name, const Arity& arity, FunctionDoc doc,
                 const FunctionOptions* default_options = NULLPTR)
      : detail::FunctionImpl<VectorKernel>(std::move(name), Function::VECTOR, arity,
                                           std::move(doc), default_options) {}

  /// \brief Add a simple kernel with given input/output types, no required
  /// state initialization, no data preallocation, and no preallocation of the
  /// validity bitmap.
  Status AddKernel(std::vector<InputType> in_types, OutputType out_type,
                   ArrayKernelExec exec, KernelInit init = NULLPTR);

  /// \brief Add a kernel (function implementation). Returns error if the
  /// kernel's signature does not match the function's arity.
  Status AddKernel(VectorKernel kernel);
};

class ARROW_EXPORT ScalarAggregateFunction
    : public detail::FunctionImpl<ScalarAggregateKernel> {
 public:
  using KernelType = ScalarAggregateKernel;

  ScalarAggregateFunction(std::string name, const Arity& arity, FunctionDoc doc,
                          const FunctionOptions* default_options = NULLPTR)
      : detail::FunctionImpl<ScalarAggregateKernel>(std::move(name),
                                                    Function::SCALAR_AGGREGATE, arity,
                                                    std::move(doc), default_options) {}

  /// \brief Add a kernel (function implementation). Returns error if the
  /// kernel's signature does not match the function's arity.
  Status AddKernel(ScalarAggregateKernel kernel);
};

class ARROW_EXPORT HashAggregateFunction
    : public detail::FunctionImpl<HashAggregateKernel> {
 public:
  using KernelType = HashAggregateKernel;

  HashAggregateFunction(std::string name, const Arity& arity, FunctionDoc doc,
                        const FunctionOptions* default_options = NULLPTR)
      : detail::FunctionImpl<HashAggregateKernel>(std::move(name),
                                                  Function::HASH_AGGREGATE, arity,
                                                  std::move(doc), default_options) {}

  /// \brief Add a kernel (function implementation). Returns error if the
  /// kernel's signature does not match the function's arity.
  Status AddKernel(HashAggregateKernel kernel);
};

/// \brief A function that dispatches to other functions. Must implement
/// MetaFunction::ExecuteImpl.
///
/// For Array, ChunkedArray, and Scalar Datum kinds, may rely on the execution
/// of concrete Function types, but must handle other Datum kinds on its own.
class ARROW_EXPORT MetaFunction : public Function {
 public:
  int num_kernels() const override { return 0; }

  Result<Datum> Execute(const std::vector<Datum>& args, const FunctionOptions* options,
                        ExecContext* ctx) const override;

  Result<Datum> Execute(const ExecBatch& batch, const FunctionOptions* options,
                        ExecContext* ctx) const override;

 protected:
  virtual Result<Datum> ExecuteImpl(const std::vector<Datum>& args,
                                    const FunctionOptions* options,
                                    ExecContext* ctx) const = 0;

  MetaFunction(std::string name, const Arity& arity, FunctionDoc doc,
               const FunctionOptions* default_options = NULLPTR)
      : Function(std::move(name), Function::META, arity, std::move(doc),
                 default_options) {}
};

/// @}

}  // namespace compute
}  // namespace arrow
