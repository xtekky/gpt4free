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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/engine/substrait/type_fwd.h"
#include "arrow/engine/substrait/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace engine {

constexpr const char* kSubstraitArithmeticFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_arithmetic.yaml";
constexpr const char* kSubstraitBooleanFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_boolean.yaml";
constexpr const char* kSubstraitComparisonFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_comparison.yaml";
constexpr const char* kSubstraitDatetimeFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_datetime.yaml";
constexpr const char* kSubstraitLogarithmicFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_logarithmic.yaml";
constexpr const char* kSubstraitRoundingFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_rounding.yaml";
constexpr const char* kSubstraitStringFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_string.yaml";
constexpr const char* kSubstraitAggregateGenericFunctionsUri =
    "https://github.com/substrait-io/substrait/blob/main/extensions/"
    "functions_aggregate_generic.yaml";

struct Id {
  std::string_view uri, name;
  bool empty() const { return uri.empty() && name.empty(); }
  std::string ToString() const;
};
struct IdHashEq {
  size_t operator()(Id id) const;
  bool operator()(Id l, Id r) const;
};

/// \brief Owning storage for ids
///
/// Substrait plans may reuse URIs and names in many places.  For convenience
/// and performance Substarit ids are typically passed around as views.  As we
/// convert a plan from Substrait to Arrow we need to copy these strings out of
/// the Substrait buffer and into owned storage.  This class serves as that owned
/// storage.
class IdStorage {
 public:
  virtual ~IdStorage() = default;
  /// \brief Get an equivalent id pointing into this storage
  ///
  /// This operation will copy the ids into storage if they do not already exist
  virtual Id Emplace(Id id) = 0;
  /// \brief Get an equivalent view pointing into this storage for a URI
  ///
  /// If no URI is found then the uri will be copied into storage
  virtual std::string_view EmplaceUri(std::string_view uri) = 0;
  /// \brief Get an equivalent id pointing into this storage
  ///
  /// If no id is found then nullopt will be returned
  virtual std::optional<Id> Find(Id id) const = 0;
  /// \brief Get an equivalent view pointing into this storage for a URI
  ///
  /// If no URI is found then nullopt will be returned
  virtual std::optional<std::string_view> FindUri(std::string_view uri) const = 0;

  static std::unique_ptr<IdStorage> Make();
};

/// \brief Describes a Substrait call
///
/// Substrait call expressions contain a list of arguments which can either
/// be enum arguments (which are serialized as strings), value arguments (which)
/// are Arrow expressions, or type arguments (not yet implemented)
class SubstraitCall {
 public:
  SubstraitCall(Id id, std::shared_ptr<DataType> output_type, bool output_nullable,
                bool is_hash = false)
      : id_(id),
        output_type_(std::move(output_type)),
        output_nullable_(output_nullable),
        is_hash_(is_hash) {}

  const Id& id() const { return id_; }
  const std::shared_ptr<DataType>& output_type() const { return output_type_; }
  bool output_nullable() const { return output_nullable_; }
  bool is_hash() const { return is_hash_; }

  bool HasEnumArg(int index) const;
  Result<std::string_view> GetEnumArg(int index) const;
  void SetEnumArg(int index, std::string enum_arg);
  Result<compute::Expression> GetValueArg(int index) const;
  bool HasValueArg(int index) const;
  void SetValueArg(int index, compute::Expression value_arg);
  std::optional<std::vector<std::string> const*> GetOption(
      std::string_view option_name) const;
  void SetOption(std::string_view option_name,
                 const std::vector<std::string_view>& option_preferences);
  int size() const { return size_; }

 private:
  Id id_;
  std::shared_ptr<DataType> output_type_;
  bool output_nullable_;
  // Only needed when converting from Substrait -> Arrow aggregates.  The
  // Arrow function name depends on whether or not there are any groups
  bool is_hash_;
  std::unordered_map<int, std::string> enum_args_;
  std::unordered_map<int, compute::Expression> value_args_;
  std::unordered_map<std::string, std::vector<std::string>> options_;
  int size_ = 0;
};

/// Substrait identifies functions and custom data types using a (uri, name) pair.
///
/// This registry is a bidirectional mapping between Substrait IDs and their
/// corresponding Arrow counterparts (arrow::DataType and function names in a function
/// registry)
///
/// Substrait extension types and variations must be registered with their
/// corresponding arrow::DataType before they can be used!
///
/// Conceptually this can be thought of as two pairs of `unordered_map`s.  One pair to
/// go back and forth between Substrait ID and arrow::DataType and another pair to go
/// back and forth between Substrait ID and Arrow function names.
///
/// Unlike an ExtensionSet this registry is not created automatically when consuming
/// Substrait plans and must be configured ahead of time (although there is a default
/// instance).
class ARROW_ENGINE_EXPORT ExtensionIdRegistry {
 public:
  using ArrowToSubstraitCall =
      std::function<Result<SubstraitCall>(const arrow::compute::Expression::Call&)>;
  using SubstraitCallToArrow =
      std::function<Result<arrow::compute::Expression>(const SubstraitCall&)>;
  using ArrowToSubstraitAggregate =
      std::function<Result<SubstraitCall>(const arrow::compute::Aggregate&)>;
  using SubstraitAggregateToArrow =
      std::function<Result<arrow::compute::Aggregate>(const SubstraitCall&)>;

  /// \brief A mapping between a Substrait ID and an arrow::DataType
  struct TypeRecord {
    Id id;
    const std::shared_ptr<DataType>& type;
  };

  /// \brief Return a uri view owned by this registry
  ///
  /// If the URI has never been emplaced it will return nullopt
  virtual std::optional<std::string_view> FindUri(std::string_view uri) const = 0;
  /// \brief Return a id view owned by this registry
  ///
  /// If the id has never been emplaced it will return nullopt
  virtual std::optional<Id> FindId(Id id) const = 0;
  virtual std::optional<TypeRecord> GetType(const DataType&) const = 0;
  virtual std::optional<TypeRecord> GetType(Id) const = 0;
  virtual Status CanRegisterType(Id, const std::shared_ptr<DataType>& type) const = 0;
  virtual Status RegisterType(Id, std::shared_ptr<DataType>) = 0;
  /// \brief Register a converter that converts an Arrow call to a Substrait call
  ///
  /// Note that there may not be 1:1 parity between ArrowToSubstraitCall and
  /// SubstraitCallToArrow because some standard functions (e.g. add) may map to
  /// multiple Arrow functions (e.g. add, add_checked)
  virtual Status AddArrowToSubstraitCall(std::string arrow_function_name,
                                         ArrowToSubstraitCall conversion_func) = 0;
  /// \brief Check to see if a converter can be registered
  ///
  /// \return Status::OK if there are no conflicts, otherwise an error is returned
  virtual Status CanAddArrowToSubstraitCall(
      const std::string& arrow_function_name) const = 0;

  /// \brief Register a converter that converts an Arrow aggregate to a Substrait
  ///        aggregate
  virtual Status AddArrowToSubstraitAggregate(
      std::string arrow_function_name, ArrowToSubstraitAggregate conversion_func) = 0;
  /// \brief Check to see if a converter can be registered
  ///
  /// \return Status::OK if there are no conflicts, otherwise an error is returned
  virtual Status CanAddArrowToSubstraitAggregate(
      const std::string& arrow_function_name) const = 0;

  /// \brief Register a converter that converts a Substrait call to an Arrow call
  virtual Status AddSubstraitCallToArrow(Id substrait_function_id,
                                         SubstraitCallToArrow conversion_func) = 0;
  /// \brief Check to see if a converter can be registered
  ///
  /// \return Status::OK if there are no conflicts, otherwise an error is returned
  virtual Status CanAddSubstraitCallToArrow(Id substrait_function_id) const = 0;
  /// \brief Register a simple mapping function
  ///
  /// All calls to the function must pass only value arguments.  The arguments
  /// will be converted to expressions and passed to the Arrow function
  virtual Status AddSubstraitCallToArrow(Id substrait_function_id,
                                         std::string arrow_function_name) = 0;

  /// \brief Register a converter that converts a Substrait aggregate to an Arrow
  ///        aggregate
  virtual Status AddSubstraitAggregateToArrow(
      Id substrait_function_id, SubstraitAggregateToArrow conversion_func) = 0;
  /// \brief Check to see if a converter can be registered
  ///
  /// \return Status::OK if there are no conflicts, otherwise an error is returned
  virtual Status CanAddSubstraitAggregateToArrow(Id substrait_function_id) const = 0;

  /// \brief Return a list of Substrait functions that have a converter
  ///
  /// The function ids are encoded as strings using the pattern {uri}#{name}
  virtual std::vector<std::string> GetSupportedSubstraitFunctions() const = 0;

  /// \brief Find a converter to map Arrow calls to Substrait calls
  /// \return A converter function or an invalid status if no converter is registered
  virtual Result<ArrowToSubstraitCall> GetArrowToSubstraitCall(
      const std::string& arrow_function_name) const = 0;

  /// \brief Find a converter to map Arrow aggregates to Substrait aggregates
  /// \return A converter function or an invalid status if no converter is registered
  virtual Result<ArrowToSubstraitAggregate> GetArrowToSubstraitAggregate(
      const std::string& arrow_function_name) const = 0;

  /// \brief Find a converter to map a Substrait aggregate to an Arrow aggregate
  /// \return A converter function or an invalid status if no converter is registered
  virtual Result<SubstraitAggregateToArrow> GetSubstraitAggregateToArrow(
      Id substrait_function_id) const = 0;

  /// \brief Find a converter to map a Substrait call to an Arrow call
  /// \return A converter function or an invalid status if no converter is registered
  virtual Result<SubstraitCallToArrow> GetSubstraitCallToArrow(
      Id substrait_function_id) const = 0;

  /// \brief Similar to \see GetSubstraitCallToArrow but only uses the name
  ///
  /// There may be multiple functions with the same name and this will return
  /// the first.  This is slower than GetSubstraitCallToArrow and should only
  /// be used when the plan does not include a URI (or the URI is "/")
  virtual Result<SubstraitCallToArrow> GetSubstraitCallToArrowFallback(
      std::string_view function_name) const = 0;

  /// \brief Similar to \see GetSubstraitAggregateToArrow but only uses the name
  ///
  /// \see GetSubstraitCallToArrowFallback for details on the fallback behavior
  virtual Result<SubstraitAggregateToArrow> GetSubstraitAggregateToArrowFallback(
      std::string_view function_name) const = 0;
};

constexpr std::string_view kArrowExtTypesUri =
    "https://github.com/apache/arrow/blob/master/format/substrait/"
    "extension_types.yaml";

/// A default registry with all supported functions and data types registered
///
/// Note: Function support is currently very minimal, see ARROW-15538
ARROW_ENGINE_EXPORT ExtensionIdRegistry* default_extension_id_registry();

/// \brief Make a nested registry with a given parent.
///
/// A nested registry supports registering types and functions other and on top of those
/// already registered in its parent registry. No conflicts in IDs and names used for
/// lookup are allowed. Normally, the given parent is the default registry.
///
/// One use case for a nested registry is for dynamic registration of functions defined
/// within a Substrait plan while keeping these registrations specific to the plan. When
/// the Substrait plan is disposed of, normally after its execution, the nested registry
/// can be disposed of as well.
ARROW_ENGINE_EXPORT std::shared_ptr<ExtensionIdRegistry> nested_extension_id_registry(
    const ExtensionIdRegistry* parent);

/// \brief A set of extensions used within a plan
///
/// Each time an extension is used within a Substrait plan the extension
/// must be included in an extension set that is defined at the root of the
/// plan.
///
/// The plan refers to a specific extension using an "anchor" which is an
/// arbitrary integer invented by the producer that has no meaning beyond a
/// plan but which should be consistent within a plan.
///
/// To support serialization and deserialization this type serves as a
/// bidirectional map between Substrait ID and "anchor"s.
///
/// When deserializing a Substrait plan the extension set should be extracted
/// after the plan has been converted from Protobuf and before the plan
/// is converted to an execution plan.
///
/// The extension set can be kept and reused during serialization if a perfect
/// round trip is required.  If serialization is not needed or round tripping
/// is not required then the extension set can be safely discarded after the
/// plan has been converted into an execution plan.
///
/// When converting an execution plan into a Substrait plan an extension set
/// can be automatically generated or a previously generated extension set can
/// be used.
///
/// ExtensionSet does not own strings; it only refers to strings in an
/// ExtensionIdRegistry.
class ARROW_ENGINE_EXPORT ExtensionSet {
 public:
  struct FunctionRecord {
    Id id;
    std::string_view name;
  };

  struct TypeRecord {
    Id id;
    std::shared_ptr<DataType> type;
  };

  /// Construct an empty ExtensionSet to be populated during serialization.
  explicit ExtensionSet(const ExtensionIdRegistry* = default_extension_id_registry());
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ExtensionSet);

  /// Construct an ExtensionSet with explicit extension ids for efficient referencing
  /// during deserialization. Note that input vectors need not be densely packed; an empty
  /// (default constructed) Id may be used as a placeholder to indicate an unused
  /// _anchor/_reference. This factory will be used to wrap the extensions declared in a
  /// substrait::Plan before deserializing the plan's relations.
  ///
  /// Views will be replaced with equivalent views pointing to memory owned by the
  /// registry.
  ///
  /// Note: This is an advanced operation.  The order of the ids, types, and functions
  /// must match the anchor numbers chosen for a plan.
  ///
  /// An extension set should instead be created using
  /// arrow::engine::GetExtensionSetFromPlan
  static Result<ExtensionSet> Make(
      std::unordered_map<uint32_t, std::string_view> uris,
      std::unordered_map<uint32_t, Id> type_ids,
      std::unordered_map<uint32_t, Id> function_ids,
      const ConversionOptions& conversion_options,
      const ExtensionIdRegistry* = default_extension_id_registry());

  const std::unordered_map<uint32_t, std::string_view>& uris() const { return uris_; }

  /// \brief Returns a data type given an anchor
  ///
  /// This is used when converting a Substrait plan to an Arrow execution plan.
  ///
  /// If the anchor does not exist in this extension set an error will be returned.
  Result<TypeRecord> DecodeType(uint32_t anchor) const;

  /// \brief Returns the number of custom type records in this extension set
  ///
  /// Note: the types are currently stored as a sparse vector, so this may return a value
  /// larger than the actual number of types. This behavior may change in the future; see
  /// ARROW-15583.
  std::size_t num_types() const { return types_.size(); }

  /// \brief Lookup the anchor for a given type
  ///
  /// This operation is used when converting an Arrow execution plan to a Substrait plan.
  /// If the type has been previously encoded then the same anchor value will returned.
  ///
  /// If the type has not been previously encoded then a new anchor value will be created.
  ///
  /// If the type does not exist in the extension id registry then an error will be
  /// returned.
  ///
  /// \return An anchor that can be used to refer to the type within a plan
  Result<uint32_t> EncodeType(const DataType& type);

  /// \brief Return a function id given an anchor
  ///
  /// This is used when converting a Substrait plan to an Arrow execution plan.
  ///
  /// If the anchor does not exist in this extension set an error will be returned.
  Result<Id> DecodeFunction(uint32_t anchor) const;

  /// \brief Lookup the anchor for a given function
  ///
  /// This operation is used when converting an Arrow execution plan to a Substrait  plan.
  /// If the function has been previously encoded then the same anchor value will be
  /// returned.
  ///
  /// If the function has not been previously encoded then a new anchor value will be
  /// created.
  ///
  /// If the function name is not in the extension id registry then an error will be
  /// returned.
  ///
  /// \return An anchor that can be used to refer to the function within a plan
  Result<uint32_t> EncodeFunction(Id function_id);

  /// \brief Return the number of custom functions in this extension set
  std::size_t num_functions() const { return functions_.size(); }

  const ExtensionIdRegistry* registry() const { return registry_; }

 private:
  const ExtensionIdRegistry* registry_;
  // If the registry is not aware of an id then we probably can't do anything
  // with it.  However, in some cases, these may represent extensions or features
  // that we can safely ignore.  For example, we can usually safely ignore
  // extension type variations if we assume the plan is valid.  These ignorable
  // ids are stored here.
  std::unique_ptr<IdStorage> plan_specific_ids_ = IdStorage::Make();

  // Map from anchor values to URI values referenced by this extension set
  std::unordered_map<uint32_t, std::string_view> uris_;
  // Map from anchor values to type definitions, used during Substrait->Arrow
  // and populated from the Substrait extension set
  std::unordered_map<uint32_t, TypeRecord> types_;
  // Map from anchor values to function ids, used during Substrait->Arrow
  // and populated from the Substrait extension set
  std::unordered_map<uint32_t, Id> functions_;
  // Map from type names to anchor values.  Used during Arrow->Substrait
  // and built as the plan is created.
  std::unordered_map<Id, uint32_t, IdHashEq, IdHashEq> types_map_;
  // Map from function names to anchor values.  Used during Arrow->Substrait
  // and built as the plan is created.
  std::unordered_map<Id, uint32_t, IdHashEq, IdHashEq> functions_map_;

  Status CheckHasUri(std::string_view uri);
  void AddUri(std::pair<uint32_t, std::string_view> uri);
  Status AddUri(Id id);
};

}  // namespace engine
}  // namespace arrow
