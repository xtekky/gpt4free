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
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "arrow/compute/type_fwd.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/engine/substrait/options.h"
#include "arrow/engine/substrait/type_fwd.h"
#include "arrow/engine/substrait/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace engine {

/// \brief Serialize an Acero Plan to a binary protobuf Substrait message
///
/// \param[in] declaration the Acero declaration to serialize.
/// This declaration is the sink relation of the Acero plan.
/// \param[in,out] ext_set the extension mapping to use; may be updated to add
/// \param[in] conversion_options options to control how the conversion is done
///
/// \return a buffer containing the protobuf serialization of the Acero relation
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Buffer>> SerializePlan(
    const compute::Declaration& declaration, ExtensionSet* ext_set,
    const ConversionOptions& conversion_options = {});

/// Factory function type for generating the node that consumes the batches produced by
/// each toplevel Substrait relation when deserializing a Substrait Plan.
using ConsumerFactory = std::function<std::shared_ptr<compute::SinkNodeConsumer>()>;

/// \brief Deserializes a Substrait Plan message to a list of ExecNode declarations
///
/// The output of each top-level Substrait relation will be sent to a caller supplied
/// consumer function provided by consumer_factory
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Plan
/// message
/// \param[in] consumer_factory factory function for generating the node that consumes
/// the batches produced by each toplevel Substrait relation
/// \param[in] registry an extension-id-registry to use, or null for the default one.
/// \param[out] ext_set_out if non-null, the extension mapping used by the Substrait
/// Plan is returned here.
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a vector of ExecNode declarations, one for each toplevel relation in the
/// Substrait Plan
ARROW_ENGINE_EXPORT Result<std::vector<compute::Declaration>> DeserializePlans(
    const Buffer& buf, const ConsumerFactory& consumer_factory,
    const ExtensionIdRegistry* registry = NULLPTR, ExtensionSet* ext_set_out = NULLPTR,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a single-relation Substrait Plan message to an execution plan
///
/// The output of each top-level Substrait relation will be sent to a caller supplied
/// consumer function provided by consumer_factory
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Plan
/// message
/// \param[in] consumer node that consumes the batches produced by each toplevel Substrait
/// relation
/// \param[in] registry an extension-id-registry to use, or null for the default one.
/// \param[out] ext_set_out if non-null, the extension mapping used by the Substrait
/// \param[in] conversion_options options to control how the conversion is to be done.
/// Plan is returned here.
/// \return an ExecNode corresponding to the single toplevel relation in the Substrait
/// Plan
ARROW_ENGINE_EXPORT Result<std::shared_ptr<compute::ExecPlan>> DeserializePlan(
    const Buffer& buf, const std::shared_ptr<compute::SinkNodeConsumer>& consumer,
    const ExtensionIdRegistry* registry = NULLPTR, ExtensionSet* ext_set_out = NULLPTR,
    const ConversionOptions& conversion_options = {});

/// Factory function type for generating the write options of a node consuming the batches
/// produced by each toplevel Substrait relation when deserializing a Substrait Plan.
using WriteOptionsFactory = std::function<std::shared_ptr<dataset::WriteNodeOptions>()>;

/// \brief Deserializes a Substrait Plan message to a list of ExecNode declarations
///
/// The output of each top-level Substrait relation will be written to a filesystem.
/// `write_options_factory` can be used to control write behavior.
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Plan
/// message
/// \param[in] write_options_factory factory function for generating the write options of
/// a node consuming the batches produced by each toplevel Substrait relation
/// \param[in] registry an extension-id-registry to use, or null for the default one.
/// \param[out] ext_set_out if non-null, the extension mapping used by the Substrait
/// Plan is returned here.
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a vector of ExecNode declarations, one for each toplevel relation in the
/// Substrait Plan
ARROW_ENGINE_EXPORT Result<std::vector<compute::Declaration>> DeserializePlans(
    const Buffer& buf, const WriteOptionsFactory& write_options_factory,
    const ExtensionIdRegistry* registry = NULLPTR, ExtensionSet* ext_set_out = NULLPTR,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a single-relation Substrait Plan message to an execution plan
///
/// The output of the single Substrait relation will be written to a filesystem.
/// `write_options_factory` can be used to control write behavior.
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Plan
/// message
/// \param[in] write_options write options of a node consuming the batches produced by
/// each toplevel Substrait relation
/// \param[in] registry an extension-id-registry to use, or null for the default one.
/// \param[out] ext_set_out if non-null, the extension mapping used by the Substrait
/// Plan is returned here.
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a vector of ExecNode declarations, one for each toplevel relation in the
/// Substrait Plan
ARROW_ENGINE_EXPORT Result<std::shared_ptr<compute::ExecPlan>> DeserializePlan(
    const Buffer& buf, const std::shared_ptr<dataset::WriteNodeOptions>& write_options,
    const ExtensionIdRegistry* registry = NULLPTR, ExtensionSet* ext_set_out = NULLPTR,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a Substrait Plan message to a Declaration
///
/// The plan will not contain any sink nodes and will be suitable for use in any
/// of the arrow::compute::DeclarationToXyz methods.
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Plan
/// message
/// \param[in] registry an extension-id-registry to use, or null for the default one.
/// \param[out] ext_set_out if non-null, the extension mapping used by the Substrait
/// Plan is returned here.
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return A declaration representing the Substrait plan
ARROW_ENGINE_EXPORT Result<compute::Declaration> DeserializePlan(
    const Buffer& buf, const ExtensionIdRegistry* registry = NULLPTR,
    ExtensionSet* ext_set_out = NULLPTR,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a Substrait Type message to the corresponding Arrow type
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait Type
/// message
/// \param[in] ext_set the extension mapping to use, normally provided by the
/// surrounding Plan message
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return the corresponding Arrow data type
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<DataType>> DeserializeType(
    const Buffer& buf, const ExtensionSet& ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Serializes an Arrow type to a Substrait Type message
///
/// \param[in] type the Arrow data type to serialize
/// \param[in,out] ext_set the extension mapping to use; may be updated to add a
/// mapping for the given type
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a buffer containing the protobuf serialization of the corresponding Substrait
/// Type message
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Buffer>> SerializeType(
    const DataType& type, ExtensionSet* ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a Substrait NamedStruct message to an Arrow schema
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait
/// NamedStruct message
/// \param[in] ext_set the extension mapping to use, normally provided by the
/// surrounding Plan message
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return the corresponding Arrow schema
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Schema>> DeserializeSchema(
    const Buffer& buf, const ExtensionSet& ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Serializes an Arrow schema to a Substrait NamedStruct message
///
/// \param[in] schema the Arrow schema to serialize
/// \param[in,out] ext_set the extension mapping to use; may be updated to add
/// mappings for the types used in the schema
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a buffer containing the protobuf serialization of the corresponding Substrait
/// NamedStruct message
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Buffer>> SerializeSchema(
    const Schema& schema, ExtensionSet* ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a Substrait Expression message to a compute expression
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait
/// Expression message
/// \param[in] ext_set the extension mapping to use, normally provided by the
/// surrounding Plan message
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return the corresponding Arrow compute expression
ARROW_ENGINE_EXPORT
Result<compute::Expression> DeserializeExpression(
    const Buffer& buf, const ExtensionSet& ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Serializes an Arrow compute expression to a Substrait Expression message
///
/// \param[in] expr the Arrow compute expression to serialize
/// \param[in,out] ext_set the extension mapping to use; may be updated to add
/// mappings for the types used in the expression
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return a buffer containing the protobuf serialization of the corresponding Substrait
/// Expression message
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Buffer>> SerializeExpression(
    const compute::Expression& expr, ExtensionSet* ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Serialize an Acero Declaration to a binary protobuf Substrait message
///
/// \param[in] declaration the Acero declaration to serialize
/// \param[in,out] ext_set the extension mapping to use; may be updated to add
/// \param[in] conversion_options options to control how the conversion is done
///
/// \return a buffer containing the protobuf serialization of the Acero relation
ARROW_ENGINE_EXPORT Result<std::shared_ptr<Buffer>> SerializeRelation(
    const compute::Declaration& declaration, ExtensionSet* ext_set,
    const ConversionOptions& conversion_options = {});

/// \brief Deserializes a Substrait Rel (relation) message to an ExecNode declaration
///
/// \param[in] buf a buffer containing the protobuf serialization of a Substrait
/// Rel message
/// \param[in] ext_set the extension mapping to use, normally provided by the
/// surrounding Plan message
/// \param[in] conversion_options options to control how the conversion is to be done.
/// \return the corresponding ExecNode declaration
ARROW_ENGINE_EXPORT Result<compute::Declaration> DeserializeRelation(
    const Buffer& buf, const ExtensionSet& ext_set,
    const ConversionOptions& conversion_options = {});

namespace internal {

/// \brief Checks whether two protobuf serializations of a particular Substrait message
/// type are equivalent
///
/// Note that a binary comparison of the two buffers is insufficient. One reason for this
/// is that the fields of a message can be specified in any order in the serialization.
///
/// \param[in] message_name the name of the Substrait message type to check
/// \param[in] l_buf buffer containing the first protobuf serialization to compare
/// \param[in] r_buf buffer containing the second protobuf serialization to compare
/// \return success if equivalent, failure if not
ARROW_ENGINE_EXPORT
Status CheckMessagesEquivalent(std::string_view message_name, const Buffer& l_buf,
                               const Buffer& r_buf);

/// \brief Utility function to convert a JSON serialization of a Substrait message to
/// its binary serialization
///
/// \param[in] type_name the name of the Substrait message type to convert
/// \param[in] json the JSON string to convert
/// \param[in] ignore_unknown_fields if true then unknown fields will be ignored and
///            will not cause an error
///
///            This should generally be true to allow consumption of plans from newer
///            producers but setting to false can be useful if you are testing
///            conformance to a specific Substrait version
/// \return a buffer filled with the binary protobuf serialization of message
ARROW_ENGINE_EXPORT
Result<std::shared_ptr<Buffer>> SubstraitFromJSON(std::string_view type_name,
                                                  std::string_view json,
                                                  bool ignore_unknown_fields = true);

/// \brief Utility function to convert a binary protobuf serialization of a Substrait
/// message to JSON
///
/// \param[in] type_name the name of the Substrait message type to convert
/// \param[in] buf the buffer containing the binary protobuf serialization of the message
/// \return a JSON string representing the message
ARROW_ENGINE_EXPORT
Result<std::string> SubstraitToJSON(std::string_view type_name, const Buffer& buf);

}  // namespace internal
}  // namespace engine
}  // namespace arrow
