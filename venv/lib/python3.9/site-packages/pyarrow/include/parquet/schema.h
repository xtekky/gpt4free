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

// This module contains the logical parquet-cpp types (independent of Thrift
// structures), schema nodes, and related type tools

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parquet/platform.h"
#include "parquet/types.h"
#include "parquet/windows_fixup.h"  // for OPTIONAL

namespace parquet {

class SchemaDescriptor;

namespace schema {

class Node;

// List encodings: using the terminology from Impala to define different styles
// of representing logical lists (a.k.a. ARRAY types) in Parquet schemas. Since
// the converted type named in the Parquet metadata is ConvertedType::LIST we
// use that terminology here. It also helps distinguish from the *_ARRAY
// primitive types.
//
// One-level encoding: Only allows required lists with required cells
//   repeated value_type name
//
// Two-level encoding: Enables optional lists with only required cells
//   <required/optional> group list
//     repeated value_type item
//
// Three-level encoding: Enables optional lists with optional cells
//   <required/optional> group bag
//     repeated group list
//       <required/optional> value_type item
//
// 2- and 1-level encoding are respectively equivalent to 3-level encoding with
// the non-repeated nodes set to required.
//
// The "official" encoding recommended in the Parquet spec is the 3-level, and
// we use that as the default when creating list types. For semantic completeness
// we allow the other two. Since all types of encodings will occur "in the
// wild" we need to be able to interpret the associated definition levels in
// the context of the actual encoding used in the file.
//
// NB: Some Parquet writers may not set ConvertedType::LIST on the repeated
// SchemaElement, which could make things challenging if we are trying to infer
// that a sequence of nodes semantically represents an array according to one
// of these encodings (versus a struct containing an array). We should refuse
// the temptation to guess, as they say.
struct ListEncoding {
  enum type { ONE_LEVEL, TWO_LEVEL, THREE_LEVEL };
};

class PARQUET_EXPORT ColumnPath {
 public:
  ColumnPath() : path_() {}
  explicit ColumnPath(const std::vector<std::string>& path) : path_(path) {}
  explicit ColumnPath(std::vector<std::string>&& path) : path_(std::move(path)) {}

  static std::shared_ptr<ColumnPath> FromDotString(const std::string& dotstring);
  static std::shared_ptr<ColumnPath> FromNode(const Node& node);

  std::shared_ptr<ColumnPath> extend(const std::string& node_name) const;
  std::string ToDotString() const;
  const std::vector<std::string>& ToDotVector() const;

 protected:
  std::vector<std::string> path_;
};

// Base class for logical schema types. A type has a name, repetition level,
// and optionally a logical type (ConvertedType in Parquet metadata parlance)
class PARQUET_EXPORT Node {
 public:
  enum type { PRIMITIVE, GROUP };

  virtual ~Node() {}

  bool is_primitive() const { return type_ == Node::PRIMITIVE; }

  bool is_group() const { return type_ == Node::GROUP; }

  bool is_optional() const { return repetition_ == Repetition::OPTIONAL; }

  bool is_repeated() const { return repetition_ == Repetition::REPEATED; }

  bool is_required() const { return repetition_ == Repetition::REQUIRED; }

  virtual bool Equals(const Node* other) const = 0;

  const std::string& name() const { return name_; }

  Node::type node_type() const { return type_; }

  Repetition::type repetition() const { return repetition_; }

  ConvertedType::type converted_type() const { return converted_type_; }

  const std::shared_ptr<const LogicalType>& logical_type() const { return logical_type_; }

  /// \brief The field_id value for the serialized SchemaElement. If the
  /// field_id is less than 0 (e.g. -1), it will not be set when serialized to
  /// Thrift.
  int field_id() const { return field_id_; }

  const Node* parent() const { return parent_; }

  const std::shared_ptr<ColumnPath> path() const;

  virtual void ToParquet(void* element) const = 0;

  // Node::Visitor abstract class for walking schemas with the visitor pattern
  class Visitor {
   public:
    virtual ~Visitor() {}

    virtual void Visit(Node* node) = 0;
  };
  class ConstVisitor {
   public:
    virtual ~ConstVisitor() {}

    virtual void Visit(const Node* node) = 0;
  };

  virtual void Visit(Visitor* visitor) = 0;
  virtual void VisitConst(ConstVisitor* visitor) const = 0;

 protected:
  friend class GroupNode;

  Node(Node::type type, const std::string& name, Repetition::type repetition,
       ConvertedType::type converted_type = ConvertedType::NONE, int field_id = -1)
      : type_(type),
        name_(name),
        repetition_(repetition),
        converted_type_(converted_type),
        field_id_(field_id),
        parent_(NULLPTR) {}

  Node(Node::type type, const std::string& name, Repetition::type repetition,
       std::shared_ptr<const LogicalType> logical_type, int field_id = -1)
      : type_(type),
        name_(name),
        repetition_(repetition),
        logical_type_(std::move(logical_type)),
        field_id_(field_id),
        parent_(NULLPTR) {}

  Node::type type_;
  std::string name_;
  Repetition::type repetition_;
  ConvertedType::type converted_type_;
  std::shared_ptr<const LogicalType> logical_type_;
  int field_id_;
  // Nodes should not be shared, they have a single parent.
  const Node* parent_;

  bool EqualsInternal(const Node* other) const;
  void SetParent(const Node* p_parent);

 private:
  PARQUET_DISALLOW_COPY_AND_ASSIGN(Node);
};

// Save our breath all over the place with these typedefs
typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodeVector;

// A type that is one of the primitive Parquet storage types. In addition to
// the other type metadata (name, repetition level, logical type), also has the
// physical storage type and their type-specific metadata (byte width, decimal
// parameters)
class PARQUET_EXPORT PrimitiveNode : public Node {
 public:
  static std::unique_ptr<Node> FromParquet(const void* opaque_element);

  // A field_id -1 (or any negative value) will be serialized as null in Thrift
  static inline NodePtr Make(const std::string& name, Repetition::type repetition,
                             Type::type type,
                             ConvertedType::type converted_type = ConvertedType::NONE,
                             int length = -1, int precision = -1, int scale = -1,
                             int field_id = -1) {
    return NodePtr(new PrimitiveNode(name, repetition, type, converted_type, length,
                                     precision, scale, field_id));
  }

  // If no logical type, pass LogicalType::None() or nullptr
  // A field_id -1 (or any negative value) will be serialized as null in Thrift
  static inline NodePtr Make(const std::string& name, Repetition::type repetition,
                             std::shared_ptr<const LogicalType> logical_type,
                             Type::type primitive_type, int primitive_length = -1,
                             int field_id = -1) {
    return NodePtr(new PrimitiveNode(name, repetition, logical_type, primitive_type,
                                     primitive_length, field_id));
  }

  bool Equals(const Node* other) const override;

  Type::type physical_type() const { return physical_type_; }

  ColumnOrder column_order() const { return column_order_; }

  void SetColumnOrder(ColumnOrder column_order) { column_order_ = column_order; }

  int32_t type_length() const { return type_length_; }

  const DecimalMetadata& decimal_metadata() const { return decimal_metadata_; }

  void ToParquet(void* element) const override;
  void Visit(Visitor* visitor) override;
  void VisitConst(ConstVisitor* visitor) const override;

 private:
  PrimitiveNode(const std::string& name, Repetition::type repetition, Type::type type,
                ConvertedType::type converted_type = ConvertedType::NONE, int length = -1,
                int precision = -1, int scale = -1, int field_id = -1);

  PrimitiveNode(const std::string& name, Repetition::type repetition,
                std::shared_ptr<const LogicalType> logical_type,
                Type::type primitive_type, int primitive_length = -1, int field_id = -1);

  Type::type physical_type_;
  int32_t type_length_;
  DecimalMetadata decimal_metadata_;
  ColumnOrder column_order_;

  // For FIXED_LEN_BYTE_ARRAY
  void SetTypeLength(int32_t length) { type_length_ = length; }

  bool EqualsInternal(const PrimitiveNode* other) const;

  FRIEND_TEST(TestPrimitiveNode, Attrs);
  FRIEND_TEST(TestPrimitiveNode, Equals);
  FRIEND_TEST(TestPrimitiveNode, PhysicalLogicalMapping);
  FRIEND_TEST(TestPrimitiveNode, FromParquet);
};

class PARQUET_EXPORT GroupNode : public Node {
 public:
  static std::unique_ptr<Node> FromParquet(const void* opaque_element,
                                           NodeVector fields = {});

  // A field_id -1 (or any negative value) will be serialized as null in Thrift
  static inline NodePtr Make(const std::string& name, Repetition::type repetition,
                             const NodeVector& fields,
                             ConvertedType::type converted_type = ConvertedType::NONE,
                             int field_id = -1) {
    return NodePtr(new GroupNode(name, repetition, fields, converted_type, field_id));
  }

  // If no logical type, pass nullptr
  // A field_id -1 (or any negative value) will be serialized as null in Thrift
  static inline NodePtr Make(const std::string& name, Repetition::type repetition,
                             const NodeVector& fields,
                             std::shared_ptr<const LogicalType> logical_type,
                             int field_id = -1) {
    return NodePtr(new GroupNode(name, repetition, fields, logical_type, field_id));
  }

  bool Equals(const Node* other) const override;

  const NodePtr& field(int i) const { return fields_[i]; }
  // Get the index of a field by its name, or negative value if not found.
  // If several fields share the same name, it is unspecified which one
  // is returned.
  int FieldIndex(const std::string& name) const;
  // Get the index of a field by its node, or negative value if not found.
  int FieldIndex(const Node& node) const;

  int field_count() const { return static_cast<int>(fields_.size()); }

  void ToParquet(void* element) const override;
  void Visit(Visitor* visitor) override;
  void VisitConst(ConstVisitor* visitor) const override;

  /// \brief Return true if this node or any child node has REPEATED repetition
  /// type
  bool HasRepeatedFields() const;

 private:
  GroupNode(const std::string& name, Repetition::type repetition,
            const NodeVector& fields,
            ConvertedType::type converted_type = ConvertedType::NONE, int field_id = -1);

  GroupNode(const std::string& name, Repetition::type repetition,
            const NodeVector& fields, std::shared_ptr<const LogicalType> logical_type,
            int field_id = -1);

  NodeVector fields_;
  bool EqualsInternal(const GroupNode* other) const;

  // Mapping between field name to the field index
  std::unordered_multimap<std::string, int> field_name_to_idx_;

  FRIEND_TEST(TestGroupNode, Attrs);
  FRIEND_TEST(TestGroupNode, Equals);
  FRIEND_TEST(TestGroupNode, FieldIndex);
  FRIEND_TEST(TestGroupNode, FieldIndexDuplicateName);
};

// ----------------------------------------------------------------------
// Convenience primitive type factory functions

#define PRIMITIVE_FACTORY(FuncName, TYPE)                                                \
  static inline NodePtr FuncName(const std::string& name,                                \
                                 Repetition::type repetition = Repetition::OPTIONAL,     \
                                 int field_id = -1) {                                    \
    return PrimitiveNode::Make(name, repetition, Type::TYPE, ConvertedType::NONE,        \
                               /*length=*/-1, /*precision=*/-1, /*scale=*/-1, field_id); \
  }

PRIMITIVE_FACTORY(Boolean, BOOLEAN)
PRIMITIVE_FACTORY(Int32, INT32)
PRIMITIVE_FACTORY(Int64, INT64)
PRIMITIVE_FACTORY(Int96, INT96)
PRIMITIVE_FACTORY(Float, FLOAT)
PRIMITIVE_FACTORY(Double, DOUBLE)
PRIMITIVE_FACTORY(ByteArray, BYTE_ARRAY)

void PARQUET_EXPORT PrintSchema(const schema::Node* schema, std::ostream& stream,
                                int indent_width = 2);

}  // namespace schema

// The ColumnDescriptor encapsulates information necessary to interpret
// primitive column data in the context of a particular schema. We have to
// examine the node structure of a column's path to the root in the schema tree
// to be able to reassemble the nested structure from the repetition and
// definition levels.
class PARQUET_EXPORT ColumnDescriptor {
 public:
  ColumnDescriptor(schema::NodePtr node, int16_t max_definition_level,
                   int16_t max_repetition_level,
                   const SchemaDescriptor* schema_descr = NULLPTR);

  bool Equals(const ColumnDescriptor& other) const;

  int16_t max_definition_level() const { return max_definition_level_; }

  int16_t max_repetition_level() const { return max_repetition_level_; }

  Type::type physical_type() const { return primitive_node_->physical_type(); }

  ConvertedType::type converted_type() const { return primitive_node_->converted_type(); }

  const std::shared_ptr<const LogicalType>& logical_type() const {
    return primitive_node_->logical_type();
  }

  ColumnOrder column_order() const { return primitive_node_->column_order(); }

  SortOrder::type sort_order() const {
    auto la = logical_type();
    auto pt = physical_type();
    return la ? GetSortOrder(la, pt) : GetSortOrder(converted_type(), pt);
  }

  const std::string& name() const { return primitive_node_->name(); }

  const std::shared_ptr<schema::ColumnPath> path() const;

  const schema::NodePtr& schema_node() const { return node_; }

  std::string ToString() const;

  int type_length() const;

  int type_precision() const;

  int type_scale() const;

 private:
  schema::NodePtr node_;
  const schema::PrimitiveNode* primitive_node_;

  int16_t max_definition_level_;
  int16_t max_repetition_level_;
};

// Container for the converted Parquet schema with a computed information from
// the schema analysis needed for file reading
//
// * Column index to Node
// * Max repetition / definition levels for each primitive node
//
// The ColumnDescriptor objects produced by this class can be used to assist in
// the reconstruction of fully materialized data structures from the
// repetition-definition level encoding of nested data
//
// TODO(wesm): this object can be recomputed from a Schema
class PARQUET_EXPORT SchemaDescriptor {
 public:
  SchemaDescriptor() {}
  ~SchemaDescriptor() {}

  // Analyze the schema
  void Init(std::unique_ptr<schema::Node> schema);
  void Init(schema::NodePtr schema);

  const ColumnDescriptor* Column(int i) const;

  // Get the index of a column by its dotstring path, or negative value if not found.
  // If several columns share the same dotstring path, it is unspecified which one
  // is returned.
  int ColumnIndex(const std::string& node_path) const;
  // Get the index of a column by its node, or negative value if not found.
  int ColumnIndex(const schema::Node& node) const;

  bool Equals(const SchemaDescriptor& other, std::ostream* diff_output = NULLPTR) const;

  // The number of physical columns appearing in the file
  int num_columns() const { return static_cast<int>(leaves_.size()); }

  const schema::NodePtr& schema_root() const { return schema_; }

  const schema::GroupNode* group_node() const { return group_node_; }

  // Returns the root (child of the schema root) node of the leaf(column) node
  const schema::Node* GetColumnRoot(int i) const;

  const std::string& name() const { return group_node_->name(); }

  std::string ToString() const;

  void updateColumnOrders(const std::vector<ColumnOrder>& column_orders);

  /// \brief Return column index corresponding to a particular
  /// PrimitiveNode. Returns -1 if not found
  int GetColumnIndex(const schema::PrimitiveNode& node) const;

  /// \brief Return true if any field or their children have REPEATED repetition
  /// type
  bool HasRepeatedFields() const;

 private:
  friend class ColumnDescriptor;

  // Root Node
  schema::NodePtr schema_;
  // Root Node
  const schema::GroupNode* group_node_;

  void BuildTree(const schema::NodePtr& node, int16_t max_def_level,
                 int16_t max_rep_level, const schema::NodePtr& base);

  // Result of leaf node / tree analysis
  std::vector<ColumnDescriptor> leaves_;

  std::unordered_map<const schema::PrimitiveNode*, int> node_to_leaf_index_;

  // Mapping between leaf nodes and root group of leaf (first node
  // below the schema's root group)
  //
  // For example, the leaf `a.b.c.d` would have a link back to `a`
  //
  // -- a  <------
  // -- -- b     |
  // -- -- -- c  |
  // -- -- -- -- d
  std::unordered_map<int, schema::NodePtr> leaf_to_base_;

  // Mapping between ColumnPath DotString to the leaf index
  std::unordered_multimap<std::string, int> leaf_to_idx_;
};

}  // namespace parquet
