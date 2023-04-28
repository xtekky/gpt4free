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

#include <memory>
#include <string>
#include <vector>

#include "arrow/extension_type.h"
#include "arrow/testing/visibility.h"
#include "arrow/util/macros.h"

namespace arrow {

class ARROW_TESTING_EXPORT UuidArray : public ExtensionArray {
 public:
  using ExtensionArray::ExtensionArray;
};

class ARROW_TESTING_EXPORT UuidType : public ExtensionType {
 public:
  UuidType() : ExtensionType(fixed_size_binary(16)) {}

  std::string extension_name() const override { return "uuid"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "uuid-serialized"; }
};

class ARROW_TESTING_EXPORT SmallintArray : public ExtensionArray {
 public:
  using ExtensionArray::ExtensionArray;
};

class ARROW_TESTING_EXPORT TinyintArray : public ExtensionArray {
 public:
  using ExtensionArray::ExtensionArray;
};

class ARROW_TESTING_EXPORT ListExtensionArray : public ExtensionArray {
 public:
  using ExtensionArray::ExtensionArray;
};

class ARROW_TESTING_EXPORT SmallintType : public ExtensionType {
 public:
  SmallintType() : ExtensionType(int16()) {}

  std::string extension_name() const override { return "smallint"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "smallint"; }
};

class ARROW_TESTING_EXPORT TinyintType : public ExtensionType {
 public:
  TinyintType() : ExtensionType(int8()) {}

  std::string extension_name() const override { return "tinyint"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "tinyint"; }
};

class ARROW_TESTING_EXPORT ListExtensionType : public ExtensionType {
 public:
  ListExtensionType() : ExtensionType(list(int32())) {}

  std::string extension_name() const override { return "list-ext"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "list-ext"; }
};

class ARROW_TESTING_EXPORT DictExtensionType : public ExtensionType {
 public:
  DictExtensionType() : ExtensionType(dictionary(int8(), utf8())) {}

  std::string extension_name() const override { return "dict-extension"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "dict-extension-serialized"; }
};

class ARROW_TESTING_EXPORT Complex128Array : public ExtensionArray {
 public:
  using ExtensionArray::ExtensionArray;
};

class ARROW_TESTING_EXPORT Complex128Type : public ExtensionType {
 public:
  Complex128Type()
      : ExtensionType(struct_({::arrow::field("real", float64(), /*nullable=*/false),
                               ::arrow::field("imag", float64(), /*nullable=*/false)})) {}

  std::string extension_name() const override { return "complex128"; }

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override { return "complex128-serialized"; }
};

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> uuid();

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> smallint();

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> tinyint();

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> list_extension_type();

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> dict_extension_type();

ARROW_TESTING_EXPORT
std::shared_ptr<DataType> complex128();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> ExampleUuid();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> ExampleSmallint();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> ExampleTinyint();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> ExampleDictExtension();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> ExampleComplex128();

ARROW_TESTING_EXPORT
std::shared_ptr<Array> MakeComplex128(const std::shared_ptr<Array>& real,
                                      const std::shared_ptr<Array>& imag);

// A RAII class that registers an extension type on construction
// and unregisters it on destruction.
class ARROW_TESTING_EXPORT ExtensionTypeGuard {
 public:
  explicit ExtensionTypeGuard(const std::shared_ptr<DataType>& type);
  explicit ExtensionTypeGuard(const DataTypeVector& types);
  ~ExtensionTypeGuard();
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ExtensionTypeGuard);

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(ExtensionTypeGuard);

  std::vector<std::string> extension_names_;
};

}  // namespace arrow
