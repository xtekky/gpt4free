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

#include "arrow/extension_type.h"
#include "arrow/util/macros.h"
#include "arrow/python/common.h"
#include "arrow/python/visibility.h"

namespace arrow {
namespace py {

class ARROW_PYTHON_EXPORT PyExtensionType : public ExtensionType {
 public:
  // Implement extensionType API
  std::string extension_name() const override { return extension_name_; }

  std::string ToString() const override;

  bool ExtensionEquals(const ExtensionType& other) const override;

  std::shared_ptr<Array> MakeArray(std::shared_ptr<ArrayData> data) const override;

  Result<std::shared_ptr<DataType>> Deserialize(
      std::shared_ptr<DataType> storage_type,
      const std::string& serialized) const override;

  std::string Serialize() const override;

  // For use from Cython
  // Assumes that `typ` is borrowed
  static Status FromClass(const std::shared_ptr<DataType> storage_type,
                          const std::string extension_name, PyObject* typ,
                          std::shared_ptr<ExtensionType>* out);

  // Return new ref
  PyObject* GetInstance() const;
  Status SetInstance(PyObject*) const;

 protected:
  PyExtensionType(std::shared_ptr<DataType> storage_type, PyObject* typ,
                  PyObject* inst = NULLPTR);
  PyExtensionType(std::shared_ptr<DataType> storage_type, std::string extension_name,
                  PyObject* typ, PyObject* inst = NULLPTR);

  std::string extension_name_;

  // These fields are mutable because of two-step initialization.
  mutable OwnedRefNoGIL type_class_;
  // A weakref or null.  Storing a strong reference to the Python extension type
  // instance would create an unreclaimable reference cycle between Python and C++
  // (the Python instance has to keep a strong reference to the C++ ExtensionType
  //  in other direction).  Instead, we store a weakref to the instance.
  // If the weakref is dead, we reconstruct the instance from its serialized form.
  mutable OwnedRefNoGIL type_instance_;
  // Empty if type_instance_ is null
  mutable std::string serialized_;
};

ARROW_PYTHON_EXPORT std::string PyExtensionName();

ARROW_PYTHON_EXPORT Status RegisterPyExtensionType(const std::shared_ptr<DataType>&);

ARROW_PYTHON_EXPORT Status UnregisterPyExtensionType(const std::string& type_name);

}  // namespace py
}  // namespace arrow
