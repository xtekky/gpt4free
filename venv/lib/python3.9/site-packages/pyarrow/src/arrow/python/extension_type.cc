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

#include <memory>
#include <sstream>
#include <utility>

#include "arrow/util/checked_cast.h"
#include "arrow/util/logging.h"
#include "arrow/python/extension_type.h"
#include "arrow/python/helpers.h"
#include "arrow/python/pyarrow.h"

namespace arrow {

using internal::checked_cast;

namespace py {

namespace {

// Serialize a Python ExtensionType instance
Status SerializeExtInstance(PyObject* type_instance, std::string* out) {
  OwnedRef res(
      cpp_PyObject_CallMethod(type_instance, "__arrow_ext_serialize__", nullptr));
  if (!res) {
    return ConvertPyError();
  }
  if (!PyBytes_Check(res.obj())) {
    return Status::TypeError(
        "__arrow_ext_serialize__ should return bytes object, "
        "got ",
        internal::PyObject_StdStringRepr(res.obj()));
  }
  *out = internal::PyBytes_AsStdString(res.obj());
  return Status::OK();
}

// Deserialize a Python ExtensionType instance
PyObject* DeserializeExtInstance(PyObject* type_class,
                                 std::shared_ptr<DataType> storage_type,
                                 const std::string& serialized_data) {
  OwnedRef storage_ref(wrap_data_type(storage_type));
  if (!storage_ref) {
    return nullptr;
  }
  OwnedRef data_ref(PyBytes_FromStringAndSize(
      serialized_data.data(), static_cast<Py_ssize_t>(serialized_data.size())));
  if (!data_ref) {
    return nullptr;
  }

  return cpp_PyObject_CallMethod(type_class, "__arrow_ext_deserialize__", "OO",
                                 storage_ref.obj(), data_ref.obj());
}

}  // namespace

static const char* kExtensionName = "arrow.py_extension_type";

std::string PyExtensionType::ToString() const {
  PyAcquireGIL lock;

  std::stringstream ss;
  OwnedRef instance(GetInstance());
  ss << "extension<" << this->extension_name() << "<" << Py_TYPE(instance.obj())->tp_name
     << ">>";
  return ss.str();
}

PyExtensionType::PyExtensionType(std::shared_ptr<DataType> storage_type, PyObject* typ,
                                 PyObject* inst)
    : ExtensionType(storage_type),
      extension_name_(kExtensionName),
      type_class_(typ),
      type_instance_(inst) {}

PyExtensionType::PyExtensionType(std::shared_ptr<DataType> storage_type,
                                 std::string extension_name, PyObject* typ,
                                 PyObject* inst)
    : ExtensionType(storage_type),
      extension_name_(std::move(extension_name)),
      type_class_(typ),
      type_instance_(inst) {}

bool PyExtensionType::ExtensionEquals(const ExtensionType& other) const {
  PyAcquireGIL lock;

  if (other.extension_name() != extension_name()) {
    return false;
  }
  const auto& other_ext = checked_cast<const PyExtensionType&>(other);
  int res = -1;
  if (!type_instance_) {
    if (other_ext.type_instance_) {
      return false;
    }
    // Compare Python types
    res = PyObject_RichCompareBool(type_class_.obj(), other_ext.type_class_.obj(), Py_EQ);
  } else {
    if (!other_ext.type_instance_) {
      return false;
    }
    // Compare Python instances
    OwnedRef left(GetInstance());
    OwnedRef right(other_ext.GetInstance());
    if (!left || !right) {
      goto error;
    }
    res = PyObject_RichCompareBool(left.obj(), right.obj(), Py_EQ);
  }
  if (res == -1) {
    goto error;
  }
  return res == 1;

error:
  // Cannot propagate error
  PyErr_WriteUnraisable(nullptr);
  return false;
}

std::shared_ptr<Array> PyExtensionType::MakeArray(std::shared_ptr<ArrayData> data) const {
  DCHECK_EQ(data->type->id(), Type::EXTENSION);
  return std::make_shared<ExtensionArray>(data);
}

std::string PyExtensionType::Serialize() const {
  DCHECK(type_instance_);
  return serialized_;
}

Result<std::shared_ptr<DataType>> PyExtensionType::Deserialize(
    std::shared_ptr<DataType> storage_type, const std::string& serialized_data) const {
  PyAcquireGIL lock;

  if (import_pyarrow()) {
    return ConvertPyError();
  }
  OwnedRef res(DeserializeExtInstance(type_class_.obj(), storage_type, serialized_data));
  if (!res) {
    return ConvertPyError();
  }
  return unwrap_data_type(res.obj());
}

PyObject* PyExtensionType::GetInstance() const {
  if (!type_instance_) {
    PyErr_SetString(PyExc_TypeError, "Not an instance");
    return nullptr;
  }
  DCHECK(PyWeakref_CheckRef(type_instance_.obj()));
  PyObject* inst = PyWeakref_GET_OBJECT(type_instance_.obj());
  if (inst != Py_None) {
    // Cached instance still alive
    Py_INCREF(inst);
    return inst;
  } else {
    // Must reconstruct from serialized form
    // XXX cache again?
    return DeserializeExtInstance(type_class_.obj(), storage_type_, serialized_);
  }
}

Status PyExtensionType::SetInstance(PyObject* inst) const {
  // Check we have the right type
  PyObject* typ = reinterpret_cast<PyObject*>(Py_TYPE(inst));
  if (typ != type_class_.obj()) {
    return Status::TypeError("Unexpected Python ExtensionType class ",
                             internal::PyObject_StdStringRepr(typ), " expected ",
                             internal::PyObject_StdStringRepr(type_class_.obj()));
  }

  PyObject* wr = PyWeakref_NewRef(inst, nullptr);
  if (wr == NULL) {
    return ConvertPyError();
  }
  type_instance_.reset(wr);
  return SerializeExtInstance(inst, &serialized_);
}

Status PyExtensionType::FromClass(const std::shared_ptr<DataType> storage_type,
                                  const std::string extension_name, PyObject* typ,
                                  std::shared_ptr<ExtensionType>* out) {
  Py_INCREF(typ);
  out->reset(new PyExtensionType(storage_type, std::move(extension_name), typ));
  return Status::OK();
}

Status RegisterPyExtensionType(const std::shared_ptr<DataType>& type) {
  DCHECK_EQ(type->id(), Type::EXTENSION);
  auto ext_type = std::dynamic_pointer_cast<ExtensionType>(type);
  return RegisterExtensionType(ext_type);
}

Status UnregisterPyExtensionType(const std::string& type_name) {
  return UnregisterExtensionType(type_name);
}

std::string PyExtensionName() { return kExtensionName; }

}  // namespace py
}  // namespace arrow
