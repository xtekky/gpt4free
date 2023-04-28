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

#include <cstdint>
#include <memory>
#include <vector>

#include "arrow/status.h"
#include "arrow/python/serialize.h"
#include "arrow/python/visibility.h"

namespace arrow {

class RecordBatch;
class Tensor;

namespace io {

class RandomAccessFile;

}  // namespace io

namespace py {

struct ARROW_PYTHON_EXPORT SparseTensorCounts {
  int coo;
  int csr;
  int csc;
  int csf;
  int ndim_csf;

  int num_total_tensors() const { return coo + csr + csc + csf; }
  int num_total_buffers() const {
    return coo * 3 + csr * 4 + csc * 4 + 2 * ndim_csf + csf;
  }
};

/// \brief Read serialized Python sequence from file interface using Arrow IPC
/// \param[in] src a RandomAccessFile
/// \param[out] out the reconstructed data
/// \return Status
ARROW_PYTHON_EXPORT
Status ReadSerializedObject(io::RandomAccessFile* src, SerializedPyObject* out);

/// \brief Reconstruct SerializedPyObject from representation produced by
/// SerializedPyObject::GetComponents.
///
/// \param[in] num_tensors number of tensors in the object
/// \param[in] num_sparse_tensors number of sparse tensors in the object
/// \param[in] num_ndarrays number of numpy Ndarrays in the object
/// \param[in] num_buffers number of buffers in the object
/// \param[in] data a list containing pyarrow.Buffer instances. It must be 1 +
/// num_tensors * 2 + num_coo_tensors * 3 + num_csr_tensors * 4 + num_csc_tensors * 4 +
/// num_csf_tensors * (2 * ndim_csf + 3) + num_buffers in length
/// \param[out] out the reconstructed object
/// \return Status
ARROW_PYTHON_EXPORT
Status GetSerializedFromComponents(int num_tensors,
                                   const SparseTensorCounts& num_sparse_tensors,
                                   int num_ndarrays, int num_buffers, PyObject* data,
                                   SerializedPyObject* out);

/// \brief Reconstruct Python object from Arrow-serialized representation
/// \param[in] context Serialization context which contains custom serialization
/// and deserialization callbacks. Can be any Python object with a
/// _serialize_callback method for serialization and a _deserialize_callback
/// method for deserialization. If context is None, no custom serialization
/// will be attempted.
/// \param[in] object Object to deserialize
/// \param[in] base a Python object holding the underlying data that any NumPy
/// arrays will reference, to avoid premature deallocation
/// \param[out] out The returned object
/// \return Status
/// This acquires the GIL
ARROW_PYTHON_EXPORT
Status DeserializeObject(PyObject* context, const SerializedPyObject& object,
                         PyObject* base, PyObject** out);

/// \brief Reconstruct Ndarray from Arrow-serialized representation
/// \param[in] object Object to deserialize
/// \param[out] out The deserialized tensor
/// \return Status
ARROW_PYTHON_EXPORT
Status DeserializeNdarray(const SerializedPyObject& object, std::shared_ptr<Tensor>* out);

ARROW_PYTHON_EXPORT
Status NdarrayFromBuffer(std::shared_ptr<Buffer> src, std::shared_ptr<Tensor>* out);

}  // namespace py
}  // namespace arrow
