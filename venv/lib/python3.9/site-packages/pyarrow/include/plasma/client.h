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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"
#include "plasma/common.h"

using arrow::Buffer;
using arrow::Status;

namespace plasma {

/// Object buffer data structure.
struct ObjectBuffer {
  /// The data buffer.
  std::shared_ptr<Buffer> data;
  /// The metadata buffer.
  std::shared_ptr<Buffer> metadata;
  /// The device number.
  int device_num;
};

class ARROW_EXPORT PlasmaClient {
 public:
  PlasmaClient();
  ~PlasmaClient();

  /// Connect to the local plasma store. Return the resulting connection.
  ///
  /// \param store_socket_name The name of the UNIX domain socket to use to
  ///        connect to the Plasma store.
  /// \param manager_socket_name The name of the UNIX domain socket to use to
  ///        connect to the local Plasma manager. If this is "", then this
  ///        function will not connect to a manager.
  ///        Note that plasma manager is no longer supported, this function
  ///        will return failure if this is not "".
  /// \param release_delay Deprecated (not used).
  /// \param num_retries number of attempts to connect to IPC socket, default 50
  /// \return The return status.
  Status Connect(const std::string& store_socket_name,
                 const std::string& manager_socket_name = "", int release_delay = 0,
                 int num_retries = -1);

  /// Set runtime options for this client.
  ///
  /// \param client_name The name of the client, used in debug messages.
  /// \param output_memory_quota The memory quota in bytes for objects created by
  ///        this client.
  Status SetClientOptions(const std::string& client_name, int64_t output_memory_quota);

  /// Create an object in the Plasma Store. Any metadata for this object must be
  /// be passed in when the object is created.
  ///
  /// \param object_id The ID to use for the newly created object.
  /// \param data_size The size in bytes of the space to be allocated for this
  /// object's
  ///        data (this does not include space used for metadata).
  /// \param metadata The object's metadata. If there is no metadata, this
  /// pointer
  ///        should be NULL.
  /// \param metadata_size The size in bytes of the metadata. If there is no
  ///        metadata, this should be 0.
  /// \param data The address of the newly created object will be written here.
  /// \param device_num The number of the device where the object is being
  ///        created.
  ///        device_num = 0 corresponds to the host,
  ///        device_num = 1 corresponds to GPU0,
  ///        device_num = 2 corresponds to GPU1, etc.
  /// \param evict_if_full Whether to evict other objects to make space for
  ///        this object.
  /// \return The return status.
  ///
  /// The returned object must be released once it is done with.  It must also
  /// be either sealed or aborted.
  Status Create(const ObjectID& object_id, int64_t data_size, const uint8_t* metadata,
                int64_t metadata_size, std::shared_ptr<Buffer>* data, int device_num = 0,
                bool evict_if_full = true);

  /// Create and seal an object in the object store. This is an optimization
  /// which allows small objects to be created quickly with fewer messages to
  /// the store.
  ///
  /// \param object_id The ID of the object to create.
  /// \param data The data for the object to create.
  /// \param metadata The metadata for the object to create.
  /// \param evict_if_full Whether to evict other objects to make space for
  ///        this object.
  /// \return The return status.
  Status CreateAndSeal(const ObjectID& object_id, const std::string& data,
                       const std::string& metadata, bool evict_if_full = true);

  /// Create and seal multiple objects in the object store. This is an optimization
  /// of CreateAndSeal to eliminate the cost of IPC per object.
  ///
  /// \param object_ids The vector of IDs of the objects to create.
  /// \param data The vector of data for the objects to create.
  /// \param metadata The vector of metadata for the objects to create.
  /// \param evict_if_full Whether to evict other objects to make space for
  ///        these objects.
  /// \return The return status.
  Status CreateAndSealBatch(const std::vector<ObjectID>& object_ids,
                            const std::vector<std::string>& data,
                            const std::vector<std::string>& metadata,
                            bool evict_if_full = true);

  /// Get some objects from the Plasma Store. This function will block until the
  /// objects have all been created and sealed in the Plasma Store or the
  /// timeout expires.
  ///
  /// If an object was not retrieved, the corresponding metadata and data
  /// fields in the ObjectBuffer structure will evaluate to false.
  /// Objects are automatically released by the client when their buffers
  /// get out of scope.
  ///
  /// \param object_ids The IDs of the objects to get.
  /// \param timeout_ms The amount of time in milliseconds to wait before this
  ///        request times out. If this value is -1, then no timeout is set.
  /// \param[out] object_buffers The object results.
  /// \return The return status.
  Status Get(const std::vector<ObjectID>& object_ids, int64_t timeout_ms,
             std::vector<ObjectBuffer>* object_buffers);

  /// Deprecated variant of Get() that doesn't automatically release buffers
  /// when they get out of scope.
  ///
  /// \param object_ids The IDs of the objects to get.
  /// \param num_objects The number of object IDs to get.
  /// \param timeout_ms The amount of time in milliseconds to wait before this
  ///        request times out. If this value is -1, then no timeout is set.
  /// \param object_buffers An array where the results will be stored.
  /// \return The return status.
  ///
  /// The caller is responsible for releasing any retrieved objects, but it
  /// should not release objects that were not retrieved.
  Status Get(const ObjectID* object_ids, int64_t num_objects, int64_t timeout_ms,
             ObjectBuffer* object_buffers);

  /// Tell Plasma that the client no longer needs the object. This should be
  /// called after Get() or Create() when the client is done with the object.
  /// After this call, the buffer returned by Get() is no longer valid.
  ///
  /// \param object_id The ID of the object that is no longer needed.
  /// \return The return status.
  Status Release(const ObjectID& object_id);

  /// Check if the object store contains a particular object and the object has
  /// been sealed. The result will be stored in has_object.
  ///
  /// @todo: We may want to indicate if the object has been created but not
  /// sealed.
  ///
  /// \param object_id The ID of the object whose presence we are checking.
  /// \param has_object The function will write true at this address if
  ///        the object is present and false if it is not present.
  /// \return The return status.
  Status Contains(const ObjectID& object_id, bool* has_object);

  /// List all the objects in the object store.
  ///
  /// This API is experimental and might change in the future.
  ///
  /// \param[out] objects ObjectTable of objects in the store. For each entry
  ///             in the map, the following fields are available:
  ///             - metadata_size: Size of the object metadata in bytes
  ///             - data_size: Size of the object data in bytes
  ///             - ref_count: Number of clients referencing the object buffer
  ///             - create_time: Unix timestamp of the object creation
  ///             - construct_duration: Object creation time in seconds
  ///             - state: Is the object still being created or already sealed?
  /// \return The return status.
  Status List(ObjectTable* objects);

  /// Abort an unsealed object in the object store. If the abort succeeds, then
  /// it will be as if the object was never created at all. The unsealed object
  /// must have only a single reference (the one that would have been removed by
  /// calling Seal).
  ///
  /// \param object_id The ID of the object to abort.
  /// \return The return status.
  Status Abort(const ObjectID& object_id);

  /// Seal an object in the object store. The object will be immutable after
  /// this
  /// call.
  ///
  /// \param object_id The ID of the object to seal.
  /// \return The return status.
  Status Seal(const ObjectID& object_id);

  /// Delete an object from the object store. This currently assumes that the
  /// object is present, has been sealed and not used by another client. Otherwise,
  /// it is a no operation.
  ///
  /// \todo We may want to allow the deletion of objects that are not present or
  ///       haven't been sealed.
  ///
  /// \param object_id The ID of the object to delete.
  /// \return The return status.
  Status Delete(const ObjectID& object_id);

  /// Delete a list of objects from the object store. This currently assumes that the
  /// object is present, has been sealed and not used by another client. Otherwise,
  /// it is a no operation.
  ///
  /// \param object_ids The list of IDs of the objects to delete.
  /// \return The return status. If all the objects are nonexistent, return OK.
  Status Delete(const std::vector<ObjectID>& object_ids);

  /// Delete objects until we have freed up num_bytes bytes or there are no more
  /// released objects that can be deleted.
  ///
  /// \param num_bytes The number of bytes to try to free up.
  /// \param num_bytes_evicted Out parameter for total number of bytes of space
  /// retrieved.
  /// \return The return status.
  Status Evict(int64_t num_bytes, int64_t& num_bytes_evicted);

  /// Bump objects up in the LRU cache, i.e. treat them as recently accessed.
  /// Objects that do not exist in the store will be ignored.
  ///
  /// \param object_ids The IDs of the objects to bump.
  /// \return The return status.
  Status Refresh(const std::vector<ObjectID>& object_ids);

  /// Compute the hash of an object in the object store.
  ///
  /// \param object_id The ID of the object we want to hash.
  /// \param digest A pointer at which to return the hash digest of the object.
  ///        The pointer must have at least kDigestSize bytes allocated.
  /// \return The return status.
  Status Hash(const ObjectID& object_id, uint8_t* digest);

  /// Subscribe to notifications when objects are sealed in the object store.
  /// Whenever an object is sealed, a message will be written to the client
  /// socket that is returned by this method.
  ///
  /// \param fd Out parameter for the file descriptor the client should use to
  /// read notifications
  ///         from the object store about sealed objects.
  /// \return The return status.
  Status Subscribe(int* fd);

  /// Receive next object notification for this client if Subscribe has been called.
  ///
  /// \param fd The file descriptor we are reading the notification from.
  /// \param object_id Out parameter, the object_id of the object that was sealed.
  /// \param data_size Out parameter, the data size of the object that was sealed.
  /// \param metadata_size Out parameter, the metadata size of the object that was sealed.
  /// \return The return status.
  Status GetNotification(int fd, ObjectID* object_id, int64_t* data_size,
                         int64_t* metadata_size);

  Status DecodeNotifications(const uint8_t* buffer, std::vector<ObjectID>* object_ids,
                             std::vector<int64_t>* data_sizes,
                             std::vector<int64_t>* metadata_sizes);

  /// Disconnect from the local plasma instance, including the local store and
  /// manager.
  ///
  /// \return The return status.
  Status Disconnect();

  /// Get the current debug string from the plasma store server.
  ///
  /// \return The debug string.
  std::string DebugString();

  /// Get the memory capacity of the store.
  ///
  /// \return Memory capacity of the store in bytes.
  int64_t store_capacity();

 private:
  friend class PlasmaBuffer;
  friend class PlasmaMutableBuffer;
  FRIEND_TEST(TestPlasmaStore, GetTest);
  FRIEND_TEST(TestPlasmaStore, LegacyGetTest);
  FRIEND_TEST(TestPlasmaStore, AbortTest);

  bool IsInUse(const ObjectID& object_id);

  class ARROW_NO_EXPORT Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace plasma
