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

namespace arrow {
namespace ipc {

enum class MetadataVersion : char {
  /// 0.1.0
  V1,

  /// 0.2.0
  V2,

  /// 0.3.0 to 0.7.1
  V3,

  /// 0.8.0 to 0.17.0
  V4,

  /// >= 1.0.0
  V5
};

class Message;
enum class MessageType {
  NONE,
  SCHEMA,
  DICTIONARY_BATCH,
  RECORD_BATCH,
  TENSOR,
  SPARSE_TENSOR
};

struct IpcReadOptions;
struct IpcWriteOptions;

class MessageReader;

class RecordBatchStreamReader;
class RecordBatchFileReader;
class RecordBatchWriter;

namespace feather {

class Reader;

}  // namespace feather
}  // namespace ipc
}  // namespace arrow
