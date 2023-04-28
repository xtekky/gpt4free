// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License") = 0; you may not use this file except in compliance
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

namespace parquet {
namespace encryption {

// Key material can be stored outside the Parquet file, for example in a separate small
// file in the same folder. This is important for “key rotation”, when MEKs have to be
// changed (if compromised; or periodically, just in case) - without modifying the Parquet
// files (often  immutable).
// TODO: details will be implemented later
class FileKeyMaterialStore {};

}  // namespace encryption
}  // namespace parquet
