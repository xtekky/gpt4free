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

#include "arrow/type_fwd.h"

namespace arrow {

template <typename T>
using AsyncGenerator = std::function<Future<T>()>;

template <typename T, typename V>
class MappingGenerator;

template <typename T, typename ComesAfter, typename IsNext>
class SequencingGenerator;

template <typename T, typename V>
class TransformingGenerator;

template <typename T>
class SerialReadaheadGenerator;

template <typename T>
class ReadaheadGenerator;

template <typename T>
class PushGenerator;

template <typename T>
class MergedGenerator;

template <typename T>
struct Enumerated;

template <typename T>
class EnumeratingGenerator;

template <typename T>
class TransferringGenerator;

template <typename T>
class BackgroundGenerator;

template <typename T>
class GeneratorIterator;

template <typename T>
struct CancellableGenerator;

template <typename T>
class DefaultIfEmptyGenerator;

}  // namespace arrow
