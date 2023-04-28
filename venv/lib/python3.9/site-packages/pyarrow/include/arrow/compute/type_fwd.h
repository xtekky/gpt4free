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

#include "arrow/util/visibility.h"

namespace arrow {

struct Datum;
struct TypeHolder;

namespace compute {

class Function;
class FunctionExecutor;
class FunctionOptions;
class FunctionRegistry;

class CastOptions;

struct ExecBatch;
class ExecContext;
class KernelContext;

struct Kernel;
struct ScalarKernel;
struct ScalarAggregateKernel;
struct VectorKernel;

struct KernelState;

struct Declaration;
class Expression;
class ExecNode;
class ExecPlan;
class ExecNodeOptions;
class ExecFactoryRegistry;
class QueryContext;
struct QueryOptions;

class SinkNodeConsumer;

ARROW_EXPORT ExecContext* default_exec_context();
ARROW_EXPORT ExecContext* threaded_exec_context();

}  // namespace compute
}  // namespace arrow
