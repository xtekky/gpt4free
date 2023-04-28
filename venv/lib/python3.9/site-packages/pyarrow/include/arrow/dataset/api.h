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

// This API is EXPERIMENTAL.

#pragma once

#include "arrow/compute/exec/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/discovery.h"
#include "arrow/dataset/file_base.h"
#ifdef ARROW_CSV
#include "arrow/dataset/file_csv.h"
#endif
#include "arrow/dataset/file_ipc.h"
#ifdef ARROW_ORC
#include "arrow/dataset/file_orc.h"
#endif
#ifdef ARROW_PARQUET
#include "arrow/dataset/file_parquet.h"
#endif
#include "arrow/dataset/scanner.h"
