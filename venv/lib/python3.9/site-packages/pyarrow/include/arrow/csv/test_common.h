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

#include "arrow/csv/parser.h"
#include "arrow/testing/visibility.h"

namespace arrow {
namespace csv {

ARROW_TESTING_EXPORT
std::string MakeCSVData(std::vector<std::string> lines);

// Make a BlockParser from a vector of lines representing a CSV file
ARROW_TESTING_EXPORT
void MakeCSVParser(std::vector<std::string> lines, ParseOptions options, int32_t num_cols,
                   std::shared_ptr<BlockParser>* out);

ARROW_TESTING_EXPORT
void MakeCSVParser(std::vector<std::string> lines, ParseOptions options,
                   std::shared_ptr<BlockParser>* out);

ARROW_TESTING_EXPORT
void MakeCSVParser(std::vector<std::string> lines, std::shared_ptr<BlockParser>* out);

// Make a BlockParser from a vector of strings representing a single CSV column
ARROW_TESTING_EXPORT
void MakeColumnParser(std::vector<std::string> items, std::shared_ptr<BlockParser>* out);

ARROW_TESTING_EXPORT
Result<std::shared_ptr<Buffer>> MakeSampleCsvBuffer(
    size_t num_rows, std::function<bool(size_t row_num)> is_valid = {});

}  // namespace csv
}  // namespace arrow
