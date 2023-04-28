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
#include <cstring>
#include <string>
#include <string_view>

#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace util {

// Convert a UTF8 string to a wstring (either UTF16 or UTF32, depending
// on the wchar_t width).
ARROW_EXPORT Result<std::wstring> UTF8ToWideString(const std::string& source);

// Similarly, convert a wstring to a UTF8 string.
ARROW_EXPORT Result<std::string> WideStringToUTF8(const std::wstring& source);

// This function needs to be called before doing UTF8 validation.
ARROW_EXPORT void InitializeUTF8();

ARROW_EXPORT bool ValidateUTF8(const uint8_t* data, int64_t size);

ARROW_EXPORT bool ValidateUTF8(const std::string_view& str);

// Skip UTF8 byte order mark, if any.
ARROW_EXPORT
Result<const uint8_t*> SkipUTF8BOM(const uint8_t* data, int64_t size);

static constexpr uint32_t kMaxUnicodeCodepoint = 0x110000;

}  // namespace util
}  // namespace arrow
