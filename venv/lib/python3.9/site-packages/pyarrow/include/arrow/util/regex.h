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

#include <cassert>
#include <initializer_list>
#include <regex>
#include <string_view>
#include <type_traits>

#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// Match regex against target and produce string_views out of matches.
inline bool RegexMatch(const std::regex& regex, std::string_view target,
                       std::initializer_list<std::string_view*> out_matches) {
  assert(regex.mark_count() == out_matches.size());

  std::match_results<decltype(target.begin())> match;
  if (!std::regex_match(target.begin(), target.end(), match, regex)) {
    return false;
  }

  // Match #0 is the whole matched sequence
  assert(regex.mark_count() + 1 == match.size());
  auto out_it = out_matches.begin();
  for (size_t i = 1; i < match.size(); ++i) {
    **out_it++ = target.substr(match.position(i), match.length(i));
  }
  return true;
}

}  // namespace internal
}  // namespace arrow
