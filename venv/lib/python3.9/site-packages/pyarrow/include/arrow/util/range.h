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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace arrow {
namespace internal {

/// Create a vector containing the values from start up to stop
template <typename T>
std::vector<T> Iota(T start, T stop) {
  if (start > stop) {
    return {};
  }
  std::vector<T> result(static_cast<size_t>(stop - start));
  std::iota(result.begin(), result.end(), start);
  return result;
}

/// Create a vector containing the values from 0 up to length
template <typename T>
std::vector<T> Iota(T length) {
  return Iota(static_cast<T>(0), length);
}

/// Create a range from a callable which takes a single index parameter
/// and returns the value of iterator on each call and a length.
/// Only iterators obtained from the same range should be compared, the
/// behaviour generally similar to other STL containers.
template <typename Generator>
class LazyRange {
 private:
  // callable which generates the values
  // has to be defined at the beginning of the class for type deduction
  const Generator gen_;
  // the length of the range
  int64_t length_;
#ifdef _MSC_VER
  // workaround to VS2010 not supporting decltype properly
  // see https://stackoverflow.com/questions/21782846/decltype-for-class-member-function
  static Generator gen_static_;
#endif

 public:
#ifdef _MSC_VER
  using return_type = decltype(gen_static_(0));
#else
  using return_type = decltype(gen_(0));
#endif

  /// Construct a new range from a callable and length
  LazyRange(Generator gen, int64_t length) : gen_(gen), length_(length) {}

  // Class of the dependent iterator, created implicitly by begin and end
  class RangeIter {
   public:
    using difference_type = int64_t;
    using value_type = return_type;
    using reference = const value_type&;
    using pointer = const value_type*;
    using iterator_category = std::forward_iterator_tag;

#ifdef _MSC_VER
    // msvc complains about unchecked iterators,
    // see https://stackoverflow.com/questions/21655496/error-c4996-checked-iterators
    using _Unchecked_type = typename LazyRange<Generator>::RangeIter;
#endif

    RangeIter() = delete;
    RangeIter(const RangeIter& other) = default;
    RangeIter& operator=(const RangeIter& other) = default;

    RangeIter(const LazyRange<Generator>& range, int64_t index)
        : range_(&range), index_(index) {}

    const return_type operator*() const { return range_->gen_(index_); }

    RangeIter operator+(difference_type length) const {
      return RangeIter(*range_, index_ + length);
    }

    // pre-increment
    RangeIter& operator++() {
      ++index_;
      return *this;
    }

    // post-increment
    RangeIter operator++(int) {
      auto copy = RangeIter(*this);
      ++index_;
      return copy;
    }

    bool operator==(const typename LazyRange<Generator>::RangeIter& other) const {
      return this->index_ == other.index_ && this->range_ == other.range_;
    }

    bool operator!=(const typename LazyRange<Generator>::RangeIter& other) const {
      return this->index_ != other.index_ || this->range_ != other.range_;
    }

    int64_t operator-(const typename LazyRange<Generator>::RangeIter& other) const {
      return this->index_ - other.index_;
    }

    bool operator<(const typename LazyRange<Generator>::RangeIter& other) const {
      return this->index_ < other.index_;
    }

   private:
    // parent range reference
    const LazyRange* range_;
    // current index
    int64_t index_;
  };

  friend class RangeIter;

  // Create a new begin const iterator
  RangeIter begin() { return RangeIter(*this, 0); }

  // Create a new end const iterator
  RangeIter end() { return RangeIter(*this, length_); }
};

/// Helper function to create a lazy range from a callable (e.g. lambda) and length
template <typename Generator>
LazyRange<Generator> MakeLazyRange(Generator&& gen, int64_t length) {
  return LazyRange<Generator>(std::forward<Generator>(gen), length);
}

}  // namespace internal
}  // namespace arrow
