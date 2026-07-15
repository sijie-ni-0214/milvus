// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace milvus::memory_usage {

inline size_t
StringDynamicBytes(const std::string& value) {
    if (value.empty()) {
        return 0;
    }
    const auto data = reinterpret_cast<uintptr_t>(value.data());
    const auto begin = reinterpret_cast<uintptr_t>(&value);
    const auto end = begin + sizeof(value);
    return data >= begin && data < end ? 0 : value.capacity() + 1;
}

template <typename T>
size_t
VectorCapacityBytes(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

template <typename Key, typename Value>
size_t
UnorderedMapBucketBytes(const std::unordered_map<Key, Value>& values) {
    return values.bucket_count() * sizeof(void*);
}

template <typename Key, typename Value>
size_t
UnorderedMapValueBytes(const std::unordered_map<Key, Value>& values) {
    return values.size() *
           sizeof(typename std::unordered_map<Key, Value>::value_type);
}

template <typename Key, typename Value>
size_t
UnorderedMapNodeOverheadEstimatedBytes(
    const std::unordered_map<Key, Value>& values) {
    return values.size() * 2 * sizeof(void*);
}

template <typename Key>
size_t
UnorderedSetBucketBytes(const std::unordered_set<Key>& values) {
    return values.bucket_count() * sizeof(void*);
}

template <typename Key>
size_t
UnorderedSetValueBytes(const std::unordered_set<Key>& values) {
    return values.size() * sizeof(Key);
}

template <typename Key>
size_t
UnorderedSetNodeOverheadEstimatedBytes(const std::unordered_set<Key>& values) {
    return values.size() * 2 * sizeof(void*);
}

template <typename Key, typename Value>
size_t
MapValueBytes(const std::map<Key, Value>& values) {
    return values.size() * sizeof(typename std::map<Key, Value>::value_type);
}

template <typename Key, typename Value>
size_t
MapNodeOverheadEstimatedBytes(const std::map<Key, Value>& values) {
    return values.size() * 4 * sizeof(void*);
}

inline constexpr size_t
SharedPtrControlBlockEstimatedBytes() {
    return 4 * sizeof(void*);
}

}  // namespace milvus::memory_usage
