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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <ostream>
#include <utility>

#include "IndexMeta.h"
#include "NamedType/named_type_impl.hpp"
#include "common/MemoryUsage.h"
#include "common/Types.h"
#include "pb/segcore.pb.h"
#include "protobuf_utils.h"

namespace milvus {

namespace {

size_t
MapStringDynamicBytes(const std::map<std::string, std::string>& values) {
    size_t bytes = 0;
    for (const auto& [key, value] : values) {
        bytes += memory_usage::StringDynamicBytes(key) +
                 memory_usage::StringDynamicBytes(value);
    }
    return bytes;
}

}  // namespace

FieldIndexMeta::FieldIndexMeta(
    FieldId fieldId,
    std::map<std::string, std::string>&& index_params,
    std::map<std::string, std::string>&& type_params) {
    fieldId_ = fieldId;
    index_params_ = std::move(index_params);
    type_params_ = std::move(type_params);
}

FieldIndexMetaMemoryUsage
FieldIndexMeta::MemoryUsage() const {
    FieldIndexMetaMemoryUsage usage;
    usage.object_bytes = sizeof(FieldIndexMeta);
    usage.index_params_count = index_params_.size();
    usage.index_params_value_bytes = memory_usage::MapValueBytes(index_params_);
    usage.index_params_node_overhead_estimated_bytes =
        memory_usage::MapNodeOverheadEstimatedBytes(index_params_);
    usage.index_params_string_dynamic_bytes =
        MapStringDynamicBytes(index_params_);
    usage.type_params_count = type_params_.size();
    usage.type_params_value_bytes = memory_usage::MapValueBytes(type_params_);
    usage.type_params_node_overhead_estimated_bytes =
        memory_usage::MapNodeOverheadEstimatedBytes(type_params_);
    usage.type_params_string_dynamic_bytes =
        MapStringDynamicBytes(type_params_);
    usage.user_index_params_count = user_index_params_.size();
    usage.user_index_params_value_bytes =
        memory_usage::MapValueBytes(user_index_params_);
    usage.user_index_params_node_overhead_estimated_bytes =
        memory_usage::MapNodeOverheadEstimatedBytes(user_index_params_);
    usage.user_index_params_string_dynamic_bytes =
        MapStringDynamicBytes(user_index_params_);
    return usage;
}

FieldIndexMeta::FieldIndexMeta(
    const milvus::proto::segcore::FieldIndexMeta& fieldIndexMeta) {
    fieldId_ = FieldId(fieldIndexMeta.fieldid());
    index_params_ = RepeatedKeyValToMap(fieldIndexMeta.index_params());
    type_params_ = RepeatedKeyValToMap(fieldIndexMeta.type_params());
    user_index_params_ =
        RepeatedKeyValToMap(fieldIndexMeta.user_index_params());
}

CollectionIndexMeta::CollectionIndexMeta(
    int64_t max_index_row_cnt, std::map<FieldId, FieldIndexMeta>&& fieldMetas)
    : max_index_row_cnt_(max_index_row_cnt),
      fieldMetas_(std::move(fieldMetas)) {
}

CollectionIndexMetaMemoryUsage
CollectionIndexMeta::MemoryUsage() const {
    CollectionIndexMetaMemoryUsage usage;
    usage.object_bytes = sizeof(CollectionIndexMeta);
    usage.shared_ptr_control_block_estimated_bytes =
        memory_usage::SharedPtrControlBlockEstimatedBytes();
    usage.field_meta_count = fieldMetas_.size();
    usage.field_meta_key_bytes = fieldMetas_.size() * sizeof(FieldId);
    usage.field_meta_node_overhead_estimated_bytes =
        memory_usage::MapNodeOverheadEstimatedBytes(fieldMetas_);
    return usage;
}

CollectionIndexMeta::CollectionIndexMeta(
    const milvus::proto::segcore::CollectionIndexMeta& collectionIndexMeta) {
    max_index_row_cnt_ = collectionIndexMeta.maxindexrowcount();
    for (const auto& filed_index_meta : collectionIndexMeta.index_metas()) {
        fieldMetas_.emplace(FieldId(filed_index_meta.fieldid()),
                            FieldIndexMeta(filed_index_meta));
    }
}

int64_t
CollectionIndexMeta::GetIndexMaxRowCount() const {
    return max_index_row_cnt_;
}

bool
CollectionIndexMeta::HasField(FieldId fieldId) const {
    return fieldMetas_.count(fieldId);
}

const FieldIndexMeta&
CollectionIndexMeta::GetFieldIndexMeta(FieldId fieldId) const {
    auto it = fieldMetas_.find(fieldId);
    assert(it != fieldMetas_.end());
    return it->second;
}

std::string
CollectionIndexMeta::ToString() {
    std::stringstream ss;
    ss << "maxRowCount : {" << max_index_row_cnt_ << "} ";
    for (const auto& filed_meta : fieldMetas_) {
        ss << "FieldId : {" << abs(filed_meta.first.get()) << " ";
        ss << "IndexParams : { ";
        for (const auto& kv : filed_meta.second.GetIndexParams()) {
            ss << kv.first << " : " << kv.second << ", ";
        }
        ss << " }";
        ss << "TypeParams : {";
        for (const auto& kv : filed_meta.second.GetTypeParams()) {
            ss << kv.first << " : " << kv.second << ", ";
        }
        ss << "}";
        ss << "}";
    }
    return ss.str();
}
}  // namespace milvus
