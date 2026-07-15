// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <google/protobuf/text_format.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "common/EasyAssert.h"
#include "common/MemoryUsage.h"
#include "glog/logging.h"
#include "log/Log.h"
#include "monitor/segcore_memory_stats_c.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "segcore/Collection.h"

namespace milvus::segcore {

namespace {

struct CollectionRegistry {
    std::mutex mutex;
    std::vector<Collection*> collections;
};

CollectionRegistry&
GetCollectionRegistry() {
    static auto* registry = new CollectionRegistry();
    return *registry;
}

void
RegisterCollection(Collection* collection) {
    auto& registry = GetCollectionRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.collections.push_back(collection);
}

void
UnregisterCollection(Collection* collection) {
    auto& registry = GetCollectionRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto it = std::find(
        registry.collections.begin(), registry.collections.end(), collection);
    if (it != registry.collections.end()) {
        registry.collections.erase(it);
    }
}

void
CopyLabel(char* dst, size_t dst_size, std::string_view src) {
    if (dst_size == 0) {
        return;
    }
    const auto size = std::min(dst_size - 1, src.size());
    std::memcpy(dst, src.data(), size);
    dst[size] = '\0';
}

bool
SameLabel(const char* current, std::string_view expected) {
    return std::string_view(current) == expected;
}

void
AddMemoryStat(SegcoreCollectionMemoryStats& stats,
              std::string_view owner,
              std::string_view field_name,
              std::string_view data_type,
              std::string_view component,
              std::string_view accuracy,
              size_t count,
              size_t bytes) {
    if (count == 0 && bytes == 0) {
        return;
    }
    for (size_t i = 0; i < stats.entry_count; ++i) {
        auto& entry = stats.entries[i];
        if (SameLabel(entry.owner, owner) &&
            SameLabel(entry.field_name, field_name) &&
            SameLabel(entry.data_type, data_type) &&
            SameLabel(entry.component, component) &&
            SameLabel(entry.accuracy, accuracy)) {
            entry.count += count;
            entry.bytes += bytes;
            return;
        }
    }
    if (stats.entry_count >= SEGCORE_COLLECTION_MEMORY_STATS_MAX_ENTRIES) {
        ++stats.overflow_count;
        return;
    }
    auto& entry = stats.entries[stats.entry_count++];
    CopyLabel(entry.owner, sizeof(entry.owner), owner);
    CopyLabel(entry.field_name, sizeof(entry.field_name), field_name);
    CopyLabel(entry.data_type, sizeof(entry.data_type), data_type);
    CopyLabel(entry.component, sizeof(entry.component), component);
    CopyLabel(entry.accuracy, sizeof(entry.accuracy), accuracy);
    entry.count = count;
    entry.bytes = bytes;
}

void
AddFieldMetaStats(SegcoreCollectionMemoryStats& stats,
                  const FieldMeta& field_meta) {
    const auto usage = field_meta.MemoryUsage();
    const auto& field_name = field_meta.get_name().get();
    const auto data_type = GetDataTypeName(field_meta.get_data_type());
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "object",
                  "exact",
                  1,
                  usage.object_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "name_dynamic",
                  "capacity",
                  1,
                  usage.name_dynamic_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "default_value_dynamic",
                  "capacity",
                  1,
                  usage.default_value_dynamic_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "string_params_values",
                  "exact",
                  usage.string_params_count,
                  usage.string_params_value_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "string_params_node_overhead",
                  "estimated",
                  usage.string_params_count,
                  usage.string_params_node_overhead_estimated_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "string_params_string_dynamic",
                  "capacity",
                  usage.string_params_count * 2,
                  usage.string_params_string_dynamic_bytes);
    AddMemoryStat(stats,
                  "field_meta",
                  field_name,
                  data_type,
                  "external_mapping_dynamic",
                  "capacity",
                  1,
                  usage.external_field_mapping_dynamic_bytes);
    AddMemoryStat(stats,
                  "schema_field",
                  field_name,
                  data_type,
                  "fields_map_key",
                  "exact",
                  1,
                  sizeof(FieldId));
    AddMemoryStat(stats,
                  "schema_field",
                  field_name,
                  data_type,
                  "fields_map_node_overhead",
                  "estimated",
                  1,
                  2 * sizeof(void*));
}

void
AddFieldIndexMetaStats(SegcoreCollectionMemoryStats& stats,
                       std::string_view field_name,
                       std::string_view data_type,
                       const FieldIndexMeta& field_meta) {
    const auto usage = field_meta.MemoryUsage();
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "object",
                  "exact",
                  1,
                  usage.object_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "index_params_values",
                  "exact",
                  usage.index_params_count,
                  usage.index_params_value_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "index_params_node_overhead",
                  "estimated",
                  usage.index_params_count,
                  usage.index_params_node_overhead_estimated_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "index_params_string_dynamic",
                  "capacity",
                  usage.index_params_count * 2,
                  usage.index_params_string_dynamic_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "type_params_values",
                  "exact",
                  usage.type_params_count,
                  usage.type_params_value_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "type_params_node_overhead",
                  "estimated",
                  usage.type_params_count,
                  usage.type_params_node_overhead_estimated_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "type_params_string_dynamic",
                  "capacity",
                  usage.type_params_count * 2,
                  usage.type_params_string_dynamic_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "user_index_params_values",
                  "exact",
                  usage.user_index_params_count,
                  usage.user_index_params_value_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "user_index_params_node_overhead",
                  "estimated",
                  usage.user_index_params_count,
                  usage.user_index_params_node_overhead_estimated_bytes);
    AddMemoryStat(stats,
                  "field_index_meta",
                  field_name,
                  data_type,
                  "user_index_params_string_dynamic",
                  "capacity",
                  usage.user_index_params_count * 2,
                  usage.user_index_params_string_dynamic_bytes);
    AddMemoryStat(stats,
                  "collection_index_field",
                  field_name,
                  data_type,
                  "field_metas_map_key",
                  "exact",
                  1,
                  sizeof(FieldId));
    AddMemoryStat(stats,
                  "collection_index_field",
                  field_name,
                  data_type,
                  "field_metas_map_node_overhead",
                  "estimated",
                  1,
                  4 * sizeof(void*));
}

void
AddSchemaStats(SegcoreCollectionMemoryStats& stats, const Schema& schema) {
    const auto usage = schema.MemoryUsage();
    const auto add = [&](std::string_view component,
                         std::string_view accuracy,
                         size_t count,
                         size_t bytes) {
        AddMemoryStat(
            stats, "schema", "", "", component, accuracy, count, bytes);
    };
    add("object", "exact", 1, usage.object_bytes);
    add("shared_ptr_control_block",
        "estimated",
        1,
        usage.shared_ptr_control_block_estimated_bytes);
    add("field_ids_capacity",
        "capacity",
        usage.field_ids_capacity,
        usage.field_ids_capacity_bytes);
    add("fields_buckets",
        "capacity",
        usage.fields_bucket_count,
        usage.fields_bucket_bytes);
    add("name_ids_values",
        "exact",
        usage.name_ids_count,
        usage.name_ids_value_bytes);
    add("name_ids_buckets",
        "capacity",
        usage.name_ids_bucket_count,
        usage.name_ids_bucket_bytes);
    add("name_ids_node_overhead",
        "estimated",
        usage.name_ids_count,
        usage.name_ids_node_overhead_estimated_bytes);
    add("name_ids_string_dynamic",
        "capacity",
        usage.name_ids_count,
        usage.name_ids_string_dynamic_bytes);
    add("id_names_values",
        "exact",
        usage.id_names_count,
        usage.id_names_value_bytes);
    add("id_names_buckets",
        "capacity",
        usage.id_names_bucket_count,
        usage.id_names_bucket_bytes);
    add("id_names_node_overhead",
        "estimated",
        usage.id_names_count,
        usage.id_names_node_overhead_estimated_bytes);
    add("id_names_string_dynamic",
        "capacity",
        usage.id_names_count,
        usage.id_names_string_dynamic_bytes);
    add("load_fields_values",
        "exact",
        usage.load_fields_count,
        usage.load_fields_value_bytes);
    add("load_fields_buckets",
        "capacity",
        usage.load_fields_bucket_count,
        usage.load_fields_bucket_bytes);
    add("load_fields_node_overhead",
        "estimated",
        usage.load_fields_count,
        usage.load_fields_node_overhead_estimated_bytes);
    add("bm25_fields_values",
        "exact",
        usage.bm25_fields_count,
        usage.bm25_fields_value_bytes);
    add("bm25_fields_buckets",
        "capacity",
        usage.bm25_fields_bucket_count,
        usage.bm25_fields_bucket_bytes);
    add("bm25_fields_node_overhead",
        "estimated",
        usage.bm25_fields_count,
        usage.bm25_fields_node_overhead_estimated_bytes);
    add("mmap_fields_values",
        "exact",
        usage.mmap_fields_count,
        usage.mmap_fields_value_bytes);
    add("mmap_fields_buckets",
        "capacity",
        usage.mmap_fields_bucket_count,
        usage.mmap_fields_bucket_bytes);
    add("mmap_fields_node_overhead",
        "estimated",
        usage.mmap_fields_count,
        usage.mmap_fields_node_overhead_estimated_bytes);
    add("struct_array_cache_values",
        "exact",
        usage.struct_array_cache_count,
        usage.struct_array_cache_value_bytes);
    add("struct_array_cache_buckets",
        "capacity",
        usage.struct_array_cache_bucket_count,
        usage.struct_array_cache_bucket_bytes);
    add("struct_array_cache_node_overhead",
        "estimated",
        usage.struct_array_cache_count,
        usage.struct_array_cache_node_overhead_estimated_bytes);
    add("struct_array_cache_string_dynamic",
        "capacity",
        usage.struct_array_cache_count,
        usage.struct_array_cache_string_dynamic_bytes);
    add("warmup_fields_values",
        "exact",
        usage.warmup_fields_count,
        usage.warmup_fields_value_bytes);
    add("warmup_fields_buckets",
        "capacity",
        usage.warmup_fields_bucket_count,
        usage.warmup_fields_bucket_bytes);
    add("warmup_fields_node_overhead",
        "estimated",
        usage.warmup_fields_count,
        usage.warmup_fields_node_overhead_estimated_bytes);
    add("warmup_fields_string_dynamic",
        "capacity",
        usage.warmup_fields_count,
        usage.warmup_fields_string_dynamic_bytes);
    add("warmup_policy_string_dynamic",
        "capacity",
        4,
        usage.warmup_policy_string_dynamic_bytes);
    add("external_string_dynamic",
        "capacity",
        2,
        usage.external_string_dynamic_bytes);
    add("arrow_schema_object",
        "exact",
        usage.arrow_schema_count,
        usage.arrow_schema_object_bytes);
    add("arrow_schema_control_block",
        "estimated",
        usage.arrow_schema_count,
        usage.arrow_schema_control_block_estimated_bytes);
    add("arrow_field_vector_capacity",
        "capacity",
        usage.arrow_field_vector_capacity,
        usage.arrow_field_vector_capacity_bytes);
    add("arrow_field_object",
        "exact",
        usage.arrow_field_count,
        usage.arrow_field_object_bytes);
    add("arrow_field_control_block",
        "estimated",
        usage.arrow_field_count,
        usage.arrow_field_control_block_estimated_bytes);
    add("arrow_field_name_dynamic",
        "capacity",
        usage.arrow_field_count,
        usage.arrow_field_name_dynamic_bytes);

    for (const auto& [field_id, field_meta] : schema.get_fields()) {
        (void)field_id;
        AddFieldMetaStats(stats, field_meta);
    }
}

void
AddIndexMetaStats(SegcoreCollectionMemoryStats& stats,
                  const Schema& schema,
                  const CollectionIndexMeta& index_meta) {
    const auto usage = index_meta.MemoryUsage();
    AddMemoryStat(stats,
                  "collection_index_meta",
                  "",
                  "",
                  "object",
                  "exact",
                  1,
                  usage.object_bytes);
    AddMemoryStat(stats,
                  "collection_index_meta",
                  "",
                  "",
                  "shared_ptr_control_block",
                  "estimated",
                  1,
                  usage.shared_ptr_control_block_estimated_bytes);
    for (const auto& [field_id, field_meta] : index_meta.GetFieldIndexMetas()) {
        const auto schema_it = schema.get_fields().find(field_id);
        if (schema_it == schema.get_fields().end()) {
            AddFieldIndexMetaStats(stats, "unknown", "unknown", field_meta);
            continue;
        }
        AddFieldIndexMetaStats(
            stats,
            schema_it->second.get_name().get(),
            GetDataTypeName(schema_it->second.get_data_type()),
            field_meta);
    }
}

}  // namespace

Collection::Collection(const milvus::proto::schema::CollectionSchema* schema) {
    Assert(schema != nullptr);
    collection_name_ = schema->name();
    schema_ = Schema::ParseFrom(*schema);
    RegisterCollection(this);
}

Collection::Collection(const std::string_view schema_proto) {
    milvus::proto::schema::CollectionSchema collection_schema;
    auto suc = google::protobuf::TextFormat::ParseFromString(
        std::string(schema_proto), &collection_schema);
    if (!suc) {
        LOG_WARN("unmarshal schema string failed");
    }
    schema_ = Schema::ParseFrom(collection_schema);
    collection_name_ = std::move(*collection_schema.mutable_name());
    RegisterCollection(this);
}

Collection::Collection(const void* schema_proto, const int64_t length) {
    Assert(schema_proto != nullptr);
    milvus::proto::schema::CollectionSchema collection_schema;
    auto suc = collection_schema.ParseFromArray(schema_proto, length);
    if (!suc) {
        LOG_WARN("unmarshal schema string failed");
    }

    schema_ = Schema::ParseFrom(collection_schema);
    collection_name_ = std::move(*collection_schema.mutable_name());
    RegisterCollection(this);
}

Collection::~Collection() {
    UnregisterCollection(this);
}

CollectionMemoryUsage
Collection::MemoryUsage() const {
    CollectionMemoryUsage usage;
    usage.object_bytes = sizeof(Collection);
    usage.collection_name_dynamic_bytes =
        memory_usage::StringDynamicBytes(collection_name_);
    return usage;
}

void
Collection::parseIndexMeta(const void* index_proto, const int64_t length) {
    Assert(index_proto != nullptr);

    milvus::proto::segcore::CollectionIndexMeta indexMeta;
    auto suc = indexMeta.ParseFromArray(index_proto, length);

    if (!suc) {
        LOG_ERROR("unmarshal index meta string failed");
        return;
    }

    auto new_index_meta = std::make_shared<CollectionIndexMeta>(indexMeta);
    LOG_INFO("index meta info: {}", new_index_meta->ToString());
    set_index_meta(new_index_meta);
}

void
Collection::parse_schema(const void* schema_proto_blob,
                         const int64_t length,
                         const uint64_t version) {
    Assert(schema_proto_blob != nullptr);

    if (version <= get_schema_version()) {
        return;
    }

    milvus::proto::schema::CollectionSchema collection_schema;
    auto suc = collection_schema.ParseFromArray(schema_proto_blob, length);

    AssertInfo(suc, "parse schema proto failed");

    auto new_schema = Schema::ParseFrom(collection_schema);
    new_schema->set_schema_version(version);
    set_schema(new_schema);
}

}  // namespace milvus::segcore

SegcoreCollectionMemoryStats
GetSegcoreCollectionMemoryStats() {
    SegcoreCollectionMemoryStats stats;
    std::memset(&stats, 0, sizeof(stats));

    auto& registry = milvus::segcore::GetCollectionRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (const auto* collection : registry.collections) {
        const auto collection_usage = collection->MemoryUsage();
        milvus::segcore::AddMemoryStat(stats,
                                       "collection",
                                       "",
                                       "",
                                       "object",
                                       "exact",
                                       1,
                                       collection_usage.object_bytes);
        milvus::segcore::AddMemoryStat(
            stats,
            "collection",
            "",
            "",
            "collection_name_dynamic",
            "capacity",
            1,
            collection_usage.collection_name_dynamic_bytes);
        const auto schema = collection->get_schema();
        if (schema != nullptr) {
            milvus::segcore::AddSchemaStats(stats, *schema);
        }
        const auto index_meta = collection->get_index_meta();
        if (schema != nullptr && index_meta != nullptr) {
            milvus::segcore::AddIndexMetaStats(stats, *schema, *index_meta);
        }
    }
    return stats;
}
