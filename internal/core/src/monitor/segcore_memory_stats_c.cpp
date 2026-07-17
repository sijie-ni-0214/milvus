// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "monitor/segcore_memory_stats_c.h"

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace {

std::atomic<int64_t> g_sealed_segment_count{0};
std::atomic<int64_t> g_sealed_segment_object_bytes{0};
std::atomic<int64_t> g_sealed_segment_runtime_estimated_bytes{0};
std::atomic<int64_t> g_sealed_segment_mmap_descriptor_bytes{0};
std::atomic<int64_t> g_sealed_segment_empty_indexing_container_bytes{0};
std::atomic<int64_t> g_sealed_segment_insert_record_bytes{0};
std::atomic<int64_t> g_sealed_segment_deleted_record_bytes{0};
std::atomic<int64_t> g_sealed_segment_load_field_data_info_bytes{0};
std::atomic<int64_t> g_sealed_segment_field_map_bytes{0};
std::atomic<int64_t> g_sealed_segment_field_shared_ptr_control_block_bytes{0};
std::atomic<int64_t> g_sealed_segment_field_data_accounted_map_bytes{0};
std::atomic<int64_t> g_sealed_segment_mmap_field_ids_bytes{0};
std::atomic<int64_t> g_segment_load_info_bytes{0};
std::atomic<int64_t> g_segment_load_info_estimated_bytes{0};
std::atomic<int64_t> g_segment_load_info_object_bytes{0};
std::atomic<int64_t> g_segment_load_info_proto_bytes{0};
std::atomic<int64_t> g_segment_load_info_converted_index_cache_bytes{0};
std::atomic<int64_t> g_segment_load_info_field_index_id_cache_bytes{0};
std::atomic<int64_t> g_segment_load_info_field_index_has_raw_data_bytes{0};
std::atomic<int64_t> g_segment_load_info_fields_filled_with_default_bytes{0};
std::atomic<int64_t> g_segment_load_info_field_binlog_cache_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_deep_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_path_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_property_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_column_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_format_bytes{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_group_count{0};
std::atomic<int64_t> g_segment_load_info_column_group_cache_file_count{0};
std::atomic<int64_t> g_segment_load_info_created_text_indexes_bytes{0};
std::atomic<int64_t> g_field_entry_count{0};

std::atomic<int64_t> g_lazy_manifest_group_count{0};
std::atomic<int64_t> g_lazy_manifest_group_estimated_bytes{0};
std::atomic<int64_t> g_lazy_manifest_proxy_count{0};
std::atomic<int64_t> g_lazy_manifest_proxy_object_bytes{0};
std::atomic<int64_t> g_lazy_manifest_projected_column_count{0};

std::atomic<int64_t> g_deferred_business_index_count{0};
std::atomic<int64_t> g_deferred_business_index_estimated_bytes{0};
std::atomic<int64_t> g_deferred_text_index_count{0};
std::atomic<int64_t> g_deferred_text_index_estimated_bytes{0};

std::atomic<int64_t> g_pk_index_slot_count{0};
std::atomic<int64_t> g_timestamp_index_slot_count{0};
std::atomic<int64_t> g_pk_index_translator_count{0};
std::atomic<int64_t> g_pk_index_translator_object_bytes{0};
std::atomic<int64_t> g_timestamp_index_translator_count{0};
std::atomic<int64_t> g_timestamp_index_translator_object_bytes{0};
std::atomic<int64_t> g_pk_index_cell_count{0};
std::atomic<int64_t> g_pk_index_cell_bytes{0};
std::atomic<int64_t> g_timestamp_index_cell_count{0};
std::atomic<int64_t> g_timestamp_index_cell_bytes{0};

struct IndexMemoryStats {
    int64_t count = 0;
    int64_t estimated_bytes = 0;
    int64_t object_bytes = 0;
    int64_t index_info_dynamic_bytes = 0;
    int64_t file_manager_context_dynamic_bytes = 0;
    int64_t config_estimated_bytes = 0;
    int64_t index_load_info_dynamic_bytes = 0;
    int64_t index_param_dynamic_bytes = 0;
    int64_t schema_proto_bytes = 0;
    int64_t string_dynamic_bytes = 0;
};

std::mutex g_index_memory_mutex;
std::map<std::pair<std::string, std::string>, IndexMemoryStats>
    g_index_memory_stats;

void
add(std::atomic<int64_t>& value, int64_t delta) {
    value.fetch_add(delta, std::memory_order_relaxed);
}

uint64_t
load_non_negative(const std::atomic<int64_t>& value) {
    auto current = value.load(std::memory_order_relaxed);
    return current > 0 ? static_cast<uint64_t>(current) : 0;
}

uint64_t
non_negative(int64_t value) {
    return value > 0 ? static_cast<uint64_t>(value) : 0;
}

void
copy_label(char* dst, size_t dst_size, const std::string& src) {
    if (dst_size == 0) {
        return;
    }
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
}

}  // namespace

SegcoreMemoryStats
GetSegcoreMemoryStats() {
    SegcoreMemoryStats stats;
    std::memset(&stats, 0, sizeof(SegcoreMemoryStats));

    stats.sealed_segment_count = load_non_negative(g_sealed_segment_count);
    stats.sealed_segment_object_bytes =
        load_non_negative(g_sealed_segment_object_bytes);
    stats.sealed_segment_runtime_estimated_bytes =
        load_non_negative(g_sealed_segment_runtime_estimated_bytes);
    stats.sealed_segment_mmap_descriptor_bytes =
        load_non_negative(g_sealed_segment_mmap_descriptor_bytes);
    stats.sealed_segment_empty_indexing_container_bytes =
        load_non_negative(g_sealed_segment_empty_indexing_container_bytes);
    stats.sealed_segment_insert_record_bytes =
        load_non_negative(g_sealed_segment_insert_record_bytes);
    stats.sealed_segment_deleted_record_bytes =
        load_non_negative(g_sealed_segment_deleted_record_bytes);
    stats.sealed_segment_load_field_data_info_bytes =
        load_non_negative(g_sealed_segment_load_field_data_info_bytes);
    stats.sealed_segment_field_map_bytes =
        load_non_negative(g_sealed_segment_field_map_bytes);
    stats.sealed_segment_field_shared_ptr_control_block_bytes =
        load_non_negative(
            g_sealed_segment_field_shared_ptr_control_block_bytes);
    stats.sealed_segment_field_data_accounted_map_bytes =
        load_non_negative(g_sealed_segment_field_data_accounted_map_bytes);
    stats.sealed_segment_mmap_field_ids_bytes =
        load_non_negative(g_sealed_segment_mmap_field_ids_bytes);
    stats.segment_load_info_bytes =
        load_non_negative(g_segment_load_info_bytes);
    stats.segment_load_info_estimated_bytes =
        load_non_negative(g_segment_load_info_estimated_bytes);
    stats.segment_load_info_object_bytes =
        load_non_negative(g_segment_load_info_object_bytes);
    stats.segment_load_info_proto_bytes =
        load_non_negative(g_segment_load_info_proto_bytes);
    stats.segment_load_info_converted_index_cache_bytes =
        load_non_negative(g_segment_load_info_converted_index_cache_bytes);
    stats.segment_load_info_field_index_id_cache_bytes =
        load_non_negative(g_segment_load_info_field_index_id_cache_bytes);
    stats.segment_load_info_field_index_has_raw_data_bytes =
        load_non_negative(g_segment_load_info_field_index_has_raw_data_bytes);
    stats.segment_load_info_fields_filled_with_default_bytes =
        load_non_negative(g_segment_load_info_fields_filled_with_default_bytes);
    stats.segment_load_info_field_binlog_cache_bytes =
        load_non_negative(g_segment_load_info_field_binlog_cache_bytes);
    stats.segment_load_info_column_group_cache_bytes =
        load_non_negative(g_segment_load_info_column_group_cache_bytes);
    stats.segment_load_info_column_group_cache_deep_bytes =
        load_non_negative(g_segment_load_info_column_group_cache_deep_bytes);
    stats.segment_load_info_column_group_cache_path_bytes =
        load_non_negative(g_segment_load_info_column_group_cache_path_bytes);
    stats.segment_load_info_column_group_cache_property_bytes =
        load_non_negative(
            g_segment_load_info_column_group_cache_property_bytes);
    stats.segment_load_info_column_group_cache_column_bytes =
        load_non_negative(g_segment_load_info_column_group_cache_column_bytes);
    stats.segment_load_info_column_group_cache_format_bytes =
        load_non_negative(g_segment_load_info_column_group_cache_format_bytes);
    stats.segment_load_info_column_group_cache_group_count =
        load_non_negative(g_segment_load_info_column_group_cache_group_count);
    stats.segment_load_info_column_group_cache_file_count =
        load_non_negative(g_segment_load_info_column_group_cache_file_count);
    stats.segment_load_info_created_text_indexes_bytes =
        load_non_negative(g_segment_load_info_created_text_indexes_bytes);
    stats.field_entry_count = load_non_negative(g_field_entry_count);

    stats.lazy_manifest_group_count =
        load_non_negative(g_lazy_manifest_group_count);
    stats.lazy_manifest_group_estimated_bytes =
        load_non_negative(g_lazy_manifest_group_estimated_bytes);
    stats.lazy_manifest_proxy_count =
        load_non_negative(g_lazy_manifest_proxy_count);
    stats.lazy_manifest_proxy_object_bytes =
        load_non_negative(g_lazy_manifest_proxy_object_bytes);
    stats.lazy_manifest_projected_column_count =
        load_non_negative(g_lazy_manifest_projected_column_count);

    stats.deferred_business_index_count =
        load_non_negative(g_deferred_business_index_count);
    stats.deferred_business_index_estimated_bytes =
        load_non_negative(g_deferred_business_index_estimated_bytes);
    stats.deferred_text_index_count =
        load_non_negative(g_deferred_text_index_count);
    stats.deferred_text_index_estimated_bytes =
        load_non_negative(g_deferred_text_index_estimated_bytes);

    stats.pk_index_slot_count = load_non_negative(g_pk_index_slot_count);
    stats.timestamp_index_slot_count =
        load_non_negative(g_timestamp_index_slot_count);
    stats.pk_index_translator_count =
        load_non_negative(g_pk_index_translator_count);
    stats.pk_index_translator_object_bytes =
        load_non_negative(g_pk_index_translator_object_bytes);
    stats.timestamp_index_translator_count =
        load_non_negative(g_timestamp_index_translator_count);
    stats.timestamp_index_translator_object_bytes =
        load_non_negative(g_timestamp_index_translator_object_bytes);
    stats.pk_index_cell_count = load_non_negative(g_pk_index_cell_count);
    stats.pk_index_cell_bytes = load_non_negative(g_pk_index_cell_bytes);
    stats.timestamp_index_cell_count =
        load_non_negative(g_timestamp_index_cell_count);
    stats.timestamp_index_cell_bytes =
        load_non_negative(g_timestamp_index_cell_bytes);

    return stats;
}

SegcoreIndexMemoryStats
GetSegcoreIndexMemoryStats() {
    SegcoreIndexMemoryStats stats;
    std::memset(&stats, 0, sizeof(SegcoreIndexMemoryStats));

    std::lock_guard<std::mutex> lock(g_index_memory_mutex);
    auto index = 0;
    for (const auto& [key, value] : g_index_memory_stats) {
        if (value.count <= 0 && value.estimated_bytes <= 0) {
            continue;
        }
        if (index >= SEGCORE_INDEX_MEMORY_STATS_MAX_ENTRIES) {
            ++stats.overflow_count;
            continue;
        }
        auto& entry = stats.entries[index++];
        copy_label(entry.data_type, sizeof(entry.data_type), key.first);
        copy_label(entry.index_type, sizeof(entry.index_type), key.second);
        entry.count = non_negative(value.count);
        entry.estimated_bytes = non_negative(value.estimated_bytes);
        entry.object_bytes = non_negative(value.object_bytes);
        entry.index_info_dynamic_bytes =
            non_negative(value.index_info_dynamic_bytes);
        entry.file_manager_context_dynamic_bytes =
            non_negative(value.file_manager_context_dynamic_bytes);
        entry.config_estimated_bytes =
            non_negative(value.config_estimated_bytes);
        entry.index_load_info_dynamic_bytes =
            non_negative(value.index_load_info_dynamic_bytes);
        entry.index_param_dynamic_bytes =
            non_negative(value.index_param_dynamic_bytes);
        entry.schema_proto_bytes = non_negative(value.schema_proto_bytes);
        entry.string_dynamic_bytes = non_negative(value.string_dynamic_bytes);
    }
    stats.entry_count = static_cast<uint64_t>(index);
    return stats;
}

namespace milvus::monitor {

void
UpdateSegcoreSealedSegment(int64_t count_delta, int64_t object_bytes_delta) {
    add(g_sealed_segment_count, count_delta);
    add(g_sealed_segment_object_bytes, object_bytes_delta);
}

void
UpdateSegcoreSealedSegmentRuntime(int64_t estimated_bytes_delta,
                                  int64_t mmap_descriptor_bytes_delta,
                                  int64_t empty_indexing_container_bytes_delta,
                                  int64_t insert_record_bytes_delta,
                                  int64_t deleted_record_bytes_delta,
                                  int64_t load_field_data_info_bytes_delta) {
    add(g_sealed_segment_runtime_estimated_bytes, estimated_bytes_delta);
    add(g_sealed_segment_mmap_descriptor_bytes, mmap_descriptor_bytes_delta);
    add(g_sealed_segment_empty_indexing_container_bytes,
        empty_indexing_container_bytes_delta);
    add(g_sealed_segment_insert_record_bytes, insert_record_bytes_delta);
    add(g_sealed_segment_deleted_record_bytes, deleted_record_bytes_delta);
    add(g_sealed_segment_load_field_data_info_bytes,
        load_field_data_info_bytes_delta);
}

void
UpdateSegcoreSealedSegmentFieldRuntime(
    int64_t field_map_bytes_delta,
    int64_t field_shared_ptr_control_block_bytes_delta,
    int64_t field_data_accounted_map_bytes_delta,
    int64_t mmap_field_ids_bytes_delta) {
    add(g_sealed_segment_field_map_bytes, field_map_bytes_delta);
    add(g_sealed_segment_field_shared_ptr_control_block_bytes,
        field_shared_ptr_control_block_bytes_delta);
    add(g_sealed_segment_field_data_accounted_map_bytes,
        field_data_accounted_map_bytes_delta);
    add(g_sealed_segment_mmap_field_ids_bytes, mmap_field_ids_bytes_delta);
}

void
UpdateSegcoreSegmentLoadInfoBytes(int64_t bytes_delta) {
    add(g_segment_load_info_bytes, bytes_delta);
}

void
UpdateSegcoreSegmentLoadInfoBreakdown(
    int64_t estimated_bytes_delta,
    int64_t object_bytes_delta,
    int64_t proto_bytes_delta,
    int64_t converted_index_cache_bytes_delta,
    int64_t field_index_id_cache_bytes_delta,
    int64_t field_index_has_raw_data_bytes_delta,
    int64_t fields_filled_with_default_bytes_delta,
    int64_t field_binlog_cache_bytes_delta,
    int64_t column_group_cache_bytes_delta,
    int64_t column_group_cache_deep_bytes_delta,
    int64_t column_group_cache_path_bytes_delta,
    int64_t column_group_cache_property_bytes_delta,
    int64_t column_group_cache_column_bytes_delta,
    int64_t column_group_cache_format_bytes_delta,
    int64_t column_group_cache_group_count_delta,
    int64_t column_group_cache_file_count_delta,
    int64_t created_text_indexes_bytes_delta) {
    add(g_segment_load_info_estimated_bytes, estimated_bytes_delta);
    add(g_segment_load_info_object_bytes, object_bytes_delta);
    add(g_segment_load_info_proto_bytes, proto_bytes_delta);
    add(g_segment_load_info_converted_index_cache_bytes,
        converted_index_cache_bytes_delta);
    add(g_segment_load_info_field_index_id_cache_bytes,
        field_index_id_cache_bytes_delta);
    add(g_segment_load_info_field_index_has_raw_data_bytes,
        field_index_has_raw_data_bytes_delta);
    add(g_segment_load_info_fields_filled_with_default_bytes,
        fields_filled_with_default_bytes_delta);
    add(g_segment_load_info_field_binlog_cache_bytes,
        field_binlog_cache_bytes_delta);
    add(g_segment_load_info_column_group_cache_bytes,
        column_group_cache_bytes_delta);
    add(g_segment_load_info_column_group_cache_deep_bytes,
        column_group_cache_deep_bytes_delta);
    add(g_segment_load_info_column_group_cache_path_bytes,
        column_group_cache_path_bytes_delta);
    add(g_segment_load_info_column_group_cache_property_bytes,
        column_group_cache_property_bytes_delta);
    add(g_segment_load_info_column_group_cache_column_bytes,
        column_group_cache_column_bytes_delta);
    add(g_segment_load_info_column_group_cache_format_bytes,
        column_group_cache_format_bytes_delta);
    add(g_segment_load_info_column_group_cache_group_count,
        column_group_cache_group_count_delta);
    add(g_segment_load_info_column_group_cache_file_count,
        column_group_cache_file_count_delta);
    add(g_segment_load_info_created_text_indexes_bytes,
        created_text_indexes_bytes_delta);
}

void
UpdateSegcoreFieldEntryCount(int64_t count_delta) {
    add(g_field_entry_count, count_delta);
}

void
UpdateSegcoreLazyManifestGroup(int64_t count_delta,
                               int64_t estimated_bytes_delta,
                               int64_t projected_column_count_delta) {
    add(g_lazy_manifest_group_count, count_delta);
    add(g_lazy_manifest_group_estimated_bytes, estimated_bytes_delta);
    add(g_lazy_manifest_projected_column_count, projected_column_count_delta);
}

void
UpdateSegcoreLazyManifestProxy(int64_t count_delta,
                               int64_t object_bytes_delta) {
    add(g_lazy_manifest_proxy_count, count_delta);
    add(g_lazy_manifest_proxy_object_bytes, object_bytes_delta);
}

void
UpdateSegcoreDeferredBusinessIndex(int64_t count_delta,
                                   int64_t estimated_bytes_delta) {
    add(g_deferred_business_index_count, count_delta);
    add(g_deferred_business_index_estimated_bytes, estimated_bytes_delta);
}

void
UpdateSegcoreDeferredTextIndex(int64_t count_delta,
                               int64_t estimated_bytes_delta) {
    add(g_deferred_text_index_count, count_delta);
    add(g_deferred_text_index_estimated_bytes, estimated_bytes_delta);
}

void
UpdateSegcorePkIndexSlot(int64_t count_delta) {
    add(g_pk_index_slot_count, count_delta);
}

void
UpdateSegcoreTimestampIndexSlot(int64_t count_delta) {
    add(g_timestamp_index_slot_count, count_delta);
}

void
UpdateSegcorePkIndexTranslator(int64_t count_delta,
                               int64_t object_bytes_delta) {
    add(g_pk_index_translator_count, count_delta);
    add(g_pk_index_translator_object_bytes, object_bytes_delta);
}

void
UpdateSegcoreTimestampIndexTranslator(int64_t count_delta,
                                      int64_t object_bytes_delta) {
    add(g_timestamp_index_translator_count, count_delta);
    add(g_timestamp_index_translator_object_bytes, object_bytes_delta);
}

void
UpdateSegcorePkIndexCell(int64_t count_delta, int64_t cell_bytes_delta) {
    add(g_pk_index_cell_count, count_delta);
    add(g_pk_index_cell_bytes, cell_bytes_delta);
}

void
UpdateSegcoreTimestampIndexCell(int64_t count_delta, int64_t cell_bytes_delta) {
    add(g_timestamp_index_cell_count, count_delta);
    add(g_timestamp_index_cell_bytes, cell_bytes_delta);
}

void
UpdateSegcoreSealedIndexTranslator(
    const char* data_type,
    const char* index_type,
    int64_t count_delta,
    int64_t estimated_bytes_delta,
    int64_t object_bytes_delta,
    int64_t index_info_dynamic_bytes_delta,
    int64_t file_manager_context_dynamic_bytes_delta,
    int64_t config_estimated_bytes_delta,
    int64_t index_load_info_dynamic_bytes_delta,
    int64_t index_param_dynamic_bytes_delta,
    int64_t schema_proto_bytes_delta,
    int64_t string_dynamic_bytes_delta) {
    std::lock_guard<std::mutex> lock(g_index_memory_mutex);
    auto& stats = g_index_memory_stats[{
        data_type == nullptr ? "unknown" : data_type,
        index_type == nullptr ? "unknown" : index_type,
    }];
    stats.count += count_delta;
    stats.estimated_bytes += estimated_bytes_delta;
    stats.object_bytes += object_bytes_delta;
    stats.index_info_dynamic_bytes += index_info_dynamic_bytes_delta;
    stats.file_manager_context_dynamic_bytes +=
        file_manager_context_dynamic_bytes_delta;
    stats.config_estimated_bytes += config_estimated_bytes_delta;
    stats.index_load_info_dynamic_bytes += index_load_info_dynamic_bytes_delta;
    stats.index_param_dynamic_bytes += index_param_dynamic_bytes_delta;
    stats.schema_proto_bytes += schema_proto_bytes_delta;
    stats.string_dynamic_bytes += string_dynamic_bytes_delta;
}

}  // namespace milvus::monitor
