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

#pragma once

#include <stdint.h>

#define SEGCORE_INDEX_MEMORY_STATS_MAX_ENTRIES 64
#define SEGCORE_COLLECTION_MEMORY_STATS_MAX_ENTRIES 256

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char data_type[24];
    char index_type[80];
    uint64_t count;
    uint64_t estimated_bytes;
    uint64_t object_bytes;
    uint64_t index_info_dynamic_bytes;
    uint64_t file_manager_context_dynamic_bytes;
    uint64_t config_estimated_bytes;
    uint64_t index_load_info_dynamic_bytes;
    uint64_t index_param_dynamic_bytes;
    uint64_t schema_proto_bytes;
    uint64_t string_dynamic_bytes;
} SegcoreIndexMemoryStatsEntry;

typedef struct {
    uint64_t entry_count;
    uint64_t overflow_count;
    SegcoreIndexMemoryStatsEntry
        entries[SEGCORE_INDEX_MEMORY_STATS_MAX_ENTRIES];
} SegcoreIndexMemoryStats;

typedef struct {
    uint64_t sealed_segment_count;
    uint64_t sealed_segment_object_bytes;
    uint64_t sealed_segment_runtime_estimated_bytes;
    uint64_t sealed_segment_mmap_descriptor_bytes;
    uint64_t sealed_segment_empty_indexing_container_bytes;
    uint64_t sealed_segment_insert_record_bytes;
    uint64_t sealed_segment_deleted_record_bytes;
    uint64_t sealed_segment_load_field_data_info_bytes;
    uint64_t sealed_segment_field_map_bytes;
    uint64_t sealed_segment_field_shared_ptr_control_block_bytes;
    uint64_t sealed_segment_field_data_accounted_map_bytes;
    uint64_t sealed_segment_mmap_field_ids_bytes;
    uint64_t segment_load_info_bytes;
    uint64_t segment_load_info_estimated_bytes;
    uint64_t segment_load_info_object_bytes;
    uint64_t segment_load_info_proto_bytes;
    uint64_t segment_load_info_converted_index_cache_bytes;
    uint64_t segment_load_info_field_index_id_cache_bytes;
    uint64_t segment_load_info_field_index_has_raw_data_bytes;
    uint64_t segment_load_info_fields_filled_with_default_bytes;
    uint64_t segment_load_info_field_binlog_cache_bytes;
    uint64_t segment_load_info_column_group_cache_bytes;
    uint64_t segment_load_info_column_group_cache_deep_bytes;
    uint64_t segment_load_info_column_group_cache_path_bytes;
    uint64_t segment_load_info_column_group_cache_property_bytes;
    uint64_t segment_load_info_column_group_cache_column_bytes;
    uint64_t segment_load_info_column_group_cache_format_bytes;
    uint64_t segment_load_info_column_group_cache_group_count;
    uint64_t segment_load_info_column_group_cache_file_count;
    uint64_t segment_load_info_created_text_indexes_bytes;
    uint64_t field_entry_count;

    uint64_t lazy_manifest_group_count;
    uint64_t lazy_manifest_group_estimated_bytes;
    uint64_t lazy_manifest_proxy_count;
    uint64_t lazy_manifest_proxy_object_bytes;
    uint64_t lazy_manifest_projected_column_count;

    uint64_t pk_index_slot_count;
    uint64_t timestamp_index_slot_count;
    uint64_t pk_index_translator_count;
    uint64_t pk_index_translator_object_bytes;
    uint64_t timestamp_index_translator_count;
    uint64_t timestamp_index_translator_object_bytes;
    uint64_t pk_index_cell_count;
    uint64_t pk_index_cell_bytes;
    uint64_t timestamp_index_cell_count;
    uint64_t timestamp_index_cell_bytes;
} SegcoreMemoryStats;

typedef struct {
    char owner[32];
    char field_name[96];
    char data_type[32];
    char component[80];
    char accuracy[16];
    uint64_t count;
    uint64_t bytes;
} SegcoreCollectionMemoryStatsEntry;

typedef struct {
    uint64_t entry_count;
    uint64_t overflow_count;
    SegcoreCollectionMemoryStatsEntry
        entries[SEGCORE_COLLECTION_MEMORY_STATS_MAX_ENTRIES];
} SegcoreCollectionMemoryStats;

SegcoreMemoryStats
GetSegcoreMemoryStats();

SegcoreIndexMemoryStats
GetSegcoreIndexMemoryStats();

SegcoreCollectionMemoryStats
GetSegcoreCollectionMemoryStats();

#ifdef __cplusplus
}

namespace milvus::monitor {

void
UpdateSegcoreSealedSegment(int64_t count_delta, int64_t object_bytes_delta);

void
UpdateSegcoreSealedSegmentRuntime(int64_t estimated_bytes_delta,
                                  int64_t mmap_descriptor_bytes_delta,
                                  int64_t empty_indexing_container_bytes_delta,
                                  int64_t insert_record_bytes_delta,
                                  int64_t deleted_record_bytes_delta,
                                  int64_t load_field_data_info_bytes_delta);

void
UpdateSegcoreSealedSegmentFieldRuntime(
    int64_t field_map_bytes_delta,
    int64_t field_shared_ptr_control_block_bytes_delta,
    int64_t field_data_accounted_map_bytes_delta,
    int64_t mmap_field_ids_bytes_delta);

void
UpdateSegcoreSegmentLoadInfoBytes(int64_t bytes_delta);

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
    int64_t created_text_indexes_bytes_delta);

void
UpdateSegcoreFieldEntryCount(int64_t count_delta);

void
UpdateSegcoreLazyManifestGroup(int64_t count_delta,
                               int64_t estimated_bytes_delta,
                               int64_t projected_column_count_delta);

void
UpdateSegcoreLazyManifestProxy(int64_t count_delta, int64_t object_bytes_delta);

void
UpdateSegcorePkIndexSlot(int64_t count_delta);

void
UpdateSegcoreTimestampIndexSlot(int64_t count_delta);

void
UpdateSegcorePkIndexTranslator(int64_t count_delta, int64_t object_bytes_delta);

void
UpdateSegcoreTimestampIndexTranslator(int64_t count_delta,
                                      int64_t object_bytes_delta);

void
UpdateSegcorePkIndexCell(int64_t count_delta, int64_t cell_bytes_delta);

void
UpdateSegcoreTimestampIndexCell(int64_t count_delta, int64_t cell_bytes_delta);

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
    int64_t string_dynamic_bytes_delta);

}  // namespace milvus::monitor
#endif
