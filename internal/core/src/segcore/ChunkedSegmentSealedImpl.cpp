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

#include "ChunkedSegmentSealedImpl.h"
#include "segcore/default_fs.h"

#include <cxxabi.h>
#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <simdjson.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <future>
#include <iosfwd>
#include <limits>
#include <map>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include <nlohmann/json.hpp>

#include "NamedType/named_type_impl.hpp"
#include "Types.h"
#include "Utils.h"
#include "arrow/array.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "bitset/bitset.h"
#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Manager.h"
#include "cachinglayer/Translator.h"
#include "common/Array.h"
#include "common/ArrayOffsets.h"
#include "common/ArrowDataWrapper.h"
#include "common/Channel.h"
#include "common/Chunk.h"
#include "common/ChunkWriter.h"
#include "common/Common.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/FieldMeta.h"
#include "common/GeometryCache.h"
#include "common/GroupChunk.h"
#include "common/Json.h"
#include "common/JsonCastType.h"
#include "common/LoadInfo.h"
#include "common/OffsetMapping.h"
#include "common/QueryInfo.h"
#include "common/Schema.h"
#include "common/ScopedTimer.h"
#include "common/Span.h"
#include "common/SystemProperty.h"
#include "common/Tracer.h"
#include "common/TypeTraits.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "common/VectorArray.h"
#include "common/resource_c.h"
#include "common/type_c.h"
#include "folly/Synchronized.h"
#include "geos_c.h"
#include "glog/logging.h"
#include "index/Index.h"
#include "index/IndexFactory.h"
#include "index/Meta.h"
#include "index/NgramInvertedIndex.h"
#include "index/json_stats/JsonKeyStats.h"
#include "index/ScalarIndex.h"
#include "index/TextMatchIndex.h"
#include "index/Utils.h"
#include "index/VectorIndex.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_static.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/version.h"
#include "log/Log.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/packed/chunk_manager.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
#include "mmap/ChunkedColumn.h"
#include "mmap/ChunkedColumnGroup.h"
#include "mmap/ChunkedColumnInterface.h"
#include "mmap/VirtualPKChunkedColumn.h"
#include "mmap/Types.h"
#include "common/VirtualPK.h"
#include "monitor/Monitor.h"
#include "monitor/scope_metric.h"
#include "parquet/metadata.h"
#include "pb/index_cgo_msg.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "prometheus/histogram.h"
#include "query/PlanImpl.h"
#include "query/SearchOnSealed.h"
#include "segcore/ConcurrentVector.h"
#include "segcore/DeletedRecord.h"
#include "segcore/SealedIndexingRecord.h"
#include "segcore/SegmentSealed.h"
#include "segcore/TimestampIndex.h"
#include "segcore/storagev1translator/ChunkTranslator.h"
#include "segcore/storagev1translator/DefaultValueChunkTranslator.h"
#include "segcore/storagev2translator/SystemIndexTranslator.h"
#include "segcore/storagev1translator/InterimSealedIndexTranslator.h"
#include "segcore/storagev1translator/TextMatchIndexTranslator.h"
#include "segcore/storagev2translator/GroupChunkTranslator.h"
#include "segcore/storagev2translator/ManifestGroupTranslator.h"
#include "segcore/TextColumnCache.h"
#include "storage/FileManager.h"
#include "storage/KeyRetriever.h"
#include "storage/LocalChunkManager.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "storage/MmapManager.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "storage/ThreadPool.h"
#include "storage/ThreadPools.h"
#include "storage/Types.h"
#include "storage/Util.h"
#include "storage/loon_ffi/property_singleton.h"
#include "storage/loon_ffi/util.h"

namespace milvus::segcore {
using namespace milvus::cachinglayer;

namespace {

constexpr int64_t kTimingLogIntervalNs = 5LL * 1000 * 1000 * 1000;

int64_t
NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t
DurationNs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

double
AvgMs(int64_t total_ns, int64_t count) {
    if (count == 0) {
        return 0.0;
    }
    return static_cast<double>(total_ns) / static_cast<double>(count) /
           1000000.0;
}

double
NsToMs(int64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

void
UpdateMax(std::atomic<int64_t>& max_value, int64_t value) {
    auto old = max_value.load(std::memory_order_relaxed);
    while (
        value > old &&
        !max_value.compare_exchange_weak(
            old, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

struct SegcoreLoadTiming {
    int64_t reopen_lock_wait_ns = 0;
    int64_t prepare_load_info_ns = 0;
    int64_t get_diff_ns = 0;
    int64_t apply_diff_ns = 0;
    int64_t total_ns = 0;
};

class SegcoreLoadTimingStats {
 public:
    void
    Record(const SegcoreLoadTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_reopen_lock_wait_ns_.fetch_add(timing.reopen_lock_wait_ns,
                                             std::memory_order_relaxed);
        total_prepare_load_info_ns_.fetch_add(timing.prepare_load_info_ns,
                                              std::memory_order_relaxed);
        total_get_diff_ns_.fetch_add(timing.get_diff_ns,
                                     std::memory_order_relaxed);
        total_apply_diff_ns_.fetch_add(timing.apply_diff_ns,
                                       std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }

        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto total_reopen_lock_wait_ns =
            total_reopen_lock_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto total_prepare_load_info_ns =
            total_prepare_load_info_ns_.exchange(0, std::memory_order_relaxed);
        auto total_get_diff_ns =
            total_get_diff_ns_.exchange(0, std::memory_order_relaxed);
        auto total_apply_diff_ns =
            total_apply_diff_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore sealed segment load timing stats count={} "
            "avgReopenLockWaitMs={:.3f} avgPrepareLoadInfoMs={:.3f} "
            "avgGetDiffMs={:.3f} avgApplyDiffMs={:.3f} avgTotalMs={:.3f} "
            "maxTotalMs={:.3f}",
            count,
            AvgMs(total_reopen_lock_wait_ns, count),
            AvgMs(total_prepare_load_info_ns, count),
            AvgMs(total_get_diff_ns, count),
            AvgMs(total_apply_diff_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns));
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_reopen_lock_wait_ns_{0};
    std::atomic<int64_t> total_prepare_load_info_ns_{0};
    std::atomic<int64_t> total_get_diff_ns_{0};
    std::atomic<int64_t> total_apply_diff_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct ApplyLoadDiffTiming {
    int64_t indexes_load_ns = 0;
    int64_t indexes_replace_ns = 0;
    int64_t reload_fields_ns = 0;
    int64_t prepare_column_groups_ns = 0;
    int64_t column_groups_load_ns = 0;
    int64_t column_groups_lazy_load_ns = 0;
    int64_t column_groups_replace_ns = 0;
    int64_t column_groups_lazy_replace_ns = 0;
    int64_t init_text_lob_ns = 0;
    int64_t binlogs_load_ns = 0;
    int64_t binlogs_replace_ns = 0;
    int64_t drop_index_ns = 0;
    int64_t text_indexes_load_ns = 0;
    int64_t json_stats_load_ns = 0;
    int64_t json_stats_replace_ns = 0;
    int64_t json_stats_drop_ns = 0;
    int64_t fill_default_ns = 0;
    int64_t text_indexes_create_ns = 0;
    int64_t field_data_drop_ns = 0;
    int64_t total_ns = 0;
    int64_t index_load_count = 0;
    int64_t index_replace_count = 0;
    int64_t column_group_count = 0;
    int64_t binlog_group_count = 0;
    int64_t text_index_count = 0;
    int64_t json_stats_count = 0;
};

template <typename T>
int64_t
CountIndexInfos(const T& field_to_indexes) {
    int64_t count = 0;
    for (const auto& [_, infos] : field_to_indexes) {
        count += infos.size();
    }
    return count;
}

template <typename T>
int64_t
CountColumnGroups(const T& groups) {
    return groups.size();
}

class ApplyLoadDiffTimingStats {
 public:
    void
    Record(const ApplyLoadDiffTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_indexes_load_ns_.fetch_add(timing.indexes_load_ns,
                                         std::memory_order_relaxed);
        total_indexes_replace_ns_.fetch_add(timing.indexes_replace_ns,
                                            std::memory_order_relaxed);
        total_reload_fields_ns_.fetch_add(timing.reload_fields_ns,
                                          std::memory_order_relaxed);
        total_prepare_column_groups_ns_.fetch_add(
            timing.prepare_column_groups_ns, std::memory_order_relaxed);
        total_column_groups_load_ns_.fetch_add(timing.column_groups_load_ns,
                                               std::memory_order_relaxed);
        total_column_groups_lazy_load_ns_.fetch_add(
            timing.column_groups_lazy_load_ns, std::memory_order_relaxed);
        total_column_groups_replace_ns_.fetch_add(
            timing.column_groups_replace_ns, std::memory_order_relaxed);
        total_column_groups_lazy_replace_ns_.fetch_add(
            timing.column_groups_lazy_replace_ns, std::memory_order_relaxed);
        total_init_text_lob_ns_.fetch_add(timing.init_text_lob_ns,
                                          std::memory_order_relaxed);
        total_binlogs_load_ns_.fetch_add(timing.binlogs_load_ns,
                                         std::memory_order_relaxed);
        total_binlogs_replace_ns_.fetch_add(timing.binlogs_replace_ns,
                                            std::memory_order_relaxed);
        total_drop_index_ns_.fetch_add(timing.drop_index_ns,
                                       std::memory_order_relaxed);
        total_text_indexes_load_ns_.fetch_add(timing.text_indexes_load_ns,
                                              std::memory_order_relaxed);
        total_json_stats_load_ns_.fetch_add(timing.json_stats_load_ns,
                                            std::memory_order_relaxed);
        total_json_stats_replace_ns_.fetch_add(timing.json_stats_replace_ns,
                                               std::memory_order_relaxed);
        total_json_stats_drop_ns_.fetch_add(timing.json_stats_drop_ns,
                                            std::memory_order_relaxed);
        total_fill_default_ns_.fetch_add(timing.fill_default_ns,
                                         std::memory_order_relaxed);
        total_text_indexes_create_ns_.fetch_add(timing.text_indexes_create_ns,
                                                std::memory_order_relaxed);
        total_field_data_drop_ns_.fetch_add(timing.field_data_drop_ns,
                                            std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_index_load_count_.fetch_add(timing.index_load_count,
                                          std::memory_order_relaxed);
        total_index_replace_count_.fetch_add(timing.index_replace_count,
                                             std::memory_order_relaxed);
        total_column_group_count_.fetch_add(timing.column_group_count,
                                            std::memory_order_relaxed);
        total_binlog_group_count_.fetch_add(timing.binlog_group_count,
                                            std::memory_order_relaxed);
        total_text_index_count_.fetch_add(timing.text_index_count,
                                          std::memory_order_relaxed);
        total_json_stats_count_.fetch_add(timing.json_stats_count,
                                          std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }

        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
#define SWAP_TIMING(field) field.exchange(0, std::memory_order_relaxed)
        auto total_indexes_load_ns = SWAP_TIMING(total_indexes_load_ns_);
        auto total_indexes_replace_ns = SWAP_TIMING(total_indexes_replace_ns_);
        auto total_reload_fields_ns = SWAP_TIMING(total_reload_fields_ns_);
        auto total_prepare_column_groups_ns =
            SWAP_TIMING(total_prepare_column_groups_ns_);
        auto total_column_groups_load_ns =
            SWAP_TIMING(total_column_groups_load_ns_);
        auto total_column_groups_lazy_load_ns =
            SWAP_TIMING(total_column_groups_lazy_load_ns_);
        auto total_column_groups_replace_ns =
            SWAP_TIMING(total_column_groups_replace_ns_);
        auto total_column_groups_lazy_replace_ns =
            SWAP_TIMING(total_column_groups_lazy_replace_ns_);
        auto total_init_text_lob_ns = SWAP_TIMING(total_init_text_lob_ns_);
        auto total_binlogs_load_ns = SWAP_TIMING(total_binlogs_load_ns_);
        auto total_binlogs_replace_ns = SWAP_TIMING(total_binlogs_replace_ns_);
        auto total_drop_index_ns = SWAP_TIMING(total_drop_index_ns_);
        auto total_text_indexes_load_ns =
            SWAP_TIMING(total_text_indexes_load_ns_);
        auto total_json_stats_load_ns = SWAP_TIMING(total_json_stats_load_ns_);
        auto total_json_stats_replace_ns =
            SWAP_TIMING(total_json_stats_replace_ns_);
        auto total_json_stats_drop_ns = SWAP_TIMING(total_json_stats_drop_ns_);
        auto total_fill_default_ns = SWAP_TIMING(total_fill_default_ns_);
        auto total_text_indexes_create_ns =
            SWAP_TIMING(total_text_indexes_create_ns_);
        auto total_field_data_drop_ns = SWAP_TIMING(total_field_data_drop_ns_);
        auto total_ns = SWAP_TIMING(total_ns_);
        auto total_index_load_count = SWAP_TIMING(total_index_load_count_);
        auto total_index_replace_count =
            SWAP_TIMING(total_index_replace_count_);
        auto total_column_group_count = SWAP_TIMING(total_column_group_count_);
        auto total_binlog_group_count = SWAP_TIMING(total_binlog_group_count_);
        auto total_text_index_count = SWAP_TIMING(total_text_index_count_);
        auto total_json_stats_count = SWAP_TIMING(total_json_stats_count_);
        auto max_total_ns = SWAP_TIMING(max_total_ns_);
#undef SWAP_TIMING

        LOG_WARN(
            "segcore apply load diff timing stats count={} "
            "avgIndexesLoadMs={:.3f} avgIndexesReplaceMs={:.3f} "
            "avgReloadFieldsMs={:.3f} avgPrepareColumnGroupsMs={:.3f} "
            "avgColumnGroupsLoadMs={:.3f} avgColumnGroupsLazyLoadMs={:.3f} "
            "avgColumnGroupsReplaceMs={:.3f} "
            "avgColumnGroupsLazyReplaceMs={:.3f} avgInitTextLobMs={:.3f} "
            "avgBinlogsLoadMs={:.3f} avgBinlogsReplaceMs={:.3f} "
            "avgDropIndexMs={:.3f} avgTextIndexesLoadMs={:.3f} "
            "avgJsonStatsLoadMs={:.3f} avgJsonStatsReplaceMs={:.3f} "
            "avgJsonStatsDropMs={:.3f} avgFillDefaultMs={:.3f} "
            "avgTextIndexesCreateMs={:.3f} avgFieldDataDropMs={:.3f} "
            "avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "avgIndexLoadCount={:.2f} avgIndexReplaceCount={:.2f} "
            "avgColumnGroupCount={:.2f} avgBinlogGroupCount={:.2f} "
            "avgTextIndexCount={:.2f} avgJsonStatsCount={:.2f}",
            count,
            AvgMs(total_indexes_load_ns, count),
            AvgMs(total_indexes_replace_ns, count),
            AvgMs(total_reload_fields_ns, count),
            AvgMs(total_prepare_column_groups_ns, count),
            AvgMs(total_column_groups_load_ns, count),
            AvgMs(total_column_groups_lazy_load_ns, count),
            AvgMs(total_column_groups_replace_ns, count),
            AvgMs(total_column_groups_lazy_replace_ns, count),
            AvgMs(total_init_text_lob_ns, count),
            AvgMs(total_binlogs_load_ns, count),
            AvgMs(total_binlogs_replace_ns, count),
            AvgMs(total_drop_index_ns, count),
            AvgMs(total_text_indexes_load_ns, count),
            AvgMs(total_json_stats_load_ns, count),
            AvgMs(total_json_stats_replace_ns, count),
            AvgMs(total_json_stats_drop_ns, count),
            AvgMs(total_fill_default_ns, count),
            AvgMs(total_text_indexes_create_ns, count),
            AvgMs(total_field_data_drop_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(total_index_load_count) / count,
            static_cast<double>(total_index_replace_count) / count,
            static_cast<double>(total_column_group_count) / count,
            static_cast<double>(total_binlog_group_count) / count,
            static_cast<double>(total_text_index_count) / count,
            static_cast<double>(total_json_stats_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_indexes_load_ns_{0};
    std::atomic<int64_t> total_indexes_replace_ns_{0};
    std::atomic<int64_t> total_reload_fields_ns_{0};
    std::atomic<int64_t> total_prepare_column_groups_ns_{0};
    std::atomic<int64_t> total_column_groups_load_ns_{0};
    std::atomic<int64_t> total_column_groups_lazy_load_ns_{0};
    std::atomic<int64_t> total_column_groups_replace_ns_{0};
    std::atomic<int64_t> total_column_groups_lazy_replace_ns_{0};
    std::atomic<int64_t> total_init_text_lob_ns_{0};
    std::atomic<int64_t> total_binlogs_load_ns_{0};
    std::atomic<int64_t> total_binlogs_replace_ns_{0};
    std::atomic<int64_t> total_drop_index_ns_{0};
    std::atomic<int64_t> total_text_indexes_load_ns_{0};
    std::atomic<int64_t> total_json_stats_load_ns_{0};
    std::atomic<int64_t> total_json_stats_replace_ns_{0};
    std::atomic<int64_t> total_json_stats_drop_ns_{0};
    std::atomic<int64_t> total_fill_default_ns_{0};
    std::atomic<int64_t> total_text_indexes_create_ns_{0};
    std::atomic<int64_t> total_field_data_drop_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_index_load_count_{0};
    std::atomic<int64_t> total_index_replace_count_{0};
    std::atomic<int64_t> total_column_group_count_{0};
    std::atomic<int64_t> total_binlog_group_count_{0};
    std::atomic<int64_t> total_text_index_count_{0};
    std::atomic<int64_t> total_json_stats_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct ColumnGroupTiming {
    int64_t schema_policy_ns = 0;
    int64_t get_chunk_reader_ns = 0;
    int64_t create_translator_ns = 0;
    int64_t create_chunk_group_ns = 0;
    int64_t proxy_column_ns = 0;
    int64_t total_ns = 0;
    int64_t field_count = 0;
    bool eager_load = false;
    bool use_mmap = false;
    bool is_vector = false;
};

class ColumnGroupTimingStats {
 public:
    void
    Record(const ColumnGroupTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_schema_policy_ns_.fetch_add(timing.schema_policy_ns,
                                          std::memory_order_relaxed);
        total_get_chunk_reader_ns_.fetch_add(timing.get_chunk_reader_ns,
                                             std::memory_order_relaxed);
        total_create_translator_ns_.fetch_add(timing.create_translator_ns,
                                              std::memory_order_relaxed);
        total_create_chunk_group_ns_.fetch_add(timing.create_chunk_group_ns,
                                               std::memory_order_relaxed);
        total_proxy_column_ns_.fetch_add(timing.proxy_column_ns,
                                         std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_field_count_.fetch_add(timing.field_count,
                                     std::memory_order_relaxed);
        eager_count_.fetch_add(timing.eager_load ? 1 : 0,
                               std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.use_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        vector_count_.fetch_add(timing.is_vector ? 1 : 0,
                                std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto schema_policy_ns =
            total_schema_policy_ns_.exchange(0, std::memory_order_relaxed);
        auto get_chunk_reader_ns =
            total_get_chunk_reader_ns_.exchange(0, std::memory_order_relaxed);
        auto create_translator_ns =
            total_create_translator_ns_.exchange(0, std::memory_order_relaxed);
        auto create_chunk_group_ns =
            total_create_chunk_group_ns_.exchange(0, std::memory_order_relaxed);
        auto proxy_column_ns =
            total_proxy_column_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto field_count =
            total_field_count_.exchange(0, std::memory_order_relaxed);
        auto eager_count = eager_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto vector_count =
            vector_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load column group timing stats count={} "
            "avgSchemaPolicyMs={:.3f} avgGetChunkReaderMs={:.3f} "
            "avgCreateTranslatorMs={:.3f} avgCreateChunkGroupMs={:.3f} "
            "avgProxyColumnMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "avgFieldCount={:.2f} eagerRatio={:.2f} mmapRatio={:.2f} "
            "vectorRatio={:.2f}",
            count,
            AvgMs(schema_policy_ns, count),
            AvgMs(get_chunk_reader_ns, count),
            AvgMs(create_translator_ns, count),
            AvgMs(create_chunk_group_ns, count),
            AvgMs(proxy_column_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(field_count) / count,
            static_cast<double>(eager_count) / count,
            static_cast<double>(mmap_count) / count,
            static_cast<double>(vector_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_schema_policy_ns_{0};
    std::atomic<int64_t> total_get_chunk_reader_ns_{0};
    std::atomic<int64_t> total_create_translator_ns_{0};
    std::atomic<int64_t> total_create_chunk_group_ns_{0};
    std::atomic<int64_t> total_proxy_column_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_field_count_{0};
    std::atomic<int64_t> eager_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> vector_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct IndexRegisterTiming {
    int64_t lock_wait_ns = 0;
    int64_t resource_ns = 0;
    int64_t register_ns = 0;
    int64_t total_ns = 0;
    bool is_vector = false;
    bool has_raw_data = false;
};

class IndexRegisterTimingStats {
 public:
    void
    Record(const IndexRegisterTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_lock_wait_ns_.fetch_add(timing.lock_wait_ns,
                                      std::memory_order_relaxed);
        total_resource_ns_.fetch_add(timing.resource_ns,
                                     std::memory_order_relaxed);
        total_register_ns_.fetch_add(timing.register_ns,
                                     std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        vector_count_.fetch_add(timing.is_vector ? 1 : 0,
                                std::memory_order_relaxed);
        raw_data_count_.fetch_add(timing.has_raw_data ? 1 : 0,
                                  std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto lock_wait_ns =
            total_lock_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto resource_ns =
            total_resource_ns_.exchange(0, std::memory_order_relaxed);
        auto register_ns =
            total_register_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto vector_count =
            vector_count_.exchange(0, std::memory_order_relaxed);
        auto raw_data_count =
            raw_data_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load index register timing stats count={} "
            "avgLockWaitMs={:.3f} avgResourceMs={:.3f} "
            "avgRegisterMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "vectorRatio={:.2f} rawDataRatio={:.2f}",
            count,
            AvgMs(lock_wait_ns, count),
            AvgMs(resource_ns, count),
            AvgMs(register_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(vector_count) / count,
            static_cast<double>(raw_data_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_lock_wait_ns_{0};
    std::atomic<int64_t> total_resource_ns_{0};
    std::atomic<int64_t> total_register_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> vector_count_{0};
    std::atomic<int64_t> raw_data_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct BatchFieldDataTiming {
    int64_t prepare_ns = 0;
    int64_t wait_ns = 0;
    int64_t total_ns = 0;
    int64_t group_count = 0;
    int64_t task_count = 0;
    int64_t skipped_group_count = 0;
    int64_t child_field_count = 0;
};

class BatchFieldDataTimingStats {
 public:
    void
    Record(const BatchFieldDataTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_prepare_ns_.fetch_add(timing.prepare_ns,
                                    std::memory_order_relaxed);
        total_wait_ns_.fetch_add(timing.wait_ns, std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_group_count_.fetch_add(timing.group_count,
                                     std::memory_order_relaxed);
        total_task_count_.fetch_add(timing.task_count,
                                    std::memory_order_relaxed);
        total_skipped_group_count_.fetch_add(timing.skipped_group_count,
                                             std::memory_order_relaxed);
        total_child_field_count_.fetch_add(timing.child_field_count,
                                           std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto prepare_ns =
            total_prepare_ns_.exchange(0, std::memory_order_relaxed);
        auto wait_ns = total_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto group_count =
            total_group_count_.exchange(0, std::memory_order_relaxed);
        auto task_count =
            total_task_count_.exchange(0, std::memory_order_relaxed);
        auto skipped_group_count =
            total_skipped_group_count_.exchange(0, std::memory_order_relaxed);
        auto child_field_count =
            total_child_field_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load batch field data timing stats count={} "
            "avgPrepareMs={:.3f} avgWaitMs={:.3f} avgTotalMs={:.3f} "
            "maxTotalMs={:.3f} avgGroupCount={:.2f} avgTaskCount={:.2f} "
            "avgSkippedGroupCount={:.2f} avgChildFieldCount={:.2f}",
            count,
            AvgMs(prepare_ns, count),
            AvgMs(wait_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(group_count) / count,
            static_cast<double>(task_count) / count,
            static_cast<double>(skipped_group_count) / count,
            static_cast<double>(child_field_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_prepare_ns_{0};
    std::atomic<int64_t> total_wait_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_group_count_{0};
    std::atomic<int64_t> total_task_count_{0};
    std::atomic<int64_t> total_skipped_group_count_{0};
    std::atomic<int64_t> total_child_field_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct FieldDataInternalTiming {
    int64_t system_remote_ns = 0;
    int64_t system_load_ns = 0;
    int64_t build_file_info_ns = 0;
    int64_t create_translator_ns = 0;
    int64_t create_cache_slot_ns = 0;
    int64_t create_column_ns = 0;
    int64_t load_common_ns = 0;
    int64_t total_ns = 0;
    bool is_system = false;
    bool enable_mmap = false;
};

class FieldDataInternalTimingStats {
 public:
    void
    Record(const FieldDataInternalTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_system_remote_ns_.fetch_add(timing.system_remote_ns,
                                          std::memory_order_relaxed);
        total_system_load_ns_.fetch_add(timing.system_load_ns,
                                        std::memory_order_relaxed);
        total_build_file_info_ns_.fetch_add(timing.build_file_info_ns,
                                            std::memory_order_relaxed);
        total_create_translator_ns_.fetch_add(timing.create_translator_ns,
                                              std::memory_order_relaxed);
        total_create_cache_slot_ns_.fetch_add(timing.create_cache_slot_ns,
                                              std::memory_order_relaxed);
        total_create_column_ns_.fetch_add(timing.create_column_ns,
                                          std::memory_order_relaxed);
        total_load_common_ns_.fetch_add(timing.load_common_ns,
                                        std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        system_count_.fetch_add(timing.is_system ? 1 : 0,
                                std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto system_remote_ns =
            total_system_remote_ns_.exchange(0, std::memory_order_relaxed);
        auto system_load_ns =
            total_system_load_ns_.exchange(0, std::memory_order_relaxed);
        auto build_file_info_ns =
            total_build_file_info_ns_.exchange(0, std::memory_order_relaxed);
        auto create_translator_ns =
            total_create_translator_ns_.exchange(0, std::memory_order_relaxed);
        auto create_cache_slot_ns =
            total_create_cache_slot_ns_.exchange(0, std::memory_order_relaxed);
        auto create_column_ns =
            total_create_column_ns_.exchange(0, std::memory_order_relaxed);
        auto load_common_ns =
            total_load_common_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto system_count =
            system_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load field data internal timing stats count={} "
            "avgSystemRemoteMs={:.3f} avgSystemLoadMs={:.3f} "
            "avgBuildFileInfoMs={:.3f} avgCreateTranslatorMs={:.3f} "
            "avgCreateCacheSlotMs={:.3f} avgCreateColumnMs={:.3f} "
            "avgLoadCommonMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "systemRatio={:.2f} mmapRatio={:.2f}",
            count,
            AvgMs(system_remote_ns, count),
            AvgMs(system_load_ns, count),
            AvgMs(build_file_info_ns, count),
            AvgMs(create_translator_ns, count),
            AvgMs(create_cache_slot_ns, count),
            AvgMs(create_column_ns, count),
            AvgMs(load_common_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(system_count) / count,
            static_cast<double>(mmap_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_system_remote_ns_{0};
    std::atomic<int64_t> total_system_load_ns_{0};
    std::atomic<int64_t> total_build_file_info_ns_{0};
    std::atomic<int64_t> total_create_translator_ns_{0};
    std::atomic<int64_t> total_create_cache_slot_ns_{0};
    std::atomic<int64_t> total_create_column_ns_{0};
    std::atomic<int64_t> total_load_common_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> system_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct StorageV2FieldDataTiming {
    int64_t resolve_fields_ns = 0;
    int64_t schema_meta_ns = 0;
    int64_t metadata_ns = 0;
    int64_t create_translator_ns = 0;
    int64_t create_chunk_group_ns = 0;
    int64_t proxy_common_ns = 0;
    int64_t total_ns = 0;
    int64_t field_count = 0;
    int64_t stats_field_count = 0;
    bool enable_mmap = false;
    bool has_vector = false;
};

class StorageV2FieldDataTimingStats {
 public:
    void
    Record(const StorageV2FieldDataTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_resolve_fields_ns_.fetch_add(timing.resolve_fields_ns,
                                           std::memory_order_relaxed);
        total_schema_meta_ns_.fetch_add(timing.schema_meta_ns,
                                        std::memory_order_relaxed);
        total_metadata_ns_.fetch_add(timing.metadata_ns,
                                     std::memory_order_relaxed);
        total_create_translator_ns_.fetch_add(timing.create_translator_ns,
                                              std::memory_order_relaxed);
        total_create_chunk_group_ns_.fetch_add(timing.create_chunk_group_ns,
                                               std::memory_order_relaxed);
        total_proxy_common_ns_.fetch_add(timing.proxy_common_ns,
                                         std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_field_count_.fetch_add(timing.field_count,
                                     std::memory_order_relaxed);
        total_stats_field_count_.fetch_add(timing.stats_field_count,
                                           std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        vector_count_.fetch_add(timing.has_vector ? 1 : 0,
                                std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto resolve_fields_ns =
            total_resolve_fields_ns_.exchange(0, std::memory_order_relaxed);
        auto schema_meta_ns =
            total_schema_meta_ns_.exchange(0, std::memory_order_relaxed);
        auto metadata_ns =
            total_metadata_ns_.exchange(0, std::memory_order_relaxed);
        auto create_translator_ns =
            total_create_translator_ns_.exchange(0, std::memory_order_relaxed);
        auto create_chunk_group_ns =
            total_create_chunk_group_ns_.exchange(0, std::memory_order_relaxed);
        auto proxy_common_ns =
            total_proxy_common_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto field_count =
            total_field_count_.exchange(0, std::memory_order_relaxed);
        auto stats_field_count =
            total_stats_field_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto vector_count =
            vector_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load storage v2 field data timing stats count={} "
            "avgResolveFieldsMs={:.3f} avgSchemaMetaMs={:.3f} "
            "avgMetadataMs={:.3f} avgCreateTranslatorMs={:.3f} "
            "avgCreateChunkGroupMs={:.3f} avgProxyCommonMs={:.3f} "
            "avgTotalMs={:.3f} maxTotalMs={:.3f} avgFieldCount={:.2f} "
            "avgStatsFieldCount={:.2f} mmapRatio={:.2f} vectorRatio={:.2f}",
            count,
            AvgMs(resolve_fields_ns, count),
            AvgMs(schema_meta_ns, count),
            AvgMs(metadata_ns, count),
            AvgMs(create_translator_ns, count),
            AvgMs(create_chunk_group_ns, count),
            AvgMs(proxy_common_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(field_count) / count,
            static_cast<double>(stats_field_count) / count,
            static_cast<double>(mmap_count) / count,
            static_cast<double>(vector_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_resolve_fields_ns_{0};
    std::atomic<int64_t> total_schema_meta_ns_{0};
    std::atomic<int64_t> total_metadata_ns_{0};
    std::atomic<int64_t> total_create_translator_ns_{0};
    std::atomic<int64_t> total_create_chunk_group_ns_{0};
    std::atomic<int64_t> total_proxy_common_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_field_count_{0};
    std::atomic<int64_t> total_stats_field_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> vector_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct FieldDataCommonTiming {
    int64_t initial_lock_ns = 0;
    int64_t nullable_ns = 0;
    int64_t memory_stats_ns = 0;
    int64_t skip_index_ns = 0;
    int64_t pk_index_ns = 0;
    int64_t interim_index_ns = 0;
    int64_t final_lock_ns = 0;
    int64_t array_offsets_ns = 0;
    int64_t total_ns = 0;
    bool system = false;
    bool vector = false;
    bool variable = false;
    bool proxy = false;
    bool mmap = false;
    bool pk = false;
    bool nullable = false;
};

class FieldDataCommonTimingStats {
 public:
    void
    Record(const FieldDataCommonTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_initial_lock_ns_.fetch_add(timing.initial_lock_ns,
                                         std::memory_order_relaxed);
        total_nullable_ns_.fetch_add(timing.nullable_ns,
                                     std::memory_order_relaxed);
        total_memory_stats_ns_.fetch_add(timing.memory_stats_ns,
                                         std::memory_order_relaxed);
        total_skip_index_ns_.fetch_add(timing.skip_index_ns,
                                       std::memory_order_relaxed);
        total_pk_index_ns_.fetch_add(timing.pk_index_ns,
                                     std::memory_order_relaxed);
        total_interim_index_ns_.fetch_add(timing.interim_index_ns,
                                          std::memory_order_relaxed);
        total_final_lock_ns_.fetch_add(timing.final_lock_ns,
                                       std::memory_order_relaxed);
        total_array_offsets_ns_.fetch_add(timing.array_offsets_ns,
                                          std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        system_count_.fetch_add(timing.system ? 1 : 0,
                                std::memory_order_relaxed);
        vector_count_.fetch_add(timing.vector ? 1 : 0,
                                std::memory_order_relaxed);
        variable_count_.fetch_add(timing.variable ? 1 : 0,
                                  std::memory_order_relaxed);
        proxy_count_.fetch_add(timing.proxy ? 1 : 0, std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.mmap ? 1 : 0, std::memory_order_relaxed);
        pk_count_.fetch_add(timing.pk ? 1 : 0, std::memory_order_relaxed);
        nullable_count_.fetch_add(timing.nullable ? 1 : 0,
                                  std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto initial_lock_ns =
            total_initial_lock_ns_.exchange(0, std::memory_order_relaxed);
        auto nullable_ns =
            total_nullable_ns_.exchange(0, std::memory_order_relaxed);
        auto memory_stats_ns =
            total_memory_stats_ns_.exchange(0, std::memory_order_relaxed);
        auto skip_index_ns =
            total_skip_index_ns_.exchange(0, std::memory_order_relaxed);
        auto pk_index_ns =
            total_pk_index_ns_.exchange(0, std::memory_order_relaxed);
        auto interim_index_ns =
            total_interim_index_ns_.exchange(0, std::memory_order_relaxed);
        auto final_lock_ns =
            total_final_lock_ns_.exchange(0, std::memory_order_relaxed);
        auto array_offsets_ns =
            total_array_offsets_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto system_count =
            system_count_.exchange(0, std::memory_order_relaxed);
        auto vector_count =
            vector_count_.exchange(0, std::memory_order_relaxed);
        auto variable_count =
            variable_count_.exchange(0, std::memory_order_relaxed);
        auto proxy_count = proxy_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto pk_count = pk_count_.exchange(0, std::memory_order_relaxed);
        auto nullable_count =
            nullable_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load field data common timing stats count={} "
            "avgInitialLockMs={:.3f} avgNullableMs={:.3f} "
            "avgMemoryStatsMs={:.3f} avgSkipIndexMs={:.3f} "
            "avgPKIndexMs={:.3f} avgInterimIndexMs={:.3f} "
            "avgFinalLockMs={:.3f} avgArrayOffsetsMs={:.3f} "
            "avgTotalMs={:.3f} maxTotalMs={:.3f} systemRatio={:.2f} "
            "vectorRatio={:.2f} variableRatio={:.2f} proxyRatio={:.2f} "
            "mmapRatio={:.2f} pkRatio={:.2f} nullableRatio={:.2f}",
            count,
            AvgMs(initial_lock_ns, count),
            AvgMs(nullable_ns, count),
            AvgMs(memory_stats_ns, count),
            AvgMs(skip_index_ns, count),
            AvgMs(pk_index_ns, count),
            AvgMs(interim_index_ns, count),
            AvgMs(final_lock_ns, count),
            AvgMs(array_offsets_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(system_count) / count,
            static_cast<double>(vector_count) / count,
            static_cast<double>(variable_count) / count,
            static_cast<double>(proxy_count) / count,
            static_cast<double>(mmap_count) / count,
            static_cast<double>(pk_count) / count,
            static_cast<double>(nullable_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_initial_lock_ns_{0};
    std::atomic<int64_t> total_nullable_ns_{0};
    std::atomic<int64_t> total_memory_stats_ns_{0};
    std::atomic<int64_t> total_skip_index_ns_{0};
    std::atomic<int64_t> total_pk_index_ns_{0};
    std::atomic<int64_t> total_interim_index_ns_{0};
    std::atomic<int64_t> total_final_lock_ns_{0};
    std::atomic<int64_t> total_array_offsets_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> system_count_{0};
    std::atomic<int64_t> vector_count_{0};
    std::atomic<int64_t> variable_count_{0};
    std::atomic<int64_t> proxy_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> pk_count_{0};
    std::atomic<int64_t> nullable_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct BatchTaskTiming {
    int64_t submit_ns = 0;
    int64_t wait_ns = 0;
    int64_t total_ns = 0;
    int64_t task_count = 0;
    int64_t field_count = 0;
    bool is_replace = false;
};

class BatchTaskTimingStats {
 public:
    explicit BatchTaskTimingStats(const char* log_name) : log_name_(log_name) {
    }

    void
    Record(const BatchTaskTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_submit_ns_.fetch_add(timing.submit_ns, std::memory_order_relaxed);
        total_wait_ns_.fetch_add(timing.wait_ns, std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_task_count_.fetch_add(timing.task_count,
                                    std::memory_order_relaxed);
        total_field_count_.fetch_add(timing.field_count,
                                     std::memory_order_relaxed);
        replace_count_.fetch_add(timing.is_replace ? 1 : 0,
                                 std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }

        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto submit_ns =
            total_submit_ns_.exchange(0, std::memory_order_relaxed);
        auto wait_ns = total_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto task_count =
            total_task_count_.exchange(0, std::memory_order_relaxed);
        auto field_count =
            total_field_count_.exchange(0, std::memory_order_relaxed);
        auto replace_count =
            replace_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "{} count={} avgSubmitMs={:.3f} avgWaitMs={:.3f} "
            "avgTotalMs={:.3f} maxTotalMs={:.3f} avgTaskCount={:.2f} "
            "avgFieldCount={:.2f} replaceRatio={:.2f}",
            log_name_,
            count,
            AvgMs(submit_ns, count),
            AvgMs(wait_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(task_count) / count,
            static_cast<double>(field_count) / count,
            static_cast<double>(replace_count) / count);
    }

    const char* log_name_;
    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_submit_ns_{0};
    std::atomic<int64_t> total_wait_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_task_count_{0};
    std::atomic<int64_t> total_field_count_{0};
    std::atomic<int64_t> replace_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct TextIndexEntryTiming {
    int64_t config_ns = 0;
    int64_t file_context_ns = 0;
    int64_t translator_ns = 0;
    int64_t cache_slot_ns = 0;
    int64_t lock_wait_ns = 0;
    int64_t register_ns = 0;
    int64_t total_ns = 0;
    int64_t file_count = 0;
    bool enable_mmap = false;
};

class TextIndexEntryTimingStats {
 public:
    void
    Record(const TextIndexEntryTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_config_ns_.fetch_add(timing.config_ns, std::memory_order_relaxed);
        total_file_context_ns_.fetch_add(timing.file_context_ns,
                                         std::memory_order_relaxed);
        total_translator_ns_.fetch_add(timing.translator_ns,
                                       std::memory_order_relaxed);
        total_cache_slot_ns_.fetch_add(timing.cache_slot_ns,
                                       std::memory_order_relaxed);
        total_lock_wait_ns_.fetch_add(timing.lock_wait_ns,
                                      std::memory_order_relaxed);
        total_register_ns_.fetch_add(timing.register_ns,
                                     std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_file_count_.fetch_add(timing.file_count,
                                    std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto config_ns =
            total_config_ns_.exchange(0, std::memory_order_relaxed);
        auto file_context_ns =
            total_file_context_ns_.exchange(0, std::memory_order_relaxed);
        auto translator_ns =
            total_translator_ns_.exchange(0, std::memory_order_relaxed);
        auto cache_slot_ns =
            total_cache_slot_ns_.exchange(0, std::memory_order_relaxed);
        auto lock_wait_ns =
            total_lock_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto register_ns =
            total_register_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto file_count =
            total_file_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load text index entry timing stats count={} "
            "avgConfigMs={:.3f} avgFileContextMs={:.3f} "
            "avgTranslatorMs={:.3f} avgCreateCacheSlotMs={:.3f} "
            "avgLockWaitMs={:.3f} avgRegisterMs={:.3f} avgTotalMs={:.3f} "
            "maxTotalMs={:.3f} avgFileCount={:.2f} mmapRatio={:.2f}",
            count,
            AvgMs(config_ns, count),
            AvgMs(file_context_ns, count),
            AvgMs(translator_ns, count),
            AvgMs(cache_slot_ns, count),
            AvgMs(lock_wait_ns, count),
            AvgMs(register_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(file_count) / count,
            static_cast<double>(mmap_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_config_ns_{0};
    std::atomic<int64_t> total_file_context_ns_{0};
    std::atomic<int64_t> total_translator_ns_{0};
    std::atomic<int64_t> total_cache_slot_ns_{0};
    std::atomic<int64_t> total_lock_wait_ns_{0};
    std::atomic<int64_t> total_register_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_file_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct CreateTextIndexTiming {
    int64_t lock_wait_ns = 0;
    int64_t create_index_ns = 0;
    int64_t build_source_ns = 0;
    int64_t finalize_ns = 0;
    int64_t publish_ns = 0;
    int64_t total_ns = 0;
    bool enable_mmap = false;
    bool source_column = false;
};

class CreateTextIndexTimingStats {
 public:
    void
    Record(const CreateTextIndexTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_lock_wait_ns_.fetch_add(timing.lock_wait_ns,
                                      std::memory_order_relaxed);
        total_create_index_ns_.fetch_add(timing.create_index_ns,
                                         std::memory_order_relaxed);
        total_build_source_ns_.fetch_add(timing.build_source_ns,
                                         std::memory_order_relaxed);
        total_finalize_ns_.fetch_add(timing.finalize_ns,
                                     std::memory_order_relaxed);
        total_publish_ns_.fetch_add(timing.publish_ns,
                                    std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        column_source_count_.fetch_add(timing.source_column ? 1 : 0,
                                       std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto lock_wait_ns =
            total_lock_wait_ns_.exchange(0, std::memory_order_relaxed);
        auto create_index_ns =
            total_create_index_ns_.exchange(0, std::memory_order_relaxed);
        auto build_source_ns =
            total_build_source_ns_.exchange(0, std::memory_order_relaxed);
        auto finalize_ns =
            total_finalize_ns_.exchange(0, std::memory_order_relaxed);
        auto publish_ns =
            total_publish_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto column_source_count =
            column_source_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore create text index timing stats count={} "
            "avgLockWaitMs={:.3f} avgCreateIndexMs={:.3f} "
            "avgBuildSourceMs={:.3f} avgFinalizeMs={:.3f} "
            "avgPublishMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "mmapRatio={:.2f} columnSourceRatio={:.2f}",
            count,
            AvgMs(lock_wait_ns, count),
            AvgMs(create_index_ns, count),
            AvgMs(build_source_ns, count),
            AvgMs(finalize_ns, count),
            AvgMs(publish_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(mmap_count) / count,
            static_cast<double>(column_source_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_lock_wait_ns_{0};
    std::atomic<int64_t> total_create_index_ns_{0};
    std::atomic<int64_t> total_build_source_ns_{0};
    std::atomic<int64_t> total_finalize_ns_{0};
    std::atomic<int64_t> total_publish_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> column_source_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

struct JsonStatsLoadTiming {
    int64_t config_ns = 0;
    int64_t file_context_ns = 0;
    int64_t create_index_ns = 0;
    int64_t load_ns = 0;
    int64_t register_ns = 0;
    int64_t total_ns = 0;
    int64_t file_count = 0;
    bool enable_mmap = false;
};

class JsonStatsLoadTimingStats {
 public:
    void
    Record(const JsonStatsLoadTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_config_ns_.fetch_add(timing.config_ns, std::memory_order_relaxed);
        total_file_context_ns_.fetch_add(timing.file_context_ns,
                                         std::memory_order_relaxed);
        total_create_index_ns_.fetch_add(timing.create_index_ns,
                                         std::memory_order_relaxed);
        total_load_ns_.fetch_add(timing.load_ns, std::memory_order_relaxed);
        total_register_ns_.fetch_add(timing.register_ns,
                                     std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_file_count_.fetch_add(timing.file_count,
                                    std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        UpdateMax(max_total_ns_, timing.total_ns);
        MaybeLog();
    }

 private:
    void
    MaybeLog() {
        auto now = NowNs();
        auto last = last_log_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now - last < kTimingLogIntervalNs) {
            return;
        }
        if (!last_log_ns_.compare_exchange_strong(last,
                                                  now,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            return;
        }
        auto count = count_.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            return;
        }
        auto config_ns =
            total_config_ns_.exchange(0, std::memory_order_relaxed);
        auto file_context_ns =
            total_file_context_ns_.exchange(0, std::memory_order_relaxed);
        auto create_index_ns =
            total_create_index_ns_.exchange(0, std::memory_order_relaxed);
        auto load_ns = total_load_ns_.exchange(0, std::memory_order_relaxed);
        auto register_ns =
            total_register_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto file_count =
            total_file_count_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore load json stats timing stats count={} "
            "avgConfigMs={:.3f} avgFileContextMs={:.3f} "
            "avgCreateIndexMs={:.3f} avgLoadMs={:.3f} "
            "avgRegisterMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "avgFileCount={:.2f} mmapRatio={:.2f}",
            count,
            AvgMs(config_ns, count),
            AvgMs(file_context_ns, count),
            AvgMs(create_index_ns, count),
            AvgMs(load_ns, count),
            AvgMs(register_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(file_count) / count,
            static_cast<double>(mmap_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_config_ns_{0};
    std::atomic<int64_t> total_file_context_ns_{0};
    std::atomic<int64_t> total_create_index_ns_{0};
    std::atomic<int64_t> total_load_ns_{0};
    std::atomic<int64_t> total_register_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_file_count_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

static SegcoreLoadTimingStats sealed_load_timing_stats;
static ApplyLoadDiffTimingStats apply_load_diff_timing_stats;
static ColumnGroupTimingStats column_group_timing_stats;
static IndexRegisterTimingStats index_register_timing_stats;
static BatchFieldDataTimingStats batch_field_data_timing_stats;
static FieldDataInternalTimingStats field_data_internal_timing_stats;
static StorageV2FieldDataTimingStats storage_v2_field_data_timing_stats;
static FieldDataCommonTimingStats field_data_common_timing_stats;
static BatchTaskTimingStats batch_column_group_timing_stats(
    "segcore load batch column group timing stats");
static BatchTaskTimingStats batch_index_timing_stats(
    "segcore load batch index timing stats");
static BatchTaskTimingStats batch_text_index_timing_stats(
    "segcore load batch text index timing stats");
static TextIndexEntryTimingStats text_index_entry_timing_stats;
static CreateTextIndexTimingStats create_text_index_timing_stats;
static JsonStatsLoadTimingStats json_stats_load_timing_stats;

}  // namespace

static inline void
set_bit(BitsetType& bitset, FieldId field_id, bool flag = true) {
    auto pos = field_id.get() - START_USER_FIELDID;
    AssertInfo(pos >= 0, "invalid field id");
    bitset[pos] = flag;
}

static inline bool
get_bit(const BitsetType& bitset, FieldId field_id) {
    auto pos = field_id.get() - START_USER_FIELDID;
    AssertInfo(pos >= 0, "invalid field id");

    return bitset[pos];
}

static inline bool
field_exists_in_schema(const SchemaPtr& schema, FieldId field_id) {
    return field_id.get() < START_USER_FIELDID ||
           schema->get_fields().find(field_id) != schema->get_fields().end();
}

static inline bool
has_bit_position(const BitsetType& bitset, FieldId field_id) {
    auto pos = field_id.get() - START_USER_FIELDID;
    return pos >= 0 && static_cast<size_t>(pos) < bitset.size();
}

static inline void
clear_bit_if_present(BitsetType& bitset, FieldId field_id) {
    if (has_bit_position(bitset, field_id)) {
        set_bit(bitset, field_id, false);
    }
}

static inline void
cancel_warmup(const index::CacheIndexBasePtr& index) {
    if (index) {
        index->CancelWarmup();
    }
}

static inline void
cancel_and_erase_scalar_index(
    std::unordered_map<FieldId, index::CacheIndexBasePtr>& scalar_indexings,
    FieldId field_id) {
    if (auto it = scalar_indexings.find(field_id);
        it != scalar_indexings.end()) {
        cancel_warmup(it->second);
        scalar_indexings.erase(it);
    }
}

static inline void
cancel_and_clear_scalar_indexings(
    std::unordered_map<FieldId, index::CacheIndexBasePtr>& scalar_indexings) {
    for (auto& [_, index] : scalar_indexings) {
        cancel_warmup(index);
    }
    scalar_indexings.clear();
}

static inline void
cancel_and_erase_ngram_index(
    std::unordered_map<
        FieldId,
        std::unordered_map<std::string, index::CacheIndexBasePtr>>&
        ngram_indexings,
    FieldId field_id,
    const std::string& nested_path) {
    auto field_it = ngram_indexings.find(field_id);
    if (field_it == ngram_indexings.end()) {
        return;
    }

    auto& path_indexings = field_it->second;
    if (auto path_it = path_indexings.find(nested_path);
        path_it != path_indexings.end()) {
        cancel_warmup(path_it->second);
        path_indexings.erase(path_it);
    }

    if (path_indexings.empty()) {
        ngram_indexings.erase(field_it);
    }
}

static inline void
cancel_and_clear_ngram_indexings(
    std::unordered_map<
        FieldId,
        std::unordered_map<std::string, index::CacheIndexBasePtr>>&
        ngram_indexings) {
    for (auto& [_, path_indexings] : ngram_indexings) {
        for (auto& [__, index] : path_indexings) {
            cancel_warmup(index);
        }
    }
    ngram_indexings.clear();
}

template <typename JsonIndexT>
static void
cancel_and_erase_json_indices(std::vector<JsonIndexT>& json_indices,
                              FieldId field_id,
                              std::string_view nested_path) {
    auto new_end = std::remove_if(
        json_indices.begin(), json_indices.end(), [&](auto& index) {
            auto matched =
                index.field_id == field_id && index.nested_path == nested_path;
            if (matched) {
                cancel_warmup(index.index);
            }
            return matched;
        });
    json_indices.erase(new_end, json_indices.end());
}

template <typename JsonIndexT>
static void
cancel_and_clear_json_indices(std::vector<JsonIndexT>& json_indices) {
    for (auto& index : json_indices) {
        cancel_warmup(index.index);
    }
    json_indices.clear();
}

PinWrapper<const storagev2translator::TimestampIndexCell*>
ChunkedSegmentSealedImpl::PinTimestampIndex(milvus::OpContext* op_ctx) const {
    auto slot = *timestamp_index_slot_.rlock();
    if (!slot) {
        return PinWrapper<const storagev2translator::TimestampIndexCell*>(
            nullptr);
    }
    auto ca = SemiInlineGet(slot->PinCells(op_ctx, {0}));
    auto* cell = ca->get_cell_of(0);
    AssertInfo(
        cell != nullptr, "timestamp index cache is corrupted, segment {}", id_);
    return PinWrapper<const storagev2translator::TimestampIndexCell*>(ca, cell);
}

PinWrapper<const storagev2translator::PkIndexCell*>
ChunkedSegmentSealedImpl::PinPkIndex(milvus::OpContext* op_ctx) const {
    auto slot = *pk_index_slot_.rlock();
    if (!slot) {
        return PinWrapper<const storagev2translator::PkIndexCell*>(nullptr);
    }
    auto ca = SemiInlineGet(slot->PinCells(op_ctx, {0}));
    auto* cell = ca->get_cell_of(0);
    AssertInfo(cell != nullptr, "pk index cache is corrupted, segment {}", id_);
    return PinWrapper<const storagev2translator::PkIndexCell*>(ca, cell);
}

bool
ChunkedSegmentSealedImpl::Contain(const PkType& pk) const {
    // Zero-storage pk2offset (VirtualPKOffsetMap) resolves PKs by bit-extract.
    // Skips PinPkIndex + sorted-pk binary search on the virtual PK column.
    if (insert_record_.pk2offset_is_zero_storage()) {
        return insert_record_.contain(pk);
    }
    auto pk_index = PinPkIndex(nullptr);
    if (pk_index.get() != nullptr && pk_index.get()->has_pk2offset()) {
        return pk_index.get()->contain(pk);
    }
    // Sorted-by-pk segment: binary search on pk column directly.
    if (is_sorted_by_pk_) {
        auto pk_field_id =
            schema_->get_primary_field_id().value_or(FieldId(-1));
        AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
        auto pk_column = get_column(pk_field_id);
        if (pk_column != nullptr) {
            auto num_chunks = pk_column->num_chunks();
            auto all_chunks = pk_column->GetAllChunks(nullptr);
            switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
                case DataType::INT64: {
                    auto target = std::get<int64_t>(pk);
                    for (int64_t i = 0; i < num_chunks; ++i) {
                        auto* src = reinterpret_cast<const int64_t*>(
                            all_chunks[i].get()->RawData());
                        auto rows = pk_column->chunk_row_nums(i);
                        auto it = std::lower_bound(src, src + rows, target);
                        if (it != src + rows && *it == target) {
                            return true;
                        }
                    }
                    return false;
                }
                case DataType::VARCHAR: {
                    auto& target = std::get<std::string>(pk);
                    for (int64_t i = 0; i < num_chunks; ++i) {
                        auto* chunk =
                            static_cast<StringChunk*>(all_chunks[i].get());
                        auto offset = chunk->binary_search_string(target);
                        if (offset != -1 && offset < chunk->RowNums() &&
                            chunk->operator[](offset) == target) {
                            return true;
                        }
                    }
                    return false;
                }
                default:
                    break;
            }
        }
    }
    return insert_record_.contain(pk);
}

bool
ChunkedSegmentSealedImpl::is_system_field_ready() const {
    if (!insert_record_.timestamps_.empty()) {
        return true;
    }
    return get_column(TimestampFieldID) != nullptr;
}

void
ChunkedSegmentSealedImpl::init_storage_v2_timestamp_index(
    const std::shared_ptr<ChunkedColumnInterface>& column,
    size_t num_rows,
    const std::string& warmup_policy) {
    std::unique_ptr<Translator<storagev2translator::TimestampIndexCell>>
        translator =
            std::make_unique<storagev2translator::TimestampIndexTranslator>(
                id_, column, num_rows, warmup_policy);
    *timestamp_index_slot_.wlock() =
        Manager::GetInstance().CreateCacheSlot(std::move(translator));

    // Provide a callback so DeletedRecord can read insert timestamps
    // from the column even when insert_record_.timestamps_ is empty
    // (StorageV2 lazy-init path). This preserves the same-timestamp
    // correctness check in DeletedRecord::InternalPush.
    auto ts_col = column;
    deleted_record_.set_get_insert_timestamp_func(
        [ts_col](int64_t row_id) -> Timestamp {
            auto num_chunks = ts_col->num_chunks();
            int64_t offset = 0;
            for (int64_t c = 0; c < num_chunks; ++c) {
                auto chunk_rows = ts_col->chunk_row_nums(c);
                if (row_id < offset + chunk_rows) {
                    auto pin = ts_col->GetChunk(nullptr, c);
                    auto* chunk_data = reinterpret_cast<const Timestamp*>(
                        pin.get()->RawData());
                    return chunk_data[row_id - offset];
                }
                offset += chunk_rows;
            }
            return 0;
        });
}

void
ChunkedSegmentSealedImpl::init_storage_v1_pk_index(
    FieldId field_id,
    const std::shared_ptr<ChunkedColumnInterface>& column,
    DataType data_type,
    bool is_replace) {
    if (schema_->get_primary_field_id().value_or(FieldId(-1)) != field_id) {
        return;
    }
    // Build compressed offset->pk for FillPrimaryKeys fast path
    insert_record_.build_offset2pk(data_type, column.get());

    if (!is_sorted_by_pk_) {
        AssertInfo(field_id.get() != -1, "Primary key is -1");
        if (!is_replace) {
            AssertInfo(insert_record_.empty_pks(),
                       "primary key records already exists, current "
                       "field id {}",
                       field_id.get());
            insert_record_.insert_pks(data_type, column.get());
            insert_record_.seal_pks();
        }
    }
}

void
ChunkedSegmentSealedImpl::init_storage_v2_pk_index(
    FieldId field_id,
    const std::shared_ptr<ChunkedColumnInterface>& column,
    DataType data_type) {
    if (schema_->get_primary_field_id().value_or(FieldId(-1)) != field_id) {
        return;
    }
    std::unique_ptr<Translator<storagev2translator::PkIndexCell>> translator =
        std::make_unique<storagev2translator::PkIndexTranslator>(
            id_, column, data_type, is_sorted_by_pk_);
    *pk_index_slot_.wlock() =
        Manager::GetInstance().CreateCacheSlot(std::move(translator));
}

void
ChunkedSegmentSealedImpl::LoadIndex(LoadIndexInfo& info) {
    LoadIndex(info, false);
}

void
ChunkedSegmentSealedImpl::LoadIndex(LoadIndexInfo& info, bool is_replace) {
    // print(info);
    // NOTE: lock only when data is ready to avoid starvation
    auto field_id = FieldId(info.field_id);
    auto& field_meta = schema_->operator[](field_id);

    if (field_meta.is_vector()) {
        LoadVecIndex(info, is_replace);
    } else {
        LoadScalarIndex(info, is_replace);
    }
}

void
ChunkedSegmentSealedImpl::LoadVecIndex(LoadIndexInfo& info, bool is_replace) {
    // NOTE: lock only when data is ready to avoid starvation
    auto total_start = std::chrono::steady_clock::now();
    IndexRegisterTiming timing;
    timing.is_vector = true;
    auto field_id = FieldId(info.field_id);

    AssertInfo(info.index_params.count("metric_type"),
               "Can't get metric_type in index_params");
    auto metric_type = info.index_params.at("metric_type");

    auto stage_start = std::chrono::steady_clock::now();
    std::unique_lock lck(mutex_);
    timing.lock_wait_ns = DurationNs(stage_start);
    if (is_replace) {
        // Drop existing vector indexing for this field before replacing
        if (get_bit(index_ready_bitset_, field_id)) {
            vector_indexings_.drop_field_indexing(field_id);
        }
        LOG_INFO("Replacing vector index for field {} in segment {}",
                 field_id.get(),
                 id_);
    } else {
        AssertInfo(
            !get_bit(index_ready_bitset_, field_id),
            "vector index has been exist at " + std::to_string(field_id.get()));
    }
    LOG_INFO(
        "Before setting field_bit for field index, fieldID:{}. "
        "segmentID:{}, ",
        info.field_id,
        id_);
    auto& field_meta = schema_->operator[](field_id);
    bool has_raw_data;
    if (info.has_raw_data.has_value()) {
        has_raw_data = *info.has_raw_data;
    } else {
        stage_start = std::chrono::steady_clock::now();
        LoadResourceRequest request =
            milvus::index::IndexFactory::GetInstance().VecIndexLoadResource(
                field_meta.get_data_type(),
                info.element_type,
                info.index_engine_version,
                info.index_size,
                info.index_params,
                info.enable_mmap,
                info.num_rows,
                info.dim);
        timing.resource_ns = DurationNs(stage_start);
        has_raw_data = request.has_raw_data;
    }
    timing.has_raw_data = has_raw_data;

    // Note: raw data lifecycle (eviction/drop) is handled by LoadDiff + ApplyLoadDiff,
    // not here. This avoids unsafe ManualEvictCache on column groups.

    stage_start = std::chrono::steady_clock::now();
    if (get_bit(binlog_index_bitset_, field_id)) {
        set_bit(binlog_index_bitset_, field_id, false);
        vector_indexings_.drop_field_indexing(field_id);
    }
    vector_indexings_.append_field_indexing(
        field_id, metric_type, std::move(info.cache_index));
    set_bit(index_ready_bitset_, field_id, true);
    index_has_raw_data_[field_id] = has_raw_data;
    LOG_INFO("Has load vec index done, fieldID:{}. segmentID:{}, ",
             info.field_id,
             id_);
    timing.register_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    index_register_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadScalarIndex(LoadIndexInfo& info,
                                          bool is_replace) {
    // NOTE: lock only when data is ready to avoid starvation
    auto total_start = std::chrono::steady_clock::now();
    IndexRegisterTiming timing;
    auto field_id = FieldId(info.field_id);
    auto& field_meta = schema_->operator[](field_id);

    auto is_pk =
        field_id == schema_->get_primary_field_id().value_or(FieldId(-1));

    LOG_INFO("LoadScalarIndex, fieldID:{}. segmentID:{}, is_pk:{}",
             info.field_id,
             id_,
             is_pk);
    // if segment is pk sorted, user created indexes bring no performance gain but extra memory usage
    if (is_pk && is_sorted_by_pk_) {
        LOG_INFO(
            "segment pk sorted, skip user index loading for primary key "
            "field");
        timing.total_ns = DurationNs(total_start);
        index_register_timing_stats.Record(timing);
        return;
    }

    auto stage_start = std::chrono::steady_clock::now();
    std::unique_lock lck(mutex_);
    timing.lock_wait_ns = DurationNs(stage_start);
    if (is_replace) {
        // Drop existing scalar indexing before replacing
        if (get_bit(index_ready_bitset_, field_id)) {
            auto [scalar_indexings, ngram_fields] = lock(
                folly::wlock(scalar_indexings_), folly::wlock(ngram_fields_));
            cancel_and_erase_scalar_index(*scalar_indexings, field_id);
            ngram_fields->erase(field_id);
        }
        LOG_INFO("Replacing scalar index for field {} in segment {}",
                 field_id.get(),
                 id_);
    } else {
        AssertInfo(
            !get_bit(index_ready_bitset_, field_id),
            "scalar index has been exist at " + std::to_string(field_id.get()));
    }

    stage_start = std::chrono::steady_clock::now();
    if (field_meta.get_data_type() == DataType::JSON) {
        auto path = info.index_params.at(JSON_PATH);
        if (auto it = info.index_params.find(index::INDEX_TYPE);
            it != info.index_params.end() &&
            it->second == index::NGRAM_INDEX_TYPE) {
            auto ngram_indexings = ngram_indexings_.wlock();
            auto& path_indexings = (*ngram_indexings)[field_id];
            if (auto path_it = path_indexings.find(path);
                path_it != path_indexings.end()) {
                cancel_warmup(path_it->second);
            }
            path_indexings[path] = std::move(info.cache_index);
            timing.register_ns = DurationNs(stage_start);
            timing.total_ns = DurationNs(total_start);
            index_register_timing_stats.Record(timing);
            return;
        } else {
            JsonIndex index;
            index.nested_path = path;
            index.field_id = field_id;
            index.index = std::move(info.cache_index);
            index.cast_type =
                JsonCastType::FromString(info.index_params.at(JSON_CAST_TYPE));
            json_indices.withWLock([&](auto& json_indexings) {
                cancel_and_erase_json_indices(json_indexings, field_id, path);
                json_indexings.push_back(std::move(index));
            });
            timing.register_ns = DurationNs(stage_start);
            timing.total_ns = DurationNs(total_start);
            index_register_timing_stats.Record(timing);
            return;
        }
    }

    if (auto it = info.index_params.find(index::INDEX_TYPE);
        it != info.index_params.end() &&
        it->second == index::NGRAM_INDEX_TYPE) {
        auto [scalar_indexings, ngram_fields] =
            lock(folly::wlock(scalar_indexings_), folly::wlock(ngram_fields_));
        ngram_fields->insert(field_id);
        scalar_indexings->insert({field_id, std::move(info.cache_index)});
    } else {
        scalar_indexings_.wlock()->insert(
            {field_id, std::move(info.cache_index)});
    }
    timing.register_ns += DurationNs(stage_start);

    bool has_raw_data;
    if (info.has_raw_data.has_value()) {
        has_raw_data = *info.has_raw_data;
    } else {
        stage_start = std::chrono::steady_clock::now();
        LoadResourceRequest request =
            milvus::index::IndexFactory::GetInstance().ScalarIndexLoadResource(
                field_meta.get_data_type(),
                info.index_engine_version,
                info.index_size,
                info.index_params,
                info.enable_mmap);
        timing.resource_ns = DurationNs(stage_start);
        has_raw_data = request.has_raw_data;
    }
    timing.has_raw_data = has_raw_data;

    stage_start = std::chrono::steady_clock::now();
    set_bit(index_ready_bitset_, field_id, true);
    index_has_raw_data_[field_id] = has_raw_data;
    // Note: raw data lifecycle (eviction/drop) is handled by LoadDiff + ApplyLoadDiff,
    // not here. This avoids unsafe ManualEvictCache on column groups.
    LOG_INFO(
        "Has load scalar index done, fieldID:{}. segmentID:{}, has_raw_data:{}",
        info.field_id,
        id_,
        has_raw_data);
    timing.register_ns += DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    index_register_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadFieldData(const LoadFieldDataInfo& load_info,
                                        milvus::OpContext* op_ctx) {
    LoadFieldData(load_info, op_ctx, false);
}

void
ChunkedSegmentSealedImpl::LoadFieldData(const LoadFieldDataInfo& load_info,
                                        milvus::OpContext* op_ctx,
                                        bool is_replace) {
    switch (load_info.storage_version) {
        case 2: {
            load_column_group_data_internal(load_info, op_ctx, is_replace);
            break;
        }
        default:
            load_field_data_internal(load_info, op_ctx, is_replace);
            break;
    }
}

void
ChunkedSegmentSealedImpl::LoadColumnGroups(const std::string& manifest_path,
                                           milvus::OpContext* op_ctx) {
    auto load_cg_start = std::chrono::high_resolution_clock::now();
    LOG_INFO(
        "[LoadColumnGroups] segment {} start, manifest {}", id_, manifest_path);
    CheckCancellation(
        op_ctx, id_, "ChunkedSegmentSealedImpl::LoadColumnGroups()");
    auto properties = std::make_shared<milvus_storage::api::Properties>(
        *milvus::storage::LoonFFIPropertiesSingleton::GetInstance()
             .GetProperties());
    auto load_info = std::atomic_load(&segment_load_info_);
    auto column_groups = load_info->GetColumnGroups();

    // External collections: inject extfs.{collectionID}.* derived from
    // external_source and external_spec only. InjectExternalSpecProperties zero-
    // initializes every extfs field so nothing is inherited from the
    // cluster's internal fs.* baseline — credentials and endpoint come
    // exclusively from spec.extfs (see refactor: [ExternalTable] isolate
    // extfs namespace from fs.* baseline). milvus_storage routes each file
    // URI to the matching extfs alias by (bucket, address); file URIs in
    // the Iceberg manifest live under external_source, so the alias always
    // matches.
    if (schema_->is_external_collection()) {
        InjectExternalSpecProperties(*properties,
                                     load_info->GetCollectionID(),
                                     schema_->get_external_source(),
                                     schema_->get_external_spec());
    }

    // Schemaless reader for external collections: pass nullptr schema and
    // let the Reader derive types from file metadata (Parquet footer).
    // FillFieldData handles Parquet-native → Milvus type conversion.
    //
    // This overload is reached only via
    // ApplyLoadDiff when load_external_manifest is set, which in turn is
    // gated on is_external_collection() — see SegmentLoadInfo.cpp where the
    // flag is assigned. The non-external path uses LoadColumnGroups(
    // column_groups, ...) and never enters here.
    auto needed_columns = schema_->GetExternalColumnNames();
    // reader_mutex_ guards reader_ against concurrent use in ExecuteTake.
    // Reopen reaches this function with mutex_ already released (see Reopen
    // for the rationale), so without this lock a concurrent ExecuteTake can
    // observe a mid-assigned shared_ptr or drop the old Reader's refcount
    // while another thread is still calling take() on it. Initial load is
    // uncontended (segment not yet ready), so the extra lock is free.
    {
        std::lock_guard<std::mutex> lock(reader_mutex_);
        reader_ = milvus_storage::api::Reader::create(column_groups,
                                                      /*arrow_schema=*/nullptr,
                                                      needed_columns,
                                                      *properties);
    }

    auto reader_create_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - load_cg_start)
            .count();
    LOG_INFO(
        "[LoadColumnGroups] segment {} reader created in {}ms, {} column "
        "groups",
        id_,
        reader_create_ms,
        column_groups->size());

    // Pre-resolve field IDs for each column group, then reuse the
    // standard LoadColumnGroup overload.
    std::vector<std::pair<int, std::vector<FieldId>>> cg_field_ids;
    cg_field_ids.reserve(column_groups->size());
    for (size_t i = 0; i < column_groups->size(); ++i) {
        auto cg = column_groups->at(i);
        std::vector<FieldId> field_ids;
        field_ids.reserve(cg->columns.size());
        for (auto& column : cg->columns) {
            field_ids.emplace_back(schema_->ResolveColumnFieldId(column));
        }
        cg_field_ids.emplace_back(static_cast<int>(i), std::move(field_ids));
    }

    // Split each column group's fields into eager (warmup=sync/async) and
    // lazy (warmup=disable) subsets, so that each subset creates its own
    // ChunkReader with column projection.  This avoids downloading all
    // columns from S3 when only a subset needs eager warming.
    struct FieldGroupTask {
        int cg_index;
        std::vector<FieldId> field_ids;
        bool eager_load;
    };
    std::vector<FieldGroupTask> tasks;

    for (const auto& pair : cg_field_ids) {
        auto cg_index = pair.first;
        const auto& all_fields = pair.second;

        std::vector<FieldId> eager_fields;
        std::vector<FieldId> lazy_fields;

        for (const auto& field_id : all_fields) {
            const auto& field_meta = (*schema_)[field_id];
            bool field_is_vector = IsVectorDataType(field_meta.get_data_type());
            auto [has_warmup, warmup_str] = schema_->WarmupPolicy(
                field_id, field_is_vector, /*is_index=*/false);
            // Resolve effective warmup using global config as fallback
            auto resolved = getCacheWarmupPolicy(has_warmup ? warmup_str : "",
                                                 field_is_vector,
                                                 /*is_index=*/false,
                                                 /*in_load_list=*/true);
            if (resolved != CacheWarmupPolicy::CacheWarmupPolicy_Disable) {
                eager_fields.push_back(field_id);
            } else {
                lazy_fields.push_back(field_id);
            }
        }

        if (!eager_fields.empty()) {
            tasks.push_back({cg_index, std::move(eager_fields), true});
        }
        // Lazy fields are emitted one-per-field so that each creates its
        // own single-column projected ChunkReader. Accessing one lazy
        // field (e.g. caption) will not co-load sibling lazy fields
        // (e.g. vector), avoiding unnecessary S3 downloads.
        for (const auto& fid : lazy_fields) {
            tasks.push_back({cg_index, {fid}, false});
        }
        if (!eager_fields.empty() && !lazy_fields.empty()) {
            LOG_INFO(
                "[LoadColumnGroups] segment {} cg {} split: {} eager, {} lazy",
                get_segment_id(),
                cg_index,
                eager_fields.size(),
                lazy_fields.size());
        }
    }

    LOG_INFO(
        "[LoadColumnGroups] segment {} external table: {} tasks from {} column "
        "groups",
        get_segment_id(),
        tasks.size(),
        cg_field_ids.size());

    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> load_group_futures;
    for (auto& task : tasks) {
        auto future = pool.Submit([this,
                                   column_groups,
                                   properties,
                                   cg_index = task.cg_index,
                                   field_ids = std::move(task.field_ids),
                                   eager_load = task.eager_load,
                                   op_ctx] {
            CheckCancellation(op_ctx,
                              id_,
                              cg_index,
                              "ChunkedSegmentSealedImpl::LoadColumnGroup()");
            LoadColumnGroup(column_groups,
                            properties,
                            cg_index,
                            field_ids,
                            eager_load,
                            op_ctx,
                            /*is_replace=*/false);
        });
        load_group_futures.emplace_back(std::move(future));
    }

    storage::WaitAllFutures(load_group_futures);

    if (schema_->is_external_collection()) {
        SynthesizeExternalSystemFields();
    }
}

void
ChunkedSegmentSealedImpl::SynthesizeExternalSystemFields() {
    int64_t num_rows = std::atomic_load(&segment_load_info_)->GetNumOfRows();
    if (num_rows == 0) {
        std::unique_lock lck(mutex_);
        update_row_count(0);
        // Initialize empty timestamps so is_system_field_ready() returns true
        insert_record_.init_timestamps_from_owned({}, TimestampIndex());
        return;
    }

    // 1. VirtualPKChunkedColumn for the synthetic primary key
    //    This is lazy — data is only materialized if DataOfChunk/Span is called.
    auto pk_field_id = schema_->get_primary_field_id().value();
    auto virtual_pk = std::make_shared<VirtualPKChunkedColumn>(id_, num_rows);
    fields_.wlock()->emplace(pk_field_id, virtual_pk);
    set_bit(field_data_ready_bitset_, pk_field_id, true);

    // 2. PK→offset index using VirtualPKOffsetMap (zero storage).
    //    Virtual PK = (seg_id << 32) | offset, so pk→offset is a simple
    //    bit-extract. This replaces the OffsetOrderedArray that would
    //    otherwise store num_rows (pk, offset) pairs (~17 GB for 1B rows).
    insert_record_.set_virtual_pk_offset_map(id_, num_rows);

    // 3. Synthetic timestamps: constant mode (all 0 — rows always visible).
    //    No data is materialized, saving ~8 GB for 1B-row external tables.
    insert_record_.init_timestamps_constant(num_rows, 0);

    // 4. Row count + readiness
    {
        std::unique_lock lck(mutex_);
        update_row_count(num_rows);
    }
}

namespace {

struct FileMetadataLoadResult {
    milvus_storage::RowGroupMetadataVector row_group_meta;
    // per field_id → per-row-group statistics; nullptr entry means the row
    // group had no statistics set for this field.
    std::map<int64_t, std::vector<std::shared_ptr<parquet::Statistics>>>
        per_field_row_group_stats;
};

}  // namespace

LoadedGroupChunkMetadata
LoadGroupChunkMetadata(const std::vector<std::string>& insert_files,
                       const std::vector<FieldId>& field_ids_for_stats,
                       const std::string& debug_key) {
    auto fs = milvus::segcore::GetDefaultArrowFileSystem();
    auto& pool = ThreadPools::GetThreadPool(ThreadPoolPriority::HIGH);

    std::vector<std::future<FileMetadataLoadResult>> futures;
    futures.reserve(insert_files.size());
    for (const auto& file : insert_files) {
        // Futures are always joined below before this function returns, so
        // capturing loader inputs by reference is safe here.
        futures.push_back(pool.Submit([&fs,
                                       file,
                                       &field_ids_for_stats,
                                       &debug_key]() {
            auto result = milvus_storage::FileRowGroupReader::Make(
                fs,
                file,
                milvus_storage::DEFAULT_READ_BUFFER_SIZE,
                storage::GetReaderProperties(),
                storage::GetArrowReaderProperties());
            AssertInfo(result.ok(),
                       "[StorageV2] Failed to create file row group reader: " +
                           result.status().ToString());

            auto reader = result.ValueOrDie();
            FileMetadataLoadResult load_result;
            auto file_metadata = reader->file_metadata();
            load_result.row_group_meta =
                file_metadata->GetRowGroupMetadataVector();

            if (!field_ids_for_stats.empty()) {
                auto field_id_mapping = file_metadata->GetFieldIDMapping();
                auto parquet_metadata = file_metadata->GetParquetMetadata();
                auto num_row_groups = parquet_metadata->num_row_groups();
                for (const auto& field_id : field_ids_for_stats) {
                    auto it = field_id_mapping.find(field_id.get());
                    AssertInfo(it != field_id_mapping.end(),
                               "field id {} not found in field id mapping",
                               field_id.get());
                    auto& per_rg =
                        load_result.per_field_row_group_stats[field_id.get()];
                    per_rg.reserve(num_row_groups);
                    for (int i = 0; i < num_row_groups; ++i) {
                        auto column_chunk =
                            parquet_metadata->RowGroup(i)->ColumnChunk(
                                it->second.col_index);
                        per_rg.push_back(column_chunk->is_stats_set()
                                             ? column_chunk->statistics()
                                             : nullptr);
                    }
                }
            }

            auto status = reader->Close();
            AssertInfo(status.ok(),
                       "[StorageV2] metadata loader {} failed to close "
                       "file reader for {} with error {}",
                       debug_key,
                       file,
                       status.ToString());
            return load_result;
        }));
    }

    auto futures_guard = folly::makeGuard([&futures]() {
        for (auto& future : futures) {
            if (future.valid()) {
                try {
                    future.get();
                } catch (...) {
                }
            }
        }
    });

    LoadedGroupChunkMetadata metadata;
    metadata.row_group_meta_list.reserve(insert_files.size());

    for (auto& future : futures) {
        auto load_result = future.get();
        metadata.row_group_meta_list.push_back(
            std::move(load_result.row_group_meta));
        // Walk files in order and replicate the original single-threaded
        // semantics: once any row group has reported stats for a field, every
        // subsequent row group (in this file or any later file) must also
        // report stats; otherwise fail.
        for (const auto& field_id : field_ids_for_stats) {
            auto& stats_vec = metadata.parquet_stats_by_field[field_id.get()];
            auto it =
                load_result.per_field_row_group_stats.find(field_id.get());
            if (it == load_result.per_field_row_group_stats.end()) {
                continue;
            }
            for (auto& stat : it->second) {
                if (stat == nullptr) {
                    AssertInfo(stats_vec.empty(),
                               "Statistics is not set for some column chunks "
                               "for field {}",
                               field_id.get());
                    continue;
                }
                stats_vec.push_back(std::move(stat));
            }
        }
    }

    return metadata;
}

void
ChunkedSegmentSealedImpl::load_column_group_data_internal(
    const LoadFieldDataInfo& load_info,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    size_t num_rows = storage::GetNumRowsForLoadInfo(load_info);
    ArrowSchemaPtr arrow_schema = schema_->ConvertToArrowSchema();
    auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();

    for (auto& [id, info] : load_info.field_infos) {
        auto total_start = std::chrono::steady_clock::now();
        auto stage_start = std::chrono::steady_clock::now();
        StorageV2FieldDataTiming timing;
        timing.enable_mmap = info.enable_mmap;

        AssertInfo(info.row_count > 0,
                   "[StorageV2] The row count of field data is 0");

        auto column_group_id = FieldId(id);
        auto insert_files = info.insert_files;
        storage::SortByPath(insert_files);
        auto fs = milvus::segcore::GetDefaultArrowFileSystem();

        milvus_storage::FieldIDList field_id_list;
        if (info.child_field_ids.size() == 0) {
            // legacy binlog meta, parse from reader
            field_id_list = storage::GetFieldIDList(
                column_group_id, insert_files[0], arrow_schema, fs);
        } else {
            field_id_list = milvus_storage::FieldIDList(info.child_field_ids);
        }
        timing.resolve_fields_ns = DurationNs(stage_start);

        // if multiple fields share same column group
        // hint for not loading certain field shall not be working for now
        // warmup will be disabled only when all columns are not in load list
        stage_start = std::chrono::steady_clock::now();
        bool merged_in_load_list = false;
        std::vector<FieldId> milvus_field_ids;
        milvus_field_ids.reserve(field_id_list.size());
        for (int i = 0; i < field_id_list.size(); ++i) {
            milvus_field_ids.emplace_back(field_id_list.Get(i));
            merged_in_load_list = merged_in_load_list ||
                                  schema_->ShouldLoadField(milvus_field_ids[i]);
        }

        auto mmap_dir_path =
            milvus::storage::LocalChunkManagerSingleton::GetInstance()
                .GetChunkManager()
                ->GetRootPath();
        auto column_group_info = FieldDataInfo(column_group_id.get(),
                                               num_rows,
                                               mmap_dir_path,
                                               merged_in_load_list);
        LOG_DEBUG(
            "[StorageV2] segment {} loads column group {} with field ids "
            "{} "
            "with "
            "num_rows "
            "{} mmap_dir_path={}",
            this->get_segment_id(),
            column_group_id.get(),
            field_id_list.ToString(),
            num_rows,
            mmap_dir_path);

        auto field_metas = schema_->get_field_metas(milvus_field_ids);

        std::vector<FieldId> fields_for_stats;
        if (ENABLE_PARQUET_STATS_SKIP_INDEX) {
            fields_for_stats = milvus_field_ids;
        } else {
            for (auto field_id : milvus_field_ids) {
                const auto& fm = field_metas.at(field_id);
                if (fm.is_nullable() && IsVectorDataType(fm.get_data_type())) {
                    fields_for_stats.push_back(field_id);
                }
            }
        }
        timing.field_count = milvus_field_ids.size();
        timing.stats_field_count = fields_for_stats.size();
        for (auto field_id : milvus_field_ids) {
            const auto& field_meta = field_metas.at(field_id);
            if (IsVectorDataType(field_meta.get_data_type())) {
                timing.has_vector = true;
                break;
            }
        }
        timing.schema_meta_ns = DurationNs(stage_start);

        stage_start = std::chrono::steady_clock::now();
        auto metadata = LoadGroupChunkMetadata(
            insert_files,
            fields_for_stats,
            fmt::format(
                "seg_{}_cg_{}", get_segment_id(), column_group_id.get()));
        auto parquet_stats_by_field =
            std::move(metadata.parquet_stats_by_field);
        timing.metadata_ns = DurationNs(stage_start);

        stage_start = std::chrono::steady_clock::now();
        auto translator =
            std::make_unique<storagev2translator::GroupChunkTranslator>(
                get_segment_id(),
                GroupChunkType::DEFAULT,
                field_metas,
                column_group_info,
                std::move(insert_files),
                std::move(metadata.row_group_meta_list),
                info.enable_mmap,
                mmap_config.GetMmapPopulate(),
                milvus_field_ids.size(),
                load_info.load_priority,
                info.warmup_policy);
        timing.create_translator_ns = DurationNs(stage_start);

        stage_start = std::chrono::steady_clock::now();
        auto chunked_column_group =
            std::make_shared<ChunkedColumnGroup>(std::move(translator));
        timing.create_chunk_group_ns = DurationNs(stage_start);

        // Create ProxyChunkColumn for each field in this column group
        stage_start = std::chrono::steady_clock::now();
        for (const auto& field_id : milvus_field_ids) {
            const auto& field_meta = field_metas.at(field_id);
            auto column = std::make_shared<ProxyChunkColumn>(
                chunked_column_group, field_id, field_meta);
            auto data_type = field_meta.get_data_type();
            std::optional<ParquetStatistics> statistics_opt;
            auto it = parquet_stats_by_field.find(field_id.get());
            if (it != parquet_stats_by_field.end()) {
                statistics_opt = std::move(it->second);
            }

            load_field_data_common(field_id,
                                   column,
                                   num_rows,
                                   data_type,
                                   info.enable_mmap,
                                   true,
                                   statistics_opt,
                                   op_ctx,
                                   is_replace);
            if (field_id == TimestampFieldID) {
                if (commit_ts_ != 0) {
                    std::vector<Timestamp> ts(num_rows, commit_ts_);
                    init_storage_v1_timestamp_index(std::move(ts), num_rows);
                } else {
                    init_storage_v2_timestamp_index(
                        column, num_rows, info.warmup_policy);
                }
            }
        }
        timing.proxy_common_ns = DurationNs(stage_start);
        timing.total_ns = DurationNs(total_start);
        storage_v2_field_data_timing_stats.Record(timing);

        if (column_group_id.get() == DEFAULT_SHORT_COLUMN_GROUP_ID) {
            stats_.mem_size += chunked_column_group->memory_size();
        }
    }
}

void
ChunkedSegmentSealedImpl::load_field_data_internal(
    const LoadFieldDataInfo& load_info,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    SCOPE_CGO_CALL_METRIC();

    auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();

    size_t num_rows = storage::GetNumRowsForLoadInfo(load_info);
    AssertInfo(
        !num_rows_.has_value() || num_rows_ == num_rows,
        "num_rows_ is set but not equal to num_rows of LoadFieldDataInfo");

    for (auto& [id, info] : load_info.field_infos) {
        auto total_start = std::chrono::steady_clock::now();
        FieldDataInternalTiming timing;
        AssertInfo(info.row_count > 0, "The row count of field data is 0");

        auto field_id = FieldId(id);
        timing.is_system = SystemProperty::Instance().IsSystem(field_id);
        timing.enable_mmap = info.enable_mmap;

        auto mmap_dir_path =
            milvus::storage::LocalChunkManagerSingleton::GetInstance()
                .GetChunkManager()
                ->GetRootPath();
        auto field_data_info =
            FieldDataInfo(field_id.get(),
                          num_rows,
                          mmap_dir_path,
                          schema_->ShouldLoadField(field_id));
        LOG_DEBUG("segment {} loads field {} with num_rows {}, sorted by pk {}",
                  this->get_segment_id(),
                  field_id.get(),
                  num_rows,
                  is_sorted_by_pk_);

        if (timing.is_system) {
            auto insert_files = info.insert_files;
            storage::SortByPath(insert_files);
            // field_data_info.arrow_reader_channel cannot have capacity
            // othersize deadlock could happen if result count is greater than cap
            // since this branch handles system only, we shall leave channel without cap for quick fix
            auto stage_start = std::chrono::steady_clock::now();
            LoadArrowReaderFromRemote(insert_files,
                                      field_data_info.arrow_reader_channel,
                                      load_info.load_priority);
            timing.system_remote_ns = DurationNs(stage_start);

            LOG_DEBUG("segment {} submits load field {} task to thread pool",
                      this->get_segment_id(),
                      field_id.get());
            stage_start = std::chrono::steady_clock::now();
            load_system_field_internal(
                field_id, field_data_info, load_info.load_priority);
            timing.system_load_ns = DurationNs(stage_start);
            LOG_DEBUG("segment {} loads system field {} mmap false done",
                      this->get_segment_id(),
                      field_id.get());
        } else {
            auto stage_start = std::chrono::steady_clock::now();
            std::vector<storagev1translator::ChunkTranslator::FileInfo>
                file_infos;
            file_infos.reserve(info.insert_files.size());
            for (int i = 0; i < info.insert_files.size(); i++) {
                file_infos.emplace_back(
                    storagev1translator::ChunkTranslator::FileInfo{
                        info.insert_files[i],
                        info.entries_nums[i],
                        info.memory_sizes[i]});
            }

            storage::SortByPath(file_infos);
            timing.build_file_info_ns = DurationNs(stage_start);

            auto field_meta = schema_->operator[](field_id);
            stage_start = std::chrono::steady_clock::now();
            std::unique_ptr<Translator<milvus::Chunk>> translator =
                std::make_unique<storagev1translator::ChunkTranslator>(
                    this->get_segment_id(),
                    field_meta,
                    field_data_info,
                    std::move(file_infos),
                    info.enable_mmap,
                    mmap_config.GetMmapPopulate(),
                    load_info.load_priority,
                    info.warmup_policy);
            timing.create_translator_ns = DurationNs(stage_start);

            auto data_type = field_meta.get_data_type();
            stage_start = std::chrono::steady_clock::now();
            auto slot = cachinglayer::Manager::GetInstance().CreateCacheSlot(
                std::move(translator), op_ctx);
            timing.create_cache_slot_ns = DurationNs(stage_start);
            stage_start = std::chrono::steady_clock::now();
            auto column =
                MakeChunkedColumnBase(data_type, std::move(slot), field_meta);
            timing.create_column_ns = DurationNs(stage_start);

            stage_start = std::chrono::steady_clock::now();
            load_field_data_common(field_id,
                                   column,
                                   num_rows,
                                   data_type,
                                   info.enable_mmap,
                                   false,
                                   std::nullopt,
                                   op_ctx,
                                   is_replace);
            timing.load_common_ns = DurationNs(stage_start);
        }
        timing.total_ns = DurationNs(total_start);
        field_data_internal_timing_stats.Record(timing);
    }
}

void
ChunkedSegmentSealedImpl::load_system_field_internal(
    FieldId field_id,
    FieldDataInfo& data,
    proto::common::LoadPriority load_priority) {
    SCOPE_CGO_CALL_METRIC();

    auto num_rows = data.row_count;
    AssertInfo(SystemProperty::Instance().IsSystem(field_id),
               "system field is not system field");
    auto system_field_type =
        SystemProperty::Instance().GetSystemFieldType(field_id);
    if (system_field_type == SystemFieldType::Timestamp) {
        std::vector<Timestamp> timestamps(num_rows);
        int64_t offset = 0;
        FieldMeta field_meta(
            FieldName(""), FieldId(0), DataType::INT64, false, std::nullopt);
        std::shared_ptr<milvus::ArrowDataWrapper> r;
        while (data.arrow_reader_channel->pop(r)) {
            auto array_vec = read_single_column_batches(r->reader);
            auto chunk = create_chunk(field_meta, array_vec);
            auto chunk_ptr = static_cast<FixedWidthChunk*>(chunk.get());
            std::copy_n(static_cast<const Timestamp*>(chunk_ptr->Span().data()),
                        chunk_ptr->Span().row_count(),
                        timestamps.data() + offset);
            offset += chunk_ptr->Span().row_count();
        }

        if (commit_ts_ != 0) {
            std::fill(timestamps.begin(), timestamps.end(), commit_ts_);
        }
        init_storage_v1_timestamp_index(std::move(timestamps), num_rows);
    } else {
        AssertInfo(system_field_type == SystemFieldType::RowId,
                   "System field type of id column is not RowId");
        // Consume rowid field data but not really load it
        // storage::CollectFieldDataChannel(data.arrow_reader_channel);
        std::shared_ptr<milvus::ArrowDataWrapper> r;
        while (data.arrow_reader_channel->pop(r)) {
        }
    }
    {
        std::unique_lock lck(mutex_);
        update_row_count(num_rows);
    }
}

void
ChunkedSegmentSealedImpl::LoadDeletedRecord(const LoadDeletedRecordInfo& info) {
    SCOPE_CGO_CALL_METRIC();

    AssertInfo(info.row_count > 0, "The row count of deleted record is 0");
    AssertInfo(info.primary_keys, "Deleted primary keys is null");
    AssertInfo(info.timestamps, "Deleted timestamps is null");
    // step 1: get pks and timestamps
    auto field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(field_id.get() != -1, "Primary key is -1");
    auto& field_meta = schema_->operator[](field_id);
    int64_t size = info.row_count;
    std::vector<PkType> pks(size);
    ParsePksFromIDs(pks, field_meta.get_data_type(), *info.primary_keys);
    auto timestamps = reinterpret_cast<const Timestamp*>(info.timestamps);

    // step 2: push delete info to delete_record
    deleted_record_.LoadPush(pks, timestamps);
}

void
ChunkedSegmentSealedImpl::AddFieldDataInfoForSealed(
    const LoadFieldDataInfo& field_data_info) {
    // copy assignment
    field_data_info_ = field_data_info;
}

int64_t
ChunkedSegmentSealedImpl::num_chunk_data(FieldId field_id) const {
    if (!get_bit(field_data_ready_bitset_, field_id)) {
        return 0;
    }
    auto column = get_column(field_id);
    return column ? column->num_chunks() : 1;
}

int64_t
ChunkedSegmentSealedImpl::num_chunk(FieldId field_id) const {
    if (!get_bit(field_data_ready_bitset_, field_id)) {
        return 1;
    }
    auto column = get_column(field_id);
    return column ? column->num_chunks() : 1;
}

int64_t
ChunkedSegmentSealedImpl::size_per_chunk() const {
    return get_row_count();
}

int64_t
ChunkedSegmentSealedImpl::chunk_size(FieldId field_id, int64_t chunk_id) const {
    if (!get_bit(field_data_ready_bitset_, field_id)) {
        return 0;
    }
    auto column = get_column(field_id);
    return column ? column->chunk_row_nums(chunk_id) : num_rows_.value();
}

std::pair<int64_t, int64_t>
ChunkedSegmentSealedImpl::get_chunk_by_offset(FieldId field_id,
                                              int64_t offset) const {
    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "field {} must exist when getting chunk by offset",
               field_id.get());
    return column->GetChunkIDByOffset(offset);
}

int64_t
ChunkedSegmentSealedImpl::num_rows_until_chunk(FieldId field_id,
                                               int64_t chunk_id) const {
    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "field {} must exist when getting rows until chunk",
               field_id.get());
    return column->GetNumRowsUntilChunk(chunk_id);
}

bool
ChunkedSegmentSealedImpl::is_mmap_field(FieldId field_id) const {
    std::shared_lock lck(mutex_);
    return mmap_field_ids_.find(field_id) != mmap_field_ids_.end();
}

void
ChunkedSegmentSealedImpl::prefetch_chunks(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    const std::vector<int64_t>& chunk_ids) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        column->PrefetchChunks(op_ctx, chunk_ids);
    }
}

PinWrapper<SpanBase>
ChunkedSegmentSealedImpl::chunk_data_impl(milvus::OpContext* op_ctx,
                                          FieldId field_id,
                                          int64_t chunk_id) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->Span(op_ctx, chunk_id);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_data_impl only used for chunk column field ");
}

PinWrapper<std::pair<std::vector<ArrayView>, FixedVector<bool>>>
ChunkedSegmentSealedImpl::chunk_array_view_impl(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    int64_t chunk_id,
    std::optional<std::pair<int64_t, int64_t>> offset_len) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->ArrayViews(op_ctx, chunk_id, offset_len);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_array_view_impl only used for chunk column field ");
}

PinWrapper<std::pair<std::vector<VectorArrayView>, FixedVector<bool>>>
ChunkedSegmentSealedImpl::chunk_vector_array_view_impl(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    int64_t chunk_id,
    std::optional<std::pair<int64_t, int64_t>> offset_len) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->VectorArrayViews(op_ctx, chunk_id, offset_len);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_vector_array_view_impl only used for chunk column field ");
}

PinWrapper<std::pair<std::vector<std::string_view>, FixedVector<bool>>>
ChunkedSegmentSealedImpl::chunk_string_view_impl(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    int64_t chunk_id,
    std::optional<std::pair<int64_t, int64_t>> offset_len) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->StringViews(op_ctx, chunk_id, offset_len);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_string_view_impl only used for variable column field ");
}

PinWrapper<std::pair<std::vector<std::string_view>, FixedVector<bool>>>
ChunkedSegmentSealedImpl::chunk_string_views_by_offsets(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    int64_t chunk_id,
    const FixedVector<int32_t>& offsets) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->StringViewsByOffsets(op_ctx, chunk_id, offsets);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_view_by_offsets only used for variable column field ");
}

PinWrapper<std::pair<std::vector<ArrayView>, FixedVector<bool>>>
ChunkedSegmentSealedImpl::chunk_array_views_by_offsets(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    int64_t chunk_id,
    const FixedVector<int32_t>& offsets) const {
    std::shared_lock lck(mutex_);
    AssertInfo(get_bit(field_data_ready_bitset_, field_id),
               "Can't get bitset element at " + std::to_string(field_id.get()));
    if (auto column = get_column(field_id)) {
        return column->ArrayViewsByOffsets(op_ctx, chunk_id, offsets);
    }
    ThrowInfo(ErrorCode::UnexpectedError,
              "chunk_array_views_by_offsets only used for variable column "
              "field ");
}

PinWrapper<index::NgramInvertedIndex*>
ChunkedSegmentSealedImpl::GetNgramIndex(milvus::OpContext* op_ctx,
                                        FieldId field_id) const {
    std::shared_lock lck(mutex_);
    auto [scalar_indexings, ngram_fields] =
        lock(folly::rlock(scalar_indexings_), folly::rlock(ngram_fields_));

    auto has = ngram_fields->find(field_id);
    if (has == ngram_fields->end()) {
        return PinWrapper<index::NgramInvertedIndex*>(nullptr);
    }

    auto iter = scalar_indexings->find(field_id);
    if (iter == scalar_indexings->end()) {
        return PinWrapper<index::NgramInvertedIndex*>(nullptr);
    }
    auto slot = iter->second.get();
    lck.unlock();

    auto ca = SemiInlineGet(slot->PinCells(op_ctx, {0}));
    auto index = dynamic_cast<index::NgramInvertedIndex*>(ca->get_cell_of(0));
    AssertInfo(index != nullptr,
               "ngram index cache is corrupted, field_id: {}",
               field_id.get());
    return PinWrapper<index::NgramInvertedIndex*>(std::move(ca), index);
}

PinWrapper<index::NgramInvertedIndex*>
ChunkedSegmentSealedImpl::GetNgramIndexForJson(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    const std::string& nested_path) const {
    std::shared_lock lck(mutex_);
    return ngram_indexings_.withRLock([&](auto& ngram_indexings) {
        auto iter = ngram_indexings.find(field_id);
        if (iter == ngram_indexings.end() ||
            iter->second.find(nested_path) == iter->second.end()) {
            return PinWrapper<index::NgramInvertedIndex*>(nullptr);
        }

        auto slot = iter->second.at(nested_path).get();

        auto ca = SemiInlineGet(slot->PinCells(op_ctx, {0}));
        auto index =
            dynamic_cast<index::NgramInvertedIndex*>(ca->get_cell_of(0));
        AssertInfo(index != nullptr,
                   "ngram index cache for json is corrupted, field_id: {}, "
                   "nested_path: {}",
                   field_id.get(),
                   nested_path);
        return PinWrapper<index::NgramInvertedIndex*>(std::move(ca), index);
    });
}

int64_t
ChunkedSegmentSealedImpl::get_row_count() const {
    std::shared_lock lck(mutex_);
    return num_rows_.value_or(0);
}

int64_t
ChunkedSegmentSealedImpl::get_deleted_count() const {
    std::shared_lock lck(mutex_);
    return deleted_record_.size();
}

const Schema&
ChunkedSegmentSealedImpl::get_schema() const {
    return *schema_;
}

void
ChunkedSegmentSealedImpl::mask_with_delete(BitsetTypeView& bitset,
                                           int64_t ins_barrier,
                                           Timestamp timestamp) const {
    deleted_record_.Query(bitset, ins_barrier, timestamp);
}

void
ChunkedSegmentSealedImpl::vector_search(SearchInfo& search_info,
                                        const void* query_data,
                                        const size_t* query_offsets,
                                        int64_t query_count,
                                        Timestamp timestamp,
                                        const BitsetView& bitset,
                                        milvus::OpContext* op_context,
                                        SearchResult& output) const {
    AssertInfo(is_system_field_ready(), "System field is not ready");
    auto field_id = search_info.field_id_;
    auto& field_meta = schema_->operator[](field_id);

    AssertInfo(field_meta.is_vector(),
               "The meta type of vector field is not vector type");

    if (get_bit(binlog_index_bitset_, field_id)) {
        AssertInfo(
            vec_binlog_config_.find(field_id) != vec_binlog_config_.end(),
            "The binlog params is not generate.");
        auto binlog_search_info =
            vec_binlog_config_.at(field_id)->GetSearchConf(search_info);

        AssertInfo(vector_indexings_.is_ready(field_id),
                   "vector indexes isn't ready for field " +
                       std::to_string(field_id.get()));
        query::SearchOnSealedIndex(*schema_,
                                   vector_indexings_,
                                   binlog_search_info,
                                   query_data,
                                   query_offsets,
                                   query_count,
                                   bitset,
                                   op_context,
                                   output);
        milvus::tracer::AddEvent(
            "finish_searching_vector_temperate_binlog_index");
    } else if (get_bit(index_ready_bitset_, field_id)) {
        if (search_info.global_refine_enable_ &&
            IsIndexRefineEnabled(op_context, field_id)) {
            search_info.topk_ = GetEffectiveSearchTopk(search_info);
        }
        AssertInfo(vector_indexings_.is_ready(field_id),
                   "vector indexes isn't ready for field " +
                       std::to_string(field_id.get()));
        query::SearchOnSealedIndex(*schema_,
                                   vector_indexings_,
                                   search_info,
                                   query_data,
                                   query_offsets,
                                   query_count,
                                   bitset,
                                   op_context,
                                   output);
        milvus::tracer::AddEvent("finish_searching_vector_index");
    } else {
        AssertInfo(
            get_bit(field_data_ready_bitset_, field_id),
            "Field Data is not loaded: " + std::to_string(field_id.get()));
        AssertInfo(num_rows_.has_value(), "Can't get row count value");
        auto row_count = num_rows_.value();
        auto vec_data = get_column(field_id);
        AssertInfo(
            vec_data != nullptr, "vector field {} not loaded", field_id.get());

        // get index params for bm25 and minhash brute force
        std::map<std::string, std::string> index_info;
        if (search_info.metric_type_ == knowhere::metric::BM25 ||
            search_info.metric_type_ == knowhere::metric::MHJACCARD) {
            index_info =
                col_index_meta_->GetFieldIndexMeta(field_id).GetIndexParams();
        }

        query::SearchOnSealedColumn(*schema_,
                                    vec_data.get(),
                                    search_info,
                                    index_info,
                                    query_data,
                                    query_offsets,
                                    query_count,
                                    row_count,
                                    bitset,
                                    op_context,
                                    output);
        milvus::tracer::AddEvent("finish_searching_vector_data");
    }
}

ChunkedSegmentSealedImpl::ValidResult
ChunkedSegmentSealedImpl::FilterVectorValidOffsets(milvus::OpContext* op_ctx,
                                                   FieldId field_id,
                                                   const int64_t* seg_offsets,
                                                   int64_t count) const {
    ValidResult result;
    result.valid_count = count;

    bool got_valid_offsets_from_index = false;
    if (vector_indexings_.is_ready(field_id)) {
        auto field_indexing = vector_indexings_.get_field_indexing(field_id);
        auto cache_index = field_indexing->indexing_;
        auto ca = SemiInlineGet(cache_index->PinCells(op_ctx, {0}));
        auto vec_index = dynamic_cast<index::VectorIndex*>(ca->get_cell_of(0));

        if (vec_index != nullptr && vec_index->HasValidData()) {
            result.valid_data = std::make_unique<bool[]>(count);
            result.valid_offsets.reserve(count);

            for (int64_t i = 0; i < count; ++i) {
                bool is_valid = vec_index->IsRowValid(seg_offsets[i]);
                result.valid_data[i] = is_valid;
                if (is_valid) {
                    int64_t physical_offset =
                        vec_index->GetPhysicalOffset(seg_offsets[i]);
                    if (physical_offset >= 0) {
                        result.valid_offsets.push_back(physical_offset);
                    }
                }
            }
            result.valid_count = result.valid_offsets.size();
            got_valid_offsets_from_index = true;
        }
    }

    if (!got_valid_offsets_from_index) {
        auto column = get_column(field_id);
        if (column != nullptr && column->IsNullable()) {
            result.valid_data = std::make_unique<bool[]>(count);
            result.valid_offsets.reserve(count);

            std::unordered_set<int64_t> touched_chunks;
            for (int64_t i = 0; i < count; ++i) {
                auto [chunk_id, _] = column->GetChunkIDByOffset(seg_offsets[i]);
                touched_chunks.insert(static_cast<int64_t>(chunk_id));
            }
            for (int64_t c : touched_chunks) {
                column->EnsureChunkOffsetMapping(c, op_ctx);
            }

            const auto& offset_mapping = column->GetOffsetMapping();
            for (int64_t i = 0; i < count; ++i) {
                bool is_valid = offset_mapping.IsValid(seg_offsets[i]);
                result.valid_data[i] = is_valid;
                if (is_valid) {
                    int64_t physical_offset =
                        offset_mapping.GetPhysicalOffset(seg_offsets[i]);
                    if (physical_offset >= 0) {
                        result.valid_offsets.push_back(physical_offset);
                    }
                }
            }
            result.valid_count = result.valid_offsets.size();
        }
    }
    return result;
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::get_vector(milvus::OpContext* op_ctx,
                                     FieldId field_id,
                                     const int64_t* ids,
                                     int64_t count) const {
    auto& field_meta = schema_->operator[](field_id);
    AssertInfo(field_meta.is_vector(), "vector field is not vector type");

    if (!get_bit(index_ready_bitset_, field_id) &&
        !get_bit(binlog_index_bitset_, field_id)) {
        return fill_with_empty(field_id, count);
    }

    AssertInfo(vector_indexings_.is_ready(field_id),
               "vector index is not ready");
    auto field_indexing = vector_indexings_.get_field_indexing(field_id);
    auto cache_index = field_indexing->indexing_;
    auto ca = SemiInlineGet(cache_index->PinCells(op_ctx, {0}));
    auto vec_index = dynamic_cast<index::VectorIndex*>(ca->get_cell_of(0));
    AssertInfo(vec_index, "invalid vector indexing");

    auto has_raw_data = vec_index->HasRawData();

    if (has_raw_data) {
        // If index has raw data, get vector from memory.
        ValidResult filter_result;
        knowhere::DataSetPtr ids_ds;
        int64_t valid_count = count;
        const bool* valid_data = nullptr;
        if (field_meta.is_nullable()) {
            filter_result =
                FilterVectorValidOffsets(op_ctx, field_id, ids, count);
            ids_ds = GenIdsDataset(filter_result.valid_count,
                                   filter_result.valid_offsets.data());
            valid_count = filter_result.valid_count;
            valid_data = filter_result.valid_data.get();
        } else {
            ids_ds = GenIdsDataset(count, ids);
        }
        if (field_meta.get_data_type() == DataType::VECTOR_SPARSE_U32_F32) {
            auto res = vec_index->GetSparseVector(ids_ds);
            return segcore::CreateVectorDataArrayFrom(
                res.get(), valid_data, count, valid_count, field_meta);
        } else {
            // dense vector:
            auto vector = vec_index->GetVector(ids_ds);
            return segcore::CreateVectorDataArrayFrom(
                vector.data(), valid_data, count, valid_count, field_meta);
        }
    }

    AssertInfo(false, "get_vector called on vector index without raw data");
    return nullptr;
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::get_emb_list(milvus::OpContext* op_ctx,
                                       FieldId field_id,
                                       const FieldMeta& field_meta,
                                       const int64_t* seg_offsets,
                                       int64_t count) const {
    AssertInfo(field_meta.get_data_type() == DataType::VECTOR_ARRAY,
               "get_emb_list only supports VECTOR_ARRAY");

    if (!get_bit(index_ready_bitset_, field_id) &&
        !get_bit(binlog_index_bitset_, field_id)) {
        return fill_with_empty(field_id, count);
    }

    AssertInfo(vector_indexings_.is_ready(field_id),
               "vector index is not ready");
    auto field_indexing = vector_indexings_.get_field_indexing(field_id);
    auto cache_index = field_indexing->indexing_;
    auto ca = SemiInlineGet(cache_index->PinCells(op_ctx, {0}));
    auto vec_index = dynamic_cast<index::VectorIndex*>(ca->get_cell_of(0));
    AssertInfo(vec_index, "invalid vector indexing");
    auto has_raw_data = vec_index->HasRawData();
    AssertInfo(has_raw_data,
               "get_emb_list called on vector index without raw data");

    auto metric_type = vec_index->GetMetricType();

    ValidResult filter_result;
    int64_t valid_count = count;
    const bool* valid_data = nullptr;
    const int64_t* valid_offsets = seg_offsets;
    if (field_meta.is_nullable()) {
        filter_result =
            FilterVectorValidOffsets(op_ctx, field_id, seg_offsets, count);
        if (filter_result.valid_data != nullptr) {
            valid_count = filter_result.valid_count;
            valid_data = filter_result.valid_data.get();
            valid_offsets = filter_result.valid_offsets.data();
        }
    }

    auto data_array =
        CreateEmptyVectorDataArray(count, valid_count, valid_data, field_meta);
    if (valid_count == 0) {
        return data_array;
    }

    // Build el_ids dataset from valid_offsets. For nullable VECTOR_ARRAY,
    // FilterVectorValidOffsets maps logical row offsets to the index's compact
    // physical embedding-list ids.
    auto ids_ds = GenIdsDataset(valid_count, valid_offsets);

    auto [raw_data, offsets] = vec_index->GetEmbListByIds(ids_ds, metric_type);
    AssertInfo(offsets.size() == static_cast<size_t>(valid_count + 1),
               "GetEmbListByIds returned invalid offsets size {}, expected {}",
               offsets.size(),
               valid_count + 1);

    auto dim = field_meta.get_dim();
    auto element_type = field_meta.get_element_type();
    const size_t vec_size_per_element =
        milvus::vector_bytes_per_element(element_type, dim);

    auto vector_array = data_array->mutable_vectors();
    auto obj = vector_array->mutable_vector_array();

    std::vector<int64_t> valid_logical_offsets;
    if (valid_data != nullptr) {
        valid_logical_offsets.reserve(valid_count);
        for (int64_t i = 0; i < count; ++i) {
            if (valid_data[i]) {
                valid_logical_offsets.push_back(i);
            }
        }
    }

    // Build a VectorFieldProto for each embedding list
    for (int64_t i = 0; i < valid_count; i++) {
        auto dst_index = valid_data != nullptr ? valid_logical_offsets[i] : i;
        auto* entry = obj->mutable_data()->Mutable(dst_index);
        entry->set_dim(dim);
        size_t vec_start = offsets[i];
        size_t vec_count = offsets[i + 1] - offsets[i];
        if (vec_count == 0) {
            continue;
        }
        size_t byte_offset = vec_start * vec_size_per_element;
        size_t byte_len = vec_count * vec_size_per_element;

        switch (element_type) {
            case DataType::VECTOR_FLOAT: {
                auto* src = reinterpret_cast<const float*>(raw_data.data() +
                                                           byte_offset);
                entry->mutable_float_vector()->mutable_data()->Add(
                    src, src + vec_count * dim);
                break;
            }
            case DataType::VECTOR_BINARY: {
                auto* src = reinterpret_cast<const char*>(raw_data.data() +
                                                          byte_offset);
                entry->mutable_binary_vector()->assign(src, byte_len);
                break;
            }
            case DataType::VECTOR_FLOAT16: {
                auto* src = reinterpret_cast<const char*>(raw_data.data() +
                                                          byte_offset);
                entry->mutable_float16_vector()->assign(src, byte_len);
                break;
            }
            case DataType::VECTOR_BFLOAT16: {
                auto* src = reinterpret_cast<const char*>(raw_data.data() +
                                                          byte_offset);
                entry->mutable_bfloat16_vector()->assign(src, byte_len);
                break;
            }
            case DataType::VECTOR_INT8: {
                auto* src = reinterpret_cast<const char*>(raw_data.data() +
                                                          byte_offset);
                entry->mutable_int8_vector()->assign(src, byte_len);
                break;
            }
            default:
                break;
        }
    }

    return data_array;
}

void
ChunkedSegmentSealedImpl::DropFieldData(const FieldId field_id) {
    AssertInfo(!SystemProperty::Instance().IsSystem(field_id),
               "Dropping system field is not supported, field id: {}",
               field_id.get());
    std::unique_lock<std::shared_mutex> lck(mutex_);
    auto schema_has_field = field_exists_in_schema(schema_, field_id);
    auto column = get_column(field_id);
    if (schema_has_field && schema_->get_primary_field_id().has_value() &&
        schema_->get_primary_field_id().value() == field_id) {
        LOG_INFO(
            "Skip dropping pk field {} in segment {}", field_id.get(), id_);
        if (has_bit_position(binlog_index_bitset_, field_id) &&
            get_bit(binlog_index_bitset_, field_id)) {
            clear_bit_if_present(binlog_index_bitset_, field_id);
            vector_indexings_.drop_field_indexing(field_id);
        }
        return;
    }
    if (column) {
        column->CancelWarmup();
        fields_.wlock()->erase(field_id);
    }
    clear_bit_if_present(field_data_ready_bitset_, field_id);
    if (has_bit_position(binlog_index_bitset_, field_id) &&
        get_bit(binlog_index_bitset_, field_id)) {
        clear_bit_if_present(binlog_index_bitset_, field_id);
        vector_indexings_.drop_field_indexing(field_id);
    }
}

void
ChunkedSegmentSealedImpl::DropIndex(const FieldId field_id) {
    AssertInfo(!SystemProperty::Instance().IsSystem(field_id),
               "Field id:" + std::to_string(field_id.get()) +
                   " isn't one of system type when drop index");
    if (field_exists_in_schema(schema_, field_id)) {
        auto& field_meta = schema_->operator[](field_id);
        AssertInfo(!field_meta.is_vector(), "vector field cannot drop index");
    }

    std::unique_lock lck(mutex_);
    auto [scalar_indexings, ngram_fields] =
        lock(folly::wlock(scalar_indexings_), folly::wlock(ngram_fields_));
    cancel_and_erase_scalar_index(*scalar_indexings, field_id);
    ngram_fields->erase(field_id);
    vector_indexings_.drop_field_indexing(field_id);

    clear_bit_if_present(index_ready_bitset_, field_id);
}

void
ChunkedSegmentSealedImpl::DropJSONIndex(const FieldId field_id,
                                        const std::string& nested_path) {
    std::unique_lock lck(mutex_);
    json_indices.withWLock([&](auto& json_indexings) {
        cancel_and_erase_json_indices(json_indexings, field_id, nested_path);
    });

    ngram_indexings_.withWLock([&](auto& ngram_indexings) {
        cancel_and_erase_ngram_index(ngram_indexings, field_id, nested_path);
    });
}

void
ChunkedSegmentSealedImpl::check_search(const query::Plan* plan) const {
    AssertInfo(plan, "Search plan is null");
    AssertInfo(plan->extra_info_opt_.has_value(),
               "Extra info of search plan doesn't have value");

    if (!is_system_field_ready()) {
        ThrowInfo(FieldNotLoaded,
                  "failed to load row ID or timestamp, potential missing "
                  "bin logs or "
                  "empty segments. Segment ID = " +
                      std::to_string(this->id_));
    }

    auto& request_fields = plan->extra_info_opt_.value().involved_fields_;
    auto field_ready_bitset =
        field_data_ready_bitset_ | index_ready_bitset_ | binlog_index_bitset_;

    // allow absent fields after supporting add fields
    AssertInfo(request_fields.size() >= field_ready_bitset.size(),
               "Request fields size less than field ready bitset size when "
               "check search");

    auto absent_fields = request_fields - field_ready_bitset;

    if (absent_fields.any()) {
        // absent_fields.find_first() returns std::optional<>
        auto field_id =
            FieldId(absent_fields.find_first().value() + START_USER_FIELDID);
        auto& field_meta = plan->schema_->operator[](field_id);
        // request field may has added field
        if (!field_meta.is_nullable()) {
            ThrowInfo(FieldNotLoaded,
                      "User Field(" + field_meta.get_name().get() +
                          ") is not loaded");
        }
    }
}

void
ChunkedSegmentSealedImpl::search_pks(BitsetType& bitset,
                                     const std::vector<PkType>& pks) const {
    if (pks.empty()) {
        return;
    }
    BitsetTypeView bitset_view(bitset);

    // See Contain() — same zero-storage pk2offset fast path.
    if (insert_record_.pk2offset_is_zero_storage()) {
        for (auto& pk : pks) {
            insert_record_.search_pk_range(
                pk, proto::plan::OpType::Equal, bitset_view);
        }
        return;
    }

    if (!is_sorted_by_pk_) {
        auto pk_index = PinPkIndex(nullptr);
        auto* pk_cell = pk_index.get();
        AssertInfo(pk_cell != nullptr || !insert_record_.empty_pks(),
                   "primary key index is not ready");
        for (auto& pk : pks) {
            if (pk_cell != nullptr) {
                pk_cell->pk2offset().find_range(
                    pk,
                    proto::plan::OpType::Equal,
                    bitset_view,
                    [](int64_t offset) { return true; });
            } else {
                insert_record_.search_pk_range(
                    pk, proto::plan::OpType::Equal, bitset_view);
            }
        }
        return;
    }

    auto pk_field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
    auto pk_column = get_column(pk_field_id);
    AssertInfo(pk_column != nullptr, "primary key column not loaded");

    switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
        case DataType::INT64:
            search_pks_with_two_pointers_impl<int64_t>(
                bitset_view, pks, pk_column);
            break;
        case DataType::VARCHAR:
            search_pks_with_two_pointers_impl<std::string>(
                bitset_view, pks, pk_column);
            break;
        default:
            ThrowInfo(
                DataTypeInvalid,
                fmt::format(
                    "unsupported type {}",
                    schema_->get_fields().at(pk_field_id).get_data_type()));
    }
}

void
ChunkedSegmentSealedImpl::search_batch_pks(
    const std::vector<PkType>& pks,
    const std::function<Timestamp(const size_t idx)>& get_timestamp,
    bool include_same_ts,
    const std::function<void(const SegOffset offset, const Timestamp ts)>&
        callback) const {
    // Helper to read a single timestamp by segment offset.
    // For import/CDC segments with commit_ts_ set: every row carries commit_ts_,
    // so short-circuit without touching the raw timestamp column or insert_record_.
    // For StorageV2: pins the timestamp column and indexes into chunks.
    // For StorageV1: reads from insert_record_ directly.
    auto effective_commit_ts = EffectiveCommitTs();
    auto ts_column =
        effective_commit_ts ? nullptr : get_column(TimestampFieldID);
    std::vector<cachinglayer::PinWrapper<Chunk*>> ts_chunk_pins;
    std::vector<int64_t> ts_chunk_offsets;
    if (ts_column) {
        ts_chunk_pins = ts_column->GetAllChunks(nullptr);
        auto num_ts_chunks = ts_column->num_chunks();
        ts_chunk_offsets.resize(num_ts_chunks + 1, 0);
        for (int64_t c = 0; c < num_ts_chunks; c++) {
            ts_chunk_offsets[c + 1] =
                ts_chunk_offsets[c] + ts_column->chunk_row_nums(c);
        }
    } else if (!effective_commit_ts) {
        AssertInfo(!insert_record_.timestamps_.empty(),
                   "timestamp data is not ready");
    }
    auto read_ts = [&](int64_t offset) -> Timestamp {
        if (effective_commit_ts) {
            return *effective_commit_ts;
        }
        if (!ts_column) {
            return insert_record_.timestamps_[offset];
        }
        auto num_ts_chunks = static_cast<int64_t>(ts_chunk_pins.size());
        int64_t c = 0;
        while (c < num_ts_chunks - 1 && offset >= ts_chunk_offsets[c + 1]) {
            ++c;
        }
        auto* data = reinterpret_cast<const Timestamp*>(
            ts_chunk_pins[c].get()->RawData());
        return data[offset - ts_chunk_offsets[c]];
    };

    // Virtual PK offset maps can resolve pk -> offset directly by bit-extract.
    // Avoid the sorted-PK column scan below: external segments synthesize PKs
    // with VirtualPKChunkedColumn, which intentionally does not support
    // GetAllChunks().
    if (insert_record_.pk2offset_is_zero_storage()) {
        auto timestamp_hit =
            include_same_ts
                ? [](Timestamp lhs, Timestamp rhs) { return lhs <= rhs; }
                : [](Timestamp lhs, Timestamp rhs) { return lhs < rhs; };
        for (size_t i = 0; i < pks.size(); i++) {
            auto timestamp = get_timestamp(i);
            for (auto offset : insert_record_.pk2offset_->find(pks[i])) {
                auto insert_ts = read_ts(offset);
                if (timestamp_hit(insert_ts, timestamp)) {
                    callback(SegOffset(offset), timestamp);
                }
            }
        }
        return;
    }

    // handle unsorted case
    if (!is_sorted_by_pk_) {
        auto pk_index = PinPkIndex(nullptr);
        auto* pk_cell = pk_index.get();
        auto timestamp_hit =
            include_same_ts
                ? [](Timestamp lhs, Timestamp rhs) { return lhs <= rhs; }
                : [](Timestamp lhs, Timestamp rhs) { return lhs < rhs; };
        for (size_t i = 0; i < pks.size(); i++) {
            auto timestamp = get_timestamp(i);
            auto offsets = pk_cell != nullptr
                               ? pk_cell->pk2offset().find(pks[i])
                               : insert_record_.pk2offset_->find(pks[i]);
            for (auto offset : offsets) {
                auto insert_ts = read_ts(offset);
                if (timestamp_hit(insert_ts, timestamp)) {
                    callback(SegOffset(offset), timestamp);
                }
            }
        }
        return;
    }

    auto pk_field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
    auto pk_column = get_column(pk_field_id);
    AssertInfo(pk_column != nullptr, "primary key column not loaded");

    auto all_chunk_pins = pk_column->GetAllChunks(nullptr);

    auto timestamp_hit = include_same_ts
                             ? [](const Timestamp& ts1,
                                  const Timestamp& ts2) { return ts1 <= ts2; }
                             : [](const Timestamp& ts1, const Timestamp& ts2) {
                                   return ts1 < ts2;
                               };

    switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
        case DataType::INT64: {
            auto num_chunk = pk_column->num_chunks();
            for (int i = 0; i < num_chunk; ++i) {
                const auto& pw = all_chunk_pins[i];
                auto src =
                    reinterpret_cast<const int64_t*>(pw.get()->RawData());
                auto chunk_row_num = pk_column->chunk_row_nums(i);
                for (size_t j = 0; j < pks.size(); j++) {
                    // get int64 pks
                    auto target = std::get<int64_t>(pks[j]);
                    auto timestamp = get_timestamp(j);
                    auto it = std::lower_bound(
                        src,
                        src + chunk_row_num,
                        target,
                        [](const int64_t& elem, const int64_t& value) {
                            return elem < value;
                        });
                    auto num_rows_until_chunk =
                        pk_column->GetNumRowsUntilChunk(i);
                    for (; it != src + chunk_row_num && *it == target; ++it) {
                        auto offset = it - src + num_rows_until_chunk;
                        auto insert_ts = read_ts(offset);
                        if (timestamp_hit(insert_ts, timestamp)) {
                            callback(SegOffset(offset), timestamp);
                        }
                    }
                }
            }

            break;
        }
        case DataType::VARCHAR: {
            auto num_chunk = pk_column->num_chunks();
            for (int i = 0; i < num_chunk; ++i) {
                // TODO @xiaocai2333, @sunby: chunk need to record the min/max.
                auto num_rows_until_chunk = pk_column->GetNumRowsUntilChunk(i);
                const auto& pw = all_chunk_pins[i];
                auto string_chunk = static_cast<StringChunk*>(pw.get());
                for (size_t j = 0; j < pks.size(); ++j) {
                    // get varchar pks
                    auto& target = std::get<std::string>(pks[j]);
                    auto timestamp = get_timestamp(j);
                    auto offset = string_chunk->binary_search_string(target);
                    for (; offset != -1 && offset < string_chunk->RowNums() &&
                           string_chunk->operator[](offset) == target;
                         ++offset) {
                        auto segment_offset = offset + num_rows_until_chunk;
                        auto insert_ts = read_ts(segment_offset);
                        if (timestamp_hit(insert_ts, timestamp)) {
                            callback(SegOffset(segment_offset), timestamp);
                        }
                    }
                }
            }
            break;
        }
        default: {
            ThrowInfo(
                DataTypeInvalid,
                fmt::format(
                    "unsupported type {}",
                    schema_->get_fields().at(pk_field_id).get_data_type()));
        }
    }
}

void
ChunkedSegmentSealedImpl::pk_range(milvus::OpContext* op_ctx,
                                   proto::plan::OpType op,
                                   const PkType& pk,
                                   BitsetTypeView& bitset) const {
    // See Contain() — same zero-storage pk2offset fast path.
    if (insert_record_.pk2offset_is_zero_storage()) {
        insert_record_.search_pk_range(pk, op, bitset);
        return;
    }
    if (!is_sorted_by_pk_) {
        auto pk_index = PinPkIndex(op_ctx);
        auto* pk_cell = pk_index.get();
        AssertInfo(pk_cell != nullptr || !insert_record_.empty_pks(),
                   "primary key index is not ready");
        if (pk_cell != nullptr) {
            pk_cell->pk2offset().find_range(
                pk, op, bitset, [](int64_t offset) { return true; });
        } else {
            insert_record_.search_pk_range(pk, op, bitset);
        }
        return;
    }

    search_sorted_pk_range(op_ctx, op, pk, bitset);
}

void
ChunkedSegmentSealedImpl::search_sorted_pk_range(milvus::OpContext* op_ctx,
                                                 proto::plan::OpType op,
                                                 const PkType& pk,
                                                 BitsetTypeView& bitset) const {
    auto pk_field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
    auto pk_column = get_column(pk_field_id);
    AssertInfo(pk_column != nullptr, "primary key column not loaded");

    switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
        case DataType::INT64:
            search_sorted_pk_range_impl<int64_t>(
                op, std::get<int64_t>(pk), pk_column, bitset);
            break;
        case DataType::VARCHAR:
            search_sorted_pk_range_impl<std::string>(
                op, std::get<std::string>(pk), pk_column, bitset);
            break;
        default:
            ThrowInfo(
                DataTypeInvalid,
                fmt::format(
                    "unsupported type {}",
                    schema_->get_fields().at(pk_field_id).get_data_type()));
    }
}

void
ChunkedSegmentSealedImpl::pk_binary_range(milvus::OpContext* op_ctx,
                                          const PkType& lower_pk,
                                          bool lower_inclusive,
                                          const PkType& upper_pk,
                                          bool upper_inclusive,
                                          BitsetTypeView& bitset) const {
    // See Contain() — same zero-storage pk2offset fast path.
    if (insert_record_.pk2offset_is_zero_storage()) {
        insert_record_.search_pk_binary_range(
            lower_pk, lower_inclusive, upper_pk, upper_inclusive, bitset);
        return;
    }
    if (!is_sorted_by_pk_) {
        auto pk_index = PinPkIndex(op_ctx);
        auto* pk_cell = pk_index.get();
        AssertInfo(pk_cell != nullptr || !insert_record_.empty_pks(),
                   "primary key index is not ready");
        if (pk_cell != nullptr) {
            auto lower_op = lower_inclusive ? proto::plan::OpType::GreaterEqual
                                            : proto::plan::OpType::GreaterThan;
            auto upper_op = upper_inclusive ? proto::plan::OpType::LessEqual
                                            : proto::plan::OpType::LessThan;
            BitsetType upper_result(bitset.size());
            auto upper_view = upper_result.view();
            pk_cell->pk2offset().find_range(
                lower_pk, lower_op, bitset, [](int64_t offset) {
                    return true;
                });
            pk_cell->pk2offset().find_range(
                upper_pk, upper_op, upper_view, [](int64_t offset) {
                    return true;
                });
            bitset &= upper_result;
        } else {
            insert_record_.search_pk_binary_range(
                lower_pk, lower_inclusive, upper_pk, upper_inclusive, bitset);
        }
        return;
    }

    // For sorted segments, use binary search
    auto pk_field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
    auto pk_column = get_column(pk_field_id);
    AssertInfo(pk_column != nullptr, "primary key column not loaded");

    switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
        case DataType::INT64:
            search_sorted_pk_binary_range_impl<int64_t>(
                std::get<int64_t>(lower_pk),
                lower_inclusive,
                std::get<int64_t>(upper_pk),
                upper_inclusive,
                pk_column,
                bitset);
            break;
        case DataType::VARCHAR:
            search_sorted_pk_binary_range_impl<std::string>(
                std::get<std::string>(lower_pk),
                lower_inclusive,
                std::get<std::string>(upper_pk),
                upper_inclusive,
                pk_column,
                bitset);
            break;
        default:
            ThrowInfo(
                DataTypeInvalid,
                fmt::format(
                    "unsupported type {}",
                    schema_->get_fields().at(pk_field_id).get_data_type()));
    }
}

std::pair<std::vector<OffsetMap::OffsetType>, bool>
ChunkedSegmentSealedImpl::find_first_n(int64_t limit,
                                       const BitsetTypeView& bitset) const {
    if (!is_sorted_by_pk_) {
        auto pk_index = PinPkIndex(nullptr);
        auto* pk_cell = pk_index.get();
        AssertInfo(pk_cell != nullptr || !insert_record_.empty_pks(),
                   "primary key index is not ready");
        if (pk_cell != nullptr) {
            return pk_cell->pk2offset().find_first_n(limit, bitset);
        }
        return insert_record_.pk2offset_->find_first_n(limit, bitset);
    }
    if (limit == Unlimited || limit == NoLimit) {
        limit = num_rows_.value();
    }

    int64_t hit_num = 0;  // avoid counting the number everytime.
    auto size = bitset.size();
    int64_t cnt = size - bitset.count();
    auto more_hit_than_limit = cnt > limit;
    limit = std::min(limit, cnt);
    std::vector<int64_t> seg_offsets;
    seg_offsets.reserve(limit);

    int64_t offset = 0;
    std::optional<size_t> result = bitset.find_first(false);
    while (result.has_value() && hit_num < limit) {
        hit_num++;
        seg_offsets.push_back(result.value());
        offset = result.value();
        if (offset >= size) {
            // In fact, this case won't happen on sealed segments.
            continue;
        }
        result = bitset.find_next(offset, false);
    }

    return {seg_offsets, more_hit_than_limit && result.has_value()};
}

std::tuple<std::vector<int64_t>, std::vector<std::vector<int32_t>>, bool>
ChunkedSegmentSealedImpl::find_first_n_element(
    int64_t limit,
    const BitsetTypeView& element_bitset,
    const IArrayOffsets* array_offsets,
    const std::optional<QueryIteratorCursor>& cursor) const {
    if (!is_sorted_by_pk_) {
        // Not sorted by PK, use pk2offset_ to iterate in PK order
        return insert_record_.pk2offset_->find_first_n_element(
            limit, element_bitset, array_offsets, cursor);
    }

    // Sorted by PK, element_id order = (PK, element_index) order
    // Directly iterate element_bitset in order
    if (limit == Unlimited || limit == NoLimit) {
        limit = static_cast<int64_t>(element_bitset.size());
    }

    // We iterate matching elements by global element id, which maps back
    // to (doc_offset, element_offset). The cursor, however, only tells us the
    // last returned PK and element offset. Find the row for that PK first; the
    // scan can then skip only elements from that row whose element offset has
    // already been returned.
    std::optional<int64_t> cursor_doc_offset;
    if (cursor.has_value()) {
        auto pk_field_id =
            schema_->get_primary_field_id().value_or(FieldId(-1));
        AssertInfo(pk_field_id.get() != -1, "Primary key is -1");
        auto pk_column = get_column(pk_field_id);
        AssertInfo(pk_column != nullptr, "primary key column not loaded");
        switch (schema_->get_fields().at(pk_field_id).get_data_type()) {
            case DataType::INT64:
                cursor_doc_offset = find_sorted_pk_doc_offset<int64_t>(
                    std::get<int64_t>(cursor->last_pk), pk_column);
                break;
            case DataType::VARCHAR:
                cursor_doc_offset = find_sorted_pk_doc_offset<std::string>(
                    std::get<std::string>(cursor->last_pk), pk_column);
                break;
            default:
                ThrowInfo(
                    DataTypeInvalid,
                    fmt::format(
                        "unsupported type {}",
                        schema_->get_fields().at(pk_field_id).get_data_type()));
        }
    }

    int64_t hit_num = 0;
    auto element_size = static_cast<int64_t>(element_bitset.size());
    int64_t cnt = element_size - element_bitset.count();
    auto more_hit_than_limit = cnt > limit;
    limit = std::min(limit, cnt);

    std::vector<int64_t> doc_offsets;
    std::vector<std::vector<int32_t>> element_indices;

    int64_t current_doc_id = -1;
    std::optional<size_t> elem_opt = element_bitset.find_first(false);
    while (elem_opt.has_value() && hit_num < limit) {
        int64_t elem_id = static_cast<int64_t>(elem_opt.value());
        auto [doc_id, elem_idx] = array_offsets->ElementIDToRowID(elem_id);
        if (cursor_doc_offset.has_value() &&
            doc_id == cursor_doc_offset.value() &&
            elem_idx <= cursor->last_element_offset) {
            elem_opt = element_bitset.find_next(elem_id, false);
            continue;
        }

        if (doc_id != current_doc_id) {
            // New document - start a new entry
            doc_offsets.push_back(doc_id);
            element_indices.push_back({static_cast<int32_t>(elem_idx)});
            current_doc_id = doc_id;
        } else {
            // Same document - append to existing entry
            element_indices.back().push_back(static_cast<int32_t>(elem_idx));
        }
        hit_num++;
        elem_opt = element_bitset.find_next(elem_id, false);
    }

    return {std::move(doc_offsets),
            std::move(element_indices),
            more_hit_than_limit && elem_opt.has_value()};
}

ChunkedSegmentSealedImpl::ChunkedSegmentSealedImpl(
    SchemaPtr schema,
    IndexMetaPtr index_meta,
    const SegcoreConfig& segcore_config,
    int64_t segment_id,
    bool is_sorted_by_pk)
    : segcore_config_(segcore_config),
      field_data_ready_bitset_(schema->get_field_id_bitset_size()),
      index_ready_bitset_(schema->get_field_id_bitset_size()),
      binlog_index_bitset_(schema->get_field_id_bitset_size()),
      ngram_fields_(std::unordered_set<FieldId>(schema->size())),
      scalar_indexings_(std::unordered_map<FieldId, index::CacheIndexBasePtr>(
          schema->size())),
      mmap_descriptor_(storage::MmapManager::GetInstance()
                           .GetMmapChunkManager()
                           ->Register()),
      insert_record_(*schema, MAX_ROW_COUNT),
      schema_(schema),
      id_(segment_id),
      col_index_meta_(index_meta),
      is_sorted_by_pk_(is_sorted_by_pk),
      deleted_record_(
          &insert_record_,
          [this](const std::vector<PkType>& pks,
                 const Timestamp* timestamps,
                 const std::function<void(const SegOffset offset,
                                          const Timestamp ts)>& callback) {
              this->search_batch_pks(
                  pks,
                  [&](const size_t idx) { return timestamps[idx]; },
                  false,
                  callback);
          },
          segment_id) {
    std::atomic_store(&segment_load_info_,
                      std::make_shared<const SegmentLoadInfo>(
                          milvus::proto::segcore::SegmentLoadInfo(), schema));
}

ChunkedSegmentSealedImpl::~ChunkedSegmentSealedImpl() {
    // Clean up geometry cache for all fields in this segment
    auto& cache_manager = milvus::exec::SimpleGeometryCacheManager::Instance();
    cache_manager.RemoveSegmentCaches(ctx_, get_segment_id());

    if (ctx_) {
        GEOS_finish_r(ctx_);
        ctx_ = nullptr;
    }

    if (mmap_descriptor_ != nullptr) {
        auto mm = storage::MmapManager::GetInstance().GetMmapChunkManager();
        mm->UnRegister(mmap_descriptor_);
    }
}

void
ChunkedSegmentSealedImpl::bulk_subscript(milvus::OpContext* op_ctx,
                                         SystemFieldType system_type,
                                         const int64_t* seg_offsets,
                                         int64_t count,
                                         void* output) const {
    AssertInfo(is_system_field_ready(),
               "System field isn't ready when do bulk_insert, segID:{}",
               id_);
    switch (system_type) {
        case SystemFieldType::Timestamp: {
            auto* dst = static_cast<Timestamp*>(output);
            // Import/CDC segments: every row carries commit_ts_, including
            // v2/v3 column-group segments where the raw timestamp column is
            // emplaced into fields_ but never overwritten. Short-circuit
            // before consulting the column.
            if (auto cts = EffectiveCommitTs()) {
                std::fill_n(dst, count, *cts);
                break;
            }
            auto ts_column = get_column(TimestampFieldID);
            if (ts_column) {
                // StorageV2: read from timestamp column directly
                auto all_chunks = ts_column->GetAllChunks(op_ctx);
                // Build prefix-sum for chunk offset lookup
                auto num_chunks = ts_column->num_chunks();
                std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);
                for (int64_t c = 0; c < num_chunks; c++) {
                    chunk_offsets[c + 1] =
                        chunk_offsets[c] + ts_column->chunk_row_nums(c);
                }
                for (int64_t i = 0; i < count; ++i) {
                    auto offset = seg_offsets[i];
                    // Find chunk via linear scan (chunks are typically few)
                    int64_t c = 0;
                    while (c < num_chunks - 1 &&
                           offset >= chunk_offsets[c + 1]) {
                        ++c;
                    }
                    auto* chunk_data = reinterpret_cast<const Timestamp*>(
                        all_chunks[c].get()->RawData());
                    dst[i] = chunk_data[offset - chunk_offsets[c]];
                }
            } else {
                // StorageV1 fallback
                AssertInfo(!insert_record_.timestamps_.empty(),
                           "timestamp data is not ready");
                auto& ts = insert_record_.timestamps_;
                for (int64_t i = 0; i < count; ++i) {
                    dst[i] = ts[seg_offsets[i]];
                }
            }
            break;
        }
        case SystemFieldType::RowId:
            ThrowInfo(ErrorCode::Unsupported, "RowId retrieve not supported");
            break;
        default:
            ThrowInfo(DataTypeInvalid,
                      fmt::format("unknown subscript fields", system_type));
    }
}

void
ChunkedSegmentSealedImpl::bulk_subscript(milvus::OpContext* op_ctx,
                                         FieldId field_id,
                                         DataType data_type,
                                         const int64_t* seg_offsets,
                                         int64_t count,
                                         void* data,
                                         TargetBitmap& valid_map,
                                         bool small_int_raw_type) const {
    auto& field_meta = schema_->operator[](field_id);
    // DO NOT directly access the column by map like: `fields_.at(field_id)->Data()`,
    // we have to clone the shared pointer, to make sure it won't get released
    // if segment released
    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "field {} must exist when doing bulk_subscript",
               field_id.get());
    if (column->IsNullable()) {
        for (auto i = 0; i < count; i++) {
            valid_map.set(i, column->IsValid(op_ctx, seg_offsets[i]));
        }
    } else {
        valid_map.set();
    }
    switch (data_type) {
        case DataType::BOOL: {
            bulk_subscript_impl<bool>(op_ctx,
                                      column.get(),
                                      seg_offsets,
                                      count,
                                      static_cast<bool*>(data));
            break;
        }
        case DataType::INT8: {
            bulk_subscript_impl<int8_t>(op_ctx,
                                        column.get(),
                                        seg_offsets,
                                        count,
                                        static_cast<int8_t*>(data),
                                        small_int_raw_type);
            break;
        }
        case DataType::INT16: {
            bulk_subscript_impl<int16_t>(op_ctx,
                                         column.get(),
                                         seg_offsets,
                                         count,
                                         static_cast<int16_t*>(data),
                                         small_int_raw_type);
            break;
        }
        case DataType::INT32: {
            bulk_subscript_impl<int32_t>(op_ctx,
                                         column.get(),
                                         seg_offsets,
                                         count,
                                         static_cast<int32_t*>(data));
            break;
        }
        case DataType::TIMESTAMPTZ:
        case DataType::INT64: {
            bulk_subscript_impl<int64_t>(op_ctx,
                                         column.get(),
                                         seg_offsets,
                                         count,
                                         static_cast<int64_t*>(data));
            break;
        }
        case DataType::FLOAT: {
            bulk_subscript_impl<float>(op_ctx,
                                       column.get(),
                                       seg_offsets,
                                       count,
                                       static_cast<float*>(data));
            break;
        }
        case DataType::DOUBLE: {
            bulk_subscript_impl<double>(op_ctx,
                                        column.get(),
                                        seg_offsets,
                                        count,
                                        static_cast<double*>(data));
            break;
        }
        case DataType::VARCHAR:
        case DataType::STRING:
        case DataType::TEXT: {
            // dst must have at least count elements; the callback's offset
            // parameter is guaranteed to be in [0, count)
            bulk_subscript_ptr_impl<std::string>(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                static_cast<std::string*>(data));
            break;
        }
        case DataType::JSON: {
            // dst must have at least count elements; the callback's offset
            // parameter is guaranteed to be in [0, count)
            bulk_subscript_ptr_impl<Json>(op_ctx,
                                          column.get(),
                                          seg_offsets,
                                          count,
                                          static_cast<Json*>(data));
            break;
        }
        case DataType::GEOMETRY: {
            // dst must have at least count elements; the callback's offset
            // parameter is guaranteed to be in [0, count)
            bulk_subscript_ptr_impl<std::string>(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                static_cast<std::string*>(data));
            break;
        }
        case DataType::ARRAY: {
            // dst must have at least count elements; the callback's index
            // parameter is guaranteed to be in [0, count)
            auto dst = static_cast<Array*>(data);
            column->BulkArrayAt(
                op_ctx,
                [dst](const ArrayView& view, size_t i) {
                    view.output_data(dst[i]);
                },
                seg_offsets,
                count);
            break;
        }
        default: {
            ThrowInfo(DataTypeInvalid,
                      fmt::format("unsupported data type {}",
                                  field_meta.get_data_type()));
        }
    }
}

template <typename S, typename T>
void
ChunkedSegmentSealedImpl::bulk_subscript_impl(milvus::OpContext* op_ctx,
                                              const void* src_raw,
                                              const int64_t* seg_offsets,
                                              int64_t count,
                                              T* dst) {
    static_assert(IsScalar<T>);
    auto src = static_cast<const S*>(src_raw);
    for (int64_t i = 0; i < count; ++i) {
        auto offset = seg_offsets[i];
        dst[i] = src[offset];
    }
}
template <typename S, typename T>
void
ChunkedSegmentSealedImpl::bulk_subscript_impl(milvus::OpContext* op_ctx,
                                              ChunkedColumnInterface* field,
                                              const int64_t* seg_offsets,
                                              int64_t count,
                                              T* dst,
                                              bool small_int_raw_type) {
    static_assert(std::is_fundamental_v<S> && std::is_fundamental_v<T>);
    // use field->data_type_ to determine the type of dst
    field->BulkPrimitiveValueAt(op_ctx,
                                static_cast<void*>(dst),
                                seg_offsets,
                                count,
                                small_int_raw_type);
}

// for dense vector
void
ChunkedSegmentSealedImpl::bulk_subscript_impl(milvus::OpContext* op_ctx,
                                              int64_t element_sizeof,
                                              ChunkedColumnInterface* field,
                                              const int64_t* seg_offsets,
                                              int64_t count,
                                              void* dst_raw) {
    auto dst_vec = reinterpret_cast<char*>(dst_raw);
    field->BulkVectorValueAt(
        op_ctx, dst_vec, seg_offsets, element_sizeof, count);
}

template <typename S>
void
ChunkedSegmentSealedImpl::bulk_subscript_ptr_impl(
    milvus::OpContext* op_ctx,
    ChunkedColumnInterface* column,
    const int64_t* seg_offsets,
    int64_t count,
    google::protobuf::RepeatedPtrField<std::string>* dst) {
    if constexpr (std::is_same_v<S, Json>) {
        column->BulkRawJsonAt(
            op_ctx,
            [&](Json json, size_t offset, bool is_valid) {
                dst->at(offset) = std::string(json.data());
            },
            seg_offsets,
            count);
    } else {
        static_assert(std::is_same_v<S, std::string>);
        column->BulkRawStringAt(
            op_ctx,
            [dst](std::string_view value, size_t offset, bool is_valid) {
                dst->at(offset) = std::string(value);
            },
            seg_offsets,
            count);
    }
}

template <typename S, typename T>
void
ChunkedSegmentSealedImpl::bulk_subscript_ptr_impl(
    milvus::OpContext* op_ctx,
    const ChunkedColumnInterface* column,
    const int64_t* seg_offsets,
    int64_t count,
    T* dst) {
    if constexpr (std::is_same_v<S, Json>) {
        column->BulkRawJsonAt(
            op_ctx,
            [&](Json json, size_t offset, bool is_valid) {
                dst[offset] = std::move(T(json));
            },
            seg_offsets,
            count);
    } else {
        static_assert(std::is_same_v<S, std::string>);
        column->BulkRawStringAt(
            op_ctx,
            [&](std::string_view value, size_t offset, bool is_valid) {
                dst[offset] = std::move(T(value));
            },
            seg_offsets,
            count);
    }
}

template <typename T>
void
ChunkedSegmentSealedImpl::bulk_subscript_array_impl(
    milvus::OpContext* op_ctx,
    ChunkedColumnInterface* column,
    const int64_t* seg_offsets,
    int64_t count,
    google::protobuf::RepeatedPtrField<T>* dst) {
    column->BulkArrayAt(
        op_ctx,
        [dst](const ArrayView& view, size_t i) {
            view.output_data(dst->at(i));
        },
        seg_offsets,
        count);
}

template <typename T>
void
ChunkedSegmentSealedImpl::bulk_subscript_vector_array_impl(
    milvus::OpContext* op_ctx,
    const ChunkedColumnInterface* column,
    const int64_t* seg_offsets,
    int64_t count,
    google::protobuf::RepeatedPtrField<T>* dst) {
    column->BulkVectorArrayAt(
        op_ctx,
        [dst](VectorFieldProto&& array, size_t i) {
            dst->at(i) = std::move(array);
        },
        seg_offsets,
        count);
}

static std::vector<std::string>
ReadTextLobBatch(
    const std::string& lob_base_path,
    const std::vector<milvus_storage::lob_column::EncodedRef>& encoded_refs) {
    if (encoded_refs.empty()) {
        return {};
    }

    auto properties = milvus::storage::LoonFFIPropertiesSingleton::GetInstance()
                          .GetProperties();
    auto fs = milvus::segcore::GetDefaultArrowFileSystem();

    auto& cache = GetGlobalTextColumnCache();
    return cache.ReadBatch(lob_base_path, fs, *properties, encoded_refs);
}

static milvus_storage::lob_column::EncodedRef
MakeTextLobEncodedRef(const void* data, size_t size) {
    return {static_cast<const uint8_t*>(data), size};
}

void
ChunkedSegmentSealedImpl::bulk_subscript_text_impl(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    const ChunkedColumnInterface* column,
    const int64_t* seg_offsets,
    int64_t count,
    google::protobuf::RepeatedPtrField<std::string>* dst) const {
    auto it = text_lob_paths_.find(field_id);
    if (it == text_lob_paths_.end()) {
        throw SegcoreError(
            ErrorCode::UnexpectedError,
            fmt::format("LOB base path not found for TEXT field {}",
                        field_id.get()));
    }
    const auto& lob_base_path = it->second;

    std::vector<milvus_storage::lob_column::EncodedRef> encoded_refs;
    std::vector<int64_t> valid_indices;
    encoded_refs.reserve(count);
    valid_indices.reserve(count);

    column->BulkRawStringAt(
        op_ctx,
        [&encoded_refs, &valid_indices](
            std::string_view value, size_t idx, bool is_valid) {
            if (!is_valid) {
                return;  // skip null values
            }
            encoded_refs.push_back(
                MakeTextLobEncodedRef(value.data(), value.size()));
            valid_indices.push_back(idx);
        },
        seg_offsets,
        count);

    if (encoded_refs.empty()) {
        return;
    }

    auto texts = ReadTextLobBatch(lob_base_path, encoded_refs);
    for (size_t i = 0; i < valid_indices.size() && i < texts.size(); i++) {
        *dst->Mutable(valid_indices[i]) = std::move(texts[i]);
    }
}

void
ChunkedSegmentSealedImpl::ClearData() {
    {
        std::unique_lock lck(mutex_);
        field_data_ready_bitset_.reset();
        index_ready_bitset_.reset();
        binlog_index_bitset_.reset();
        index_has_raw_data_.clear();
        num_rows_ = std::nullopt;
        ngram_fields_.wlock()->clear();
        scalar_indexings_.withWLock([&](auto& scalar_indexings) {
            cancel_and_clear_scalar_indexings(scalar_indexings);
        });
        vector_indexings_.clear();
        ngram_indexings_.withWLock([&](auto& ngram_indexings) {
            cancel_and_clear_ngram_indexings(ngram_indexings);
        });
        json_indices.withWLock([&](auto& json_indexings) {
            cancel_and_clear_json_indices(json_indexings);
        });
        insert_record_.clear();
        timestamp_index_slot_.wlock()->reset();
        pk_index_slot_.wlock()->reset();
        fields_.wlock()->clear();
        variable_fields_avg_size_.clear();
        stats_.mem_size = 0;
    }
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::fill_with_empty(FieldId field_id,
                                          int64_t count,
                                          int64_t valid_count,
                                          const void* valid_data) const {
    auto& field_meta = schema_->operator[](field_id);
    if (IsVectorDataType(field_meta.get_data_type())) {
        return CreateEmptyVectorDataArray(
            count, valid_count, valid_data, field_meta);
    }
    return CreateEmptyScalarDataArray(count, field_meta);
}

void
ChunkedSegmentSealedImpl::CreateTextIndex(FieldId field_id,
                                          milvus::OpContext* op_ctx) {
    auto total_start = std::chrono::steady_clock::now();
    CreateTextIndexTiming timing;
    // Check for cancellation before starting
    CheckCancellation(op_ctx,
                      id_,
                      field_id.get(),
                      "ChunkedSegmentSealedImpl::CreateTextIndex()");

    auto stage_start = std::chrono::steady_clock::now();
    std::unique_lock lck(mutex_);
    timing.lock_wait_ns = DurationNs(stage_start);

    // Guard against re-entry on a field whose temp text index was already
    // built: the path `<mmap>/<segment_id>_<field_id>` is shared with the
    // live holder in text_indexes_, and rebuilding there races with its
    // destructor's RemoveDir (issue #49076).
    AssertInfo(text_indexes_.find(field_id) == text_indexes_.end(),
               "text index for field {} already exists, refusing to rebuild",
               field_id.get());

    const auto& field_meta = schema_->operator[](field_id);
    auto& cfg = storage::MmapManager::GetInstance().GetMmapConfig();
    timing.enable_mmap = cfg.GetScalarIndexEnableMmap();
    std::unique_ptr<index::TextMatchIndex> index;
    std::string unique_id = GetUniqueFieldId(field_meta.get_id().get());
    stage_start = std::chrono::steady_clock::now();
    if (!cfg.GetScalarIndexEnableMmap()) {
        // build text index in ram.
        index = std::make_unique<index::TextMatchIndex>(
            std::numeric_limits<int64_t>::max(),
            unique_id.c_str(),
            "milvus_tokenizer",
            field_meta.get_analyzer_params().c_str());
    } else {
        // build text index using mmap.
        index = std::make_unique<index::TextMatchIndex>(
            cfg.GetMmapPath(),
            unique_id.c_str(),
            // todo: make it configurable
            index::TANTIVY_INDEX_LATEST_VERSION,
            "milvus_tokenizer",
            field_meta.get_analyzer_params().c_str());
    }
    timing.create_index_ns = DurationNs(stage_start);

    {
        // build
        stage_start = std::chrono::steady_clock::now();
        auto column = get_column(field_id);
        if (column) {
            timing.source_column = true;
            // Check for cancellation before bulk operation
            CheckCancellation(op_ctx,
                              id_,
                              field_id.get(),
                              "ChunkedSegmentSealedImpl::CreateTextIndex()");
            column->BulkRawStringAt(
                nullptr,
                [&](std::string_view value, size_t offset, bool is_valid) {
                    index->AddTextSealed(std::string(value), is_valid, offset);
                });
        } else {  // fetch raw data from index.
            auto field_index_iter =
                scalar_indexings_.withRLock([&](auto& mapping) {
                    auto iter = mapping.find(field_id);
                    AssertInfo(iter != mapping.end(),
                               "failed to create text index, neither "
                               "raw data nor "
                               "index are found");
                    return iter;
                });
            auto accessor =
                SemiInlineGet(field_index_iter->second->PinCells(op_ctx, {0}));
            auto ptr = accessor->get_cell_of(0);
            AssertInfo(ptr->HasRawData(),
                       "text raw data not found, trying to create text index "
                       "from index, but this index don't contain raw data");
            auto impl = dynamic_cast<index::ScalarIndex<std::string>*>(ptr);
            AssertInfo(impl != nullptr,
                       "failed to create text index, field index cannot be "
                       "converted to string index");
            auto n = impl->Size();
            for (size_t i = 0; i < n; i++) {
                auto raw = impl->Reverse_Lookup(i);
                if (!raw.has_value()) {
                    index->AddNullSealed(i);
                }
                index->AddTextSealed(raw.value(), true, i);
            }
        }
        timing.build_source_ns = DurationNs(stage_start);
    }

    // Check for cancellation before finalizing
    CheckCancellation(op_ctx,
                      id_,
                      field_id.get(),
                      "ChunkedSegmentSealedImpl::CreateTextIndex()");

    // create index reader.
    stage_start = std::chrono::steady_clock::now();
    index->CreateReader(milvus::index::SetBitsetSealed);
    // release index writer.
    index->Finish();

    index->Reload();

    index->RegisterAnalyzer("milvus_tokenizer",
                            field_meta.get_analyzer_params().c_str());
    timing.finalize_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    text_indexes_[field_id] = std::make_shared<index::TextMatchIndexHolder>(
        std::move(index), cfg.GetScalarIndexEnableMmap());
    // Publish the CreatedTextIndexes update to the atomic segment_load_info_
    // snapshot. Uses CAS loop so it is safe even when CreateTextIndex is
    // invoked directly by tests without an outer Reopen/Load holding
    // reopen_mutex_.
    RecordTextIndexCreated(field_id);
    timing.publish_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    create_text_index_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::RecordDefaultFieldsFilled(
    const std::vector<FieldId>& field_ids) {
    if (field_ids.empty()) {
        return;
    }

    auto current = std::atomic_load(&segment_load_info_);
    std::shared_ptr<const SegmentLoadInfo> next;
    do {
        auto copy = std::make_shared<SegmentLoadInfo>(*current);
        for (auto field_id : field_ids) {
            copy->SetFieldFilledWithDefault(field_id);
        }
        next = std::const_pointer_cast<const SegmentLoadInfo>(copy);
    } while (!std::atomic_compare_exchange_weak(
        &segment_load_info_, &current, next));
}

void
ChunkedSegmentSealedImpl::RecordTextIndexCreated(FieldId field_id) {
    auto current = std::atomic_load(&segment_load_info_);
    std::shared_ptr<const SegmentLoadInfo> next;
    do {
        auto copy = std::make_shared<SegmentLoadInfo>(*current);
        copy->SetTextIndexCreated(field_id);
        next = std::const_pointer_cast<const SegmentLoadInfo>(copy);
    } while (!std::atomic_compare_exchange_weak(
        &segment_load_info_, &current, next));
}

void
ChunkedSegmentSealedImpl::LoadTextIndex(
    milvus::OpContext* op_ctx,
    std::shared_ptr<milvus::proto::indexcgo::LoadTextIndexInfo> info_proto) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    TextIndexEntryTiming timing;
    timing.file_count = info_proto->files_size();
    timing.enable_mmap = info_proto->enable_mmap();

    // Check for cancellation before starting
    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::LoadTextIndex()");

    milvus::storage::FieldDataMeta field_data_meta{info_proto->collectionid(),
                                                   info_proto->partitionid(),
                                                   this->get_segment_id(),
                                                   info_proto->fieldid(),
                                                   info_proto->schema()};
    milvus::storage::IndexMeta index_meta{this->get_segment_id(),
                                          info_proto->fieldid(),
                                          info_proto->buildid(),
                                          info_proto->version()};
    auto field_meta = milvus::FieldMeta::ParseFrom(info_proto->schema());
    auto remote_chunk_manager =
        milvus::storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager();
    auto fs = milvus::segcore::GetDefaultArrowFileSystem();
    AssertInfo(fs != nullptr, "arrow file system is null");

    milvus::Config config;
    std::vector<std::string> files;
    for (const auto& f : info_proto->files()) {
        files.push_back(f);
    }
    config[milvus::index::INDEX_FILES] = files;
    config[milvus::LOAD_PRIORITY] = info_proto->load_priority();
    config[milvus::index::ENABLE_MMAP] = info_proto->enable_mmap();
    config[milvus::index::COLLECTION_ID] = info_proto->collectionid();
    if (info_proto->warmup_policy() != "") {
        config[milvus::index::WARMUP] = info_proto->warmup_policy();
    }
    if (!info_proto->base_path().empty()) {
        config[STATS_BASE_PATH_KEY] = info_proto->base_path();
    }
    timing.config_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    milvus::storage::FileManagerContext file_ctx(
        field_data_meta, index_meta, remote_chunk_manager, fs);
    if (!info_proto->base_path().empty()) {
        file_ctx.set_stats_base_path(info_proto->base_path());
    }
    timing.file_context_ns = DurationNs(stage_start);

    auto field_id = milvus::FieldId(info_proto->fieldid());
    // const auto& field_meta = schema_->operator[](field_id);
    milvus::segcore::storagev1translator::TextMatchIndexLoadInfo load_info{
        info_proto->enable_mmap(),
        this->get_segment_id(),
        info_proto->fieldid(),
        field_meta.get_analyzer_params(),
        info_proto->index_size(),
        info_proto->warmup_policy()};

    stage_start = std::chrono::steady_clock::now();
    std::unique_ptr<
        milvus::cachinglayer::Translator<milvus::index::TextMatchIndex>>
        translator = std::make_unique<
            milvus::segcore::storagev1translator::TextMatchIndexTranslator>(
            load_info, file_ctx, config);
    timing.translator_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    auto cache_slot =
        milvus::cachinglayer::Manager::GetInstance().CreateCacheSlot(
            std::move(translator), op_ctx);
    timing.cache_slot_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    std::unique_lock lck(mutex_);
    timing.lock_wait_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    text_indexes_[field_id] = std::move(cache_slot);
    timing.register_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    text_index_entry_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadJsonKeyIndex(
    milvus::OpContext* op_ctx,
    std::shared_ptr<milvus::proto::indexcgo::LoadJsonKeyIndexInfo> info_proto) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    JsonStatsLoadTiming timing;
    timing.file_count = info_proto->files_size();
    timing.enable_mmap = info_proto->enable_mmap();
    auto field_id = milvus::FieldId(info_proto->fieldid());
    CheckCancellation(op_ctx,
                      id_,
                      field_id.get(),
                      "ChunkedSegmentSealedImpl::LoadJsonKeyIndex()");

    if (!JSON_KEY_STATS_ENABLED.load()) {
        LOG_WARN(
            "skip load json key stats because json key stats is disabled, "
            "segment:{}, field:{}, build:{}, version:{}",
            id_,
            info_proto->fieldid(),
            info_proto->buildid(),
            info_proto->version());
        return;
    }

    LOG_INFO(
        "start load json key stats, segment:{}, field:{}, build:{}, "
        "version:{}, "
        "file_count:{}, base_path:{}, enable_mmap:{}, stats_size:{}",
        id_,
        info_proto->fieldid(),
        info_proto->buildid(),
        info_proto->version(),
        info_proto->files_size(),
        info_proto->base_path(),
        info_proto->enable_mmap(),
        info_proto->stats_size());

    milvus::storage::FieldDataMeta field_data_meta{info_proto->collectionid(),
                                                   info_proto->partitionid(),
                                                   this->get_segment_id(),
                                                   info_proto->fieldid(),
                                                   info_proto->schema()};
    milvus::storage::IndexMeta index_meta{this->get_segment_id(),
                                          info_proto->fieldid(),
                                          info_proto->buildid(),
                                          info_proto->version()};
    auto remote_chunk_manager =
        milvus::storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager();
    auto fs = milvus::segcore::GetDefaultArrowFileSystem();
    AssertInfo(fs != nullptr, "arrow file system is null");

    milvus::Config config;
    std::vector<std::string> files;
    files.reserve(info_proto->files_size());
    for (const auto& f : info_proto->files()) {
        files.push_back(f);
    }
    config[milvus::index::INDEX_FILES] = files;
    config[milvus::LOAD_PRIORITY] = info_proto->load_priority();
    config[milvus::index::ENABLE_MMAP] = info_proto->enable_mmap();
    if (info_proto->enable_mmap()) {
        config[milvus::index::MMAP_FILE_PATH] = info_proto->mmap_dir_path();
    }
    if (!info_proto->warmup_policy().empty()) {
        config[milvus::index::WARMUP] = info_proto->warmup_policy();
    }
    config[milvus::index::INDEX_SIZE] = info_proto->stats_size();
    if (!info_proto->base_path().empty()) {
        config[STATS_BASE_PATH_KEY] = info_proto->base_path();
    }
    timing.config_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    milvus::storage::FileManagerContext file_ctx(
        field_data_meta, index_meta, remote_chunk_manager, fs);
    timing.file_context_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    auto index = std::make_shared<milvus::index::JsonKeyStats>(file_ctx, true);
    timing.create_index_ns = DurationNs(stage_start);
    milvus::tracer::TraceContext trace_ctx;
    try {
        milvus::ScopedTimer timer(
            "json_stats_load",
            [](double us) {
                milvus::monitor::internal_json_stats_latency_load.Observe(
                    us / 1000.0);
            },
            milvus::ScopedTimer::LogLevel::Info);
        stage_start = std::chrono::steady_clock::now();
        index->Load(trace_ctx, config);
        timing.load_ns = DurationNs(stage_start);
    } catch (std::exception& e) {
        LOG_WARN(
            "failed load json key stats, segment:{}, field:{}, build:{}, "
            "version:{}, error:{}",
            id_,
            info_proto->fieldid(),
            info_proto->buildid(),
            info_proto->version(),
            e.what());
        throw;
    }

    stage_start = std::chrono::steady_clock::now();
    LoadJsonStats(field_id, std::move(index));
    timing.register_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    json_stats_load_timing_stats.Record(timing);
    LOG_INFO(
        "load json key stats success, segment:{}, field:{}, build:{}, "
        "version:{}",
        id_,
        info_proto->fieldid(),
        info_proto->buildid(),
        info_proto->version());
}

void
ChunkedSegmentSealedImpl::LoadBatchJsonKeyIndexes(
    milvus::OpContext* op_ctx,
    const std::unordered_map<
        FieldId,
        std::shared_ptr<milvus::proto::indexcgo::LoadJsonKeyIndexInfo>>&
        infos) {
    for (const auto& [field_id, info_proto] : infos) {
        AssertInfo(field_exists_in_schema(schema_, field_id),
                   "field {} not found in schema when loading json stats",
                   field_id.get());
        LoadJsonKeyIndex(op_ctx, info_proto);
    }
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::get_raw_data(milvus::OpContext* op_ctx,
                                       FieldId field_id,
                                       const FieldMeta& field_meta,
                                       const int64_t* seg_offsets,
                                       int64_t count) const {
    // DO NOT directly access the column by map like: `fields_.at(field_id)->Data()`,
    // we have to clone the shared pointer,
    // to make sure it won't get released if segment released
    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "field {} must exist when getting raw data",
               field_id.get());

    int64_t valid_count = count;
    const bool* valid_data = nullptr;
    const int64_t* valid_offsets = seg_offsets;
    ValidResult filter_result;

    if (field_meta.is_vector() && field_meta.is_nullable()) {
        filter_result =
            FilterVectorValidOffsets(op_ctx, field_id, seg_offsets, count);
        valid_count = filter_result.valid_count;
        valid_data = filter_result.valid_data.get();
        valid_offsets = filter_result.valid_offsets.data();
    }
    auto ret = fill_with_empty(field_id, count, valid_count, valid_data);
    if (field_meta.is_vector() && valid_count == 0) {
        return ret;
    }

    if (!field_meta.is_vector() && column->IsNullable()) {
        auto dst = ret->mutable_valid_data()->mutable_data();
        column->BulkIsValid(
            op_ctx,
            [&](bool is_valid, size_t offset) { dst[offset] = is_valid; },
            seg_offsets,
            count);
    }

    switch (field_meta.get_data_type()) {
        case DataType::VARCHAR:
        case DataType::STRING: {
            bulk_subscript_ptr_impl<std::string>(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                ret->mutable_scalars()->mutable_string_data()->mutable_data());
            break;
        }

        case DataType::TEXT: {
            // TEXT type is only supported in StorageV3 with LOB files.
            auto it = text_lob_paths_.find(field_id);
            AssertInfo(it != text_lob_paths_.end(),
                       "TEXT field {} has no LOB path. TEXT type requires "
                       "StorageV3 with manifest. segment_id={}",
                       field_id.get(),
                       id_);
            bulk_subscript_text_impl(
                op_ctx,
                field_id,
                column.get(),
                seg_offsets,
                count,
                ret->mutable_scalars()->mutable_string_data()->mutable_data());
            break;
        }

        case DataType::JSON: {
            bulk_subscript_ptr_impl<Json>(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                ret->mutable_scalars()->mutable_json_data()->mutable_data());
            break;
        }

        case DataType::GEOMETRY: {
            bulk_subscript_ptr_impl<std::string>(op_ctx,
                                                 column.get(),
                                                 seg_offsets,
                                                 count,
                                                 ret->mutable_scalars()
                                                     ->mutable_geometry_data()
                                                     ->mutable_data());
            break;
        }

        case DataType::ARRAY: {
            // Carry the element type into the response so the caller (and
            // SDK) can dispatch on it. Without this, callers parse a
            // headless ScalarField_ArrayData and reject it with
            // "unsupported element type None". Regression fix for #48619.
            ret->mutable_scalars()->mutable_array_data()->set_element_type(
                static_cast<milvus::proto::schema::DataType>(
                    field_meta.get_element_type()));
            bulk_subscript_array_impl(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                ret->mutable_scalars()->mutable_array_data()->mutable_data());
            break;
        }

        case DataType::BOOL: {
            bulk_subscript_impl<bool, bool>(op_ctx,
                                            column.get(),
                                            seg_offsets,
                                            count,
                                            ret->mutable_scalars()
                                                ->mutable_bool_data()
                                                ->mutable_data()
                                                ->mutable_data());
            break;
        }
        case DataType::INT8: {
            bulk_subscript_impl<int8_t, int32_t>(op_ctx,
                                                 column.get(),
                                                 seg_offsets,
                                                 count,
                                                 ret->mutable_scalars()
                                                     ->mutable_int_data()
                                                     ->mutable_data()
                                                     ->mutable_data());
            break;
        }
        case DataType::INT16: {
            bulk_subscript_impl<int16_t, int32_t>(op_ctx,
                                                  column.get(),
                                                  seg_offsets,
                                                  count,
                                                  ret->mutable_scalars()
                                                      ->mutable_int_data()
                                                      ->mutable_data()
                                                      ->mutable_data());
            break;
        }
        case DataType::INT32: {
            bulk_subscript_impl<int32_t, int32_t>(op_ctx,
                                                  column.get(),
                                                  seg_offsets,
                                                  count,
                                                  ret->mutable_scalars()
                                                      ->mutable_int_data()
                                                      ->mutable_data()
                                                      ->mutable_data());
            break;
        }
        case DataType::INT64: {
            bulk_subscript_impl<int64_t, int64_t>(op_ctx,
                                                  column.get(),
                                                  seg_offsets,
                                                  count,
                                                  ret->mutable_scalars()
                                                      ->mutable_long_data()
                                                      ->mutable_data()
                                                      ->mutable_data());
            break;
        }
        case DataType::FLOAT: {
            bulk_subscript_impl<float, float>(op_ctx,
                                              column.get(),
                                              seg_offsets,
                                              count,
                                              ret->mutable_scalars()
                                                  ->mutable_float_data()
                                                  ->mutable_data()
                                                  ->mutable_data());
            break;
        }
        case DataType::DOUBLE: {
            bulk_subscript_impl<double, double>(op_ctx,
                                                column.get(),
                                                seg_offsets,
                                                count,
                                                ret->mutable_scalars()
                                                    ->mutable_double_data()
                                                    ->mutable_data()
                                                    ->mutable_data());
            break;
        }
        case DataType::TIMESTAMPTZ: {
            bulk_subscript_impl<int64_t, int64_t>(
                op_ctx,
                column.get(),
                seg_offsets,
                count,
                ret->mutable_scalars()
                    ->mutable_timestamptz_data()
                    ->mutable_data()
                    ->mutable_data());
            break;
        }
        case DataType::VECTOR_FLOAT: {
            bulk_subscript_impl(op_ctx,
                                field_meta.get_sizeof(),
                                column.get(),
                                valid_offsets,
                                valid_count,
                                ret->mutable_vectors()
                                    ->mutable_float_vector()
                                    ->mutable_data()
                                    ->mutable_data());
            break;
        }
        case DataType::VECTOR_FLOAT16: {
            bulk_subscript_impl(
                op_ctx,
                field_meta.get_sizeof(),
                column.get(),
                valid_offsets,
                valid_count,
                ret->mutable_vectors()->mutable_float16_vector()->data());
            break;
        }
        case DataType::VECTOR_BFLOAT16: {
            bulk_subscript_impl(
                op_ctx,
                field_meta.get_sizeof(),
                column.get(),
                valid_offsets,
                valid_count,
                ret->mutable_vectors()->mutable_bfloat16_vector()->data());
            break;
        }
        case DataType::VECTOR_BINARY: {
            bulk_subscript_impl(
                op_ctx,
                field_meta.get_sizeof(),
                column.get(),
                valid_offsets,
                valid_count,
                ret->mutable_vectors()->mutable_binary_vector()->data());
            break;
        }
        case DataType::VECTOR_INT8: {
            bulk_subscript_impl(
                op_ctx,
                field_meta.get_sizeof(),
                column.get(),
                valid_offsets,
                valid_count,
                ret->mutable_vectors()->mutable_int8_vector()->data());
            break;
        }
        case DataType::VECTOR_SPARSE_U32_F32: {
            auto dst = ret->mutable_vectors()->mutable_sparse_float_vector();
            int64_t max_dim = 0;
            column->BulkValueAt(
                op_ctx,
                [&](const char* value, size_t i) mutable {
                    auto offset = valid_offsets[i];
                    auto row =
                        offset != INVALID_SEG_OFFSET
                            ? static_cast<const knowhere::sparse::SparseRow<
                                  SparseValueType>*>(
                                  static_cast<const void*>(value))
                            : nullptr;
                    if (row == nullptr) {
                        dst->add_contents();
                        return;
                    }
                    max_dim = std::max(max_dim, row->dim());
                    dst->add_contents(row->data(), row->data_byte_size());
                },
                valid_offsets,
                valid_count);
            dst->set_dim(max_dim);
            ret->mutable_vectors()->set_dim(dst->dim());
            break;
        }
        case DataType::VECTOR_ARRAY: {
            auto dst =
                ret->mutable_vectors()->mutable_vector_array()->mutable_data();
            if (field_meta.is_nullable() && valid_data != nullptr) {
                if (valid_count == 0) {
                    break;
                }
                std::vector<int64_t> valid_logical_offsets;
                valid_logical_offsets.reserve(valid_count);
                for (int64_t i = 0; i < count; ++i) {
                    if (valid_data[i]) {
                        valid_logical_offsets.push_back(i);
                    }
                }
                column->BulkVectorArrayAt(
                    op_ctx,
                    [dst, &valid_logical_offsets](VectorFieldProto&& array,
                                                  size_t i) {
                        dst->at(valid_logical_offsets[i]) = std::move(array);
                    },
                    valid_offsets,
                    valid_count);
            } else {
                bulk_subscript_vector_array_impl(
                    op_ctx, column.get(), seg_offsets, count, dst);
            }
            break;
        }
        default: {
            ThrowInfo(DataTypeInvalid,
                      fmt::format("unsupported data type {}",
                                  field_meta.get_data_type()));
        }
    }
    return ret;
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::bulk_subscript(milvus::OpContext* op_ctx,
                                         FieldId field_id,
                                         const int64_t* seg_offsets,
                                         int64_t count) const {
    auto& field_meta = schema_->operator[](field_id);
    // if count == 0, return empty data array
    if (count == 0) {
        return fill_with_empty(field_id, count);
    }

    // Fast path for int64 PK field: use compressed offset2pk index
    auto pk_field_id = schema_->get_primary_field_id();
    auto pk_index = PinPkIndex(op_ctx);
    if (pk_field_id.has_value() && pk_field_id.value() == field_id &&
        field_meta.get_data_type() == DataType::INT64 &&
        (pk_index.get() != nullptr ? pk_index.get()->has_int64_pk_index()
                                   : insert_record_.has_int64_pk_index())) {
        auto ret = fill_with_empty(field_id, count);
        auto* output = ret->mutable_scalars()
                           ->mutable_long_data()
                           ->mutable_data()
                           ->mutable_data();
        if (pk_index.get() != nullptr) {
            pk_index.get()->bulk_get_int64_pks_by_offsets(
                seg_offsets, count, output);
        } else {
            insert_record_.bulk_get_int64_pks_by_offsets(
                seg_offsets, count, output);
        }
        return ret;
    }

    // Decide once whether to serve this retrieve from column data instead of
    // the index-backed raw data. The flag is off by default, so short-circuit
    // before touching HasFieldData — that call takes a shared_lock and would
    // otherwise be paid on every retrieve regardless of the flag.
    bool use_field_data =
        SegcoreConfig::default_config()
            .get_prefer_field_data_when_index_has_raw_data() &&
        HasFieldData(field_id);

    if (!IsVectorDataType(field_meta.get_data_type())) {
        // === Scalar field ===
        if (!use_field_data) {
            // Try index first: if scalar index exists and has raw data, read from index
            PinWrapper<const index::IndexBase*> pin_scalar_index_ptr;
            auto scalar_indexes = PinIndex(op_ctx, field_id);
            if (!scalar_indexes.empty()) {
                pin_scalar_index_ptr = std::move(scalar_indexes[0]);
                if (IndexHasRawData(field_id)) {
                    return ReverseDataFromIndex(pin_scalar_index_ptr.get(),
                                                seg_offsets,
                                                count,
                                                field_meta);
                }
            }
        }
        return get_raw_data(op_ctx, field_id, field_meta, seg_offsets, count);
    }

    // === Vector field ===
    std::chrono::high_resolution_clock::time_point get_vector_start =
        std::chrono::high_resolution_clock::now();

    std::unique_ptr<DataArray> vector{nullptr};
    // Try index first: if vector index exists and has raw data, read from index
    if (!use_field_data && IndexHasRawData(field_id)) {
        if (IsVectorArrayDataType(field_meta.get_data_type())) {
            vector =
                get_emb_list(op_ctx, field_id, field_meta, seg_offsets, count);
        } else {
            vector = get_vector(op_ctx, field_id, seg_offsets, count);
        }
    } else {
        vector = get_raw_data(op_ctx, field_id, field_meta, seg_offsets, count);
    }

    std::chrono::high_resolution_clock::time_point get_vector_end =
        std::chrono::high_resolution_clock::now();
    double get_vector_cost = std::chrono::duration<double, std::micro>(
                                 get_vector_end - get_vector_start)
                                 .count();
    milvus::monitor::internal_core_get_vector_latency.Observe(get_vector_cost /
                                                              1000);

    return vector;
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::bulk_subscript(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    const int64_t* seg_offsets,
    int64_t count,
    const std::vector<std::string>& dynamic_field_names) const {
    Assert(!dynamic_field_names.empty());
    if (count == 0) {
        return fill_with_empty(field_id, 0);
    }

    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "json field {} must exist when bulk_subscript",
               field_id.get());
    auto ret = fill_with_empty(field_id, count);
    if (column->IsNullable()) {
        auto dst = ret->mutable_valid_data()->mutable_data();
        column->BulkIsValid(
            op_ctx,
            [&](bool is_valid, size_t offset) { dst[offset] = is_valid; },
            seg_offsets,
            count);
    }
    auto dst = ret->mutable_scalars()->mutable_json_data()->mutable_data();
    column->BulkRawJsonAt(
        op_ctx,
        [&](Json json, size_t offset, bool is_valid) {
            dst->at(offset) = ExtractSubJson(json.data(), dynamic_field_names);
        },
        seg_offsets,
        count);
    return ret;
}

bool
ChunkedSegmentSealedImpl::HasIndex(FieldId field_id) const {
    std::shared_lock lck(mutex_);
    return get_bit(index_ready_bitset_, field_id) ||
           get_bit(binlog_index_bitset_, field_id);
}

bool
ChunkedSegmentSealedImpl::HasJsonIndex(FieldId field_id) const {
    // JSON indexes (JsonFlatIndex + JSON-cast) live in a separate per-segment
    // vector rather than in either index bitset. Kept as a distinct API so
    // HasIndex() preserves its narrower "scalar/vector/binlog index exists"
    // semantics for ReorderConjunctExpr and other consumers.
    return json_indices.withRLock([&](const auto& vec) {
        for (const auto& index : vec) {
            if (index.field_id == field_id) {
                return true;
            }
        }
        return false;
    });
}

bool
ChunkedSegmentSealedImpl::HasFieldData(FieldId field_id) const {
    std::shared_lock lck(mutex_);
    if (SystemProperty::Instance().IsSystem(field_id)) {
        return is_system_field_ready();
    } else {
        return get_bit(field_data_ready_bitset_, field_id);
    }
}

std::pair<std::shared_ptr<ChunkedColumnInterface>, bool>
ChunkedSegmentSealedImpl::GetFieldDataIfExist(FieldId field_id) const {
    std::shared_lock lck(mutex_);
    bool exists;
    if (SystemProperty::Instance().IsSystem(field_id)) {
        exists = is_system_field_ready();
    } else {
        exists = get_bit(field_data_ready_bitset_, field_id);
    }
    if (!exists) {
        return {nullptr, false};
    }
    auto column = get_column(field_id);
    AssertInfo(column != nullptr,
               "field {} must exist if bitset is set",
               field_id.get());
    return {column, exists};
}

bool
ChunkedSegmentSealedImpl::HasRawData(int64_t field_id) const {
    std::shared_lock lck(mutex_);
    auto fieldID = FieldId(field_id);
    const auto& field_meta = schema_->operator[](fieldID);
    if (IsVectorDataType(field_meta.get_data_type())) {
        if (get_bit(index_ready_bitset_, fieldID)) {
            AssertInfo(vector_indexings_.is_ready(fieldID),
                       "vector index is not ready");
            AssertInfo(
                index_has_raw_data_.find(fieldID) != index_has_raw_data_.end(),
                "index_has_raw_data_ is not set for fieldID: " +
                    std::to_string(fieldID.get()));
            return index_has_raw_data_.at(fieldID);
        } else if (get_bit(binlog_index_bitset_, fieldID)) {
            AssertInfo(vector_indexings_.is_ready(fieldID),
                       "interim index is not ready");
            AssertInfo(
                index_has_raw_data_.find(fieldID) != index_has_raw_data_.end(),
                "index_has_raw_data_ is not set for fieldID: " +
                    std::to_string(fieldID.get()));
            return index_has_raw_data_.at(fieldID) ||
                   get_bit(field_data_ready_bitset_, fieldID);
        }
    } else if (IsJsonDataType(field_meta.get_data_type())) {
        return get_bit(field_data_ready_bitset_, fieldID);
    } else {
        auto has_scalar_index = scalar_indexings_.withRLock([&](auto& mapping) {
            return mapping.find(fieldID) != mapping.end();
        });
        if (has_scalar_index) {
            AssertInfo(
                index_has_raw_data_.find(fieldID) != index_has_raw_data_.end(),
                "index_has_raw_data_ is not set for fieldID: " +
                    std::to_string(fieldID.get()));
            return index_has_raw_data_.at(fieldID);
        }
    }
    return true;
}

bool
ChunkedSegmentSealedImpl::IndexHasRawData(FieldId field_id) const {
    std::shared_lock lck(mutex_);
    auto it = index_has_raw_data_.find(field_id);
    if (it == index_has_raw_data_.end()) {
        return false;
    }
    return it->second;
}

bool
ChunkedSegmentSealedImpl::CalcDistByIDs(
    milvus::OpContext* op_ctx,
    FieldId field_id,
    const knowhere::DataSetPtr& query_dataset,
    const int64_t* seg_offsets,
    size_t count,
    bool is_cosine,
    float* distances) const {
    if (!vector_indexings_.is_ready(field_id)) {
        return false;
    }
    auto field_indexing = vector_indexings_.get_field_indexing(field_id);
    auto accessor =
        SemiInlineGet(field_indexing->indexing_->PinCells(op_ctx, {0}));
    auto vec_index =
        dynamic_cast<index::VectorIndex*>(accessor->get_cell_of(0));
    if (vec_index == nullptr) {
        return false;
    }
    // Callers pass logical offsets (already translated from physical by
    // SearchOnIndex). When the index carries an offset_mapping (nullable
    // vector), the underlying knowhere index operates on physical offsets,
    // so translate logical -> physical before the call.
    const auto& offset_mapping = vec_index->GetOffsetMapping();
    std::vector<int64_t> physical_offsets;
    const int64_t* labels = seg_offsets;
    if (offset_mapping.IsEnabled()) {
        physical_offsets.resize(count);
        for (size_t i = 0; i < count; ++i) {
            physical_offsets[i] =
                seg_offsets[i] < 0
                    ? seg_offsets[i]
                    : offset_mapping.GetPhysicalOffset(seg_offsets[i]);
        }
        labels = physical_offsets.data();
    }
    auto res = vec_index->CalcDistByIDs(
        query_dataset, BitsetView(), labels, count, is_cosine, op_ctx);
    if (!res.has_value()) {
        return false;
    }
    auto result_distances = res.value()->GetDistance();
    if (result_distances == nullptr) {
        return false;
    }
    std::memcpy(distances, result_distances, count * sizeof(float));
    return true;
}

bool
ChunkedSegmentSealedImpl::IsIndexRefineEnabled(milvus::OpContext* op_ctx,
                                               FieldId field_id) const {
    if (!vector_indexings_.is_ready(field_id)) {
        return false;
    }
    auto field_indexing = vector_indexings_.get_field_indexing(field_id);
    auto accessor =
        SemiInlineGet(field_indexing->indexing_->PinCells(op_ctx, {0}));
    auto vec_index =
        dynamic_cast<index::VectorIndex*>(accessor->get_cell_of(0));
    return vec_index != nullptr && vec_index->IsIndexRefineEnabled();
}

DataType
ChunkedSegmentSealedImpl::GetFieldDataType(milvus::FieldId field_id) const {
    auto& field_meta = schema_->operator[](field_id);
    return field_meta.get_data_type();
}

void
ChunkedSegmentSealedImpl::search_ids(BitsetType& bitset,
                                     const IdArray& id_array) const {
    auto field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(field_id.get() != -1, "Primary key is -1");
    auto& field_meta = schema_->operator[](field_id);
    auto data_type = field_meta.get_data_type();
    auto ids_size = GetSizeOfIdArray(id_array);
    std::vector<PkType> pks(ids_size);
    ParsePksFromIDs(pks, data_type, id_array);

    this->search_pks(bitset, pks);
}

SegcoreError
ChunkedSegmentSealedImpl::Delete(int64_t size,
                                 const IdArray* ids,
                                 const Timestamp* timestamps_raw) {
    auto field_id = schema_->get_primary_field_id().value_or(FieldId(-1));
    AssertInfo(field_id.get() != -1, "Primary key is -1");
    auto& field_meta = schema_->operator[](field_id);
    std::vector<PkType> pks(size);
    ParsePksFromIDs(pks, field_meta.get_data_type(), *ids);

    // filter out the deletions that the primary key not exists
    std::vector<std::tuple<Timestamp, PkType>> ordering(size);
    for (int i = 0; i < size; i++) {
        ordering[i] = std::make_tuple(timestamps_raw[i], pks[i]);
    }
    // if insert record is empty (may be only-load meta but not data for lru-cache at go side),
    // filtering may cause the deletion lost, skip the filtering to avoid it.
    auto pk_index = PinPkIndex(nullptr);
    auto has_pk_index = pk_index.get() != nullptr ? !pk_index.get()->empty_pks()
                                                  : !insert_record_.empty_pks();
    if (has_pk_index) {
        auto end = std::remove_if(
            ordering.begin(),
            ordering.end(),
            [&](const std::tuple<Timestamp, PkType>& record) {
                if (pk_index.get() != nullptr) {
                    return !pk_index.get()->contain(std::get<1>(record));
                }
                return !insert_record_.contain(std::get<1>(record));
            });
        size = end - ordering.begin();
        ordering.resize(size);
    }
    if (size == 0) {
        return SegcoreError::success();
    }

    // step 1: sort timestamp
    std::sort(ordering.begin(), ordering.end());
    std::vector<PkType> sort_pks(size);
    std::vector<Timestamp> sort_timestamps(size);

    for (int i = 0; i < size; i++) {
        auto [t, pk] = ordering[i];
        sort_timestamps[i] = t;
        sort_pks[i] = pk;
    }

    deleted_record_.StreamPush(sort_pks, sort_timestamps.data());
    return SegcoreError::success();
}

void
ChunkedSegmentSealedImpl::LoadSegmentMeta(
    const proto::segcore::LoadSegmentMeta& segment_meta) {
    std::unique_lock lck(mutex_);
    std::vector<int64_t> slice_lengths;
    for (auto& info : segment_meta.metas()) {
        slice_lengths.push_back(info.row_count());
    }
    insert_record_.timestamp_index_.set_length_meta(std::move(slice_lengths));
    ThrowInfo(NotImplemented, "unimplemented");
}

int64_t
ChunkedSegmentSealedImpl::get_active_count(Timestamp ts) const {
    // TODO optimize here to reduce expr search range
    return this->get_row_count();
}

// Helper: apply a per-element timestamp scan over a range [beg, end),
// calling `pred(global_offset, ts_value)` for each row.
// Overload for TimestampData (StorageV1 / growing segment path).
template <typename Pred>
static void
scan_timestamp_range(const TimestampData& ts,
                     int64_t beg,
                     int64_t end,
                     Pred pred) {
    for (int64_t c = 0; c < ts.num_chunks(); c++) {
        auto chunk_start = ts.chunk_start_offset(c);
        auto chunk_end = chunk_start + ts.chunk_row_count(c);
        auto overlap_beg = std::max(beg, chunk_start);
        auto overlap_end = std::min(end, chunk_end);
        if (overlap_beg >= overlap_end) {
            continue;
        }
        auto* data = ts.chunk_data(c);
        auto local = overlap_beg - chunk_start;
        for (int64_t i = overlap_beg; i < overlap_end; ++i, ++local) {
            pred(i, data[local]);
        }
    }
}

// Overload for ChunkedColumnInterface (StorageV2 sealed segment path).
// Pins each chunk on demand and releases after scanning.
template <typename Pred>
static void
scan_timestamp_range(const ChunkedColumnInterface& column,
                     int64_t beg,
                     int64_t end,
                     Pred pred) {
    auto num_chunks = column.num_chunks();
    int64_t chunk_start = 0;
    for (int64_t c = 0; c < num_chunks; c++) {
        auto chunk_rows = column.chunk_row_nums(c);
        auto chunk_end = chunk_start + chunk_rows;
        auto overlap_beg = std::max(beg, chunk_start);
        auto overlap_end = std::min(end, chunk_end);
        if (overlap_beg >= overlap_end) {
            chunk_start = chunk_end;
            continue;
        }
        auto pw = column.DataOfChunk(nullptr, c);
        auto* data = reinterpret_cast<const Timestamp*>(pw.get());
        auto local = overlap_beg - chunk_start;
        for (int64_t i = overlap_beg; i < overlap_end; ++i, ++local) {
            pred(i, data[local]);
        }
        chunk_start = chunk_end;
    }
}

void
ChunkedSegmentSealedImpl::mask_with_timestamps(BitsetTypeView& bitset_chunk,
                                               Timestamp timestamp,
                                               Timestamp collection_ttl) const {
    // External collections have no timestamps; all data is always visible
    if (schema_->is_external_collection()) {
        return;
    }
    auto ts_index = PinTimestampIndex(nullptr);
    auto* ts_cell = ts_index.get();
    AssertInfo(ts_cell != nullptr || !insert_record_.timestamps_.empty(),
               "timestamp index is not ready");
    auto& ts_index_data = ts_cell != nullptr ? ts_cell->timestamp_index()
                                             : insert_record_.timestamp_index_;
    auto effective_commit_ts = EffectiveCommitTs();
    // When commit_ts_ is set, the per-row scan must use commit_ts_, not the
    // raw v2/v3 timestamp column (which still holds the original row_ts). The
    // index itself is already commit_ts-overwritten at load time, so the
    // get_active_range narrowing above is consistent — only the per-bit scan
    // below needs the override.
    auto ts_column = (ts_cell != nullptr && !effective_commit_ts)
                         ? get_column(TimestampFieldID)
                         : nullptr;
    auto total_size = static_cast<int64_t>(get_row_count());

    // Lambda to dispatch scan_timestamp_range to the right overload, or to
    // apply the predicate uniformly with commit_ts_ when set.
    auto do_scan = [&](int64_t beg, int64_t end, auto pred) {
        if (effective_commit_ts) {
            for (int64_t i = beg; i < end; ++i) {
                pred(i, *effective_commit_ts);
            }
            return;
        }
        if (ts_column) {
            scan_timestamp_range(*ts_column, beg, end, pred);
        } else {
            scan_timestamp_range(insert_record_.timestamps_, beg, end, pred);
        }
    };

    if (collection_ttl > 0) {
        auto range = ts_index_data.get_active_range(collection_ttl);
        if (range.first == range.second && range.first == total_size) {
            bitset_chunk.set();
            return;
        } else {
            // TTL bitset: [0, beg) = true, [beg, end) = check, [end, size) = false
            BitsetType ttl_mask;
            ttl_mask.reserve(total_size);
            ttl_mask.resize(range.first, true);
            ttl_mask.resize(total_size, false);
            do_scan(range.first, range.second, [&](int64_t i, Timestamp val) {
                ttl_mask[i] = val <= collection_ttl;
            });
            bitset_chunk |= ttl_mask;
        }
    }

    AssertInfo(total_size == get_row_count(),
               fmt::format("Timestamp size not equal to row count: {}, {}",
                           total_size,
                           get_row_count()));
    auto range = ts_index_data.get_active_range(timestamp);

    // range == (size_, size_): all data is useful, no filtering needed.
    if (range.first == range.second && range.first == total_size) {
        return;
    }
    // range == (0, 0): all data is too new, mask everything out.
    if (range.first == range.second && range.first == 0) {
        bitset_chunk.set();
        return;
    }
    // [0, beg) = false, [beg, end) = check, [end, size) = true
    BitsetType mask;
    mask.reserve(total_size);
    mask.resize(range.first, false);
    mask.resize(total_size, true);
    do_scan(range.first, range.second, [&](int64_t i, Timestamp val) {
        mask[i] = val > timestamp;
    });
    bitset_chunk |= mask;
}

bool
ChunkedSegmentSealedImpl::generate_interim_index(const FieldId field_id,
                                                 int64_t num_rows) {
    if (col_index_meta_ == nullptr || !col_index_meta_->HasField(field_id)) {
        return false;
    }
    auto& field_meta = schema_->operator[](field_id);
    auto& field_index_meta = col_index_meta_->GetFieldIndexMeta(field_id);
    auto& index_params = field_index_meta.GetIndexParams();

    bool is_sparse =
        field_meta.get_data_type() == DataType::VECTOR_SPARSE_U32_F32;

    bool enable_growing_mmap = storage::MmapManager::GetInstance()
                                   .GetMmapConfig()
                                   .GetEnableGrowingMmap();

    auto enable_binlog_index = [&]() {
        // check milvus config
        if (!segcore_config_.get_enable_interim_segment_index() ||
            enable_growing_mmap) {
            return false;
        }
        // check data type
        if (field_meta.get_data_type() != DataType::VECTOR_FLOAT &&
            field_meta.get_data_type() != DataType::VECTOR_FLOAT16 &&
            field_meta.get_data_type() != DataType::VECTOR_BFLOAT16 &&
            !is_sparse) {
            return false;
        }
        // check index type
        if (index_params.find(knowhere::meta::INDEX_TYPE) ==
                index_params.end() ||
            field_index_meta.IsFlatIndex()) {
            return false;
        }
        // check index exist
        if (vector_indexings_.is_ready(field_id)) {
            return false;
        }
        return true;
    };
    if (!enable_binlog_index()) {
        return false;
    }
    try {
        std::shared_ptr<ChunkedColumnInterface> vec_data = get_column(field_id);
        AssertInfo(
            vec_data != nullptr, "vector field {} not loaded", field_id.get());
        int64_t row_count = field_meta.is_nullable()
                                ? vec_data->GetOffsetMapping().GetValidCount()
                                : num_rows;

        // generate index params
        auto field_binlog_config = std::unique_ptr<VecIndexConfig>(
            new VecIndexConfig(row_count,
                               field_index_meta,
                               segcore_config_,
                               SegmentType::Sealed,
                               is_sparse));
        if (row_count < field_binlog_config->GetBuildThreshold()) {
            return false;
        }
        auto dim = is_sparse ? std::numeric_limits<uint32_t>::max()
                             : field_meta.get_dim();
        auto interim_index_type = field_binlog_config->GetIndexType();
        auto build_config =
            field_binlog_config->GetBuildBaseParams(field_meta.get_data_type());
        build_config[knowhere::meta::DIM] = std::to_string(dim);
        build_config[knowhere::meta::NUM_BUILD_THREAD] = std::to_string(1);
        auto index_metric = field_binlog_config->GetMetricType();

        if (enable_binlog_index()) {
            std::unique_lock lck(mutex_);

            std::unique_ptr<
                milvus::cachinglayer::Translator<milvus::index::IndexBase>>
                translator =
                    std::make_unique<milvus::segcore::storagev1translator::
                                         InterimSealedIndexTranslator>(
                        vec_data,
                        id_,
                        field_id.get(),
                        interim_index_type,
                        index_metric,
                        build_config,
                        dim,
                        is_sparse,
                        field_meta.get_data_type());

            auto interim_index_cache_slot =
                milvus::cachinglayer::Manager::GetInstance().CreateCacheSlot(
                    std::move(translator));
            // TODO: how to handle the binlog index?
            vector_indexings_.append_field_indexing(
                field_id, index_metric, std::move(interim_index_cache_slot));

            vec_binlog_config_[field_id] = std::move(field_binlog_config);
            set_bit(binlog_index_bitset_, field_id, true);
            auto index_version =
                knowhere::Version::GetCurrentVersion().VersionNumber();
            if (is_sparse ||
                field_meta.get_data_type() == DataType::VECTOR_FLOAT) {
                index_has_raw_data_[field_id] =
                    knowhere::IndexStaticFaced<float>::HasRawData(
                        interim_index_type, index_version, build_config);
            } else if (field_meta.get_data_type() == DataType::VECTOR_FLOAT16) {
                index_has_raw_data_[field_id] =
                    knowhere::IndexStaticFaced<float16>::HasRawData(
                        interim_index_type, index_version, build_config);
            } else if (field_meta.get_data_type() ==
                       DataType::VECTOR_BFLOAT16) {
                index_has_raw_data_[field_id] =
                    knowhere::IndexStaticFaced<bfloat16>::HasRawData(
                        interim_index_type, index_version, build_config);
            }

            LOG_INFO(
                "replace binlog with intermin index in segment {}, "
                "field {}.",
                this->get_segment_id(),
                field_id.get());
        }
        return true;
    } catch (std::exception& e) {
        LOG_WARN("fail to generate intermin index, because {}", e.what());
        return false;
    }
}
void
ChunkedSegmentSealedImpl::RemoveFieldFile(const FieldId field_id) {
}

void
ChunkedSegmentSealedImpl::LazyCheckSchema(SchemaPtr sch,
                                          milvus::OpContext* op_ctx) {
    if (!sch) {
        return;
    }

    uint64_t current_schema_version;
    {
        std::shared_lock lck(mutex_);
        current_schema_version = schema_->get_schema_version();
    }

    if (sch->get_schema_version() > current_schema_version) {
        LOG_INFO(
            "lazy check schema segment {} found newer schema version, "
            "current "
            "schema version {}, new schema version {}",
            id_,
            current_schema_version,
            sch->get_schema_version());
        Reopen(op_ctx, std::move(sch));
    }
}

void
ChunkedSegmentSealedImpl::load_field_data_common(
    FieldId field_id,
    const std::shared_ptr<ChunkedColumnInterface>& column,
    size_t num_rows,
    DataType data_type,
    bool enable_mmap,
    bool is_proxy_column,
    std::optional<ParquetStatistics> statistics,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    FieldDataCommonTiming timing;
    timing.system = SystemProperty::Instance().IsSystem(field_id);
    timing.vector = IsVectorDataType(data_type);
    timing.variable = IsVariableDataType(data_type);
    timing.proxy = is_proxy_column;
    timing.mmap = enable_mmap;
    timing.pk = schema_->get_primary_field_id().value_or(FieldId(-1)).get() ==
                field_id.get();
    timing.nullable = column->IsNullable();

    {
        std::unique_lock lck(mutex_);
        if (is_replace) {
            // Subtract old column memory before replacing
            auto old_column = get_column(field_id);
            if (old_column && !enable_mmap) {
                if (!is_proxy_column ||
                    (is_proxy_column &&
                     field_id.get() != DEFAULT_SHORT_COLUMN_GROUP_ID)) {
                    stats_.mem_size -= old_column->DataByteSize();
                }
            }
            fields_.wlock()->insert_or_assign(field_id, column);
            LOG_INFO(
                "Replacing field {} data in segment {}", field_id.get(), id_);
        } else {
            AssertInfo(SystemProperty::Instance().IsSystem(field_id) ||
                           !get_bit(field_data_ready_bitset_, field_id),
                       "non system field {} data already loaded",
                       field_id.get());
            bool already_exists = false;
            fields_.withRLock([&](auto& fields) {
                already_exists = fields.find(field_id) != fields.end();
            });
            AssertInfo(!already_exists,
                       "field {} column already exists",
                       field_id.get());
            fields_.wlock()->emplace(field_id, column);
        }
        if (enable_mmap) {
            mmap_field_ids_.insert(field_id);
        }
    }
    timing.initial_lock_ns = DurationNs(stage_start);

    // system field only needs to emplace column to fields_ map
    if (timing.system) {
        timing.total_ns = DurationNs(total_start);
        field_data_common_timing_stats.Record(timing);
        return;
    }

    stage_start = std::chrono::steady_clock::now();
    if (timing.nullable && timing.vector) {
        bool lazy_inited = false;
        // For VECTOR_ARRAY, Parquet num_values is the child vector count, not
        // the outer row count. Empty non-null rows would corrupt physical row
        // mapping if we derived valid counts from those statistics.
        if (data_type != DataType::VECTOR_ARRAY && statistics.has_value() &&
            !statistics.value().empty()) {
            const auto& stats = statistics.value();
            bool any_null = false;
            for (const auto& s : stats) {
                if (s == nullptr) {
                    any_null = true;
                    break;
                }
            }
            if (!any_null) {
                lazy_inited = column->TryInitValidRowIdsFromRowGroups(
                    stats.size(),
                    [&](size_t i) {
                        return stats[i]->num_values() + stats[i]->null_count();
                    },
                    [&](size_t i) { return stats[i]->null_count(); });
            }
        }
        if (!lazy_inited) {
            column->BuildValidRowIds(op_ctx);
        }
    }
    timing.nullable_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    if (!enable_mmap) {
        if (!is_proxy_column ||
            (is_proxy_column &&
             field_id.get() != DEFAULT_SHORT_COLUMN_GROUP_ID)) {
            stats_.mem_size += column->DataByteSize();
        }
        if (IsVariableDataType(data_type)) {
            // update average row data size
            SegmentInternalInterface::set_field_avg_size(
                field_id, num_rows, column->DataByteSize());
        }
    }
    timing.memory_stats_ns = DurationNs(stage_start);

    // Skip index construction: for proxy columns (external tables) with no
    // statistics, skip building the skip index during load to avoid triggering
    // S3 data fetches when warmup=disable. The skip index will be unavailable
    // for these segments, which only affects skip-index-based pruning.
    stage_start = std::chrono::steady_clock::now();
    if (!IsVariableDataType(data_type) || IsStringDataType(data_type)) {
        if (statistics) {
            LoadSkipIndexFromStatistics(
                field_id, data_type, statistics.value());
        } else if (!is_proxy_column) {
            LoadSkipIndex(field_id, data_type, column);
        }
    }
    timing.skip_index_ns = DurationNs(stage_start);

    // set pks to offset
    stage_start = std::chrono::steady_clock::now();
    if (timing.pk) {
        if (std::atomic_load(&segment_load_info_)->GetStorageVersion() >=
            STORAGE_V2) {
            init_storage_v2_pk_index(field_id, column, data_type);
        } else {
            init_storage_v1_pk_index(field_id, column, data_type, is_replace);
        }
    }
    timing.pk_index_ns = DurationNs(stage_start);

    // now interim index does not touch column warmup
    stage_start = std::chrono::steady_clock::now();
    generate_interim_index(field_id, num_rows);
    timing.interim_index_ns = DurationNs(stage_start);

    std::string struct_name;
    const FieldMeta* field_meta_ptr = nullptr;

    stage_start = std::chrono::steady_clock::now();
    {
        std::unique_lock lck(mutex_);
        if (!is_replace) {
            AssertInfo(!get_bit(field_data_ready_bitset_, field_id),
                       "field {} data already loaded",
                       field_id.get());
        }
        set_bit(field_data_ready_bitset_, field_id, true);
        update_row_count(num_rows);

        if (data_type == DataType::GEOMETRY &&
            segcore_config_.get_enable_geometry_cache()) {
            // Construct GeometryCache for the entire field
            LoadGeometryCache(field_id, column);
        }

        // Check if need to build ArrayOffsetsSealed for struct array fields.
        auto& field_meta = schema_->operator[](field_id);
        if (auto parsed_struct_name = GetStructNameForArrayField(field_meta);
            parsed_struct_name.has_value()) {
            struct_name = *parsed_struct_name;

            auto it = struct_to_array_offsets_.find(struct_name);
            if (it != struct_to_array_offsets_.end()) {
                array_offsets_map_[field_id] = it->second;
            } else {
                field_meta_ptr = &field_meta;  // need to build
            }
        }
    }
    timing.final_lock_ns = DurationNs(stage_start);

    // Build ArrayOffsetsSealed outside lock (expensive operation)
    stage_start = std::chrono::steady_clock::now();
    if (field_meta_ptr) {
        auto new_offsets =
            ArrayOffsetsSealed::BuildFromSegment(this, *field_meta_ptr);

        std::unique_lock lck(mutex_);
        // Double-check after re-acquiring lock
        auto it = struct_to_array_offsets_.find(struct_name);
        if (it == struct_to_array_offsets_.end()) {
            struct_to_array_offsets_[struct_name] = new_offsets;
            array_offsets_map_[field_id] = new_offsets;
        } else {
            array_offsets_map_[field_id] = it->second;
        }
    }
    timing.array_offsets_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    field_data_common_timing_stats.Record(timing);
}

static TimestampIndex
build_timestamp_index(const Timestamp* data, size_t num_rows) {
    TimestampIndex index;
    auto min_slice_length = num_rows < 4096 ? 1 : 4096;
    auto meta = GenerateFakeSlices(data, num_rows, min_slice_length);
    index.set_length_meta(std::move(meta));
    index.build_with(data, num_rows);
    return index;
}

void
ChunkedSegmentSealedImpl::init_storage_v1_timestamp_index(
    std::vector<Timestamp> timestamps, size_t num_rows) {
    auto index = build_timestamp_index(timestamps.data(), num_rows);
    std::unique_lock lck(mutex_);
    AssertInfo(insert_record_.timestamps_.empty(), "already exists");
    insert_record_.init_timestamps_from_owned(std::move(timestamps),
                                              std::move(index));
    stats_.mem_size += sizeof(Timestamp) * num_rows;
}

void
ChunkedSegmentSealedImpl::ApplySchemaForReopen(SchemaPtr sch) {
    if (!sch) {
        return;
    }

    std::unique_lock lck(mutex_);
    if (sch->get_schema_version() <= schema_->get_schema_version()) {
        return;
    }

    field_data_ready_bitset_.resize(sch->get_field_id_bitset_size());
    index_ready_bitset_.resize(sch->get_field_id_bitset_size());
    binlog_index_bitset_.resize(sch->get_field_id_bitset_size());
    schema_ = std::move(sch);
}

void
ChunkedSegmentSealedImpl::Reopen(SchemaPtr sch) {
    milvus::OpContext op_ctx;
    Reopen(&op_ctx, std::move(sch));
}

void
ChunkedSegmentSealedImpl::Reopen(milvus::OpContext* op_ctx, SchemaPtr sch) {
    if (!sch) {
        return;
    }

    std::lock_guard<std::mutex> reopen_guard(reopen_mutex_);
    SchemaPtr current_schema;
    {
        std::shared_lock lck(mutex_);
        current_schema = schema_;
    }
    // Schema-only reopen carries no load-info updates, so equal-version input
    // cannot produce work. Reopen(load_info, schema) still accepts equal schema
    // versions because load info may have changed independently.
    if (sch->get_schema_version() <= current_schema->get_schema_version()) {
        return;
    }

    auto current = std::atomic_load(&segment_load_info_);
    SegmentLoadInfo current_mutable(*current);
    SegmentLoadInfo new_local(current->GetProto(), sch);
    for (auto fid : current->GetCreatedTextIndexes()) {
        new_local.SetTextIndexCreated(fid);
    }

    auto diff = current_mutable.ComputeDiff(new_local);
    new_local.SetFieldsFilledWithDefault(
        current_mutable.GetDefaultFilledFieldsForNewInfo(new_local));
    LOG_INFO(
        "Schema-only reopen segment {} with diff {}", id_, diff.ToString());

    auto published = std::make_shared<const SegmentLoadInfo>(new_local);
    std::atomic_store(&segment_load_info_, published);
    use_take_for_output_.store(published->GetUseTakeForOutput(),
                               std::memory_order_relaxed);

    ApplySchemaForReopen(sch);
    ApplyLoadDiff(op_ctx, new_local, diff);

    LOG_INFO("Schema-only reopen segment {} done", id_);
}

void
ChunkedSegmentSealedImpl::Reopen(
    milvus::OpContext* op_ctx,
    const milvus::proto::segcore::SegmentLoadInfo& new_load_info) {
    Reopen(op_ctx, new_load_info, nullptr);
}

void
ChunkedSegmentSealedImpl::Reopen(
    milvus::OpContext* op_ctx,
    const milvus::proto::segcore::SegmentLoadInfo& new_load_info,
    SchemaPtr new_schema) {
    // reopen_mutex_ serializes top-level writers of segment_load_info_.
    // It is held across ApplyLoadDiff so two Reopens never interleave their
    // resource mutations. Readers are unaffected — they snapshot via
    // std::atomic_load and never touch this mutex.
    std::lock_guard<std::mutex> reopen_guard(reopen_mutex_);

    SchemaPtr current_schema;
    {
        std::shared_lock lck(mutex_);
        current_schema = schema_;
    }
    if (new_schema && new_schema->get_schema_version() <
                          current_schema->get_schema_version()) {
        LOG_WARN(
            "Skip stale reopen segment {}, current schema version {}, incoming "
            "schema version {}",
            id_,
            current_schema->get_schema_version(),
            new_schema->get_schema_version());
        return;
    }

    auto current = std::atomic_load(&segment_load_info_);
    auto target_schema = new_schema ? std::move(new_schema) : current_schema;

    SegmentLoadInfo current_mutable(*current);
    SegmentLoadInfo new_local(new_load_info, target_schema);
    for (auto fid : current->GetCreatedTextIndexes()) {
        new_local.SetTextIndexCreated(fid);
    }

    auto diff = current_mutable.ComputeDiff(new_local);
    new_local.SetFieldsFilledWithDefault(
        current_mutable.GetDefaultFilledFieldsForNewInfo(new_local));
    LOG_INFO("Reopen segment {} with diff {}", id_, diff.ToString());

    auto published = std::make_shared<const SegmentLoadInfo>(new_local);
    std::atomic_store(&segment_load_info_, published);
    use_take_for_output_.store(published->GetUseTakeForOutput(),
                               std::memory_order_relaxed);

    ApplySchemaForReopen(target_schema);
    ApplyLoadDiff(op_ctx, new_local, diff);

    LOG_INFO("Reopen segment {} done", id_);
}

void
ChunkedSegmentSealedImpl::ApplyLoadDiff(milvus::OpContext* op_ctx,
                                        SegmentLoadInfo& segment_load_info,
                                        LoadDiff& diff) {
    auto total_start = std::chrono::steady_clock::now();
    ApplyLoadDiffTiming timing;
    timing.index_load_count = CountIndexInfos(diff.indexes_to_load);
    timing.index_replace_count = CountIndexInfos(diff.indexes_to_replace);
    timing.column_group_count =
        CountColumnGroups(diff.column_groups_to_load) +
        CountColumnGroups(diff.column_groups_to_replace) +
        CountColumnGroups(diff.column_groups_to_lazyload) +
        CountColumnGroups(diff.column_groups_to_lazyreplace);
    timing.binlog_group_count =
        diff.binlogs_to_load.size() + diff.binlogs_to_replace.size();
    timing.text_index_count =
        diff.text_indexes_to_load.size() + diff.text_indexes_to_create.size();
    timing.json_stats_count =
        diff.json_stats_to_load.size() + diff.json_stats_to_replace.size();

    // TODO: pass trace_ctx separately when needed
    milvus::tracer::TraceContext trace_ctx;

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Load new indexes (fields without existing index)
    auto stage_start = std::chrono::steady_clock::now();
    if (!diff.indexes_to_load.empty()) {
        LoadBatchIndexes(trace_ctx, diff.indexes_to_load, op_ctx);
    }
    timing.indexes_load_ns = DurationNs(stage_start);
    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Replace indexes (fields that already have an index loaded)
    stage_start = std::chrono::steady_clock::now();
    if (!diff.indexes_to_replace.empty()) {
        LoadBatchIndexes(trace_ctx, diff.indexes_to_replace, op_ctx, true);
    }
    timing.indexes_replace_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // reload fields (warmup for fields already in memory)
    stage_start = std::chrono::steady_clock::now();
    if (!diff.fields_to_reload.empty()) {
        ReloadColumns(diff.fields_to_reload, op_ctx);
    }
    timing.reload_fields_ns = DurationNs(stage_start);

    // Load field data from storage BEFORE dropping indexes, so that queries
    // always have a data source available during the transition.

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // load column groups
    if (diff.load_external_manifest) {
        // External collections: load via manifest path
        stage_start = std::chrono::steady_clock::now();
        LoadColumnGroups(segment_load_info.GetManifestPath(), op_ctx);
        timing.column_groups_load_ns = DurationNs(stage_start);
    } else {
        bool has_cg_changes = !diff.column_groups_to_load.empty() ||
                              !diff.column_groups_to_replace.empty() ||
                              !diff.column_groups_to_lazyload.empty() ||
                              !diff.column_groups_to_lazyreplace.empty();
        if (has_cg_changes) {
            stage_start = std::chrono::steady_clock::now();
            auto properties =
                milvus::storage::LoonFFIPropertiesSingleton::GetInstance()
                    .GetProperties();
            auto column_groups = segment_load_info.GetColumnGroups();
            auto arrow_schema = schema_->ConvertToLoonArrowSchema();
            auto needed_columns = std::make_shared<std::vector<std::string>>();
            for (const auto& field_id : schema_->get_field_ids()) {
                needed_columns->push_back(std::to_string(field_id.get()));
            }
            // reader_mutex_ guards reader_ against concurrent ExecuteTake.
            // ApplyLoadDiff is invoked from Reopen AFTER mutex_ has been
            // released (Reopen:lck.unlock()), so a concurrent take() on the
            // old reader_ could otherwise observe a half-assigned shared_ptr
            // or lose its referent mid-call.
            {
                std::lock_guard<std::mutex> lock(reader_mutex_);
                reader_ = milvus_storage::api::Reader::create(
                    column_groups, arrow_schema, needed_columns, *properties);
            }
            timing.prepare_column_groups_ns = DurationNs(stage_start);
            // New column group fields
            stage_start = std::chrono::steady_clock::now();
            if (!diff.column_groups_to_load.empty()) {
                LoadColumnGroups(column_groups,
                                 properties,
                                 diff.column_groups_to_load,
                                 true,
                                 op_ctx);
            }
            timing.column_groups_load_ns = DurationNs(stage_start);
            stage_start = std::chrono::steady_clock::now();
            if (!diff.column_groups_to_lazyload.empty()) {
                LoadColumnGroups(column_groups,
                                 properties,
                                 diff.column_groups_to_lazyload,
                                 false,
                                 op_ctx);
            }
            timing.column_groups_lazy_load_ns = DurationNs(stage_start);
            // Replace column group fields
            stage_start = std::chrono::steady_clock::now();
            if (!diff.column_groups_to_replace.empty()) {
                LoadColumnGroups(column_groups,
                                 properties,
                                 diff.column_groups_to_replace,
                                 true,
                                 op_ctx,
                                 true);
            }
            timing.column_groups_replace_ns = DurationNs(stage_start);
            stage_start = std::chrono::steady_clock::now();
            if (!diff.column_groups_to_lazyreplace.empty()) {
                LoadColumnGroups(column_groups,
                                 properties,
                                 diff.column_groups_to_lazyreplace,
                                 false,
                                 op_ctx,
                                 true);
            }
            timing.column_groups_lazy_replace_ns = DurationNs(stage_start);
        }
    }

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Initialize LOB paths for TEXT fields after any column group loading
    stage_start = std::chrono::steady_clock::now();
    if (segment_load_info.HasManifestPath()) {
        InitTextLobPaths(segment_load_info.GetManifestPath());
    }
    timing.init_text_lob_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Load new field binlogs
    stage_start = std::chrono::steady_clock::now();
    if (!diff.binlogs_to_load.empty()) {
        LoadBatchFieldData(trace_ctx, diff.binlogs_to_load, op_ctx);
    }
    timing.binlogs_load_ns = DurationNs(stage_start);
    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Replace field binlogs
    stage_start = std::chrono::steady_clock::now();
    if (!diff.binlogs_to_replace.empty()) {
        LoadBatchFieldData(trace_ctx, diff.binlogs_to_replace, op_ctx, true);
    }
    timing.binlogs_replace_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // drop index — field data is already loaded/restored above, so queries
    // can fall back to raw data after the index is dropped.
    stage_start = std::chrono::steady_clock::now();
    if (!diff.indexes_to_drop.empty()) {
        for (auto field_id : diff.indexes_to_drop) {
            // Skip drop if this field already has a replacement or new index loaded
            if (diff.indexes_to_replace.count(field_id) > 0 ||
                diff.indexes_to_load.count(field_id) > 0) {
                continue;
            }
            DropIndex(field_id);
        }
    }
    timing.drop_index_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // load pre-built text indexes
    stage_start = std::chrono::steady_clock::now();
    if (!diff.text_indexes_to_load.empty()) {
        LoadBatchTextIndexes(op_ctx, diff.text_indexes_to_load);
    }
    timing.text_indexes_load_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    stage_start = std::chrono::steady_clock::now();
    if (!diff.json_stats_to_load.empty()) {
        LoadBatchJsonKeyIndexes(op_ctx, diff.json_stats_to_load);
    }
    timing.json_stats_load_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    if (!diff.json_stats_to_replace.empty()) {
        LoadBatchJsonKeyIndexes(op_ctx, diff.json_stats_to_replace);
    }
    timing.json_stats_replace_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    if (!diff.json_stats_to_drop.empty()) {
        for (auto field_id : diff.json_stats_to_drop) {
            if (diff.json_stats_to_load.count(field_id) > 0 ||
                diff.json_stats_to_replace.count(field_id) > 0) {
                LOG_INFO(
                    "skip drop json key stats because replacement is loaded, "
                    "segment:{}, field:{}",
                    id_,
                    field_id.get());
                continue;
            }
            LOG_INFO("drop json key stats, segment:{}, field:{}",
                     id_,
                     field_id.get());
            RemoveJsonStats(field_id);
        }
    }
    timing.json_stats_drop_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // fill default values for fields without data sources (schema evolution)
    stage_start = std::chrono::steady_clock::now();
    if (!diff.fields_to_fill_default.empty()) {
        FillDefaultValueFields(diff.fields_to_fill_default);
        RecordDefaultFieldsFilled(diff.fields_to_fill_default);
    }
    timing.fill_default_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // create text indexes from raw data
    stage_start = std::chrono::steady_clock::now();
    if (!diff.text_indexes_to_create.empty()) {
        for (const auto& field_id : diff.text_indexes_to_create) {
            CreateTextIndex(field_id, op_ctx);
        }
    }
    timing.text_indexes_create_ns = DurationNs(stage_start);

    CheckCancellation(op_ctx, id_, "ChunkedSegmentSealedImpl::ApplyLoadDiff()");

    // Drop field data — only for schema evolution scenarios where
    // the field has been removed from the data source (binlogs/column_groups).
    stage_start = std::chrono::steady_clock::now();
    if (!diff.field_data_to_drop.empty()) {
        for (auto field_id : diff.field_data_to_drop) {
            DropFieldData(field_id);
        }
    }
    timing.field_data_drop_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    apply_load_diff_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::fill_empty_field(const FieldMeta& field_meta) {
    auto field_id = field_meta.get_id();
    auto data_type = field_meta.get_data_type();
    LOG_INFO(
        "start fill empty field {} (data type {}) for sealed segment "
        "{}",
        data_type,
        field_id.get(),
        id_);
    auto [field_has_setting, field_mmap_enabled] =
        schema_->MmapEnabled(field_id);
    auto is_vector = IsVectorDataType(field_meta.get_data_type());
    auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();
    bool global_use_mmap = is_vector ? mmap_config.GetVectorFieldEnableMmap()
                                     : mmap_config.GetScalarFieldEnableMmap();
    bool use_mmap = field_has_setting ? field_mmap_enabled : global_use_mmap;
    auto mmap_dir_path =
        milvus::storage::LocalChunkManagerSingleton::GetInstance()
            .GetChunkManager()
            ->GetRootPath();
    int64_t size = num_rows_.value();
    AssertInfo(size > 0, "Chunked Sealed segment must have more than 0 row");
    auto field_data_info = FieldDataInfo(field_id.get(), size, mmap_dir_path);

    auto [field_has_warmup, field_warmup_policy] = schema_->WarmupPolicy(
        field_id, IsVectorDataType(data_type), /*is_index=*/false);
    std::string warmup_policy = field_has_warmup ? field_warmup_policy : "";
    std::unique_ptr<Translator<milvus::Chunk>> translator =
        std::make_unique<storagev1translator::DefaultValueChunkTranslator>(
            get_segment_id(),
            field_meta,
            field_data_info,
            use_mmap,
            mmap_config.GetMmapPopulate(),
            warmup_policy);
    auto slot = cachinglayer::Manager::GetInstance().CreateCacheSlot(
        std::move(translator), nullptr);
    auto column = MakeChunkedColumnBase(data_type, std::move(slot), field_meta);

    if (column->IsNullable() && IsVectorDataType(data_type)) {
        column->BuildValidRowIds(nullptr);
    }

    fields_.wlock()->emplace(field_id, column);
    set_bit(field_data_ready_bitset_, field_id, true);
    LOG_INFO(
        "fill empty field {} (data type {}) for growing segment {} "
        "done",
        field_meta.get_data_type(),
        field_id.get(),
        id_);
}

void
ChunkedSegmentSealedImpl::EnsureArrayOffsetsForStructField(
    const FieldMeta& field_meta, int64_t row_count) {
    auto struct_name = GetStructNameForArrayField(field_meta);
    if (!struct_name.has_value()) {
        return;
    }

    auto it = struct_to_array_offsets_.find(*struct_name);
    if (it == struct_to_array_offsets_.end()) {
        std::vector<int32_t> row_to_element_start(row_count + 1, 0);
        auto array_offsets = std::make_shared<ArrayOffsetsSealed>(
            std::move(row_to_element_start));
        it =
            struct_to_array_offsets_.emplace(*struct_name, array_offsets).first;
    }

    array_offsets_map_[field_meta.get_id()] = it->second;
}

void
ChunkedSegmentSealedImpl::FillDefaultValueFields(
    const std::vector<FieldId>& field_ids) {
    std::unique_lock lck(mutex_);
    for (const auto& field_id : field_ids) {
        // Skip if field data already loaded
        if (get_bit(field_data_ready_bitset_, field_id)) {
            continue;
        }
        // Skip if index has raw data
        if (get_bit(index_ready_bitset_, field_id) &&
            index_has_raw_data_[field_id]) {
            continue;
        }
        const auto& field_meta = schema_->operator[](field_id);
        fill_empty_field(field_meta);
        EnsureArrayOffsetsForStructField(field_meta, num_rows_.value_or(0));
    }
}

void
ChunkedSegmentSealedImpl::LoadGeometryCache(
    FieldId field_id, const std::shared_ptr<ChunkedColumnInterface>& column) {
    try {
        // Get geometry cache for this segment+field
        auto& geometry_cache =
            milvus::exec::SimpleGeometryCacheManager::Instance()
                .GetOrCreateCache(get_segment_id(), field_id);

        // Iterate through all chunks and collect WKB data
        auto num_chunks = column->num_chunks();
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            // Get all string views from this chunk
            auto pw = column->StringViews(nullptr, chunk_id);
            auto [string_views, valid_data] = pw.get();

            // Add each string view to the geometry cache
            for (size_t i = 0; i < string_views.size(); ++i) {
                if (valid_data.empty() || valid_data[i]) {
                    // Valid geometry data
                    const auto& wkb_data = string_views[i];
                    geometry_cache.AppendData(
                        ctx_, wkb_data.data(), wkb_data.size());
                } else {
                    // Null/invalid geometry
                    geometry_cache.AppendData(ctx_, nullptr, 0);
                }
            }
        }

        LOG_INFO(
            "Successfully loaded geometry cache for segment {} field {} "
            "with "
            "{} geometries",
            get_segment_id(),
            field_id.get(),
            geometry_cache.Size());

    } catch (const std::exception& e) {
        ThrowInfo(UnexpectedError,
                  "Failed to load geometry cache for segment {} field {}: {}",
                  get_segment_id(),
                  field_id.get(),
                  e.what());
    }
}

void
ChunkedSegmentSealedImpl::SetCommitTimestamp(uint64_t ts) {
    std::unique_lock lck(mutex_);
    commit_ts_ = ts;
}

uint64_t
ChunkedSegmentSealedImpl::GetCommitTimestamp() const {
    return commit_ts_;
}

void
ChunkedSegmentSealedImpl::SetLoadInfo(
    proto::segcore::SegmentLoadInfo load_info) {
    // reopen_mutex_ serializes with Reopen(pb)/Load/other SetLoadInfo so the
    // published snapshot and use_take_for_output_ bit stay in sync.
    std::lock_guard<std::mutex> reopen_guard(reopen_mutex_);
    auto commit_ts =
        static_cast<milvus::Timestamp>(load_info.commit_timestamp());
    {
        std::unique_lock lck(mutex_);
        commit_ts_ = commit_ts;
    }
    auto published =
        std::make_shared<const SegmentLoadInfo>(std::move(load_info), schema_);
    std::atomic_store(&segment_load_info_, published);
    use_take_for_output_.store(published->GetUseTakeForOutput(),
                               std::memory_order_relaxed);
    LOG_INFO(
        "SetLoadInfo for segment {}, num_rows: {}, index count: {}, "
        "storage_version: {}, use_take_for_output: {}, commit_ts: {}",
        id_,
        published->GetNumOfRows(),
        published->GetIndexInfoCount(),
        published->GetStorageVersion(),
        use_take_for_output_.load(std::memory_order_relaxed),
        commit_ts);
}

void
ChunkedSegmentSealedImpl::LoadManifest(const std::string& manifest_path) {
    LOG_INFO(
        "Loading segment {} field data with manifest {}", id_, manifest_path);
    auto properties = milvus::storage::LoonFFIPropertiesSingleton::GetInstance()
                          .GetProperties();

    auto column_groups =
        std::atomic_load(&segment_load_info_)->GetColumnGroups();

    auto arrow_schema = schema_->ConvertToArrowSchema();
    reader_ = milvus_storage::api::Reader::create(
        column_groups, arrow_schema, nullptr, *properties);

    std::vector<std::pair<int, std::vector<FieldId>>> cg_field_ids;
    for (int i = 0; i < column_groups->size(); ++i) {
        auto column_group = column_groups->at(i);
        std::vector<FieldId> milvus_field_ids;
        for (auto& column : column_group->columns) {
            auto field_id = std::stoll(column);
            milvus_field_ids.emplace_back(field_id);
        }
        cg_field_ids.emplace_back(i, std::move(milvus_field_ids));
    }

    LoadColumnGroups(column_groups, properties, cg_field_ids, true);

    // initialize LOB paths for TEXT fields
    InitTextLobPaths(manifest_path);
}

void
ChunkedSegmentSealedImpl::InitTextLobPaths(const std::string& manifest_path) {
    std::vector<FieldId> text_field_ids;
    for (auto& [field_id, field_meta] : schema_->get_fields()) {
        if (field_meta.get_data_type() == DataType::TEXT) {
            text_field_ids.push_back(field_id);
        }
    }

    if (text_field_ids.empty()) {
        return;
    }

    std::string segment_base_path;
    try {
        nlohmann::json j = nlohmann::json::parse(manifest_path);
        segment_base_path = j.at("base_path").get<std::string>();
    } catch (const std::exception& e) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "Failed to parse manifest path for TEXT columns: {}",
                  e.what());
    }

    // segment_base_path format: {root}/{collectionID}/{partitionID}/{segmentID}
    // lob_base_path format: {root}/{collectionID}/{partitionID}/lobs/{field_id}
    std::filesystem::path segment_fs_path(segment_base_path);
    std::filesystem::path partition_path = segment_fs_path.parent_path();

    for (auto field_id : text_field_ids) {
        std::filesystem::path lob_base_path =
            partition_path / "lobs" / std::to_string(field_id.get());
        text_lob_paths_[field_id] = lob_base_path.string();
        LOG_INFO("Initialized TEXT LOB path for segment {} field {}: {}",
                 id_,
                 field_id.get(),
                 lob_base_path.string());
    }
}

void
ChunkedSegmentSealedImpl::LoadColumnGroups(
    const std::shared_ptr<milvus_storage::api::ColumnGroups>& column_groups,
    const std::shared_ptr<milvus_storage::api::Properties>& properties,
    std::vector<std::pair<int, std::vector<FieldId>>>& cg_field_ids,
    bool eager_load,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    BatchTaskTiming timing;
    timing.task_count = cg_field_ids.size();
    timing.is_replace = is_replace;
    for (const auto& pair : cg_field_ids) {
        timing.field_count += pair.second.size();
    }

    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> load_group_futures;
    load_group_futures.reserve(cg_field_ids.size());
    for (const auto& pair : cg_field_ids) {
        auto cg_index = pair.first;
        const auto& field_ids = pair.second;
        auto future = pool.Submit([this,
                                   column_groups,
                                   properties,
                                   cg_index,
                                   field_ids,
                                   eager_load,
                                   op_ctx,
                                   is_replace]() {
            // Early exit if cancelled while queued
            CheckCancellation(op_ctx,
                              id_,
                              cg_index,
                              "ChunkedSegmentSealedImpl::LoadColumnGroup()");
            LoadColumnGroup(column_groups,
                            properties,
                            cg_index,
                            field_ids,
                            eager_load,
                            op_ctx,
                            is_replace);
        });
        load_group_futures.emplace_back(std::move(future));
    }
    timing.submit_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    storage::WaitAllFutures(load_group_futures);
    timing.wait_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    batch_column_group_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadColumnGroup(
    const std::shared_ptr<milvus_storage::api::ColumnGroups>& column_groups,
    const std::shared_ptr<milvus_storage::api::Properties>& properties,
    int64_t index,
    const std::vector<FieldId>& milvus_field_ids,
    bool eager_load,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    auto total_start = std::chrono::steady_clock::now();
    ColumnGroupTiming timing;
    timing.field_count = milvus_field_ids.size();
    timing.eager_load = eager_load;

    AssertInfo(index < column_groups->size(),
               "load column group index out of range");
    AssertInfo(!milvus_field_ids.empty(),
               "load column group with empty field list");
    auto column_group = column_groups->at(index);
    auto load_info = std::atomic_load(&segment_load_info_);

    for (const auto& field_id : milvus_field_ids) {
        AssertInfo(field_exists_in_schema(schema_, field_id),
                   "field {} not found in schema when loading column group",
                   field_id.get());
    }

    auto field_metas = schema_->get_field_metas(milvus_field_ids);

    // assumption: vector field occupies whole column group
    bool is_vector = false;
    bool has_mmap_setting = false;
    bool mmap_enabled = false;
    bool has_warmup_setting = false;
    std::string aggregated_warmup_policy = "disable";
    auto stage_start = std::chrono::steady_clock::now();
    for (auto& [field_id, field_meta] : field_metas) {
        if (IsVectorDataType(field_meta.get_data_type())) {
            is_vector = true;
        }
        std::shared_lock lck(mutex_);

        // if field has mmap setting, use it
        // - mmap setting at collection level, then all field are the same
        // - mmap setting at field level, we define that as long as one field shall be mmap, then whole group shall be mmaped
        auto [field_has_setting, field_mmap_enabled] =
            schema_->MmapEnabled(field_id);
        has_mmap_setting = has_mmap_setting || field_has_setting;
        mmap_enabled = mmap_enabled || field_mmap_enabled;

        // if field has warmup setting, use it
        // - warmup setting at collection level, uses appropriate key based on field type
        // - warmup setting at field level, use the most aggressive policy (sync > async > disable)
        // Note: this is for field data loading, not index (is_index = false)
        bool field_is_vector = IsVectorDataType(field_meta.get_data_type());
        auto [field_has_warmup, field_warmup_policy] = schema_->WarmupPolicy(
            field_id, field_is_vector, /*is_index=*/false);
        if (field_has_warmup) {
            has_warmup_setting = true;
            if (field_warmup_policy == "sync") {
                aggregated_warmup_policy = "sync";
            } else if (field_warmup_policy == "async" &&
                       aggregated_warmup_policy != "sync") {
                aggregated_warmup_policy = "async";
            }
        }
    }
    timing.schema_policy_ns = DurationNs(stage_start);
    timing.is_vector = is_vector;

    auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();
    bool global_use_mmap = is_vector ? mmap_config.GetVectorFieldEnableMmap()
                                     : mmap_config.GetScalarFieldEnableMmap();
    auto use_mmap = has_mmap_setting ? mmap_enabled : global_use_mmap;
    timing.use_mmap = use_mmap;

    // The set of columns this entry projects is exactly the field_ids the
    // diff handed us. For lazy entries, SegmentLoadInfo::ComputeDiffColumnGroups
    // emits one entry per field, so each lazy entry produces a single-column
    // projected ChunkReader — touching one lazy field will not co-load chunks
    // for sibling lazy fields in the same column group.
    auto needed_columns = std::make_shared<std::vector<std::string>>();
    needed_columns->reserve(milvus_field_ids.size());
    for (const auto& fid : milvus_field_ids) {
        needed_columns->push_back(schema_->get_storage_column_name(fid));
    }
    stage_start = std::chrono::steady_clock::now();
    auto chunk_reader_result = reader_->get_chunk_reader(index, needed_columns);
    AssertInfo(chunk_reader_result.ok(),
               "get chunk reader failed, segment {}, column group index {}, "
               "status msg: {}",
               get_segment_id(),
               index,
               chunk_reader_result.status().ToString());

    auto chunk_reader = std::move(chunk_reader_result).ValueOrDie();
    timing.get_chunk_reader_ns = DurationNs(stage_start);

    LOG_INFO("[StorageV2] segment {} loads manifest cg index {}",
             this->get_segment_id(),
             index);
    auto mmap_dir_path =
        milvus::storage::LocalChunkManagerSingleton::GetInstance()
            .GetChunkManager()
            ->GetRootPath();

    // Determine warmup policy: use per-field settings if any,
    // otherwise pass empty string to fall back to global config
    std::string warmup_policy =
        has_warmup_setting ? aggregated_warmup_policy : "";

    // Multiple lazy entries can share the same column-group index (one per
    // field), so the translator cache key must be disambiguated by the
    // field-id of this entry. Eager entries are still one-per-cg, so they
    // keep the unsuffixed key.
    std::string cache_key_suffix;
    if (!eager_load) {
        cache_key_suffix = std::to_string(milvus_field_ids.front().get());
    }

    stage_start = std::chrono::steady_clock::now();
    auto translator =
        std::make_unique<storagev2translator::ManifestGroupTranslator>(
            get_segment_id(),
            GroupChunkType::DEFAULT,
            index,
            std::move(chunk_reader),
            field_metas,
            use_mmap,
            mmap_config.GetMmapPopulate(),
            mmap_dir_path,
            milvus_field_ids.size(),
            load_info->GetPriority(),
            eager_load,
            warmup_policy,
            cache_key_suffix,
            load_info->GetEstimatedBytesPerRow());
    timing.create_translator_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    auto chunked_column_group =
        std::make_shared<ChunkedColumnGroup>(std::move(translator));
    timing.create_chunk_group_ns = DurationNs(stage_start);

    // Create ProxyChunkColumn for each field
    stage_start = std::chrono::steady_clock::now();
    for (const auto& field_id : milvus_field_ids) {
        const auto& field_meta = field_metas.at(field_id);
        auto column = std::make_shared<ProxyChunkColumn>(
            chunked_column_group, field_id, field_meta);
        auto data_type = field_meta.get_data_type();
        load_field_data_common(
            field_id,
            column,
            load_info->GetNumOfRows(),
            data_type,
            use_mmap,
            true,
            std::
                nullopt,  // manifest cannot provide parquet skip index directly
            op_ctx,
            is_replace);
        if (field_id == TimestampFieldID) {
            int64_t num_rows = load_info->GetNumOfRows();
            if (commit_ts_ != 0) {
                std::vector<Timestamp> ts(num_rows, commit_ts_);
                init_storage_v1_timestamp_index(std::move(ts), num_rows);
            } else {
                init_storage_v2_timestamp_index(column, num_rows);
            }
        }
    }
    timing.proxy_column_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    column_group_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::ReloadColumns(const std::vector<FieldId>& field_ids,
                                        milvus::OpContext* op_ctx) {
    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> reload_futures;
    for (auto& field_id : field_ids) {
        auto future = pool.Submit([this, field_id, op_ctx]() {
            auto column = get_column(field_id);
            AssertInfo(column != nullptr,
                       "cannot reload non-existing field column {}",
                       field_id.get());
            auto num_chunks = column->num_chunks();
            std::vector<int64_t> chunk_ids(num_chunks);
            for (int64_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
                chunk_ids[chunk_id] = chunk_id;
            }
            column->PrefetchChunks(op_ctx, chunk_ids);
        });
        reload_futures.push_back(std::move(future));
    }

    storage::WaitAllFutures(reload_futures);
}

void
ChunkedSegmentSealedImpl::LoadBatchTextIndexes(
    milvus::OpContext* op_ctx,
    std::unordered_map<FieldId,
                       std::shared_ptr<proto::indexcgo::LoadTextIndexInfo>>&
        text_indexes_to_load) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    BatchTaskTiming timing;
    timing.task_count = text_indexes_to_load.size();
    timing.field_count = text_indexes_to_load.size();

    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> load_index_futures;

    load_index_futures.reserve(text_indexes_to_load.size());
    for (auto& [field_id, load_text_index_info] : text_indexes_to_load) {
        AssertInfo(field_exists_in_schema(schema_, field_id),
                   "field {} not found in schema when loading text index",
                   field_id.get());
        auto future = pool.Submit(
            [this, op_ctx, info = std::move(load_text_index_info)]() mutable
            -> void { LoadTextIndex(op_ctx, std::move(info)); });
        load_index_futures.emplace_back(std::move(future));
    }
    timing.submit_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    storage::WaitAllFutures(load_index_futures);
    timing.wait_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    batch_text_index_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadBatchIndexes(
    milvus::tracer::TraceContext& trace_ctx,
    std::unordered_map<FieldId, std::vector<LoadIndexInfo>>&
        field_id_to_index_info,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    BatchTaskTiming timing;
    timing.is_replace = is_replace;
    for (const auto& pair : field_id_to_index_info) {
        timing.field_count++;
        timing.task_count += pair.second.size();
    }

    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> load_index_futures;
    load_index_futures.reserve(timing.task_count);

    for (auto& pair : field_id_to_index_info) {
        auto field_id = pair.first;
        AssertInfo(field_exists_in_schema(schema_, field_id),
                   "field {} not found in schema when loading index",
                   field_id.get());
        auto& index_infos = pair.second;
        for (auto& load_index_info : index_infos) {
            auto* load_index_info_ptr = &load_index_info;
            auto future = pool.Submit([this,
                                       trace_ctx,
                                       field_id,
                                       load_index_info_ptr,
                                       op_ctx,
                                       is_replace]() mutable -> void {
                // Early exit if cancelled while queued
                CheckCancellation(op_ctx, id_, field_id.get(), "LoadIndex");

                LOG_INFO("Loading index for segment {} field {} with {} files",
                         id_,
                         field_id.get(),
                         load_index_info_ptr->index_files.size());

                // Download & compose index
                LoadIndexData(trace_ctx, load_index_info_ptr, op_ctx);

                // Load index into segment
                LoadIndex(*load_index_info_ptr, is_replace);
            });

            load_index_futures.push_back(std::move(future));
        }
    }
    timing.submit_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    storage::WaitAllFutures(load_index_futures);
    timing.wait_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    batch_index_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::LoadBatchFieldData(
    milvus::tracer::TraceContext& trace_ctx,
    std::vector<std::pair<std::vector<FieldId>, proto::segcore::FieldBinlog>>&
        field_binlog_to_load,
    milvus::OpContext* op_ctx,
    bool is_replace) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    BatchFieldDataTiming timing;
    timing.group_count = field_binlog_to_load.size();
    LOG_INFO("Loading field binlog for {} fields in segment {}",
             field_binlog_to_load.size(),
             id_);

    // When the flag is on, the loader must keep the column resident alongside
    // the index so bulk_subscript can serve retrieve from field data.
    auto prefer_field_data =
        SegcoreConfig::default_config()
            .get_prefer_field_data_when_index_has_raw_data();

    auto load_info_snapshot = std::atomic_load(&segment_load_info_);
    std::map<FieldId, LoadFieldDataInfo> field_data_to_load;
    for (auto& [field_ids, field_binlog] : field_binlog_to_load) {
        LoadFieldDataInfo load_field_data_info;
        load_field_data_info.storage_version =
            load_info_snapshot->GetStorageVersion();
        auto fields_to_load = field_ids;
        AssertInfo(!fields_to_load.empty(),
                   "load field data with empty field list");
        timing.child_field_count += fields_to_load.size();
        for (const auto& field_id : fields_to_load) {
            AssertInfo(field_exists_in_schema(schema_, field_id),
                       "field {} not found in schema when loading field data",
                       field_id.get());
        }

        bool index_has_raw_data = true;
        bool has_mmap_setting = false;
        bool mmap_enabled = false;
        bool is_vector = false;

        bool has_warmup_setting = false;
        std::string aggregated_warmup_policy = "disable";
        for (const auto& child_field_id : fields_to_load) {
            auto& field_meta = schema_->operator[](child_field_id);
            if (IsVectorDataType(field_meta.get_data_type())) {
                is_vector = true;
            }

            // if field has mmap setting, use it
            // - mmap setting at collection level, then all field are the same
            // - mmap setting at field level, we define that as long as one field shall be mmap, then whole group shall be mmaped
            auto [field_has_setting, field_mmap_enabled] =
                schema_->MmapEnabled(child_field_id);
            has_mmap_setting = has_mmap_setting || field_has_setting;
            mmap_enabled = mmap_enabled || field_mmap_enabled;

            auto iter = index_has_raw_data_.find(child_field_id);
            if (iter != index_has_raw_data_.end()) {
                index_has_raw_data = index_has_raw_data && iter->second;
            } else {
                index_has_raw_data = false;
            }

            auto [field_has_warmup, field_warmup_policy] =
                schema_->WarmupPolicy(
                    child_field_id,
                    IsVectorDataType(field_meta.get_data_type()),
                    /*is_index=*/false);
            if (field_has_warmup) {
                has_warmup_setting = true;
                if (field_warmup_policy == "sync") {
                    aggregated_warmup_policy = "sync";
                } else if (field_warmup_policy == "async" &&
                           aggregated_warmup_policy != "sync") {
                    aggregated_warmup_policy = "async";
                }
            }
        }

        auto group_id = field_binlog.fieldid();
        // Normally we skip loading field data when the index already carries
        // raw data, but prefer_field_data_when_index_has_raw_data opts into
        // keeping both resident so retrieve can read the column directly.
        if (index_has_raw_data && !prefer_field_data) {
            LOG_INFO(
                "Skip loading fielddata for segment {} group {} because "
                "index "
                "has raw data",
                id_,
                group_id);
            timing.skipped_group_count++;
            continue;
        }

        // Build FieldBinlogInfo
        FieldBinlogInfo field_binlog_info;
        field_binlog_info.field_id = group_id;
        field_binlog_info.child_field_ids.reserve(fields_to_load.size());
        for (const auto& field_id : fields_to_load) {
            field_binlog_info.child_field_ids.push_back(field_id.get());
        }

        // Calculate total row count and collect binlog paths
        int64_t total_entries = 0;
        auto binlog_count = field_binlog.binlogs().size();
        field_binlog_info.insert_files.reserve(binlog_count);
        field_binlog_info.entries_nums.reserve(binlog_count);
        field_binlog_info.memory_sizes.reserve(binlog_count);
        for (const auto& binlog : field_binlog.binlogs()) {
            field_binlog_info.insert_files.push_back(binlog.log_path());
            field_binlog_info.entries_nums.push_back(binlog.entries_num());
            field_binlog_info.memory_sizes.push_back(binlog.memory_size());
            total_entries += binlog.entries_num();
        }
        field_binlog_info.row_count = total_entries;

        auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();
        auto global_use_mmap = is_vector
                                   ? mmap_config.GetVectorFieldEnableMmap()
                                   : mmap_config.GetScalarFieldEnableMmap();
        field_binlog_info.enable_mmap =
            has_mmap_setting ? mmap_enabled : global_use_mmap;

        // Determine group warmup policy: use per-field settings if any,
        // otherwise fall back to global warmup policy
        field_binlog_info.warmup_policy =
            has_warmup_setting ? aggregated_warmup_policy : "";

        // Store in map
        load_field_data_info.field_infos[group_id] = field_binlog_info;

        field_data_to_load[FieldId(group_id)] = load_field_data_info;
    }

    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    std::vector<std::future<void>> load_field_futures;
    load_field_futures.reserve(field_data_to_load.size());

    for (const auto& [field_id, load_field_data_info] : field_data_to_load) {
        // Create local copies to capture in lambda (C++17 compatible)
        const auto field_data = load_field_data_info;
        const auto captured_field_id = field_id;
        auto future = pool.Submit(
            [this, field_data, captured_field_id, op_ctx, is_replace]()
                -> void {
                // Early exit if cancelled while queued
                CheckCancellation(op_ctx,
                                  id_,
                                  captured_field_id.get(),
                                  "ChunkedSegmentSealedImpl::LoadFieldData()");
                LoadFieldData(field_data, op_ctx, is_replace);
            });

        load_field_futures.push_back(std::move(future));
    }

    timing.task_count = field_data_to_load.size();
    timing.prepare_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    storage::WaitAllFutures(load_field_futures);
    timing.wait_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    batch_field_data_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::Load(milvus::tracer::TraceContext& trace_ctx,
                               milvus::OpContext* op_ctx) {
    auto total_start = std::chrono::steady_clock::now();
    SegcoreLoadTiming timing;

    // Serialize with Reopen(pb)/SetLoadInfo. Runtime-only updates produced by
    // ApplyLoadDiff are committed through COW helpers after the data is loaded.
    auto lock_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> reopen_guard(reopen_mutex_);
    timing.reopen_lock_wait_ns = DurationNs(lock_start);

    auto snapshot = std::atomic_load(&segment_load_info_);
    auto num_rows = snapshot->GetNumOfRows();
    LOG_INFO("Loading segment {} with {} rows", id_, num_rows);

    // reopen_mutex_ synchronizes this read with all schema_ writers.
    auto stage_start = std::chrono::steady_clock::now();
    SegmentLoadInfo mutable_copy(*snapshot);
    mutable_copy.SetFieldsFilledWithDefault(
        snapshot->GetFieldsFilledWithDefault());
    for (auto fid : snapshot->GetCreatedTextIndexes()) {
        mutable_copy.SetTextIndexCreated(fid);
    }
    timing.prepare_load_info_ns = DurationNs(stage_start);
    stage_start = std::chrono::steady_clock::now();
    auto diff = mutable_copy.GetLoadDiff();
    timing.get_diff_ns = DurationNs(stage_start);
    LOG_WARN("Load segment {} with diff {}", id_, diff.ToString());

    stage_start = std::chrono::steady_clock::now();
    ApplyLoadDiff(op_ctx, mutable_copy, diff);
    timing.apply_diff_ns = DurationNs(stage_start);

    LOG_INFO("Successfully loaded segment {} with {} rows", id_, num_rows);
    timing.total_ns = DurationNs(total_start);
    sealed_load_timing_stats.Record(timing);
}

void
ChunkedSegmentSealedImpl::FillTargetEntry(const query::Plan* plan,
                                          SearchResult& results,
                                          milvus::OpContext* op_ctx) const {
    std::shared_lock lck(mutex_);
    AssertInfo(plan, "empty plan");
    auto size = results.distances_.size();
    AssertInfo(results.seg_offsets_.size() == size,
               "Size of result distances is not equal to size of ids");

    segcore::CheckCancellation(op_ctx, get_segment_id(), "FillTargetEntry");

    // Try take() for eligible output fields. Fields not filled by take still
    // go through bulk_subscript below.
    bool used_take = TryTakeForSearch(
        plan, results.seg_offsets_.data(), size, results, op_ctx);

    std::unique_ptr<DataArray> field_data;
    // Per-call OpContext keeps storage_usage scoped to this segment;
    // sharing op_ctx across segments would double-count bytes. See
    // SegmentInternalInterface::FillPrimaryKeys for the same pattern.
    milvus::OpContext local_ctx;
    if (op_ctx != nullptr) {
        local_ctx.cancellation_token = op_ctx->cancellation_token;
        local_ctx.runtime_load_priority = op_ctx->runtime_load_priority;
    }
    for (auto field_id : plan->target_entries_) {
        // Skip fields already filled by take
        if (used_take && results.output_fields_data_.count(field_id) > 0) {
            continue;
        }
        segcore::CheckCancellation(
            op_ctx, get_segment_id(), field_id.get(), "FillTargetEntry");
        auto& field_meta = plan->schema_->operator[](field_id);
        if (plan->schema_->get_dynamic_field_id().has_value() &&
            plan->schema_->get_dynamic_field_id().value() == field_id &&
            !plan->target_dynamic_fields_.empty()) {
            auto& target_dynamic_fields = plan->target_dynamic_fields_;
            field_data = bulk_subscript(&local_ctx,
                                        field_id,
                                        results.seg_offsets_.data(),
                                        size,
                                        target_dynamic_fields);
        } else if (!is_field_exist(field_id)) {
            field_data = bulk_subscript_not_exist_field(field_meta, size);
        } else {
            field_data = bulk_subscript(
                &local_ctx, field_id, results.seg_offsets_.data(), size);
        }
        results.output_fields_data_[field_id] = std::move(field_data);
    }
    results.search_storage_cost_.scanned_remote_bytes +=
        local_ctx.storage_usage.scanned_cold_bytes.load();
    results.search_storage_cost_.scanned_total_bytes +=
        local_ctx.storage_usage.scanned_total_bytes.load();
}

// ---- Shared helpers for TryTakeForRetrieve / TryTakeForSearch ----

static inline void
LogTakeFallback(const char* caller_tag,
                int64_t segment_id,
                int64_t rows,
                size_t unique_rows,
                size_t field_count,
                std::string_view reason) {
    LOG_INFO(
        "[TakeAPI] {} fallback to bulk_subscript for segment {}: "
        "reason={}, rows={}, unique_rows={}, fields={}",
        caller_tag,
        segment_id,
        reason,
        rows,
        unique_rows,
        field_count);
}

static bool
ShouldProjectInternalTakeDynamicField(
    const Schema& schema,
    FieldId field_id,
    const std::vector<std::string>& target_dynamic_fields) {
    if (schema.is_external_collection() || target_dynamic_fields.empty()) {
        return false;
    }
    auto dynamic_field_id = schema.get_dynamic_field_id();
    return dynamic_field_id.has_value() && dynamic_field_id.value() == field_id;
}

ChunkedSegmentSealedImpl::TakeContext
ChunkedSegmentSealedImpl::BuildTakeContext(const int64_t* offsets,
                                           int64_t size) {
    struct OffsetEntry {
        int64_t offset;
        int64_t orig_pos;
    };
    std::vector<OffsetEntry> entries;
    entries.reserve(size);
    for (int64_t i = 0; i < size; i++) {
        entries.push_back({offsets[i], i});
    }
    std::sort(entries.begin(),
              entries.end(),
              [](const OffsetEntry& a, const OffsetEntry& b) {
                  return a.offset < b.offset;
              });

    TakeContext ctx;
    ctx.unique_offsets.reserve(size);
    ctx.result_mapping.resize(size);
    for (auto& e : entries) {
        if (ctx.unique_offsets.empty() ||
            ctx.unique_offsets.back() != e.offset) {
            ctx.unique_offsets.push_back(e.offset);
        }
        ctx.result_mapping[e.orig_pos] =
            static_cast<int64_t>(ctx.unique_offsets.size() - 1);
    }
    return ctx;
}

std::unique_ptr<DataArray>
ChunkedSegmentSealedImpl::ArrowToDataArray(
    const std::shared_ptr<arrow::Array>& arr,
    const FieldMeta& field_meta,
    const std::vector<int64_t>& result_mapping,
    int64_t size,
    const std::vector<std::string>* dynamic_field_names,
    const std::string* text_lob_path) {
    auto data_array = std::make_unique<DataArray>();
    data_array->set_type(
        static_cast<proto::schema::DataType>(field_meta.get_data_type()));

    switch (field_meta.get_data_type()) {
        case DataType::BOOL: {
            auto typed = std::static_pointer_cast<arrow::BooleanArray>(arr);
            auto obj = data_array->mutable_scalars()->mutable_bool_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::INT8: {
            auto typed = std::static_pointer_cast<arrow::Int8Array>(arr);
            auto obj = data_array->mutable_scalars()->mutable_int_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(
                    static_cast<int32_t>(typed->Value(result_mapping[i])));
            }
            break;
        }
        case DataType::INT16: {
            auto typed = std::static_pointer_cast<arrow::Int16Array>(arr);
            auto obj = data_array->mutable_scalars()->mutable_int_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(
                    static_cast<int32_t>(typed->Value(result_mapping[i])));
            }
            break;
        }
        case DataType::INT32: {
            auto typed = std::static_pointer_cast<arrow::Int32Array>(arr);
            auto obj = data_array->mutable_scalars()->mutable_int_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::INT64: {
            auto typed = std::static_pointer_cast<arrow::Int64Array>(arr);
            auto obj = data_array->mutable_scalars()->mutable_long_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::FLOAT: {
            auto typed = std::static_pointer_cast<arrow::FloatArray>(arr);
            auto obj = data_array->mutable_scalars()->mutable_float_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::DOUBLE: {
            auto typed = std::static_pointer_cast<arrow::DoubleArray>(arr);
            auto obj = data_array->mutable_scalars()->mutable_double_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::TEXT: {
            auto obj = data_array->mutable_scalars()->mutable_string_data();
            if (text_lob_path != nullptr) {
                std::vector<milvus_storage::lob_column::EncodedRef>
                    encoded_refs;
                encoded_refs.reserve(size);
                std::vector<std::string> string_refs;
                string_refs.reserve(size);

                if (arr->type()->id() == arrow::Type::STRING) {
                    auto typed =
                        std::static_pointer_cast<arrow::StringArray>(arr);
                    for (int64_t i = 0; i < size; i++) {
                        auto idx = result_mapping[i];
                        if (typed->IsNull(idx)) {
                            encoded_refs.push_back(
                                MakeTextLobEncodedRef(nullptr, 0));
                            continue;
                        }
                        string_refs.emplace_back(typed->GetString(idx));
                        auto& ref = string_refs.back();
                        encoded_refs.push_back(
                            MakeTextLobEncodedRef(ref.data(), ref.size()));
                    }
                } else if (arr->type()->id() == arrow::Type::BINARY) {
                    auto typed =
                        std::static_pointer_cast<arrow::BinaryArray>(arr);
                    for (int64_t i = 0; i < size; i++) {
                        auto idx = result_mapping[i];
                        if (typed->IsNull(idx)) {
                            encoded_refs.push_back(
                                MakeTextLobEncodedRef(nullptr, 0));
                            continue;
                        }
                        auto val = typed->Value(idx);
                        encoded_refs.push_back(MakeTextLobEncodedRef(
                            val.data(), static_cast<size_t>(val.size())));
                    }
                } else {
                    return nullptr;
                }

                auto texts = ReadTextLobBatch(*text_lob_path, encoded_refs);
                for (auto& text : texts) {
                    obj->add_data(std::move(text));
                }
                break;
            }

            auto typed = std::static_pointer_cast<arrow::StringArray>(arr);
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->GetString(result_mapping[i]));
            }
            break;
        }
        case DataType::VARCHAR:
        case DataType::STRING: {
            auto typed = std::static_pointer_cast<arrow::StringArray>(arr);
            auto obj = data_array->mutable_scalars()->mutable_string_data();
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->GetString(result_mapping[i]));
            }
            break;
        }
        case DataType::JSON: {
            // NormalizeExternalArrow already converted String→Binary.
            auto obj = data_array->mutable_scalars()->mutable_json_data();
            auto typed = std::static_pointer_cast<arrow::BinaryArray>(arr);
            for (int64_t i = 0; i < size; i++) {
                auto val = typed->Value(result_mapping[i]);
                if (dynamic_field_names != nullptr &&
                    !dynamic_field_names->empty()) {
                    auto projected = ExtractSubJson(
                        std::string_view(
                            reinterpret_cast<const char*>(val.data()),
                            val.size()),
                        *dynamic_field_names);
                    obj->add_data(std::move(projected));
                } else {
                    obj->add_data(val.data(), val.size());
                }
            }
            break;
        }
        case DataType::GEOMETRY: {
            // NormalizeExternalArrow already converted WKT→WKB if needed.
            auto obj = data_array->mutable_scalars()->mutable_geometry_data();
            auto typed = std::static_pointer_cast<arrow::BinaryArray>(arr);
            for (int64_t i = 0; i < size; i++) {
                auto val = typed->Value(result_mapping[i]);
                obj->add_data(val.data(), val.size());
            }
            break;
        }
        case DataType::TIMESTAMPTZ: {
            // NormalizeExternalArrow already converted Timestamp→Int64.
            auto obj =
                data_array->mutable_scalars()->mutable_timestamptz_data();
            auto typed = std::static_pointer_cast<arrow::Int64Array>(arr);
            for (int64_t i = 0; i < size; i++) {
                obj->add_data(typed->Value(result_mapping[i]));
            }
            break;
        }
        case DataType::ARRAY: {
            // NormalizeExternalArrow already converted List→Binary(protobuf).
            auto obj = data_array->mutable_scalars()->mutable_array_data();
            // Same element_type carry-through as the chunked sealed path
            // above; without it the SDK rejects the response. Fix for #48619.
            obj->set_element_type(static_cast<milvus::proto::schema::DataType>(
                field_meta.get_element_type()));
            auto typed = std::static_pointer_cast<arrow::BinaryArray>(arr);
            for (int64_t i = 0; i < size; i++) {
                auto val = typed->Value(result_mapping[i]);
                auto* sf = obj->add_data();
                sf->ParseFromArray(val.data(), static_cast<int>(val.size()));
            }
            break;
        }
        case DataType::VECTOR_FLOAT: {
            int dim = field_meta.get_dim();
            auto vectors = data_array->mutable_vectors();
            vectors->set_dim(dim);
            auto float_data = vectors->mutable_float_vector();
            int64_t valid_count = size;
            if (field_meta.is_nullable()) {
                valid_count = 0;
                for (int64_t i = 0; i < size; i++) {
                    if (arr->IsValid(result_mapping[i])) {
                        valid_count++;
                    }
                }
            }
            float_data->mutable_data()->Resize(valid_count * dim, 0.0f);
            int64_t data_pos = 0;
            for (int64_t i = 0; i < size; i++) {
                auto idx = result_mapping[i];
                if (arr->IsNull(idx)) {
                    continue;
                }
                const uint8_t* val = nullptr;
                if (arr->type_id() == arrow::Type::FIXED_SIZE_BINARY) {
                    val = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(
                              arr)
                              ->Value(idx);
                } else {
                    auto bin_val =
                        std::static_pointer_cast<arrow::BinaryArray>(arr)
                            ->Value(idx);
                    val = reinterpret_cast<const uint8_t*>(bin_val.data());
                }
                auto floats = reinterpret_cast<const float*>(val);
                std::copy(floats,
                          floats + dim,
                          float_data->mutable_data()->mutable_data() +
                              data_pos * dim);
                data_pos++;
            }
            break;
        }
        case DataType::VECTOR_BINARY:
        case DataType::VECTOR_FLOAT16:
        case DataType::VECTOR_BFLOAT16:
        case DataType::VECTOR_INT8: {
            int dim = field_meta.get_dim();
            auto byte_width = field_meta.get_sizeof();
            auto vectors = data_array->mutable_vectors();
            vectors->set_dim(dim);
            std::string* vector_data = nullptr;
            switch (field_meta.get_data_type()) {
                case DataType::VECTOR_BINARY:
                    vector_data = vectors->mutable_binary_vector();
                    break;
                case DataType::VECTOR_FLOAT16:
                    vector_data = vectors->mutable_float16_vector();
                    break;
                case DataType::VECTOR_BFLOAT16:
                    vector_data = vectors->mutable_bfloat16_vector();
                    break;
                case DataType::VECTOR_INT8:
                    vector_data = vectors->mutable_int8_vector();
                    break;
                default:
                    break;
            }
            int64_t valid_count = size;
            if (field_meta.is_nullable()) {
                valid_count = 0;
                for (int64_t i = 0; i < size; i++) {
                    if (arr->IsValid(result_mapping[i])) {
                        valid_count++;
                    }
                }
            }
            vector_data->resize(valid_count * byte_width);
            int64_t data_pos = 0;
            for (int64_t i = 0; i < size; i++) {
                auto idx = result_mapping[i];
                if (arr->IsNull(idx)) {
                    continue;
                }
                const uint8_t* val = nullptr;
                if (arr->type_id() == arrow::Type::FIXED_SIZE_BINARY) {
                    val = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(
                              arr)
                              ->Value(idx);
                } else {
                    auto bin_val =
                        std::static_pointer_cast<arrow::BinaryArray>(arr)
                            ->Value(idx);
                    AssertInfo(
                        static_cast<size_t>(bin_val.size()) == byte_width,
                        "vector byte width mismatch, expected {}, actual {}",
                        byte_width,
                        bin_val.size());
                    val = reinterpret_cast<const uint8_t*>(bin_val.data());
                }
                std::memcpy(vector_data->data() + data_pos * byte_width,
                            val,
                            byte_width);
                data_pos++;
            }
            break;
        }
        case DataType::VECTOR_SPARSE_U32_F32: {
            auto vectors = data_array->mutable_vectors();
            auto sparse_data = vectors->mutable_sparse_float_vector();
            auto typed = std::static_pointer_cast<arrow::BinaryArray>(arr);
            int64_t max_dim = 0;
            for (int64_t i = 0; i < size; i++) {
                auto idx = result_mapping[i];
                if (arr->IsNull(idx)) {
                    continue;
                }
                auto val = typed->Value(idx);
                sparse_data->add_contents(val.data(), val.size());
                auto row = CopyAndWrapSparseRow(val.data(), val.size(), true);
                max_dim = std::max(max_dim, row.dim());
            }
            sparse_data->set_dim(max_dim);
            vectors->set_dim(sparse_data->dim());
            break;
        }
        case DataType::VECTOR_ARRAY: {
            // After normalize, arr is List<FixedSizeBinaryArray>.
            auto outer_list = std::static_pointer_cast<arrow::ListArray>(arr);
            auto inner_values =
                std::static_pointer_cast<arrow::FixedSizeBinaryArray>(
                    outer_list->values());
            int dim = field_meta.get_dim();
            auto element_type = field_meta.get_element_type();
            auto* va = data_array->mutable_vectors()
                           ->mutable_vector_array()
                           ->mutable_data();
            data_array->mutable_vectors()->set_dim(dim);
            for (int64_t i = 0; i < size; i++) {
                auto idx = result_mapping[i];
                int64_t start = outer_list->value_offset(idx);
                int64_t end = outer_list->value_offset(idx + 1);
                int64_t num_vectors = end - start;
                VectorArray vec_arr(inner_values->GetValue(start),
                                    num_vectors,
                                    dim,
                                    element_type);
                auto* vf = va->Add();
                *vf = vec_arr.output_data();
            }
            break;
        }
        default:
            return nullptr;  // unsupported type
    }

    // Populate valid_data for nullable fields so clients can identify nulls.
    if (field_meta.is_nullable()) {
        auto* vd = data_array->mutable_valid_data();
        vd->Reserve(size);
        for (int64_t i = 0; i < size; i++) {
            vd->Add(arr->IsValid(result_mapping[i]));
        }
    }

    return data_array;
}

std::shared_ptr<arrow::Table>
ChunkedSegmentSealedImpl::ExecuteTake(
    const std::vector<int64_t>& unique_offsets,
    const std::shared_ptr<std::vector<std::string>>& needed_columns,
    const char* caller_tag,
    double& elapsed_ms,
    milvus::OpContext* op_ctx) const {
    // reader_->take() issues remote reads and can take seconds under slow
    // object storage. Bail out if the upstream reduce has already been
    // cancelled so we don't waste IO on a doomed request. caller_tag is a
    // short static string ("search" / "retrieve"); pass it directly to
    // avoid an extra fmt::format allocation on every call.
    segcore::CheckCancellation(op_ctx, id_, caller_tag);

    // reader_->take() is NOT thread-safe — concurrent retrieve and search
    // workers may hit the same segment simultaneously under load. Also,
    // Reopen/ApplyLoadDiff can reassign reader_ under reader_mutex_ while a
    // take() is in flight. Serialize both the null check and the take() call
    // through reader_mutex_ so we never observe a half-assigned shared_ptr
    // and the old Reader cannot be destroyed mid-call.
    std::lock_guard<std::mutex> lock(reader_mutex_);
    if (!reader_) {
        LOG_WARN("[TakeAPI] {} reader is null for segment {}", caller_tag, id_);
        return nullptr;
    }
    auto take_start = std::chrono::high_resolution_clock::now();
    auto result = reader_->take(unique_offsets, 1, needed_columns);
    elapsed_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - take_start)
                     .count();
    if (!result.ok()) {
        LOG_WARN("[TakeAPI] {} take() failed for segment {}: {}",
                 caller_tag,
                 id_,
                 result.status().ToString());
        return nullptr;
    }
    return *result;
}

// ---- End shared helpers ----

bool
ChunkedSegmentSealedImpl::TryTakeForRetrieve(
    const query::RetrievePlan* plan,
    const std::unique_ptr<proto::segcore::RetrieveResults>& results,
    const int64_t* offsets,
    int64_t size,
    bool ignore_non_pk,
    bool fill_ids,
    milvus::OpContext* op_ctx) const {
    if (size == 0 || !use_take_for_output_.load(std::memory_order_relaxed)) {
        return false;
    }
    const bool is_external_collection = schema_->is_external_collection();

    auto pk_field_id = plan->schema_->get_primary_field_id();
    auto is_pk_field = [&](const FieldId& fid) {
        return pk_field_id.has_value() && pk_field_id.value() == fid;
    };

    // Collect needed columns and their field IDs. External collections use
    // user-provided external column names; internal storage v2 uses field-id
    // strings in the Loon Arrow schema.
    auto needed_columns = std::make_shared<std::vector<std::string>>();
    std::vector<FieldId> take_field_ids;
    std::vector<std::string> take_column_names;
    for (auto field_id : plan->field_ids_) {
        if (SystemProperty::Instance().IsSystem(field_id)) {
            continue;
        }
        if (ignore_non_pk && !is_pk_field(field_id)) {
            continue;
        }
        auto& field_meta = schema_->operator[](field_id);
        if (!field_meta.is_external_field() &&
            !schema_->is_function_output(field_id) && is_external_collection) {
            continue;
        }
        auto column_name = schema_->get_storage_column_name(field_id);
        needed_columns->push_back(column_name);
        take_field_ids.push_back(field_id);
        take_column_names.push_back(std::move(column_name));
    }
    if (take_field_ids.empty()) {
        return false;
    }

    auto ctx = BuildTakeContext(offsets, size);

    double take_elapsed_ms = 0;
    auto table = ExecuteTake(ctx.unique_offsets,
                             needed_columns,
                             "retrieve",
                             take_elapsed_ms,
                             op_ctx);
    if (!table) {
        LogTakeFallback("retrieve",
                        id_,
                        size,
                        ctx.unique_offsets.size(),
                        take_field_ids.size(),
                        "take returned no table");
        return false;
    }

    // Cancellation can become observable between reader_->take() returning
    // and the Arrow Concatenate/NormalizeExternalArrow pass below, which
    // can itself be expensive on wide result sets.
    segcore::CheckCancellation(
        op_ctx, id_, "TryTakeForRetrieve(pre-arrow-convert)");

    // Convert Arrow Table columns to DataArray results
    auto fields_data = results->mutable_fields_data();
    auto ids = results->mutable_ids();

    // Build lookup from field_id to index in take_field_ids.
    std::unordered_map<int64_t, size_t> ext_field_idx;
    for (size_t fi = 0; fi < take_field_ids.size(); fi++) {
        ext_field_idx[take_field_ids[fi].get()] = fi;
    }

    // Pre-combine Arrow chunks for each external / function-output column
    std::vector<std::shared_ptr<arrow::Array>> combined_arrays(
        take_field_ids.size());
    for (size_t fi = 0; fi < take_field_ids.size(); fi++) {
        auto column_name = take_column_names[fi];
        auto col = table->GetColumnByName(column_name);
        if (!col || col->num_chunks() == 0) {
            LOG_WARN(
                "[TakeAPI] column '{}' not found in take result for "
                "segment {}",
                column_name,
                id_);
            LogTakeFallback("retrieve",
                            id_,
                            size,
                            ctx.unique_offsets.size(),
                            take_field_ids.size(),
                            fmt::format("missing column '{}'", column_name));
            return false;
        }
        if (col->num_chunks() == 1) {
            combined_arrays[fi] = col->chunk(0);
        } else {
            auto combined_result = arrow::Concatenate(col->chunks());
            if (!combined_result.ok()) {
                LOG_WARN("[TakeAPI] concatenate failed: {}",
                         combined_result.status().ToString());
                LogTakeFallback(
                    "retrieve",
                    id_,
                    size,
                    ctx.unique_offsets.size(),
                    take_field_ids.size(),
                    fmt::format("concatenate failed: {}",
                                combined_result.status().ToString()));
                return false;
            }
            combined_arrays[fi] = *combined_result;
        }
    }

    // Emit fields in plan->field_ids_ order so the positional index
    // matches outputFieldsID in the Proxy's afterReduce.
    for (auto field_id : plan->field_ids_) {
        if (SystemProperty::Instance().IsSystem(field_id)) {
            auto system_type =
                SystemProperty::Instance().GetSystemFieldType(field_id);
            FixedVector<int64_t> output(size);
            milvus::OpContext op_ctx;
            bulk_subscript(&op_ctx, system_type, offsets, size, output.data());
            auto data_array = std::make_unique<DataArray>();
            data_array->set_field_id(field_id.get());
            data_array->set_type(milvus::proto::schema::DataType::Int64);
            auto obj = data_array->mutable_scalars()->mutable_long_data();
            auto data = reinterpret_cast<const int64_t*>(output.data());
            obj->mutable_data()->Add(data, data + size);
            fields_data->AddAllocated(data_array.release());
            continue;
        }

        if (ignore_non_pk && !is_pk_field(field_id)) {
            continue;
        }

        auto& field_meta = schema_->operator[](field_id);

        // External virtual PK field (not external, not function-output).
        if (!field_meta.is_external_field() &&
            !schema_->is_function_output(field_id) && is_external_collection) {
            if (is_pk_field(field_id) &&
                field_meta.get_data_type() == DataType::INT64) {
                auto data_array = std::make_unique<DataArray>();
                data_array->set_field_id(field_id.get());
                data_array->set_type(milvus::proto::schema::DataType::Int64);
                auto obj = data_array->mutable_scalars()->mutable_long_data();
                for (int64_t i = 0; i < size; i++) {
                    obj->add_data(GetVirtualPK(id_, offsets[i]));
                }
                if (!ignore_non_pk) {
                    fields_data->AddAllocated(data_array.release());
                }
                if (fill_ids) {
                    auto int_ids = ids->mutable_int_id();
                    for (int64_t i = 0; i < size; i++) {
                        int_ids->add_data(GetVirtualPK(id_, offsets[i]));
                    }
                }
            }
            continue;
        }

        // Convert from take() result
        auto it = ext_field_idx.find(field_id.get());
        if (it == ext_field_idx.end()) {
            continue;
        }
        size_t fi = it->second;
        auto arr = combined_arrays[fi];

        // Normalize external arrow types to Milvus internal format.
        if (is_external_collection) {
            arr = storage::NormalizeExternalArrow(arr, field_meta);
        }

        auto dynamic_field_names =
            ShouldProjectInternalTakeDynamicField(
                *schema_, field_id, plan->target_dynamic_fields_)
                ? &plan->target_dynamic_fields_
                : nullptr;
        const std::string* text_lob_path = nullptr;
        if (!is_external_collection &&
            field_meta.get_data_type() == DataType::TEXT) {
            auto path_it = text_lob_paths_.find(field_id);
            if (path_it == text_lob_paths_.end()) {
                LogTakeFallback(
                    "retrieve",
                    id_,
                    size,
                    ctx.unique_offsets.size(),
                    take_field_ids.size(),
                    fmt::format("missing TEXT LOB path for field {}",
                                field_id.get()));
                results->clear_fields_data();
                results->clear_ids();
                return false;
            }
            text_lob_path = &path_it->second;
        }
        auto data_array = ArrowToDataArray(arr,
                                           field_meta,
                                           ctx.result_mapping,
                                           size,
                                           dynamic_field_names,
                                           text_lob_path);
        if (!data_array) {
            LOG_WARN(
                "[TakeAPI] unsupported data type {} for field '{}', "
                "falling back",
                static_cast<int>(field_meta.get_data_type()),
                take_column_names[fi]);
            LogTakeFallback(
                "retrieve",
                id_,
                size,
                ctx.unique_offsets.size(),
                take_field_ids.size(),
                fmt::format("unsupported data type {} for column '{}'",
                            static_cast<int>(field_meta.get_data_type()),
                            take_column_names[fi]));
            results->clear_fields_data();
            results->clear_ids();
            return false;
        }
        data_array->set_field_id(field_id.get());

        if (fill_ids && is_pk_field(field_id)) {
            switch (field_meta.get_data_type()) {
                case DataType::INT64: {
                    auto int_ids = ids->mutable_int_id();
                    auto& src_data = data_array->scalars().long_data();
                    int_ids->mutable_data()->Add(src_data.data().begin(),
                                                 src_data.data().end());
                    break;
                }
                case DataType::VARCHAR: {
                    auto str_ids = ids->mutable_str_id();
                    auto& src_data = data_array->scalars().string_data();
                    for (auto i = 0; i < src_data.data_size(); ++i) {
                        *(str_ids->mutable_data()->Add()) = src_data.data(i);
                    }
                    break;
                }
                default: {
                    ThrowInfo(DataTypeInvalid,
                              fmt::format("unsupported datatype {}",
                                          field_meta.get_data_type()));
                }
            }
        }

        if (!ignore_non_pk) {
            fields_data->AddAllocated(data_array.release());
        }
    }

    LOG_DEBUG(
        "[TakeAPI] segment {} used take() for {} rows ({} unique), "
        "{} fields, elapsed={:.2f}ms",
        id_,
        size,
        ctx.unique_offsets.size(),
        take_field_ids.size(),
        take_elapsed_ms);
    return true;
}

bool
ChunkedSegmentSealedImpl::TryTakeForSearch(const query::Plan* plan,
                                           const int64_t* seg_offsets,
                                           int64_t size,
                                           SearchResult& results,
                                           milvus::OpContext* op_ctx) const {
    if (size == 0 || !use_take_for_output_.load(std::memory_order_relaxed)) {
        return false;
    }
    const bool is_external_collection = schema_->is_external_collection();

    // Collect needed columns. External collections use external field names;
    // internal storage v2 uses field-id strings.
    auto needed_columns = std::make_shared<std::vector<std::string>>();
    std::vector<FieldId> take_field_ids;
    std::vector<const FieldMeta*> take_field_metas;
    std::vector<std::string> take_column_names;
    for (auto field_id : plan->target_entries_) {
        auto& field_meta = schema_->operator[](field_id);
        if (!field_meta.is_external_field() &&
            !schema_->is_function_output(field_id) && is_external_collection) {
            continue;
        }
        auto column_name = schema_->get_storage_column_name(field_id);
        needed_columns->push_back(column_name);
        take_field_ids.push_back(field_id);
        take_field_metas.push_back(&field_meta);
        take_column_names.push_back(std::move(column_name));
    }
    if (take_field_ids.empty()) {
        return false;
    }

    auto ctx = BuildTakeContext(seg_offsets, size);

    double take_elapsed_ms = 0;
    auto table = ExecuteTake(
        ctx.unique_offsets, needed_columns, "search", take_elapsed_ms, op_ctx);
    if (!table) {
        LogTakeFallback("search",
                        id_,
                        size,
                        ctx.unique_offsets.size(),
                        take_field_ids.size(),
                        "take returned no table");
        return false;
    }

    // Cancellation can become observable between reader_->take() returning
    // and the Arrow Concatenate/NormalizeExternalArrow pass below, which
    // can itself be expensive on wide result sets.
    segcore::CheckCancellation(
        op_ctx, id_, "TryTakeForSearch(pre-arrow-convert)");

    // Convert Arrow Table columns to DataArray and store in SearchResult
    for (size_t fi = 0; fi < take_field_ids.size(); fi++) {
        auto field_id = take_field_ids[fi];
        auto& field_meta = *take_field_metas[fi];
        auto column_name = take_column_names[fi];
        auto col = table->GetColumnByName(column_name);
        if (!col || col->num_chunks() == 0) {
            LOG_WARN("[TakeAPI] search column '{}' not found for segment {}",
                     column_name,
                     id_);
            LogTakeFallback("search",
                            id_,
                            size,
                            ctx.unique_offsets.size(),
                            take_field_ids.size(),
                            fmt::format("missing column '{}'", column_name));
            return false;
        }

        std::shared_ptr<arrow::Array> arr;
        if (col->num_chunks() == 1) {
            arr = col->chunk(0);
        } else {
            auto combined_result = arrow::Concatenate(col->chunks());
            if (!combined_result.ok()) {
                LogTakeFallback(
                    "search",
                    id_,
                    size,
                    ctx.unique_offsets.size(),
                    take_field_ids.size(),
                    fmt::format("concatenate failed: {}",
                                combined_result.status().ToString()));
                return false;
            }
            arr = *combined_result;
        }

        // Normalize external arrow types to Milvus internal format.
        if (is_external_collection) {
            arr = storage::NormalizeExternalArrow(arr, field_meta);
        }

        auto dynamic_field_names =
            ShouldProjectInternalTakeDynamicField(
                *schema_, field_id, plan->target_dynamic_fields_)
                ? &plan->target_dynamic_fields_
                : nullptr;
        const std::string* text_lob_path = nullptr;
        if (!is_external_collection &&
            field_meta.get_data_type() == DataType::TEXT) {
            auto path_it = text_lob_paths_.find(field_id);
            if (path_it == text_lob_paths_.end()) {
                LogTakeFallback(
                    "search",
                    id_,
                    size,
                    ctx.unique_offsets.size(),
                    take_field_ids.size(),
                    fmt::format("missing TEXT LOB path for field {}",
                                field_id.get()));
                results.output_fields_data_.clear();
                return false;
            }
            text_lob_path = &path_it->second;
        }
        auto data_array = ArrowToDataArray(arr,
                                           field_meta,
                                           ctx.result_mapping,
                                           size,
                                           dynamic_field_names,
                                           text_lob_path);
        if (!data_array) {
            LOG_WARN(
                "[TakeAPI] search: unsupported type {} for '{}', "
                "falling back",
                static_cast<int>(field_meta.get_data_type()),
                column_name);
            LogTakeFallback(
                "search",
                id_,
                size,
                ctx.unique_offsets.size(),
                take_field_ids.size(),
                fmt::format("unsupported data type {} for column '{}'",
                            static_cast<int>(field_meta.get_data_type()),
                            column_name));
            results.output_fields_data_.clear();
            return false;
        }
        data_array->set_field_id(field_id.get());
        results.output_fields_data_[field_id] = std::move(data_array);
    }

    LOG_DEBUG(
        "[TakeAPI] search: segment {} used take() for {} rows ({} unique), "
        "{} fields, elapsed={:.2f}ms",
        id_,
        size,
        ctx.unique_offsets.size(),
        take_field_ids.size(),
        take_elapsed_ms);
    return true;
}

}  // namespace milvus::segcore
