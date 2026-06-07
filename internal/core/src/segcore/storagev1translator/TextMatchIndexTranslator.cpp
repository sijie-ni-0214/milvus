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

#include "segcore/storagev1translator/TextMatchIndexTranslator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <string_view>
#include <type_traits>
#include <utility>

#include "common/ScopedTimer.h"
#include "fmt/core.h"
#include "glog/logging.h"
#include "index/TextMatchIndex.h"
#include "log/Log.h"
#include "segcore/Utils.h"

namespace milvus::segcore::storagev1translator {
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
UpdateMax(std::atomic<int64_t>& target, int64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

struct TextMatchGetCellsTiming {
    int64_t create_index_ns = 0;
    int64_t load_ns = 0;
    int64_t register_ns = 0;
    int64_t set_cell_size_ns = 0;
    int64_t result_ns = 0;
    int64_t total_ns = 0;
    int64_t cid_count = 0;
    int64_t index_size = 0;
    bool enable_mmap = false;
};

class TextMatchGetCellsTimingStats {
 public:
    void
    Record(const TextMatchGetCellsTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_create_index_ns_.fetch_add(timing.create_index_ns,
                                         std::memory_order_relaxed);
        total_load_ns_.fetch_add(timing.load_ns, std::memory_order_relaxed);
        total_register_ns_.fetch_add(timing.register_ns,
                                     std::memory_order_relaxed);
        total_set_cell_size_ns_.fetch_add(timing.set_cell_size_ns,
                                          std::memory_order_relaxed);
        total_result_ns_.fetch_add(timing.result_ns, std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_cid_count_.fetch_add(timing.cid_count, std::memory_order_relaxed);
        total_index_size_.fetch_add(timing.index_size,
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
        auto create_index_ns =
            total_create_index_ns_.exchange(0, std::memory_order_relaxed);
        auto load_ns = total_load_ns_.exchange(0, std::memory_order_relaxed);
        auto register_ns =
            total_register_ns_.exchange(0, std::memory_order_relaxed);
        auto set_cell_size_ns =
            total_set_cell_size_ns_.exchange(0, std::memory_order_relaxed);
        auto result_ns =
            total_result_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto cid_count =
            total_cid_count_.exchange(0, std::memory_order_relaxed);
        auto index_size =
            total_index_size_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore text match get cells timing stats count={} "
            "avgCreateIndexMs={:.3f} avgLoadMs={:.3f} "
            "avgRegisterMs={:.3f} avgSetCellSizeMs={:.3f} "
            "avgResultMs={:.3f} avgTotalMs={:.3f} maxTotalMs={:.3f} "
            "avgCidCount={:.2f} avgIndexSizeBytes={:.2f} mmapRatio={:.2f}",
            count,
            AvgMs(create_index_ns, count),
            AvgMs(load_ns, count),
            AvgMs(register_ns, count),
            AvgMs(set_cell_size_ns, count),
            AvgMs(result_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(cid_count) / count,
            static_cast<double>(index_size) / count,
            static_cast<double>(mmap_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_create_index_ns_{0};
    std::atomic<int64_t> total_load_ns_{0};
    std::atomic<int64_t> total_register_ns_{0};
    std::atomic<int64_t> total_set_cell_size_ns_{0};
    std::atomic<int64_t> total_result_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_cid_count_{0};
    std::atomic<int64_t> total_index_size_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

static TextMatchGetCellsTimingStats text_match_get_cells_timing_stats;

}  // namespace

TextMatchIndexTranslator::TextMatchIndexTranslator(
    TextMatchIndexLoadInfo load_info,
    milvus::storage::FileManagerContext file_manager_context,
    milvus::Config config)
    : load_info_(std::move(load_info)),
      file_manager_context_(std::move(file_manager_context)),
      config_(std::move(config)),
      key_(fmt::format(
          "seg_{}_textindex_{}", load_info_.segment_id, load_info_.field_id)),
      meta_(load_info_.enable_mmap ? milvus::cachinglayer::StorageType::DISK
                                   : milvus::cachinglayer::StorageType::MEMORY,
            milvus::cachinglayer::CellIdMappingMode::ALWAYS_ZERO,
            milvus::segcore::getCellDataType(/* is_vector */ false,
                                             /* is_index */ true),
            milvus::segcore::getCacheWarmupPolicy(load_info_.warmup_policy,
                                                  /* is_vector */ false,
                                                  /* is_index */ true),
            /* support_eviction */ true) {
}

size_t
TextMatchIndexTranslator::num_cells() const {
    return 1;
}

milvus::cachinglayer::cid_t
TextMatchIndexTranslator::cell_id_of(milvus::cachinglayer::uid_t) const {
    return 0;
}

std::pair<milvus::cachinglayer::ResourceUsage,
          milvus::cachinglayer::ResourceUsage>
TextMatchIndexTranslator::estimated_byte_size_of_cell(
    milvus::cachinglayer::cid_t) const {
    // ignore the cid checking, because there is only one cell
    if (load_info_.enable_mmap) {
        return {{0, load_info_.index_size}, {load_info_.index_size, 0}};
    } else {
        // The reason the maximum disk usage is not zero is that the text match index
        // is first written to the disk, then loaded into memory. Only after that are
        // the disk files deleted.
        return {{load_info_.index_size, 0}, {0, load_info_.index_size}};
    }
}

int64_t
TextMatchIndexTranslator::cells_storage_bytes(
    const std::vector<milvus::cachinglayer::cid_t>&) const {
    // ignore the cids checking, because there is only one cell
    constexpr int64_t MIN_STORAGE_BYTES = 1 * 1024 * 1024;
    return std::max(load_info_.index_size, MIN_STORAGE_BYTES);
}

const std::string&
TextMatchIndexTranslator::key() const {
    return key_;
}

std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<milvus::index::TextMatchIndex>>>
TextMatchIndexTranslator::get_cells(
    milvus::OpContext* ctx,
    const std::vector<milvus::cachinglayer::cid_t>& cids) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    TextMatchGetCellsTiming timing;
    timing.cid_count = cids.size();
    timing.index_size = load_info_.index_size;
    timing.enable_mmap = load_info_.enable_mmap;

    // Check for cancellation before loading text match index
    CheckCancellation(
        ctx, load_info_.segment_id, "TextMatchIndexTranslator::get_cells()");

    auto index =
        std::make_unique<milvus::index::TextMatchIndex>(file_manager_context_);
    timing.create_index_ns = DurationNs(stage_start);

    {
        milvus::ScopedTimer timer(
            "text_match_index_load",
            [](double /*us*/) {
                // no specific metric defined for text match index load yet
            },
            milvus::ScopedTimer::LogLevel::Info);
        stage_start = std::chrono::steady_clock::now();
        index->Load(config_);
        timing.load_ns = DurationNs(stage_start);
        stage_start = std::chrono::steady_clock::now();
        index->RegisterAnalyzer("milvus_tokenizer",
                                load_info_.analyzer_params.c_str());
        timing.register_ns = DurationNs(stage_start);
    }

    LOG_INFO("load text match index success for field:{} of segment:{}",
             load_info_.field_id,
             load_info_.segment_id);

    stage_start = std::chrono::steady_clock::now();
    if (load_info_.enable_mmap) {
        index->SetCellSize({0, index->ByteSize()});
    } else {
        index->SetCellSize({index->ByteSize(), 0});
    }
    timing.set_cell_size_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    std::vector<std::pair<milvus::cachinglayer::cid_t,
                          std::unique_ptr<milvus::index::TextMatchIndex>>>
        result;
    result.emplace_back(std::make_pair(0, std::move(index)));
    timing.result_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    text_match_get_cells_timing_stats.Record(timing);
    return result;
}

milvus::cachinglayer::Meta*
TextMatchIndexTranslator::meta() {
    return &meta_;
}

}  // namespace milvus::segcore::storagev1translator
