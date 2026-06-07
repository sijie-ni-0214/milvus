#include "segcore/storagev1translator/SealedIndexTranslator.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <utility>

#include "common/EasyAssert.h"
#include "common/common_type_c.h"
#include "common/resource_c.h"
#include "fmt/core.h"
#include "glog/logging.h"
#include "index/Index.h"
#include "index/IndexFactory.h"
#include "index/Meta.h"
#include "index/Utils.h"
#include "log/Log.h"
#include "nlohmann/json.hpp"
#include "segcore/Types.h"
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

struct SealedIndexGetCellsTiming {
    int64_t create_index_ns = 0;
    int64_t resource_ns = 0;
    int64_t config_ns = 0;
    int64_t load_ns = 0;
    int64_t result_ns = 0;
    int64_t total_ns = 0;
    int64_t cid_count = 0;
    int64_t index_size = 0;
    bool enable_mmap = false;
    bool is_vector = false;
    bool use_unified = false;
};

class SealedIndexGetCellsTimingStats {
 public:
    void
    Record(const SealedIndexGetCellsTiming& timing) {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_create_index_ns_.fetch_add(timing.create_index_ns,
                                         std::memory_order_relaxed);
        total_resource_ns_.fetch_add(timing.resource_ns,
                                     std::memory_order_relaxed);
        total_config_ns_.fetch_add(timing.config_ns, std::memory_order_relaxed);
        total_load_ns_.fetch_add(timing.load_ns, std::memory_order_relaxed);
        total_result_ns_.fetch_add(timing.result_ns, std::memory_order_relaxed);
        total_ns_.fetch_add(timing.total_ns, std::memory_order_relaxed);
        total_cid_count_.fetch_add(timing.cid_count, std::memory_order_relaxed);
        total_index_size_.fetch_add(timing.index_size,
                                    std::memory_order_relaxed);
        mmap_count_.fetch_add(timing.enable_mmap ? 1 : 0,
                              std::memory_order_relaxed);
        vector_count_.fetch_add(timing.is_vector ? 1 : 0,
                                std::memory_order_relaxed);
        unified_count_.fetch_add(timing.use_unified ? 1 : 0,
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
        auto resource_ns =
            total_resource_ns_.exchange(0, std::memory_order_relaxed);
        auto config_ns =
            total_config_ns_.exchange(0, std::memory_order_relaxed);
        auto load_ns = total_load_ns_.exchange(0, std::memory_order_relaxed);
        auto result_ns =
            total_result_ns_.exchange(0, std::memory_order_relaxed);
        auto total_ns = total_ns_.exchange(0, std::memory_order_relaxed);
        auto cid_count =
            total_cid_count_.exchange(0, std::memory_order_relaxed);
        auto index_size =
            total_index_size_.exchange(0, std::memory_order_relaxed);
        auto mmap_count = mmap_count_.exchange(0, std::memory_order_relaxed);
        auto vector_count =
            vector_count_.exchange(0, std::memory_order_relaxed);
        auto unified_count =
            unified_count_.exchange(0, std::memory_order_relaxed);
        auto max_total_ns =
            max_total_ns_.exchange(0, std::memory_order_relaxed);

        LOG_WARN(
            "segcore sealed index get cells timing stats count={} "
            "avgCreateIndexMs={:.3f} avgResourceMs={:.3f} "
            "avgConfigMs={:.3f} avgLoadMs={:.3f} avgResultMs={:.3f} "
            "avgTotalMs={:.3f} maxTotalMs={:.3f} avgCidCount={:.2f} "
            "avgIndexSizeBytes={:.2f} mmapRatio={:.2f} vectorRatio={:.2f} "
            "unifiedRatio={:.2f}",
            count,
            AvgMs(create_index_ns, count),
            AvgMs(resource_ns, count),
            AvgMs(config_ns, count),
            AvgMs(load_ns, count),
            AvgMs(result_ns, count),
            AvgMs(total_ns, count),
            NsToMs(max_total_ns),
            static_cast<double>(cid_count) / count,
            static_cast<double>(index_size) / count,
            static_cast<double>(mmap_count) / count,
            static_cast<double>(vector_count) / count,
            static_cast<double>(unified_count) / count);
    }

    std::atomic<int64_t> count_{0};
    std::atomic<int64_t> total_create_index_ns_{0};
    std::atomic<int64_t> total_resource_ns_{0};
    std::atomic<int64_t> total_config_ns_{0};
    std::atomic<int64_t> total_load_ns_{0};
    std::atomic<int64_t> total_result_ns_{0};
    std::atomic<int64_t> total_ns_{0};
    std::atomic<int64_t> total_cid_count_{0};
    std::atomic<int64_t> total_index_size_{0};
    std::atomic<int64_t> mmap_count_{0};
    std::atomic<int64_t> vector_count_{0};
    std::atomic<int64_t> unified_count_{0};
    std::atomic<int64_t> max_total_ns_{0};
    std::atomic<int64_t> last_log_ns_{0};
};

static SealedIndexGetCellsTimingStats sealed_index_get_cells_timing_stats;

}  // namespace

SealedIndexTranslator::SealedIndexTranslator(
    milvus::index::CreateIndexInfo index_info,
    const milvus::segcore::LoadIndexInfo* load_index_info,
    milvus::tracer::TraceContext ctx,
    milvus::storage::FileManagerContext file_manager_context,
    Config config)
    : index_info_(std::move(index_info)),
      ctx_(ctx),
      file_manager_context_(std::move(file_manager_context)),
      config_(std::move(config)),
      index_key_(fmt::format("seg_{}_si_{}",
                             load_index_info->segment_id,
                             load_index_info->field_id)),
      index_load_info_({load_index_info->enable_mmap,
                        load_index_info->mmap_dir_path,
                        load_index_info->field_type,
                        load_index_info->element_type,
                        load_index_info->index_params,
                        load_index_info->index_size,
                        load_index_info->index_engine_version,
                        std::to_string(load_index_info->index_id),
                        std::to_string(load_index_info->segment_id),
                        std::to_string(load_index_info->field_id),
                        load_index_info->num_rows,
                        load_index_info->dim,
                        load_index_info->warmup_policy}),
      meta_(
          load_index_info->enable_mmap
              ? milvus::cachinglayer::StorageType::DISK
              : milvus::cachinglayer::StorageType::MEMORY,
          milvus::cachinglayer::CellIdMappingMode::ALWAYS_ZERO,
          milvus::segcore::getCellDataType(
              /* is_vector */ IsVectorDataType(load_index_info->field_type),
              /* is_index */ true),
          // if index data supports lazy load internally, we always use sync for index metadata
          // warmup policy will be used for index internally
          // currently only vector index is possible to support lazy load
          (IsVectorDataType(load_index_info->field_type) &&
           knowhere::IndexFactory::Instance().FeatureCheck(
               index_info_.index_type, knowhere::feature::LAZY_LOAD))
              ? CacheWarmupPolicy::CacheWarmupPolicy_Sync
              : milvus::segcore::getCacheWarmupPolicy(
                    load_index_info->warmup_policy,
                    /* is_vector */
                    IsVectorDataType(load_index_info->field_type),
                    /* is_index */ true),
          /* support_eviction */
          // if index data supports lazy load internally, we don't need to support eviction for index metadata
          // currently only vector index is possible to support lazy load
          !(IsVectorDataType(load_index_info->field_type) &&
            knowhere::IndexFactory::Instance().FeatureCheck(
                index_info_.index_type, knowhere::feature::LAZY_LOAD))) {
}

size_t
SealedIndexTranslator::num_cells() const {
    return 1;
}

milvus::cachinglayer::cid_t
SealedIndexTranslator::cell_id_of(milvus::cachinglayer::uid_t uid) const {
    return 0;
}

std::pair<milvus::cachinglayer::ResourceUsage,
          milvus::cachinglayer::ResourceUsage>
SealedIndexTranslator::estimated_byte_size_of_cell(
    milvus::cachinglayer::cid_t cid) const {
    LoadResourceRequest request =
        milvus::index::IndexFactory::GetInstance().IndexLoadResource(
            index_load_info_.field_type,
            index_load_info_.element_type,
            index_load_info_.index_engine_version,
            index_load_info_.index_size,
            index_load_info_.index_params,
            index_load_info_.enable_mmap,
            index_load_info_.num_rows,
            index_load_info_.dim);
    // this is an estimation, error could be up to 20%.
    return {milvus::cachinglayer::ResourceUsage(request.final_memory_cost,
                                                request.final_disk_cost),
            milvus::cachinglayer::ResourceUsage(
                request.max_memory_cost - request.final_memory_cost,
                request.max_disk_cost * 2 - request.final_disk_cost)};
}

const std::string&
SealedIndexTranslator::key() const {
    return index_key_;
}

std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<milvus::index::IndexBase>>>
SealedIndexTranslator::get_cells(milvus::OpContext* ctx,
                                 const std::vector<cid_t>& cids) {
    auto total_start = std::chrono::steady_clock::now();
    auto stage_start = std::chrono::steady_clock::now();
    SealedIndexGetCellsTiming timing;
    timing.cid_count = cids.size();
    timing.index_size = index_load_info_.index_size;
    timing.enable_mmap = index_load_info_.enable_mmap;
    timing.is_vector = IsVectorDataType(index_load_info_.field_type);
    int64_t segment_id = std::stoll(index_load_info_.segment_id);

    std::unique_ptr<milvus::index::IndexBase> index =
        milvus::index::IndexFactory::GetInstance().CreateIndex(
            index_info_, file_manager_context_);
    timing.create_index_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    LoadResourceRequest request =
        milvus::index::IndexFactory::GetInstance().IndexLoadResource(
            index_load_info_.field_type,
            index_load_info_.element_type,
            index_load_info_.index_engine_version,
            index_load_info_.index_size,
            index_load_info_.index_params,
            index_load_info_.enable_mmap,
            index_load_info_.num_rows,
            index_load_info_.dim);
    index->SetCellSize(milvus::cachinglayer::ResourceUsage(
        request.final_memory_cost, request.final_disk_cost));
    timing.resource_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    if (index_load_info_.enable_mmap && index->IsMmapSupported()) {
        AssertInfo(!index_load_info_.mmap_dir_path.empty(),
                   "mmap directory path is empty");
        auto base_path = std::filesystem::path(index_load_info_.mmap_dir_path) /
                         "index_files" / index_load_info_.index_id /
                         index_load_info_.segment_id /
                         index_load_info_.field_id;
        config_[milvus::index::ENABLE_MMAP] = "true";
        config_[milvus::index::MMAP_FILE_PATH] = (base_path / "index").string();
        config_[milvus::index::EMB_LIST_META_PATH] =
            (base_path / index::EMB_LIST_META_FILE_NAME).string();
        config_[milvus::index::EMB_LIST_RAW_INDEX_PATH] =
            (base_path / index::EMB_LIST_RAW_INDEX_FILE_NAME).string();
    } else {
        config_[milvus::index::ENABLE_MMAP] = "false";
    }
    timing.config_ns = DurationNs(stage_start);

    // Check for cancellation before loading index data
    CheckCancellation(ctx, segment_id, "LoadIndex");

    // Check scalar index engine version for V3 routing
    auto scalar_version =
        milvus::index::GetValueFromConfig<int32_t>(
            config_, milvus::index::SCALAR_INDEX_ENGINE_VERSION)
            .value_or(1);
    stage_start = std::chrono::steady_clock::now();
    if (scalar_version >= 3 && !IsVectorDataType(index_info_.field_type)) {
        timing.use_unified = true;
        config_[milvus::index::COLLECTION_ID] =
            file_manager_context_.fieldDataMeta.collection_id;
        LOG_INFO("load V3 scalar index with configs: {}", config_.dump());
        index->LoadUnified(config_);
    } else {
        LOG_INFO("load index with configs: {}", config_.dump());
        index->Load(ctx_, config_);
    }
    timing.load_ns = DurationNs(stage_start);

    stage_start = std::chrono::steady_clock::now();
    std::vector<std::pair<cid_t, std::unique_ptr<milvus::index::IndexBase>>>
        result;
    result.emplace_back(std::make_pair(0, std::move(index)));
    timing.result_ns = DurationNs(stage_start);
    timing.total_ns = DurationNs(total_start);
    sealed_index_get_cells_timing_stats.Record(timing);
    return result;
}

Meta*
SealedIndexTranslator::meta() {
    return &meta_;
}
}  // namespace milvus::segcore::storagev1translator
