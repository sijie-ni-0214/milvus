#include "segcore/storagev1translator/SealedIndexTranslator.h"

#include <filesystem>
#include <limits>
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
#include "monitor/segcore_memory_stats_c.h"
#include "nlohmann/json.hpp"
#include "segcore/Types.h"
#include "segcore/Utils.h"

namespace {

int64_t
ToMetricDelta(size_t value) {
    return value > static_cast<size_t>(std::numeric_limits<int64_t>::max())
               ? std::numeric_limits<int64_t>::max()
               : static_cast<int64_t>(value);
}

size_t
ApproxStringDynamicBytes(const std::string& value) {
    return value.capacity() + 1;
}

size_t
ApproxStringMapBytes(const std::map<std::string, std::string>& values) {
    if (values.empty()) {
        return 0;
    }
    size_t bytes =
        values.size() * (sizeof(std::pair<const std::string, std::string>) +
                         3 * sizeof(void*) + sizeof(bool));
    for (const auto& [key, value] : values) {
        bytes += ApproxStringDynamicBytes(key);
        bytes += ApproxStringDynamicBytes(value);
    }
    return bytes;
}

size_t
ApproxStringVectorBytes(const std::vector<std::string>& values) {
    size_t bytes = values.capacity() * sizeof(std::string);
    for (const auto& value : values) {
        bytes += ApproxStringDynamicBytes(value);
    }
    return bytes;
}

size_t
ApproxJsonBytes(const milvus::Config& value) {
    size_t bytes = sizeof(milvus::Config);
    if (value.is_object()) {
        bytes += value.size() *
                 (sizeof(std::pair<const std::string, milvus::Config>) +
                  3 * sizeof(void*) + sizeof(bool));
        for (auto it = value.begin(); it != value.end(); ++it) {
            bytes += ApproxStringDynamicBytes(it.key());
            bytes += ApproxJsonBytes(it.value());
        }
        return bytes;
    }
    if (value.is_array()) {
        bytes += value.size() * sizeof(milvus::Config);
        for (const auto& item : value) {
            bytes += ApproxJsonBytes(item);
        }
        return bytes;
    }
    if (value.is_string()) {
        if (auto string_value = value.get_ptr<const std::string*>()) {
            bytes += ApproxStringDynamicBytes(*string_value);
        }
    }
    return bytes;
}

size_t
ApproxCreateIndexInfoDynamicBytes(
    const milvus::index::CreateIndexInfo& index_info) {
    return ApproxStringDynamicBytes(index_info.field_name) +
           ApproxStringDynamicBytes(index_info.index_type) +
           ApproxStringDynamicBytes(index_info.metric_type) +
           ApproxStringDynamicBytes(index_info.json_path) +
           ApproxStringDynamicBytes(index_info.json_cast_function) +
           ApproxStringDynamicBytes(index_info.analyzer_extra_info);
}

size_t
ApproxIndexMetaDynamicBytes(const milvus::storage::IndexMeta& index_meta) {
    return ApproxStringDynamicBytes(index_meta.key) +
           ApproxStringDynamicBytes(index_meta.field_name);
}

size_t
ApproxFileManagerContextDynamicBytes(
    const milvus::storage::FileManagerContext& context) {
    return context.fieldDataMeta.field_schema.SpaceUsedLong() +
           ApproxIndexMetaDynamicBytes(context.indexMeta) +
           ApproxStringDynamicBytes(context.stats_base_path);
}

template <typename IndexLoadInfoT>
size_t
ApproxIndexLoadInfoDynamicBytes(const IndexLoadInfoT& index_load_info) {
    return ApproxStringDynamicBytes(index_load_info.mmap_dir_path) +
           ApproxStringMapBytes(index_load_info.index_params) +
           ApproxStringDynamicBytes(index_load_info.index_id) +
           ApproxStringDynamicBytes(index_load_info.segment_id) +
           ApproxStringDynamicBytes(index_load_info.field_id) +
           ApproxStringDynamicBytes(index_load_info.warmup_policy);
}

template <typename IndexLoadInfoT>
size_t
ApproxIndexLoadInfoStringDynamicBytes(const IndexLoadInfoT& index_load_info) {
    size_t bytes = ApproxStringDynamicBytes(index_load_info.mmap_dir_path) +
                   ApproxStringDynamicBytes(index_load_info.index_id) +
                   ApproxStringDynamicBytes(index_load_info.segment_id) +
                   ApproxStringDynamicBytes(index_load_info.field_id) +
                   ApproxStringDynamicBytes(index_load_info.warmup_policy);
    for (const auto& [key, value] : index_load_info.index_params) {
        bytes += ApproxStringDynamicBytes(key);
        bytes += ApproxStringDynamicBytes(value);
    }
    return bytes;
}

}  // namespace

namespace milvus::segcore::storagev1translator {

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
                        load_index_info->warmup_policy,
                        load_index_info->has_load_resource_request,
                        load_index_info->load_resource_request}),
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
    memory_data_type_ = IsVectorDataType(load_index_info->field_type)
                            ? "vector_index"
                            : "scalar_index";
    memory_index_type_ =
        index_info_.index_type.empty() ? "unknown" : index_info_.index_type;

    memory_usage_.object_bytes = sizeof(SealedIndexTranslator);
    memory_usage_.index_info_dynamic_bytes =
        ApproxCreateIndexInfoDynamicBytes(index_info_);
    memory_usage_.file_manager_context_dynamic_bytes =
        ApproxFileManagerContextDynamicBytes(file_manager_context_);
    memory_usage_.config_estimated_bytes = ApproxJsonBytes(config_);
    memory_usage_.index_load_info_dynamic_bytes =
        ApproxIndexLoadInfoDynamicBytes(index_load_info_);
    memory_usage_.index_param_dynamic_bytes =
        ApproxStringMapBytes(index_load_info_.index_params);
    memory_usage_.schema_proto_bytes =
        file_manager_context_.fieldDataMeta.field_schema.SpaceUsedLong();
    memory_usage_.string_dynamic_bytes =
        ApproxStringDynamicBytes(index_key_) +
        ApproxCreateIndexInfoDynamicBytes(index_info_) +
        ApproxIndexMetaDynamicBytes(file_manager_context_.indexMeta) +
        ApproxStringDynamicBytes(file_manager_context_.stats_base_path) +
        ApproxIndexLoadInfoStringDynamicBytes(index_load_info_);
    memory_usage_.estimated_bytes =
        memory_usage_.object_bytes + memory_usage_.index_info_dynamic_bytes +
        memory_usage_.file_manager_context_dynamic_bytes +
        memory_usage_.config_estimated_bytes +
        memory_usage_.index_load_info_dynamic_bytes +
        ApproxStringDynamicBytes(index_key_);

    milvus::monitor::UpdateSegcoreSealedIndexTranslator(
        memory_data_type_.c_str(),
        memory_index_type_.c_str(),
        1,
        ToMetricDelta(memory_usage_.estimated_bytes),
        ToMetricDelta(memory_usage_.object_bytes),
        ToMetricDelta(memory_usage_.index_info_dynamic_bytes),
        ToMetricDelta(memory_usage_.file_manager_context_dynamic_bytes),
        ToMetricDelta(memory_usage_.config_estimated_bytes),
        ToMetricDelta(memory_usage_.index_load_info_dynamic_bytes),
        ToMetricDelta(memory_usage_.index_param_dynamic_bytes),
        ToMetricDelta(memory_usage_.schema_proto_bytes),
        ToMetricDelta(memory_usage_.string_dynamic_bytes));
}

SealedIndexTranslator::~SealedIndexTranslator() {
    milvus::monitor::UpdateSegcoreSealedIndexTranslator(
        memory_data_type_.c_str(),
        memory_index_type_.c_str(),
        -1,
        -ToMetricDelta(memory_usage_.estimated_bytes),
        -ToMetricDelta(memory_usage_.object_bytes),
        -ToMetricDelta(memory_usage_.index_info_dynamic_bytes),
        -ToMetricDelta(memory_usage_.file_manager_context_dynamic_bytes),
        -ToMetricDelta(memory_usage_.config_estimated_bytes),
        -ToMetricDelta(memory_usage_.index_load_info_dynamic_bytes),
        -ToMetricDelta(memory_usage_.index_param_dynamic_bytes),
        -ToMetricDelta(memory_usage_.schema_proto_bytes),
        -ToMetricDelta(memory_usage_.string_dynamic_bytes));
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
    LoadResourceRequest request{};
    if (index_load_info_.has_load_resource_request) {
        request = index_load_info_.load_resource_request;
    } else {
        request = milvus::index::IndexFactory::GetInstance().IndexLoadResource(
            index_load_info_.field_type,
            index_load_info_.element_type,
            index_load_info_.index_engine_version,
            index_load_info_.index_size,
            index_load_info_.index_params,
            index_load_info_.enable_mmap,
            index_load_info_.num_rows,
            index_load_info_.dim);
    }
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
    int64_t segment_id = std::stoll(index_load_info_.segment_id);

    std::unique_ptr<milvus::index::IndexBase> index =
        milvus::index::IndexFactory::GetInstance().CreateIndex(
            index_info_, file_manager_context_);
    LoadResourceRequest request{};
    if (index_load_info_.has_load_resource_request) {
        request = index_load_info_.load_resource_request;
    } else {
        request = milvus::index::IndexFactory::GetInstance().IndexLoadResource(
            index_load_info_.field_type,
            index_load_info_.element_type,
            index_load_info_.index_engine_version,
            index_load_info_.index_size,
            index_load_info_.index_params,
            index_load_info_.enable_mmap,
            index_load_info_.num_rows,
            index_load_info_.dim);
    }
    index->SetCellSize(milvus::cachinglayer::ResourceUsage(
        request.final_memory_cost, request.final_disk_cost));
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

    // Check for cancellation before loading index data
    CheckCancellation(ctx, segment_id, "LoadIndex");

    // Check scalar index engine version for V3 routing
    auto scalar_version =
        milvus::index::GetValueFromConfig<int32_t>(
            config_, milvus::index::SCALAR_INDEX_ENGINE_VERSION)
            .value_or(1);
    if (scalar_version >= 3 && !IsVectorDataType(index_info_.field_type)) {
        config_[milvus::index::COLLECTION_ID] =
            file_manager_context_.fieldDataMeta.collection_id;
        LOG_INFO("load V3 scalar index with configs: {}", config_.dump());
        index->LoadUnified(config_);
    } else {
        LOG_INFO("load index with configs: {}", config_.dump());
        index->Load(ctx_, config_);
    }

    std::vector<std::pair<cid_t, std::unique_ptr<milvus::index::IndexBase>>>
        result;
    result.emplace_back(std::make_pair(0, std::move(index)));
    return result;
}

Meta*
SealedIndexTranslator::meta() {
    return &meta_;
}
}  // namespace milvus::segcore::storagev1translator
