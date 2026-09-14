// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cachinglayer/CacheSlot.h"
#include "segcore/CacheMetricAttribution.h"

namespace milvus::segcore {
namespace {

using namespace milvus::cachinglayer;
namespace cache_monitor = milvus::cachinglayer::monitor;

constexpr int64_t kDiskBytes = 128;
constexpr auto kShard = "cache-metric-attribution-disabled";

struct DiskCell {
    ResourceUsage
    CellByteSize() const {
        return {0, kDiskBytes};
    }
};

class DiskTranslator : public Translator<DiskCell> {
 public:
    size_t
    num_cells() const override {
        return 1;
    }

    cid_t
    cell_id_of(milvus::cachinglayer::uid_t) const override {
        return 0;
    }

    std::pair<ResourceUsage, ResourceUsage>
    estimated_byte_size_of_cell(cid_t) const override {
        return {{0, kDiskBytes}, {0, 0}};
    }

    const std::string&
    key() const override {
        return key_;
    }

    Meta*
    meta() override {
        return &meta_;
    }

    int64_t
    cells_storage_bytes(const std::vector<cid_t>& cids) const override {
        return cids.size() * kDiskBytes;
    }

    std::vector<std::pair<cid_t, std::unique_ptr<DiskCell>>>
    get_cells(OpContext*, const std::vector<cid_t>& cids) override {
        std::vector<std::pair<cid_t, std::unique_ptr<DiskCell>>> cells;
        for (const auto cid : cids) {
            cells.emplace_back(cid, std::make_unique<DiskCell>());
        }
        return cells;
    }

 private:
    const std::string key_ = kShard;
    Meta meta_{StorageType::DISK,
               CellIdMappingMode::ALWAYS_ZERO,
               CellDataType::SCALAR_FIELD,
               CacheWarmupPolicy::CacheWarmupPolicy_Disable,
               true,
               std::nullopt,
               MetricAttributionFromShard(kShard)};
};

size_t
ShardGaugeCount() {
    for (const auto& family :
         milvus::monitor::getPrometheusClient().GetRegistry().Collect()) {
        if (family.name == "internal_cache_shard_disk_usage_bytes") {
            return family.metric.size();
        }
    }
    return 0;
}

TEST(CacheMetricAttribution, DisablesShardGaugeWithoutDisablingDiskAccounting) {
    const auto original_series = ShardGaugeCount();
    auto& loaded_bytes = cache_monitor::cache_loaded_bytes(
        CellDataType::SCALAR_FIELD, StorageType::DISK);
    const auto original_bytes = loaded_bytes.Value();
    const ResourceUsage limit{0, 1024};
    auto dlist = std::make_shared<milvus::cachinglayer::internal::DList>(
        true, limit, limit, limit, EvictionConfig{10, false, 600});
    auto slot = std::make_shared<CacheSlot<DiskCell>>(
        std::make_unique<DiskTranslator>(),
        dlist.get(),
        true,
        true,
        true,
        std::chrono::milliseconds(1000),
        std::chrono::milliseconds(0));

    auto expect_no_shard_tracking = [&] {
        EXPECT_EQ(ShardGaugeCount(), original_series);
        EXPECT_EQ(cache_monitor::cache_shard_disk_usage_bytes_value(
                      CellDataType::SCALAR_FIELD, kShard),
                  std::nullopt);
    };
    expect_no_shard_tracking();
    for (int cycle = 0; cycle < 2; ++cycle) {
        {
            OpContext ctx;
            auto accessor = slot->PinCellsDirect(&ctx, {0});
            ASSERT_NE(accessor->get_ith_cell(0), nullptr);
            EXPECT_EQ(loaded_bytes.Value(), original_bytes + kDiskBytes);
            expect_no_shard_tracking();
        }
        if (cycle == 0) {
            ASSERT_TRUE(slot->ManualEvictAll());
        } else {
            slot.reset();
        }
        EXPECT_EQ(loaded_bytes.Value(), original_bytes);
        expect_no_shard_tracking();
    }
}

}  // namespace
}  // namespace milvus::segcore
