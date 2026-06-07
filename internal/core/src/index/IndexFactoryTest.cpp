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

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "common/Consts.h"
#include "common/Types.h"
#include "index/IndexFactory.h"
#include "index/IndexInfo.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/version.h"

namespace milvus::index {
namespace {

void
ExpectIndexHasRawDataMatchesResource(
    DataType field_type,
    DataType element_type,
    const std::map<std::string, std::string>& index_params,
    bool mmap_enable = false) {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto request = IndexFactory::GetInstance().IndexLoadResource(field_type,
                                                                 element_type,
                                                                 version,
                                                                 1024,
                                                                 index_params,
                                                                 mmap_enable,
                                                                 1000,
                                                                 128);
    auto has_raw_data = IndexFactory::GetInstance().IndexHasRawData(
        field_type, element_type, version, index_params, mmap_enable);
    EXPECT_EQ(request.has_raw_data, has_raw_data);
}

}  // namespace

TEST(IndexFactoryTest, IndexHasRawDataMatchesLoadResource) {
    ExpectIndexHasRawDataMatchesResource(
        DataType::VECTOR_FLOAT,
        DataType::NONE,
        {{INDEX_TYPE, knowhere::IndexEnum::INDEX_HNSW},
         {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
         {knowhere::indexparam::M, "16"},
         {knowhere::indexparam::EF, "10"}});

    ExpectIndexHasRawDataMatchesResource(
        DataType::INT64, DataType::NONE, {{INDEX_TYPE, ASCENDING_SORT}});

    ExpectIndexHasRawDataMatchesResource(
        DataType::VARCHAR, DataType::NONE, {{INDEX_TYPE, INVERTED_INDEX_TYPE}});

    ExpectIndexHasRawDataMatchesResource(
        DataType::INT64, DataType::NONE, {{INDEX_TYPE, BITMAP_INDEX_TYPE}});

    ExpectIndexHasRawDataMatchesResource(
        DataType::VECTOR_ARRAY,
        DataType::VECTOR_FLOAT,
        {{INDEX_TYPE, knowhere::IndexEnum::INDEX_HNSW},
         {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
         {knowhere::indexparam::M, "16"},
         {knowhere::indexparam::EF, "10"}});
}

}  // namespace milvus::index
