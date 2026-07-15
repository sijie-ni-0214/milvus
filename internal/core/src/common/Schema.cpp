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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "Schema.h"
#include "arrow/type.h"
#include "arrow/util/key_value_metadata.h"
#include "common/Consts.h"
#include "common/FieldMeta.h"
#include "common/MemoryUsage.h"
#include "milvus-storage/common/constants.h"
#include "pb/common.pb.h"
#include "protobuf_utils.h"

namespace milvus {

using std::string;
const std::string namespace_field_name = "$namespace_id";

SchemaMemoryUsage
Schema::MemoryUsage() const {
    SchemaMemoryUsage usage;
    usage.object_bytes = sizeof(Schema);
    usage.shared_ptr_control_block_estimated_bytes =
        memory_usage::SharedPtrControlBlockEstimatedBytes();
    usage.field_count = fields_.size();
    usage.field_ids_size = field_ids_.size();
    usage.field_ids_capacity = field_ids_.capacity();
    usage.field_ids_capacity_bytes =
        memory_usage::VectorCapacityBytes(field_ids_);

    usage.fields_bucket_count = fields_.bucket_count();
    usage.fields_bucket_bytes = memory_usage::UnorderedMapBucketBytes(fields_);
    usage.fields_key_bytes = fields_.size() * sizeof(FieldId);
    usage.fields_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(fields_);

    usage.name_ids_count = name_ids_.size();
    usage.name_ids_bucket_count = name_ids_.bucket_count();
    usage.name_ids_bucket_bytes =
        memory_usage::UnorderedMapBucketBytes(name_ids_);
    usage.name_ids_value_bytes =
        memory_usage::UnorderedMapValueBytes(name_ids_);
    usage.name_ids_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(name_ids_);
    for (const auto& [name, field_id] : name_ids_) {
        (void)field_id;
        usage.name_ids_string_dynamic_bytes +=
            memory_usage::StringDynamicBytes(name.get());
    }

    usage.id_names_count = id_names_.size();
    usage.id_names_bucket_count = id_names_.bucket_count();
    usage.id_names_bucket_bytes =
        memory_usage::UnorderedMapBucketBytes(id_names_);
    usage.id_names_value_bytes =
        memory_usage::UnorderedMapValueBytes(id_names_);
    usage.id_names_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(id_names_);
    for (const auto& [field_id, name] : id_names_) {
        (void)field_id;
        usage.id_names_string_dynamic_bytes +=
            memory_usage::StringDynamicBytes(name.get());
    }

    usage.load_fields_count = load_fields_.size();
    usage.load_fields_bucket_count = load_fields_.bucket_count();
    usage.load_fields_bucket_bytes =
        memory_usage::UnorderedSetBucketBytes(load_fields_);
    usage.load_fields_value_bytes =
        memory_usage::UnorderedSetValueBytes(load_fields_);
    usage.load_fields_node_overhead_estimated_bytes =
        memory_usage::UnorderedSetNodeOverheadEstimatedBytes(load_fields_);

    usage.bm25_fields_count = bm25_function_output_fields_.size();
    usage.bm25_fields_bucket_count =
        bm25_function_output_fields_.bucket_count();
    usage.bm25_fields_bucket_bytes =
        memory_usage::UnorderedSetBucketBytes(bm25_function_output_fields_);
    usage.bm25_fields_value_bytes =
        memory_usage::UnorderedSetValueBytes(bm25_function_output_fields_);
    usage.bm25_fields_node_overhead_estimated_bytes =
        memory_usage::UnorderedSetNodeOverheadEstimatedBytes(
            bm25_function_output_fields_);

    usage.mmap_fields_count = mmap_fields_.size();
    usage.mmap_fields_bucket_count = mmap_fields_.bucket_count();
    usage.mmap_fields_bucket_bytes =
        memory_usage::UnorderedMapBucketBytes(mmap_fields_);
    usage.mmap_fields_value_bytes =
        memory_usage::UnorderedMapValueBytes(mmap_fields_);
    usage.mmap_fields_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(mmap_fields_);

    usage.struct_array_cache_count = struct_array_field_cache_.size();
    usage.struct_array_cache_bucket_count =
        struct_array_field_cache_.bucket_count();
    usage.struct_array_cache_bucket_bytes =
        memory_usage::UnorderedMapBucketBytes(struct_array_field_cache_);
    usage.struct_array_cache_value_bytes =
        memory_usage::UnorderedMapValueBytes(struct_array_field_cache_);
    usage.struct_array_cache_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(
            struct_array_field_cache_);
    for (const auto& [name, field_id] : struct_array_field_cache_) {
        (void)field_id;
        usage.struct_array_cache_string_dynamic_bytes +=
            memory_usage::StringDynamicBytes(name);
    }

    usage.warmup_fields_count = warmup_fields_.size();
    usage.warmup_fields_bucket_count = warmup_fields_.bucket_count();
    usage.warmup_fields_bucket_bytes =
        memory_usage::UnorderedMapBucketBytes(warmup_fields_);
    usage.warmup_fields_value_bytes =
        memory_usage::UnorderedMapValueBytes(warmup_fields_);
    usage.warmup_fields_node_overhead_estimated_bytes =
        memory_usage::UnorderedMapNodeOverheadEstimatedBytes(warmup_fields_);
    for (const auto& [field_id, policy] : warmup_fields_) {
        (void)field_id;
        usage.warmup_fields_string_dynamic_bytes +=
            memory_usage::StringDynamicBytes(policy);
    }

    const auto add_optional_string = [&](const auto& value) {
        if (value.has_value()) {
            usage.warmup_policy_string_dynamic_bytes +=
                memory_usage::StringDynamicBytes(*value);
        }
    };
    add_optional_string(warmup_vector_index_);
    add_optional_string(warmup_scalar_index_);
    add_optional_string(warmup_scalar_field_);
    add_optional_string(warmup_vector_field_);
    usage.external_string_dynamic_bytes =
        memory_usage::StringDynamicBytes(external_source_) +
        memory_usage::StringDynamicBytes(external_spec_);

    std::lock_guard<std::mutex> lock(arrow_schema_cache_mutex_);
    if (loon_arrow_schema_cache_ != nullptr) {
        usage.arrow_schema_count = 1;
        usage.arrow_schema_object_bytes = sizeof(arrow::Schema);
        usage.arrow_schema_control_block_estimated_bytes =
            memory_usage::SharedPtrControlBlockEstimatedBytes();
        const auto& arrow_fields = loon_arrow_schema_cache_->fields();
        usage.arrow_field_count = arrow_fields.size();
        usage.arrow_field_vector_capacity = arrow_fields.capacity();
        usage.arrow_field_vector_capacity_bytes =
            memory_usage::VectorCapacityBytes(arrow_fields);
        usage.arrow_field_object_bytes =
            arrow_fields.size() * sizeof(arrow::Field);
        usage.arrow_field_control_block_estimated_bytes =
            arrow_fields.size() *
            memory_usage::SharedPtrControlBlockEstimatedBytes();
        for (const auto& field : arrow_fields) {
            usage.arrow_field_name_dynamic_bytes +=
                memory_usage::StringDynamicBytes(field->name());
        }
    }
    return usage;
}

Schema::Schema(const Schema& other) {
    std::lock_guard<std::mutex> lock(other.arrow_schema_cache_mutex_);
    debug_id = other.debug_id;
    field_ids_ = other.field_ids_;
    fields_.insert(other.fields_.begin(), other.fields_.end());
    name_ids_ = other.name_ids_;
    id_names_ = other.id_names_;
    primary_field_id_opt_ = other.primary_field_id_opt_;
    dynamic_field_id_opt_ = other.dynamic_field_id_opt_;
    namespace_field_id_opt_ = other.namespace_field_id_opt_;
    ttl_field_id_opt_ = other.ttl_field_id_opt_;
    load_fields_ = other.load_fields_;
    bm25_function_output_fields_ = other.bm25_function_output_fields_;
    schema_version_ = other.schema_version_;
    has_mmap_setting_ = other.has_mmap_setting_;
    mmap_enabled_ = other.mmap_enabled_;
    mmap_fields_ = other.mmap_fields_;
    struct_array_field_cache_ = other.struct_array_field_cache_;
    warmup_vector_index_ = other.warmup_vector_index_;
    warmup_scalar_index_ = other.warmup_scalar_index_;
    warmup_scalar_field_ = other.warmup_scalar_field_;
    warmup_vector_field_ = other.warmup_vector_field_;
    warmup_fields_ = other.warmup_fields_;
    external_source_ = other.external_source_;
    external_spec_ = other.external_spec_;
}

Schema&
Schema::operator=(const Schema& other) {
    if (this == &other) {
        return *this;
    }

    std::scoped_lock lock(arrow_schema_cache_mutex_,
                          other.arrow_schema_cache_mutex_);
    auto copied_fields = other.fields_;
    debug_id = other.debug_id;
    field_ids_ = other.field_ids_;
    fields_.swap(copied_fields);
    name_ids_ = other.name_ids_;
    id_names_ = other.id_names_;
    primary_field_id_opt_ = other.primary_field_id_opt_;
    dynamic_field_id_opt_ = other.dynamic_field_id_opt_;
    namespace_field_id_opt_ = other.namespace_field_id_opt_;
    ttl_field_id_opt_ = other.ttl_field_id_opt_;
    load_fields_ = other.load_fields_;
    bm25_function_output_fields_ = other.bm25_function_output_fields_;
    schema_version_ = other.schema_version_;
    has_mmap_setting_ = other.has_mmap_setting_;
    mmap_enabled_ = other.mmap_enabled_;
    mmap_fields_ = other.mmap_fields_;
    struct_array_field_cache_ = other.struct_array_field_cache_;
    warmup_vector_index_ = other.warmup_vector_index_;
    warmup_scalar_index_ = other.warmup_scalar_index_;
    warmup_scalar_field_ = other.warmup_scalar_field_;
    warmup_vector_field_ = other.warmup_vector_field_;
    warmup_fields_ = other.warmup_fields_;
    external_source_ = other.external_source_;
    external_spec_ = other.external_spec_;
    loon_arrow_schema_cache_.reset();
    return *this;
}

std::shared_ptr<Schema>
Schema::ParseFrom(const milvus::proto::schema::CollectionSchema& schema_proto) {
    auto schema = std::make_shared<Schema>();
    // schema->set_auto_id(schema_proto.autoid());

    // NOTE: only two system

    auto process_field = [&schema, &schema_proto](const auto& child) {
        auto field_id = FieldId(child.fieldid());

        auto f = FieldMeta::ParseFrom(child);
        schema->AddField(std::move(f));

        if (child.is_primary_key()) {
            AssertInfo(!schema->get_primary_field_id().has_value(),
                       "repetitive primary key");
            schema->set_primary_field_id(field_id);
        }

        if (child.is_dynamic()) {
            Assert(schema_proto.enable_dynamic_field());
            AssertInfo(!schema->get_dynamic_field_id().has_value(),
                       "repetitive dynamic field");
            schema->set_dynamic_field_id(field_id);
        }
        if (child.name() == namespace_field_name) {
            schema->set_namespace_field_id(field_id);
        }

        auto [has_setting, enabled] =
            GetBoolFromRepeatedKVs(child.type_params(), MMAP_ENABLED_KEY);
        if (has_setting) {
            schema->mmap_fields_[field_id] = enabled;
        }

        // Parse warmup policy for the field (key: "warmup")
        auto warmup_policy =
            GetStringFromRepeatedKVs(child.type_params(), WARMUP_KEY);
        if (warmup_policy.has_value()) {
            schema->warmup_fields_[field_id] = std::move(warmup_policy).value();
        }
    };

    for (const milvus::proto::schema::FieldSchema& child :
         schema_proto.fields()) {
        process_field(child);
    }

    for (const milvus::proto::schema::StructArrayFieldSchema& child :
         schema_proto.struct_array_fields()) {
        for (const auto& sub_field : child.fields()) {
            process_field(sub_field);
        }
    }

    for (const auto& function : schema_proto.functions()) {
        if (function.type() != milvus::proto::schema::BM25) {
            continue;
        }
        for (const auto output_field_id : function.output_field_ids()) {
            schema->bm25_function_output_fields_.emplace(output_field_id);
        }
    }

    std::tie(schema->has_mmap_setting_, schema->mmap_enabled_) =
        GetBoolFromRepeatedKVs(schema_proto.properties(), MMAP_ENABLED_KEY);

    std::optional<std::string> ttl_field_name;
    for (const auto& property : schema_proto.properties()) {
        if (property.key() == COLLECTION_TTL_FIELD_KEY) {
            ttl_field_name = property.value();
            break;
        }
    }
    if (ttl_field_name.has_value()) {
        bool found = false;
        for (const milvus::proto::schema::FieldSchema& child :
             schema_proto.fields()) {
            if (child.name() == ttl_field_name.value()) {
                schema->set_ttl_field_id(FieldId(child.fieldid()));
                found = true;
                break;
            }
        }
        AssertInfo(found, "ttl field name not found in schema fields");
    }
    // Parse collection-level warmup policies
    schema->warmup_vector_index_ = GetStringFromRepeatedKVs(
        schema_proto.properties(), WARMUP_VECTOR_INDEX_KEY);
    schema->warmup_scalar_index_ = GetStringFromRepeatedKVs(
        schema_proto.properties(), WARMUP_SCALAR_INDEX_KEY);
    schema->warmup_scalar_field_ = GetStringFromRepeatedKVs(
        schema_proto.properties(), WARMUP_SCALAR_FIELD_KEY);
    schema->warmup_vector_field_ = GetStringFromRepeatedKVs(
        schema_proto.properties(), WARMUP_VECTOR_FIELD_KEY);

    AssertInfo(schema->get_primary_field_id().has_value(),
               "primary key should be specified");

    // Parse external collection properties
    if (!schema_proto.external_source().empty()) {
        schema->set_external_source(schema_proto.external_source());
        schema->set_external_spec(schema_proto.external_spec());
    }

    return schema;
}

const FieldMeta FieldMeta::RowIdMeta(
    FieldName("RowID"), RowFieldID, DataType::INT64, false, std::nullopt);

const ArrowSchemaPtr
Schema::ConvertToArrowSchema() const {
    arrow::FieldVector arrow_fields;
    arrow_fields.reserve(field_ids_.size());
    for (const auto& field_id : field_ids_) {
        const auto& meta = fields_.at(field_id);
        int dim = IsVectorDataType(meta.get_data_type()) &&
                          !IsSparseFloatVectorDataType(meta.get_data_type())
                      ? meta.get_dim()
                      : 1;

        std::shared_ptr<arrow::DataType> arrow_data_type = nullptr;
        auto data_type = meta.get_data_type();
        if (data_type == DataType::VECTOR_ARRAY) {
            arrow_data_type = GetArrowDataTypeForVectorArray(
                meta.get_element_type(), meta.get_dim());
        } else {
            arrow_data_type = GetArrowDataType(data_type, dim);
        }

        auto arrow_field = std::make_shared<arrow::Field>(
            meta.get_name().get(),
            arrow_data_type,
            meta.is_nullable(),
            arrow::key_value_metadata({milvus_storage::ARROW_FIELD_ID_KEY},
                                      {std::to_string(meta.get_id().get())}));
        arrow_fields.push_back(arrow_field);
    }
    return arrow::schema(arrow_fields);
}

const ArrowSchemaPtr
Schema::ConvertToLoonArrowSchema() const {
    std::lock_guard<std::mutex> lock(arrow_schema_cache_mutex_);
    if (loon_arrow_schema_cache_ != nullptr) {
        return loon_arrow_schema_cache_;
    }

    arrow::FieldVector arrow_fields;
    arrow_fields.reserve(field_ids_.size());
    for (const auto& field_id : field_ids_) {
        const auto& meta = fields_.at(field_id);
        int dim = IsVectorDataType(meta.get_data_type()) &&
                          !IsSparseFloatVectorDataType(meta.get_data_type())
                      ? meta.get_dim()
                      : 1;

        std::shared_ptr<arrow::DataType> arrow_data_type = nullptr;
        auto data_type = meta.get_data_type();
        if (data_type == DataType::VECTOR_ARRAY) {
            arrow_data_type = GetArrowDataTypeForVectorArray(
                meta.get_element_type(), meta.get_dim());
        } else {
            arrow_data_type = GetArrowDataType(data_type, dim);
        }

        auto arrow_field =
            std::make_shared<arrow::Field>(std::to_string(field_id.get()),
                                           arrow_data_type,
                                           meta.is_nullable());
        arrow_fields.push_back(arrow_field);
    }
    loon_arrow_schema_cache_ = arrow::schema(arrow_fields);
    return loon_arrow_schema_cache_;
}

proto::schema::CollectionSchema
Schema::ToProto() const {
    proto::schema::CollectionSchema schema_proto;
    schema_proto.set_enable_dynamic_field(dynamic_field_id_opt_.has_value());

    for (const auto& field_id : field_ids_) {
        const auto& meta = fields_.at(field_id);
        auto* field_proto = schema_proto.add_fields();
        *field_proto = meta.ToProto();

        if (primary_field_id_opt_.has_value() &&
            field_id == primary_field_id_opt_.value()) {
            field_proto->set_is_primary_key(true);
        }
        if (dynamic_field_id_opt_.has_value() &&
            field_id == dynamic_field_id_opt_.value()) {
            field_proto->set_is_dynamic(true);
        }
    }

    return schema_proto;
}

std::unique_ptr<std::vector<FieldMeta>>
Schema::AbsentFields(Schema& old_schema) const {
    std::vector<FieldMeta> result;
    result.reserve(fields_.size());
    for (const auto& [field_id, field_meta] : fields_) {
        auto it = old_schema.fields_.find(field_id);
        if (it == old_schema.fields_.end()) {
            result.emplace_back(field_meta);
        }
    }

    return std::make_unique<std::vector<FieldMeta>>(std::move(result));
}

std::shared_ptr<std::vector<std::string>>
Schema::GetExternalColumnNames() const {
    auto columns = std::make_shared<std::vector<std::string>>();
    for (const auto& field_id : field_ids_) {
        auto it = fields_.find(field_id);
        if (it != fields_.end() && it->second.is_external_field()) {
            columns->push_back(it->second.get_external_field());
        }
    }
    return columns;
}

FieldId
Schema::ResolveColumnFieldId(const std::string& column_name) const {
    if (is_external_collection()) {
        for (const auto& [fid, meta] : fields_) {
            if (meta.is_external_field() &&
                meta.get_external_field() == column_name) {
                return fid;
            }
        }
        ThrowInfo(ErrorCode::DataFormatBroken,
                  "external column '{}' not found in schema",
                  column_name);
    }
    return FieldId(std::stoll(column_name));
}

std::pair<bool, bool>
Schema::MmapEnabled(const FieldId& field_id) const {
    auto it = mmap_fields_.find(field_id);
    // fallback to  collection-level config
    if (it == mmap_fields_.end()) {
        return {has_mmap_setting_, mmap_enabled_};
    }
    return {true, it->second};
}

const FieldMeta&
Schema::GetFirstArrayFieldInStruct(const std::string& struct_name) const {
    auto cache_it = struct_array_field_cache_.find(struct_name);
    if (cache_it != struct_array_field_cache_.end()) {
        return fields_.at(cache_it->second);
    }

    ThrowInfo(ErrorCode::UnexpectedError,
              "No array field found in struct: {}",
              struct_name);
}

std::pair<bool, std::string>
Schema::WarmupPolicy(const FieldId& field_id,
                     bool is_vector,
                     bool is_index) const {
    // First check field-level warmup policy
    auto it = warmup_fields_.find(field_id);
    if (it != warmup_fields_.end()) {
        return {true, it->second};
    }

    // Fallback to appropriate collection-level config based on field type
    if (is_vector) {
        if (is_index) {
            return {warmup_vector_index_.has_value(),
                    warmup_vector_index_.value_or("")};
        }
        return {warmup_vector_field_.has_value(),
                warmup_vector_field_.value_or("")};
    }
    if (is_index) {
        return {warmup_scalar_index_.has_value(),
                warmup_scalar_index_.value_or("")};
    }
    return {warmup_scalar_field_.has_value(),
            warmup_scalar_field_.value_or("")};
}

}  // namespace milvus
