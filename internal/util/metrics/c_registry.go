/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
 * # regarding copyright ownership. The ASF licenses this file
 * # to you under the Apache License, Version 2.0 (the
 * # "License"); you may not use this file except in compliance
 * # with the License. You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing, software
 * # distributed under the License is distributed on an "AS IS" BASIS,
 * # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * # See the License for the specific language governing permissions and
 * # limitations under the License.
 */

package metrics

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>
#include "segcore/metrics_c.h"
#include "monitor/monitor_c.h"
#include "monitor/jemalloc_stats_c.h"
#include "monitor/segcore_memory_stats_c.h"

*/
import "C"

import (
	"sort"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"go.uber.org/zap"
	"golang.org/x/exp/maps"
	"google.golang.org/protobuf/proto"

	_ "github.com/milvus-io/milvus/internal/util/cgo"
	"github.com/milvus-io/milvus/pkg/v3/log"
)

// metricSorter is a sortable slice of *dto.Metric.
type metricSorter []*dto.Metric

func (s metricSorter) Len() int {
	return len(s)
}

func (s metricSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s metricSorter) Less(i, j int) bool {
	if len(s[i].Label) != len(s[j].Label) {
		// This should not happen. The metrics are
		// inconsistent. However, we have to deal with the fact, as
		// people might use custom collectors or metric family injection
		// to create inconsistent metrics. So let's simply compare the
		// number of labels in this case. That will still yield
		// reproducible sorting.
		return len(s[i].Label) < len(s[j].Label)
	}
	for n, lp := range s[i].Label {
		vi := lp.GetValue()
		vj := s[j].Label[n].GetValue()
		if vi != vj {
			return vi < vj
		}
	}

	// We should never arrive here. Multiple metrics with the same
	// label set in the same scrape will lead to undefined ingestion
	// behavior. However, as above, we have to provide stable sorting
	// here, even for inconsistent metrics. So sort equal metrics
	// by their timestamp, with missing timestamps (implying "now")
	// coming last.
	if s[i].TimestampMs == nil {
		return false
	}
	if s[j].TimestampMs == nil {
		return true
	}
	return s[i].GetTimestampMs() < s[j].GetTimestampMs()
}

// NormalizeMetricFamilies returns a MetricFamily slice with empty
// MetricFamilies pruned and the remaining MetricFamilies sorted by name within
// the slice, with the contained Metrics sorted within each MetricFamily.
func NormalizeMetricFamilies(metricFamiliesByName map[string]*dto.MetricFamily) []*dto.MetricFamily {
	for _, mf := range metricFamiliesByName {
		sort.Sort(metricSorter(mf.Metric))
	}
	names := make([]string, 0, len(metricFamiliesByName))
	for name, mf := range metricFamiliesByName {
		if len(mf.Metric) > 0 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	result := make([]*dto.MetricFamily, 0, len(names))
	for _, name := range names {
		result = append(result, metricFamiliesByName[name])
	}
	return result
}

// Jemalloc metrics cache to avoid frequent C calls
var (
	jemallocMetricsCache struct {
		sync.RWMutex
		metrics   map[string]*dto.MetricFamily
		timestamp time.Time
	}
	// Cache TTL: 10 seconds to balance performance and data freshness
	// This reduces mallctl("epoch") calls from every scrape to once per 10s
	jemallocMetricsCacheTTL = 10 * time.Second
)

func NewCRegistry() *CRegistry {
	return &CRegistry{
		Registry: prometheus.NewRegistry(),
	}
}

// only re-write the implementation of Gather()
type CRegistry struct {
	*prometheus.Registry
	mtx sync.RWMutex
}

// Gather implements Gatherer.
func (r *CRegistry) Gather() (res []*dto.MetricFamily, err error) {
	var parser expfmt.TextParser

	r.mtx.RLock()
	defer r.mtx.RUnlock()

	cMetricsStr := C.GetKnowhereMetrics()
	metricsStr := C.GoString(cMetricsStr)
	C.free(unsafe.Pointer(cMetricsStr))

	out, err := parser.TextToMetricFamilies(strings.NewReader(metricsStr))
	if err != nil {
		log.Error("fail to parse knowhere prometheus metrics", zap.Error(err))
		return
	}

	cMetricsStr = C.GetCoreMetrics()
	metricsStr = C.GoString(cMetricsStr)
	C.free(unsafe.Pointer(cMetricsStr))

	out1, err := parser.TextToMetricFamilies(strings.NewReader(metricsStr))
	if err != nil {
		log.Error("fail to parse storage prometheus metrics", zap.Error(err))
		return
	}

	maps.Copy(out, out1)

	// Add jemalloc stats metrics
	jemallocMetrics := gatherJemallocMetrics()
	for name, mf := range jemallocMetrics {
		out[name] = mf
	}

	segcoreMemoryMetrics := gatherSegcoreMemoryMetrics()
	for name, mf := range segcoreMemoryMetrics {
		out[name] = mf
	}

	res = NormalizeMetricFamilies(out)
	return
}

// gatherJemallocMetrics collects comprehensive jemalloc stats and returns them as metric families.
// Uses a 10-second cache to avoid expensive mallctl("epoch") calls on every Prometheus scrape.
func gatherJemallocMetrics() map[string]*dto.MetricFamily {
	// Fast path: check if cache is still valid
	jemallocMetricsCache.RLock()
	if time.Since(jemallocMetricsCache.timestamp) < jemallocMetricsCacheTTL && jemallocMetricsCache.metrics != nil {
		cached := jemallocMetricsCache.metrics
		jemallocMetricsCache.RUnlock()
		log.Debug("using cached jemalloc metrics",
			zap.Duration("age", time.Since(jemallocMetricsCache.timestamp)))
		return cached
	}
	jemallocMetricsCache.RUnlock()

	// Slow path: cache expired, collect fresh metrics from C
	// This involves expensive mallctl("epoch") call which can take 100-5000μs
	result := make(map[string]*dto.MetricFamily)

	cStats := C.GetJemallocStats()
	if !bool(cStats.success) {
		log.Debug("jemalloc stats not available (may be running on macOS or jemalloc is disabled)")
		return result
	}

	gaugeType := dto.MetricType_GAUGE

	// Helper function to create a gauge metric family
	createGaugeFamily := func(name, help string, value float64) *dto.MetricFamily {
		return &dto.MetricFamily{
			Name: proto.String(name),
			Help: proto.String(help),
			Type: &gaugeType,
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{
						Value: proto.Float64(value),
					},
				},
			},
		}
	}

	// Define all jemalloc metrics (8 comprehensive metrics)
	metrics := []struct {
		name  string
		help  string
		value uint64
	}{
		// Core metrics from jemalloc
		{"milvus_jemalloc_allocated_bytes", "Total number of bytes allocated by the application", uint64(cStats.allocated)},
		{"milvus_jemalloc_active_bytes", "Total number of bytes in active pages allocated by the application (includes fragmentation)", uint64(cStats.active)},
		{"milvus_jemalloc_metadata_bytes", "Total number of bytes dedicated to jemalloc metadata", uint64(cStats.metadata)},
		{"milvus_jemalloc_resident_bytes", "Total number of bytes in physically resident data pages mapped by the allocator", uint64(cStats.resident)},
		{"milvus_jemalloc_mapped_bytes", "Total number of bytes in virtual memory mappings", uint64(cStats.mapped)},
		{"milvus_jemalloc_retained_bytes", "Total number of bytes in retained virtual memory mappings (could be returned to OS)", uint64(cStats.retained)},
		// Derived metrics (calculated in C code)
		{"milvus_jemalloc_fragmentation_bytes", "Internal fragmentation in bytes (active - allocated)", uint64(cStats.fragmentation)},
		{"milvus_jemalloc_overhead_bytes", "Memory overhead in bytes (resident - active)", uint64(cStats.overhead)},
	}

	for _, m := range metrics {
		result[m.name] = createGaugeFamily(m.name, m.help, float64(m.value))
	}

	// Update cache with fresh metrics
	jemallocMetricsCache.Lock()
	jemallocMetricsCache.metrics = result
	jemallocMetricsCache.timestamp = time.Now()
	jemallocMetricsCache.Unlock()

	log.Debug("refreshed jemalloc metrics cache",
		zap.Int("num_metrics", len(result)))

	return result
}

func gatherSegcoreMemoryMetrics() map[string]*dto.MetricFamily {
	result := make(map[string]*dto.MetricFamily)
	cStats := C.GetSegcoreMemoryStats()
	gaugeType := dto.MetricType_GAUGE

	createGaugeFamily := func(name, help string, value float64) *dto.MetricFamily {
		return &dto.MetricFamily{
			Name: proto.String(name),
			Help: proto.String(help),
			Type: &gaugeType,
			Metric: []*dto.Metric{
				{
					Gauge: &dto.Gauge{
						Value: proto.Float64(value),
					},
				},
			},
		}
	}

	metrics := []struct {
		name  string
		help  string
		value uint64
	}{
		{"milvus_segcore_sealed_segment_count", "Live sealed segment objects in segcore", uint64(cStats.sealed_segment_count)},
		{"milvus_segcore_sealed_segment_object_bytes", "Live sealed segment object bytes estimated by sizeof(ChunkedSegmentSealedImpl)", uint64(cStats.sealed_segment_object_bytes)},
		{"milvus_segcore_sealed_segment_runtime_estimated_bytes", "Live sealed segment runtime bytes estimated from mmap descriptor, empty containers, InsertRecord, DeletedRecord, and LoadFieldDataInfo", uint64(cStats.sealed_segment_runtime_estimated_bytes)},
		{"milvus_segcore_sealed_segment_mmap_descriptor_bytes", "Live sealed segment mmap descriptor and manager-entry bytes estimated from object and container shells", uint64(cStats.sealed_segment_mmap_descriptor_bytes)},
		{"milvus_segcore_sealed_segment_empty_indexing_container_bytes", "Live sealed segment empty indexing container bytes estimated from bucket arrays", uint64(cStats.sealed_segment_empty_indexing_container_bytes)},
		{"milvus_segcore_sealed_segment_insert_record_bytes", "Live sealed segment InsertRecord runtime bytes estimated from PK/timestamp helper containers", uint64(cStats.sealed_segment_insert_record_bytes)},
		{"milvus_segcore_sealed_segment_deleted_record_bytes", "Live sealed segment DeletedRecord runtime bytes estimated from skiplist shell, bitset, and snapshots", uint64(cStats.sealed_segment_deleted_record_bytes)},
		{"milvus_segcore_sealed_segment_load_field_data_info_bytes", "Live sealed segment LoadFieldDataInfo bytes estimated from field metadata and path containers", uint64(cStats.sealed_segment_load_field_data_info_bytes)},
		{"milvus_segcore_sealed_segment_field_map_bytes", "Live sealed segment fields_ unordered_map bytes estimated from buckets and nodes", uint64(cStats.sealed_segment_field_map_bytes)},
		{"milvus_segcore_sealed_segment_field_shared_ptr_control_block_bytes", "Live sealed segment field column shared_ptr control block bytes estimated per field entry", uint64(cStats.sealed_segment_field_shared_ptr_control_block_bytes)},
		{"milvus_segcore_sealed_segment_field_data_accounted_map_bytes", "Live sealed segment field_data_accounted_bytes_ unordered_map bytes estimated from buckets and nodes", uint64(cStats.sealed_segment_field_data_accounted_map_bytes)},
		{"milvus_segcore_sealed_segment_mmap_field_ids_bytes", "Live sealed segment mmap_field_ids_ unordered_set bytes estimated from buckets and nodes", uint64(cStats.sealed_segment_mmap_field_ids_bytes)},
		{"milvus_segcore_segment_load_info_bytes", "Live SegmentLoadInfo protobuf bytes estimated by SpaceUsedLong", uint64(cStats.segment_load_info_bytes)},
		{"milvus_segcore_segment_load_info_estimated_bytes", "Live SegmentLoadInfo bytes estimated by object size, protobuf SpaceUsedLong, and runtime cache containers", uint64(cStats.segment_load_info_estimated_bytes)},
		{"milvus_segcore_segment_load_info_object_bytes", "Live SegmentLoadInfo object bytes estimated by sizeof(SegmentLoadInfo)", uint64(cStats.segment_load_info_object_bytes)},
		{"milvus_segcore_segment_load_info_proto_bytes", "Live SegmentLoadInfo protobuf bytes estimated by SpaceUsedLong", uint64(cStats.segment_load_info_proto_bytes)},
		{"milvus_segcore_segment_load_info_converted_index_cache_bytes", "Live SegmentLoadInfo converted index cache bytes estimated from container capacity", uint64(cStats.segment_load_info_converted_index_cache_bytes)},
		{"milvus_segcore_segment_load_info_field_index_id_cache_bytes", "Live SegmentLoadInfo field index id cache bytes estimated from container capacity", uint64(cStats.segment_load_info_field_index_id_cache_bytes)},
		{"milvus_segcore_segment_load_info_field_index_has_raw_data_bytes", "Live SegmentLoadInfo raw-data field set bytes estimated from container nodes", uint64(cStats.segment_load_info_field_index_has_raw_data_bytes)},
		{"milvus_segcore_segment_load_info_fields_filled_with_default_bytes", "Live SegmentLoadInfo default-filled field set bytes estimated from container nodes", uint64(cStats.segment_load_info_fields_filled_with_default_bytes)},
		{"milvus_segcore_segment_load_info_field_binlog_cache_bytes", "Live SegmentLoadInfo field binlog cache bytes estimated from container nodes", uint64(cStats.segment_load_info_field_binlog_cache_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_bytes", "Live SegmentLoadInfo column group cache shell bytes estimated from mutex/shared-pointer/object shells", uint64(cStats.segment_load_info_column_group_cache_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_deep_bytes", "Live SegmentLoadInfo column group cache deep bytes estimated from vector, ColumnGroup, ColumnGroupFile, strings, properties, and shared_ptr control blocks", uint64(cStats.segment_load_info_column_group_cache_deep_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_path_bytes", "Live SegmentLoadInfo column group file path dynamic string bytes", uint64(cStats.segment_load_info_column_group_cache_path_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_property_bytes", "Live SegmentLoadInfo column group file property map bytes including key/value strings", uint64(cStats.segment_load_info_column_group_cache_property_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_column_bytes", "Live SegmentLoadInfo column group column-name vector bytes", uint64(cStats.segment_load_info_column_group_cache_column_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_format_bytes", "Live SegmentLoadInfo column group format string dynamic bytes", uint64(cStats.segment_load_info_column_group_cache_format_bytes)},
		{"milvus_segcore_segment_load_info_column_group_cache_group_count", "Live SegmentLoadInfo cached column group count", uint64(cStats.segment_load_info_column_group_cache_group_count)},
		{"milvus_segcore_segment_load_info_column_group_cache_file_count", "Live SegmentLoadInfo cached column group file count", uint64(cStats.segment_load_info_column_group_cache_file_count)},
		{"milvus_segcore_segment_load_info_created_text_indexes_bytes", "Live SegmentLoadInfo created text index set bytes estimated from container nodes", uint64(cStats.segment_load_info_created_text_indexes_bytes)},
		{"milvus_segcore_field_entry_count", "Live segcore field map entries across sealed segments", uint64(cStats.field_entry_count)},
		{"milvus_segcore_lazy_manifest_group_count", "Live lazy manifest column group shells", uint64(cStats.lazy_manifest_group_count)},
		{"milvus_segcore_lazy_manifest_group_estimated_bytes", "Live lazy manifest column group shell bytes including estimated dynamic metadata", uint64(cStats.lazy_manifest_group_estimated_bytes)},
		{"milvus_segcore_lazy_manifest_proxy_count", "Live lazy manifest proxy columns", uint64(cStats.lazy_manifest_proxy_count)},
		{"milvus_segcore_lazy_manifest_proxy_object_bytes", "Live lazy manifest proxy column object bytes estimated by sizeof(LazyManifestProxyColumn)", uint64(cStats.lazy_manifest_proxy_object_bytes)},
		{"milvus_segcore_lazy_manifest_projected_column_count", "Projected columns referenced by live lazy manifest group shells", uint64(cStats.lazy_manifest_projected_column_count)},
		{"milvus_segcore_pk_index_slot_count", "Live primary-key index cache slots attached to sealed segments", uint64(cStats.pk_index_slot_count)},
		{"milvus_segcore_timestamp_index_slot_count", "Live timestamp index cache slots attached to sealed segments", uint64(cStats.timestamp_index_slot_count)},
		{"milvus_segcore_pk_index_translator_count", "Live primary-key index translators", uint64(cStats.pk_index_translator_count)},
		{"milvus_segcore_pk_index_translator_object_bytes", "Live primary-key index translator object bytes estimated by sizeof(PkIndexTranslator)", uint64(cStats.pk_index_translator_object_bytes)},
		{"milvus_segcore_timestamp_index_translator_count", "Live timestamp index translators", uint64(cStats.timestamp_index_translator_count)},
		{"milvus_segcore_timestamp_index_translator_object_bytes", "Live timestamp index translator object bytes estimated by sizeof(TimestampIndexTranslator)", uint64(cStats.timestamp_index_translator_object_bytes)},
		{"milvus_segcore_pk_index_cell_count", "Live materialized primary-key index cells", uint64(cStats.pk_index_cell_count)},
		{"milvus_segcore_pk_index_cell_bytes", "Live materialized primary-key index cell bytes reported by the cell", uint64(cStats.pk_index_cell_bytes)},
		{"milvus_segcore_timestamp_index_cell_count", "Live materialized timestamp index cells", uint64(cStats.timestamp_index_cell_count)},
		{"milvus_segcore_timestamp_index_cell_bytes", "Live materialized timestamp index cell bytes reported by the cell", uint64(cStats.timestamp_index_cell_bytes)},
	}

	for _, m := range metrics {
		result[m.name] = createGaugeFamily(m.name, m.help, float64(m.value))
	}

	indexMetrics := gatherSegcoreIndexMemoryMetrics()
	for name, mf := range indexMetrics {
		result[name] = mf
	}
	collectionMetrics := gatherSegcoreCollectionMemoryMetrics()
	for name, mf := range collectionMetrics {
		result[name] = mf
	}

	return result
}

type segcoreCollectionMemoryRecord struct {
	owner     string
	fieldName string
	dataType  string
	component string
	accuracy  string
	count     uint64
	bytes     uint64
}

func gatherSegcoreCollectionMemoryMetrics() map[string]*dto.MetricFamily {
	cStats := C.GetSegcoreCollectionMemoryStats()
	records := make([]segcoreCollectionMemoryRecord, 0, int(cStats.entry_count))
	for i := 0; i < int(cStats.entry_count); i++ {
		entry := cStats.entries[i]
		records = append(records, segcoreCollectionMemoryRecord{
			owner:     C.GoString((*C.char)(unsafe.Pointer(&entry.owner[0]))),
			fieldName: C.GoString((*C.char)(unsafe.Pointer(&entry.field_name[0]))),
			dataType:  C.GoString((*C.char)(unsafe.Pointer(&entry.data_type[0]))),
			component: C.GoString((*C.char)(unsafe.Pointer(&entry.component[0]))),
			accuracy:  C.GoString((*C.char)(unsafe.Pointer(&entry.accuracy[0]))),
			count:     uint64(entry.count),
			bytes:     uint64(entry.bytes),
		})
	}

	gaugeType := dto.MetricType_GAUGE
	createFamily := func(name, help string, value func(segcoreCollectionMemoryRecord) uint64) *dto.MetricFamily {
		family := &dto.MetricFamily{
			Name: proto.String(name),
			Help: proto.String(help),
			Type: &gaugeType,
		}
		for _, record := range records {
			family.Metric = append(family.Metric, &dto.Metric{
				Label: []*dto.LabelPair{
					{Name: proto.String("owner"), Value: proto.String(record.owner)},
					{Name: proto.String("field_name"), Value: proto.String(record.fieldName)},
					{Name: proto.String("data_type"), Value: proto.String(record.dataType)},
					{Name: proto.String("component"), Value: proto.String(record.component)},
					{Name: proto.String("accuracy"), Value: proto.String(record.accuracy)},
				},
				Gauge: &dto.Gauge{Value: proto.Float64(float64(value(record)))},
			})
		}
		return family
	}

	overflowFamily := &dto.MetricFamily{
		Name: proto.String("milvus_segcore_collection_memory_overflow_count"),
		Help: proto.String("Collection memory statistic entries omitted because the fixed snapshot was full"),
		Type: &gaugeType,
		Metric: []*dto.Metric{{
			Gauge: &dto.Gauge{Value: proto.Float64(float64(cStats.overflow_count))},
		}},
	}

	return map[string]*dto.MetricFamily{
		"milvus_segcore_collection_memory_bytes": createFamily(
			"milvus_segcore_collection_memory_bytes",
			"Live collection-owned memory split by object member and measurement accuracy",
			func(record segcoreCollectionMemoryRecord) uint64 { return record.bytes },
		),
		"milvus_segcore_collection_memory_count": createFamily(
			"milvus_segcore_collection_memory_count",
			"Live collection-owned object, element, capacity, bucket, or node counts",
			func(record segcoreCollectionMemoryRecord) uint64 { return record.count },
		),
		"milvus_segcore_collection_memory_overflow_count": overflowFamily,
	}
}

type segcoreIndexMemoryRecord struct {
	dataType                       string
	indexType                      string
	count                          uint64
	estimatedBytes                 uint64
	objectBytes                    uint64
	indexInfoDynamicBytes          uint64
	fileManagerContextDynamicBytes uint64
	configEstimatedBytes           uint64
	indexLoadInfoDynamicBytes      uint64
	indexParamDynamicBytes         uint64
	schemaProtoBytes               uint64
	stringDynamicBytes             uint64
}

func gatherSegcoreIndexMemoryMetrics() map[string]*dto.MetricFamily {
	cStats := C.GetSegcoreIndexMemoryStats()
	records := make([]segcoreIndexMemoryRecord, 0, int(cStats.entry_count))
	for i := 0; i < int(cStats.entry_count); i++ {
		entry := cStats.entries[i]
		records = append(records, segcoreIndexMemoryRecord{
			dataType:                       C.GoString((*C.char)(unsafe.Pointer(&entry.data_type[0]))),
			indexType:                      C.GoString((*C.char)(unsafe.Pointer(&entry.index_type[0]))),
			count:                          uint64(entry.count),
			estimatedBytes:                 uint64(entry.estimated_bytes),
			objectBytes:                    uint64(entry.object_bytes),
			indexInfoDynamicBytes:          uint64(entry.index_info_dynamic_bytes),
			fileManagerContextDynamicBytes: uint64(entry.file_manager_context_dynamic_bytes),
			configEstimatedBytes:           uint64(entry.config_estimated_bytes),
			indexLoadInfoDynamicBytes:      uint64(entry.index_load_info_dynamic_bytes),
			indexParamDynamicBytes:         uint64(entry.index_param_dynamic_bytes),
			schemaProtoBytes:               uint64(entry.schema_proto_bytes),
			stringDynamicBytes:             uint64(entry.string_dynamic_bytes),
		})
	}

	gaugeType := dto.MetricType_GAUGE
	createFamily := func(name, help string, value func(segcoreIndexMemoryRecord) uint64) *dto.MetricFamily {
		family := &dto.MetricFamily{
			Name: proto.String(name),
			Help: proto.String(help),
			Type: &gaugeType,
		}
		for _, record := range records {
			family.Metric = append(family.Metric, &dto.Metric{
				Label: []*dto.LabelPair{
					{Name: proto.String("data_type"), Value: proto.String(record.dataType)},
					{Name: proto.String("index_type"), Value: proto.String(record.indexType)},
				},
				Gauge: &dto.Gauge{Value: proto.Float64(float64(value(record)))},
			})
		}
		return family
	}

	return map[string]*dto.MetricFamily{
		"milvus_segcore_sealed_index_translator_count": createFamily(
			"milvus_segcore_sealed_index_translator_count",
			"Live sealed index translators grouped by index data type and index type",
			func(record segcoreIndexMemoryRecord) uint64 { return record.count },
		),
		"milvus_segcore_sealed_index_translator_estimated_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_estimated_bytes",
			"Live sealed index translator bytes estimated from object size and copied metadata",
			func(record segcoreIndexMemoryRecord) uint64 { return record.estimatedBytes },
		),
		"milvus_segcore_sealed_index_translator_object_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_object_bytes",
			"Live sealed index translator object bytes estimated by sizeof(SealedIndexTranslator)",
			func(record segcoreIndexMemoryRecord) uint64 { return record.objectBytes },
		),
		"milvus_segcore_sealed_index_translator_index_info_dynamic_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_index_info_dynamic_bytes",
			"Live sealed index translator dynamic bytes copied in CreateIndexInfo",
			func(record segcoreIndexMemoryRecord) uint64 { return record.indexInfoDynamicBytes },
		),
		"milvus_segcore_sealed_index_translator_file_manager_context_dynamic_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_file_manager_context_dynamic_bytes",
			"Live sealed index translator dynamic bytes copied in FileManagerContext",
			func(record segcoreIndexMemoryRecord) uint64 { return record.fileManagerContextDynamicBytes },
		),
		"milvus_segcore_sealed_index_translator_config_estimated_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_config_estimated_bytes",
			"Live sealed index translator nlohmann json config bytes estimated recursively",
			func(record segcoreIndexMemoryRecord) uint64 { return record.configEstimatedBytes },
		),
		"milvus_segcore_sealed_index_translator_index_load_info_dynamic_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_index_load_info_dynamic_bytes",
			"Live sealed index translator dynamic bytes copied in IndexLoadInfo",
			func(record segcoreIndexMemoryRecord) uint64 { return record.indexLoadInfoDynamicBytes },
		),
		"milvus_segcore_sealed_index_translator_index_param_dynamic_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_index_param_dynamic_bytes",
			"Live sealed index translator index_params map bytes; subset of IndexLoadInfo dynamic bytes",
			func(record segcoreIndexMemoryRecord) uint64 { return record.indexParamDynamicBytes },
		),
		"milvus_segcore_sealed_index_translator_schema_proto_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_schema_proto_bytes",
			"Live sealed index translator FieldSchema protobuf bytes; subset of FileManagerContext dynamic bytes",
			func(record segcoreIndexMemoryRecord) uint64 { return record.schemaProtoBytes },
		),
		"milvus_segcore_sealed_index_translator_string_dynamic_bytes": createFamily(
			"milvus_segcore_sealed_index_translator_string_dynamic_bytes",
			"Live sealed index translator string dynamic bytes across copied metadata; explanatory subset",
			func(record segcoreIndexMemoryRecord) uint64 { return record.stringDynamicBytes },
		),
	}
}
