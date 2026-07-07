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

package segments

import (
	"sort"
	"sync"
	"unsafe"

	"go.uber.org/atomic"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/querynodev2/pkoracle"
	"github.com/milvus-io/milvus/internal/querynodev2/segments/state"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	goMemPartSegmentCount                    = "segment_count"
	goMemPartBaseSegmentObject               = "base_segment_object"
	goMemPartLocalSegmentObjectExtra         = "local_segment_object_extra"
	goMemPartSegmentAtomicObject             = "segment_atomic_object"
	goMemPartLoadStateLockObject             = "load_state_lock_object"
	goMemPartSegmentContainerShell           = "segment_container_shell"
	goMemPartSegmentManagerReference         = "segment_manager_reference"
	goMemPartBM25StatsMapPayload             = "bm25_stats_map_payload"
	goMemPartResourceUsageCacheObject        = "resource_usage_cache_object"
	goMemPartResourceUsageCacheSlice         = "resource_usage_cache_slice_backing"
	goMemPartCompactLoadInfoObject           = "compact_load_info_object"
	goMemPartCompactLoadInfoString           = "compact_load_info_string"
	goMemPartCompactLoadInfoCompactionSlice  = "compact_load_info_compaction_slice_backing"
	goMemPartCompactLoadInfoDeltalogSlice    = "compact_load_info_deltalog_slice_backing"
	goMemPartCompactLoadInfoPositionObject   = "compact_load_info_position_object"
	goMemPartCompactLoadInfoPositionString   = "compact_load_info_position_string"
	goMemPartCompactLoadInfoPositionMsgID    = "compact_load_info_position_msgid_bytes"
	goMemPartLoadInfoDeltalogObject          = "load_info_deltalog_field_binlog_object"
	goMemPartLoadInfoDeltalogBinlogSlice     = "load_info_deltalog_binlog_slice_backing"
	goMemPartLoadInfoDeltalogChildFieldSlice = "load_info_deltalog_child_field_slice_backing"
	goMemPartLoadInfoDeltalogBinlogObject    = "load_info_deltalog_binlog_object"
	goMemPartLoadInfoDeltalogPathString      = "load_info_deltalog_path_string"
	goMemPartLoadInfoDeltalogNullCount       = "load_info_deltalog_null_count_payload"
	goMemPartFieldInfoObject                 = "field_info_object"
	goMemPartFieldBinlogObject               = "field_binlog_object"
	goMemPartFieldBinlogBinlogSlice          = "field_binlog_binlog_slice_backing"
	goMemPartFieldBinlogChildFieldSlice      = "field_binlog_child_field_slice_backing"
	goMemPartFieldBinlogBinlogObject         = "field_binlog_binlog_object"
	goMemPartFieldBinlogPathString           = "field_binlog_path_string"
	goMemPartFieldBinlogNullCount            = "field_binlog_null_count_payload"
	goMemPartIndexedFieldInfoObject          = "indexed_field_info_object"
	goMemPartFieldIndexInfoObject            = "field_index_info_object"
	goMemPartFieldIndexInfoNameString        = "field_index_info_name_string"
	goMemPartFieldIndexInfoParamSlice        = "field_index_info_param_slice_backing"
	goMemPartFieldIndexInfoFilePathSlice     = "field_index_info_file_path_slice_backing"
	goMemPartFieldIndexInfoFilePathString    = "field_index_info_file_path_string"
	goMemPartIndexParamKeyValueObject        = "index_param_key_value_object"
	goMemPartIndexParamKeyString             = "index_param_key_string"
	goMemPartIndexParamValueString           = "index_param_value_string"
	goMemPartFieldJSONStatsObject            = "field_json_stats_object"
	goMemPartFieldJSONStatsMapPayload        = "field_json_stats_map_payload"
	goMemPartManifestLazyKeyString           = "manifest_lazy_key_string"
	goMemPartPKCandidateObject               = "pk_candidate_object"
)

type goSegmentRuntimeMetricKey struct {
	part string
}

type goSegmentRuntimeMetricValue struct {
	count int64
	bytes int64
}

func (mgr *segmentManager) collectGoSegmentRuntimeMemoryStats() []metrics.QueryNodeGoSegmentRuntimeMemoryStats {
	aggregates := make(map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue)
	collect := func(_ typeutil.UniqueID, segment Segment) bool {
		local, ok := segment.(*LocalSegment)
		if ok {
			collectLocalSegmentRuntimeMemoryStats(aggregates, local)
		}
		return true
	}

	mgr.globalSegments.sealedSegments.Range(collect)
	mgr.globalSegments.growingSegments.Range(collect)

	stats := make([]metrics.QueryNodeGoSegmentRuntimeMemoryStats, 0, len(aggregates))
	for key, value := range aggregates {
		stats = append(stats, metrics.QueryNodeGoSegmentRuntimeMemoryStats{
			Part:  key.part,
			Count: float64(value.count),
			Bytes: float64(value.bytes),
		})
	}
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].Part < stats[j].Part
	})
	return stats
}

func collectLocalSegmentRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, segment *LocalSegment) {
	addGoSegmentRuntimeMetric(aggregates, goMemPartSegmentCount, 1, 0)
	addGoSegmentRuntimeMetric(aggregates, goMemPartBaseSegmentObject, 1, int64(unsafe.Sizeof(baseSegment{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartLocalSegmentObjectExtra, 1, int64(unsafe.Sizeof(LocalSegment{})-unsafe.Sizeof(baseSegment{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartLoadStateLockObject, 1, loadStateLockRuntimeBytes())
	addGoSegmentRuntimeMetric(aggregates, goMemPartSegmentAtomicObject, 9, segmentAtomicRuntimeBytes())
	addGoSegmentRuntimeMetric(aggregates, goMemPartSegmentContainerShell, 1, segmentContainerShellRuntimeBytes())
	addGoSegmentRuntimeMetric(aggregates, goMemPartSegmentManagerReference, 2, segmentManagerReferenceRuntimeBytes(2))
	collectBM25StatsRuntimeMemoryStats(aggregates, segment)
	collectResourceUsageCacheRuntimeMemoryStats(aggregates, segment)

	if _, ok := segment.pkCandidate.(*pkoracle.BloomFilterSet); ok {
		addGoSegmentRuntimeMetric(aggregates, goMemPartPKCandidateObject, 1, int64(unsafe.Sizeof(pkoracle.BloomFilterSet{})))
	}

	collectLoadInfoRuntimeMemoryStats(aggregates, segment.LoadInfo())
	collectFieldRuntimeMemoryStats(aggregates, segment)
	collectManifestLazyKeyRuntimeMemoryStats(aggregates, segment)
}

func collectLoadInfoRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, loadInfo *querypb.SegmentLoadInfo) {
	if loadInfo == nil {
		return
	}
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoObject, 1, int64(unsafe.Sizeof(querypb.SegmentLoadInfo{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoString, 2, int64(len(loadInfo.GetInsertChannel())+len(loadInfo.GetManifestPath())))
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoCompactionSlice, int64(len(loadInfo.GetCompactionFrom())), int64(cap(loadInfo.GetCompactionFrom()))*int64(unsafe.Sizeof(int64(0))))
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoDeltalogSlice, int64(len(loadInfo.GetDeltalogs())), slicePointerBackingBytes(cap(loadInfo.GetDeltalogs())))
	collectMsgPositionRuntimeMemoryStats(aggregates, loadInfo.GetStartPosition())
	collectMsgPositionRuntimeMemoryStats(aggregates, loadInfo.GetDeltaPosition())
	for _, fieldBinlog := range loadInfo.GetDeltalogs() {
		collectFieldBinlogRuntimeMemoryStats(aggregates, fieldBinlog, fieldBinlogRuntimeParts{
			object:          goMemPartLoadInfoDeltalogObject,
			binlogSlice:     goMemPartLoadInfoDeltalogBinlogSlice,
			childFieldSlice: goMemPartLoadInfoDeltalogChildFieldSlice,
			binlogObject:    goMemPartLoadInfoDeltalogBinlogObject,
			pathString:      goMemPartLoadInfoDeltalogPathString,
			nullCount:       goMemPartLoadInfoDeltalogNullCount,
		})
	}
}

func collectMsgPositionRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, position *msgpb.MsgPosition) {
	if position == nil {
		return
	}
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoPositionObject, 1, int64(unsafe.Sizeof(msgpb.MsgPosition{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoPositionString, 2, int64(len(position.GetChannelName())+len(position.GetMsgGroup())))
	addGoSegmentRuntimeMetric(aggregates, goMemPartCompactLoadInfoPositionMsgID, 1, int64(len(position.GetMsgID())))
}

func collectFieldRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, segment *LocalSegment) {
	segment.fields.Range(func(_ int64, info *FieldInfo) bool {
		if info == nil {
			return true
		}
		addGoSegmentRuntimeMetric(aggregates, goMemPartFieldInfoObject, 1, int64(unsafe.Sizeof(FieldInfo{})))
		collectFieldBinlogRuntimeMemoryStats(aggregates, info.FieldBinlog, fieldBinlogRuntimeParts{
			object:          goMemPartFieldBinlogObject,
			binlogSlice:     goMemPartFieldBinlogBinlogSlice,
			childFieldSlice: goMemPartFieldBinlogChildFieldSlice,
			binlogObject:    goMemPartFieldBinlogBinlogObject,
			pathString:      goMemPartFieldBinlogPathString,
			nullCount:       goMemPartFieldBinlogNullCount,
		})
		return true
	})

	segment.fieldIndexes.Range(func(_ int64, info *IndexedFieldInfo) bool {
		if info == nil {
			return true
		}
		addGoSegmentRuntimeMetric(aggregates, goMemPartIndexedFieldInfoObject, 1, int64(unsafe.Sizeof(IndexedFieldInfo{})))
		collectFieldBinlogRuntimeMemoryStats(aggregates, info.FieldBinlog, fieldBinlogRuntimeParts{
			object:          goMemPartFieldBinlogObject,
			binlogSlice:     goMemPartFieldBinlogBinlogSlice,
			childFieldSlice: goMemPartFieldBinlogChildFieldSlice,
			binlogObject:    goMemPartFieldBinlogBinlogObject,
			pathString:      goMemPartFieldBinlogPathString,
			nullCount:       goMemPartFieldBinlogNullCount,
		})
		collectFieldIndexInfoRuntimeMemoryStats(aggregates, info.IndexInfo)
		return true
	})

	segment.fieldJSONStatsMu.RLock()
	addGoSegmentRuntimeMetric(aggregates, goMemPartFieldJSONStatsMapPayload, int64(len(segment.fieldJSONStats)), int64(len(segment.fieldJSONStats))*(int64(unsafe.Sizeof(int64(0)))+int64(unsafe.Sizeof(uintptr(0)))))
	for _, info := range segment.fieldJSONStats {
		if info != nil {
			addGoSegmentRuntimeMetric(aggregates, goMemPartFieldJSONStatsObject, 1, int64(unsafe.Sizeof(querypb.JsonStatsInfo{})))
		}
	}
	segment.fieldJSONStatsMu.RUnlock()
}

func collectFieldIndexInfoRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, info *querypb.FieldIndexInfo) {
	if info == nil {
		return
	}
	addGoSegmentRuntimeMetric(aggregates, goMemPartFieldIndexInfoObject, 1, int64(unsafe.Sizeof(querypb.FieldIndexInfo{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartFieldIndexInfoNameString, 1, int64(len(info.GetIndexName())))
	addGoSegmentRuntimeMetric(aggregates, goMemPartFieldIndexInfoParamSlice, int64(len(info.GetIndexParams())), slicePointerBackingBytes(cap(info.GetIndexParams())))
	addGoSegmentRuntimeMetric(aggregates, goMemPartFieldIndexInfoFilePathSlice, int64(len(info.GetIndexFilePaths())), sliceStringBackingBytes(cap(info.GetIndexFilePaths())))
	for _, pair := range info.GetIndexParams() {
		if pair == nil {
			continue
		}
		addGoSegmentRuntimeMetric(aggregates, goMemPartIndexParamKeyValueObject, 1, int64(unsafe.Sizeof(commonpb.KeyValuePair{})))
		addGoSegmentRuntimeMetric(aggregates, goMemPartIndexParamKeyString, 1, int64(len(pair.GetKey())))
		addGoSegmentRuntimeMetric(aggregates, goMemPartIndexParamValueString, 1, int64(len(pair.GetValue())))
	}
	for _, path := range info.GetIndexFilePaths() {
		addGoSegmentRuntimeMetric(aggregates, goMemPartFieldIndexInfoFilePathString, 1, int64(len(path)))
	}
}

type fieldBinlogRuntimeParts struct {
	object          string
	binlogSlice     string
	childFieldSlice string
	binlogObject    string
	pathString      string
	nullCount       string
}

func collectFieldBinlogRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, fieldBinlog *datapb.FieldBinlog, parts fieldBinlogRuntimeParts) {
	if fieldBinlog == nil {
		return
	}
	addGoSegmentRuntimeMetric(aggregates, parts.object, 1, int64(unsafe.Sizeof(datapb.FieldBinlog{})))
	addGoSegmentRuntimeMetric(aggregates, parts.binlogSlice, int64(len(fieldBinlog.GetBinlogs())), slicePointerBackingBytes(cap(fieldBinlog.GetBinlogs())))
	addGoSegmentRuntimeMetric(aggregates, parts.childFieldSlice, int64(len(fieldBinlog.GetChildFields())), int64(cap(fieldBinlog.GetChildFields()))*int64(unsafe.Sizeof(int64(0))))
	for _, binlog := range fieldBinlog.GetBinlogs() {
		if binlog == nil {
			continue
		}
		addGoSegmentRuntimeMetric(aggregates, parts.binlogObject, 1, int64(unsafe.Sizeof(datapb.Binlog{})))
		addGoSegmentRuntimeMetric(aggregates, parts.pathString, 1, int64(len(binlog.GetLogPath())))
		// FieldNullCounts map buckets are excluded; count key/value payload only.
		addGoSegmentRuntimeMetric(aggregates, parts.nullCount, int64(len(binlog.GetFieldNullCounts())), int64(len(binlog.GetFieldNullCounts()))*2*int64(unsafe.Sizeof(int64(0))))
	}
}

func collectBM25StatsRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, segment *LocalSegment) {
	addGoSegmentRuntimeMetric(aggregates, goMemPartBM25StatsMapPayload, int64(len(segment.bm25Stats)), int64(len(segment.bm25Stats))*(int64(unsafe.Sizeof(int64(0)))+int64(unsafe.Sizeof(uintptr(0)))))
}

func collectResourceUsageCacheRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, segment *LocalSegment) {
	usage := segment.resourceUsageCache.Load()
	if usage == nil {
		return
	}
	addGoSegmentRuntimeMetric(aggregates, goMemPartResourceUsageCacheObject, 1, int64(unsafe.Sizeof(ResourceUsage{})))
	addGoSegmentRuntimeMetric(aggregates, goMemPartResourceUsageCacheSlice, int64(len(usage.FieldGpuMemorySize)), int64(cap(usage.FieldGpuMemorySize))*int64(unsafe.Sizeof(uint64(0))))
}

func collectManifestLazyKeyRuntimeMemoryStats(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, segment *LocalSegment) {
	bytes := int64(len(segment.manifestDeltaKey) + len(segment.manifestStatsKey))
	count := int64(0)
	if segment.manifestDeltaKey != "" {
		count++
	}
	if segment.manifestStatsKey != "" {
		count++
	}
	addGoSegmentRuntimeMetric(aggregates, goMemPartManifestLazyKeyString, count, bytes)
}

func addGoSegmentRuntimeMetric(aggregates map[goSegmentRuntimeMetricKey]goSegmentRuntimeMetricValue, part string, count int64, bytes int64) {
	if count == 0 && bytes == 0 {
		return
	}
	key := goSegmentRuntimeMetricKey{part: part}
	value := aggregates[key]
	value.count += count
	value.bytes += bytes
	aggregates[key] = value
}

func loadStateLockRuntimeBytes() int64 {
	return int64(unsafe.Sizeof(state.LoadStateLock{})) +
		int64(unsafe.Sizeof(sync.RWMutex{})) +
		int64(unsafe.Sizeof(atomic.Int32{}))
}

func segmentAtomicRuntimeBytes() int64 {
	return 6*int64(unsafe.Sizeof(atomic.Int64{})) +
		int64(unsafe.Sizeof(atomic.Uint64{})) +
		int64(unsafe.Sizeof(atomic.Pointer[querypb.SegmentLoadInfo]{})) +
		int64(unsafe.Sizeof(atomic.Pointer[ResourceUsage]{}))
}

func segmentContainerShellRuntimeBytes() int64 {
	return 2*int64(unsafe.Sizeof(typeutil.ConcurrentMap[int64, *FieldInfo]{})) +
		int64(unsafe.Sizeof(map[int64]*querypb.JsonStatsInfo{}))
}

func segmentManagerReferenceRuntimeBytes(count int64) int64 {
	return count * (int64(unsafe.Sizeof(int64(0))) + int64(unsafe.Sizeof(any(nil))))
}

func slicePointerBackingBytes(capacity int) int64 {
	return int64(capacity) * int64(unsafe.Sizeof(uintptr(0)))
}

func sliceStringBackingBytes(capacity int) int64 {
	return int64(capacity) * int64(unsafe.Sizeof(""))
}
