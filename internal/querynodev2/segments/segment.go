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

/*
#cgo pkg-config: milvus_core

#include "futures/future_c.h"
#include "segcore/collection_c.h"
#include "segcore/load_index_c.h"
#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"
#include "segcore/segment_c.h"
#include "common/init_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/cockroachdb/errors"
	"go.opentelemetry.io/otel"
	"go.uber.org/atomic"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/pkoracle"
	"github.com/milvus-io/milvus/internal/querynodev2/segments/state"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/cgopb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexcgopb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/segcorepb"
	"github.com/milvus-io/milvus/pkg/v3/util/contextutil"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/indexparams"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type SegmentType = commonpb.SegmentState

const (
	SegmentTypeGrowing = commonpb.SegmentState_Growing
	SegmentTypeSealed  = commonpb.SegmentState_Sealed
)

var (
	newSegmentTiming        = newNewSegmentTimingStats()
	initializeSegmentTiming = newInitializeSegmentTimingStats()
)

type newSegmentTimingStats struct {
	count             *atomic.Int64
	totalBase         *atomic.Int64
	totalCreateSubmit *atomic.Int64
	totalCreateCgo    *atomic.Int64
	totalStruct       *atomic.Int64
	totalInitialize   *atomic.Int64
	totalSegment      *atomic.Int64
	maxBase           *atomic.Int64
	maxCreateSubmit   *atomic.Int64
	maxCreateCgo      *atomic.Int64
	maxStruct         *atomic.Int64
	maxInitialize     *atomic.Int64
	maxSegment        *atomic.Int64
	lastLogUnixNano   *atomic.Int64
}

func newNewSegmentTimingStats() *newSegmentTimingStats {
	return &newSegmentTimingStats{
		count:             atomic.NewInt64(0),
		totalBase:         atomic.NewInt64(0),
		totalCreateSubmit: atomic.NewInt64(0),
		totalCreateCgo:    atomic.NewInt64(0),
		totalStruct:       atomic.NewInt64(0),
		totalInitialize:   atomic.NewInt64(0),
		totalSegment:      atomic.NewInt64(0),
		maxBase:           atomic.NewInt64(0),
		maxCreateSubmit:   atomic.NewInt64(0),
		maxCreateCgo:      atomic.NewInt64(0),
		maxStruct:         atomic.NewInt64(0),
		maxInitialize:     atomic.NewInt64(0),
		maxSegment:        atomic.NewInt64(0),
		lastLogUnixNano:   atomic.NewInt64(time.Now().UnixNano()),
	}
}

func (s *newSegmentTimingStats) record(baseDur, createSubmitDur, createCgoDur, structDur, initializeDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalBase.Add(int64(baseDur))
	s.totalCreateSubmit.Add(int64(createSubmitDur))
	s.totalCreateCgo.Add(int64(createCgoDur))
	s.totalStruct.Add(int64(structDur))
	s.totalInitialize.Add(int64(initializeDur))
	s.totalSegment.Add(int64(totalDur))
	updateMaxDuration(s.maxBase, baseDur)
	updateMaxDuration(s.maxCreateSubmit, createSubmitDur)
	updateMaxDuration(s.maxCreateCgo, createCgoDur)
	updateMaxDuration(s.maxStruct, structDur)
	updateMaxDuration(s.maxInitialize, initializeDur)
	updateMaxDuration(s.maxSegment, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}
	windowDur := time.Duration(now.UnixNano() - last)

	count := s.count.Swap(0)
	totalBase := s.totalBase.Swap(0)
	totalCreateSubmit := s.totalCreateSubmit.Swap(0)
	totalCreateCgo := s.totalCreateCgo.Swap(0)
	totalStruct := s.totalStruct.Swap(0)
	totalInitialize := s.totalInitialize.Swap(0)
	totalSegment := s.totalSegment.Swap(0)
	maxBase := s.maxBase.Swap(0)
	maxCreateSubmit := s.maxCreateSubmit.Swap(0)
	maxCreateCgo := s.maxCreateCgo.Swap(0)
	maxStruct := s.maxStruct.Swap(0)
	maxInitialize := s.maxInitialize.Swap(0)
	maxSegment := s.maxSegment.Swap(0)
	if count == 0 {
		return
	}

	totalAccounted := totalBase + totalCreateSubmit + totalStruct + totalInitialize
	log.Warn("[SN recovery] load timing stats",
		zap.String("phase", "segment.new_segment"),
		zap.Duration("windowDur", windowDur),
		zap.Int64("count", count),
		zap.Duration("avgBaseDur", avgDuration(totalBase, count)),
		zap.Duration("avgCreateSubmitDur", avgDuration(totalCreateSubmit, count)),
		zap.Duration("avgCreateCgoDur", avgDuration(totalCreateCgo, count)),
		zap.Duration("avgCreateQueueDur", avgDuration(totalCreateSubmit-totalCreateCgo, count)),
		zap.Duration("avgStructDur", avgDuration(totalStruct, count)),
		zap.Duration("avgInitializeDur", avgDuration(totalInitialize, count)),
		zap.Duration("avgOtherDur", avgDuration(totalSegment-totalAccounted, count)),
		zap.Duration("avgTotalDur", avgDuration(totalSegment, count)),
		zap.Duration("maxBaseDur", time.Duration(maxBase)),
		zap.Duration("maxCreateSubmitDur", time.Duration(maxCreateSubmit)),
		zap.Duration("maxCreateCgoDur", time.Duration(maxCreateCgo)),
		zap.Duration("maxStructDur", time.Duration(maxStruct)),
		zap.Duration("maxInitializeDur", time.Duration(maxInitialize)),
		zap.Duration("maxTotalDur", time.Duration(maxSegment)),
	)
}

type initializeSegmentTimingStats struct {
	count                *atomic.Int64
	totalIndexInfo       *atomic.Int64
	totalFieldBinlog     *atomic.Int64
	totalSeparate        *atomic.Int64
	totalSchemaHelper    *atomic.Int64
	totalIndexLoop       *atomic.Int64
	totalSchemaGet       *atomic.Int64
	totalInsertIndex     *atomic.Int64
	totalHasRawData      *atomic.Int64
	totalInsertField     *atomic.Int64
	totalFieldBinlogLoop *atomic.Int64
	totalStoreInsert     *atomic.Int64
	totalInitialize      *atomic.Int64
	maxInitialize        *atomic.Int64
	lastLogUnixNano      *atomic.Int64
}

func newInitializeSegmentTimingStats() *initializeSegmentTimingStats {
	return &initializeSegmentTimingStats{
		count:                atomic.NewInt64(0),
		totalIndexInfo:       atomic.NewInt64(0),
		totalFieldBinlog:     atomic.NewInt64(0),
		totalSeparate:        atomic.NewInt64(0),
		totalSchemaHelper:    atomic.NewInt64(0),
		totalIndexLoop:       atomic.NewInt64(0),
		totalSchemaGet:       atomic.NewInt64(0),
		totalInsertIndex:     atomic.NewInt64(0),
		totalHasRawData:      atomic.NewInt64(0),
		totalInsertField:     atomic.NewInt64(0),
		totalFieldBinlogLoop: atomic.NewInt64(0),
		totalStoreInsert:     atomic.NewInt64(0),
		totalInitialize:      atomic.NewInt64(0),
		maxInitialize:        atomic.NewInt64(0),
		lastLogUnixNano:      atomic.NewInt64(time.Now().UnixNano()),
	}
}

func (s *initializeSegmentTimingStats) record(indexInfoCount, fieldBinlogCount int, separateDur, schemaHelperDur, indexLoopDur, schemaGetDur, insertIndexDur, hasRawDataDur, insertFieldDur, fieldBinlogLoopDur, storeInsertDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalIndexInfo.Add(int64(indexInfoCount))
	s.totalFieldBinlog.Add(int64(fieldBinlogCount))
	s.totalSeparate.Add(int64(separateDur))
	s.totalSchemaHelper.Add(int64(schemaHelperDur))
	s.totalIndexLoop.Add(int64(indexLoopDur))
	s.totalSchemaGet.Add(int64(schemaGetDur))
	s.totalInsertIndex.Add(int64(insertIndexDur))
	s.totalHasRawData.Add(int64(hasRawDataDur))
	s.totalInsertField.Add(int64(insertFieldDur))
	s.totalFieldBinlogLoop.Add(int64(fieldBinlogLoopDur))
	s.totalStoreInsert.Add(int64(storeInsertDur))
	s.totalInitialize.Add(int64(totalDur))
	updateMaxDuration(s.maxInitialize, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}
	windowDur := time.Duration(now.UnixNano() - last)

	count := s.count.Swap(0)
	totalIndexInfo := s.totalIndexInfo.Swap(0)
	totalFieldBinlog := s.totalFieldBinlog.Swap(0)
	totalSeparate := s.totalSeparate.Swap(0)
	totalSchemaHelper := s.totalSchemaHelper.Swap(0)
	totalIndexLoop := s.totalIndexLoop.Swap(0)
	totalSchemaGet := s.totalSchemaGet.Swap(0)
	totalInsertIndex := s.totalInsertIndex.Swap(0)
	totalHasRawData := s.totalHasRawData.Swap(0)
	totalInsertField := s.totalInsertField.Swap(0)
	totalFieldBinlogLoop := s.totalFieldBinlogLoop.Swap(0)
	totalStoreInsert := s.totalStoreInsert.Swap(0)
	totalInitialize := s.totalInitialize.Swap(0)
	maxInitialize := s.maxInitialize.Swap(0)
	if count == 0 {
		return
	}

	totalAccounted := totalSeparate + totalSchemaHelper + totalIndexLoop + totalFieldBinlogLoop + totalStoreInsert
	log.Warn("[SN recovery] load timing stats",
		zap.String("phase", "segment.initialize"),
		zap.Duration("windowDur", windowDur),
		zap.Int64("count", count),
		zap.Int64("avgIndexInfoCount", totalIndexInfo/count),
		zap.Int64("avgFieldBinlogCount", totalFieldBinlog/count),
		zap.Duration("avgSeparateDur", avgDuration(totalSeparate, count)),
		zap.Duration("avgSchemaHelperDur", avgDuration(totalSchemaHelper, count)),
		zap.Duration("avgIndexLoopDur", avgDuration(totalIndexLoop, count)),
		zap.Duration("avgSchemaGetDur", avgDuration(totalSchemaGet, count)),
		zap.Duration("avgInsertIndexDur", avgDuration(totalInsertIndex, count)),
		zap.Duration("avgHasRawDataDur", avgDuration(totalHasRawData, count)),
		zap.Duration("avgInsertFieldDur", avgDuration(totalInsertField, count)),
		zap.Duration("avgFieldBinlogLoopDur", avgDuration(totalFieldBinlogLoop, count)),
		zap.Duration("avgStoreInsertDur", avgDuration(totalStoreInsert, count)),
		zap.Duration("avgOtherDur", avgDuration(totalInitialize-totalAccounted, count)),
		zap.Duration("avgTotalDur", avgDuration(totalInitialize, count)),
		zap.Duration("maxTotalDur", time.Duration(maxInitialize)),
	)
}

var ErrSegmentUnhealthy = errors.New("segment unhealthy")

// IndexedFieldInfo contains binlog info of vector field
type IndexedFieldInfo struct {
	FieldBinlog *datapb.FieldBinlog
	IndexInfo   *querypb.FieldIndexInfo
	IsLoaded    bool
}

type baseSegment struct {
	collection *Collection
	version    *atomic.Int64

	segmentType   SegmentType
	pkCandidate   pkoracle.Candidate // PK candidate: BloomFilterSet for regular collections, ExternalSegmentCandidate for external collections
	loadInfo      *atomic.Pointer[querypb.SegmentLoadInfo]
	skipGrowingBF bool // Skip generating or maintaining BF for growing segments; deletion checks will be handled in segcore.
	channel       metautil.Channel

	bm25Stats map[int64]*storage.BM25Stats

	resourceUsageCache *atomic.Pointer[ResourceUsage]

	needUpdatedVersion *atomic.Int64 // only for lazy load mode update index
}

func newBaseSegment(collection *Collection, segmentType SegmentType, version int64, loadInfo *querypb.SegmentLoadInfo) (baseSegment, error) {
	channel, err := metautil.ParseChannel(loadInfo.GetInsertChannel(), channelMapper)
	if err != nil {
		return baseSegment{}, err
	}
	bs := baseSegment{
		collection:    collection,
		loadInfo:      atomic.NewPointer[querypb.SegmentLoadInfo](loadInfo),
		version:       atomic.NewInt64(version),
		segmentType:   segmentType,
		pkCandidate:   pkoracle.NewBloomFilterSet(loadInfo.GetSegmentID(), loadInfo.GetPartitionID(), segmentType),
		bm25Stats:     make(map[int64]*storage.BM25Stats),
		channel:       channel,
		skipGrowingBF: segmentType == SegmentTypeGrowing && paramtable.Get().QueryNodeCfg.SkipGrowingSegmentBF.GetAsBool(),

		resourceUsageCache: atomic.NewPointer[ResourceUsage](nil),
		needUpdatedVersion: atomic.NewInt64(0),
	}
	return bs, nil
}

// ID returns the identity number.
func (s *baseSegment) ID() int64 {
	return s.loadInfo.Load().GetSegmentID()
}

func (s *baseSegment) Collection() int64 {
	return s.loadInfo.Load().GetCollectionID()
}

func (s *baseSegment) GetCollection() *Collection {
	return s.collection
}

func (s *baseSegment) Partition() int64 {
	return s.loadInfo.Load().GetPartitionID()
}

func (s *baseSegment) DatabaseName() string {
	return s.collection.GetDBName()
}

func (s *baseSegment) ResourceGroup() string {
	return s.collection.GetResourceGroup()
}

func (s *baseSegment) Shard() metautil.Channel {
	return s.channel
}

func (s *baseSegment) Type() SegmentType {
	return s.segmentType
}

func (s *baseSegment) Level() datapb.SegmentLevel {
	return s.loadInfo.Load().GetLevel()
}

func (s *baseSegment) IsSorted() bool {
	return s.loadInfo.Load().GetIsSorted()
}

func (s *baseSegment) StartPosition() *msgpb.MsgPosition {
	return s.loadInfo.Load().GetStartPosition()
}

func (s *baseSegment) Version() int64 {
	return s.version.Load()
}

func (s *baseSegment) CASVersion(old, newVersion int64) bool {
	return s.version.CompareAndSwap(old, newVersion)
}

func (s *baseSegment) LoadInfo() *querypb.SegmentLoadInfo {
	return s.loadInfo.Load()
}

func cloneMsgPosition(position *msgpb.MsgPosition) *msgpb.MsgPosition {
	if position == nil {
		return nil
	}
	return proto.Clone(position).(*msgpb.MsgPosition)
}

func compactSegmentLoadInfoForRuntime(loadInfo *querypb.SegmentLoadInfo) *querypb.SegmentLoadInfo {
	if loadInfo == nil {
		return nil
	}
	return &querypb.SegmentLoadInfo{
		SegmentID:       loadInfo.GetSegmentID(),
		PartitionID:     loadInfo.GetPartitionID(),
		CollectionID:    loadInfo.GetCollectionID(),
		NumOfRows:       loadInfo.GetNumOfRows(),
		SegmentSize:     loadInfo.GetSegmentSize(),
		InsertChannel:   loadInfo.GetInsertChannel(),
		StartPosition:   cloneMsgPosition(loadInfo.GetStartPosition()),
		ReadableVersion: loadInfo.GetReadableVersion(),
		Level:           loadInfo.GetLevel(),
		StorageVersion:  loadInfo.GetStorageVersion(),
		IsSorted:        loadInfo.GetIsSorted(),
		Priority:        loadInfo.GetPriority(),
		ManifestPath:    loadInfo.GetManifestPath(),
		DataVersion:     loadInfo.GetDataVersion(),
		CommitTimestamp: loadInfo.GetCommitTimestamp(),
	}
}

func (s *baseSegment) compactLoadInfoForRuntime() {
	if s.segmentType != SegmentTypeSealed || s.Level() == datapb.SegmentLevel_L0 {
		return
	}
	s.ResourceUsageEstimate()
	s.loadInfo.Store(compactSegmentLoadInfoForRuntime(s.LoadInfo()))
}

func (s *baseSegment) SetPKCandidate(candidate pkoracle.Candidate) {
	s.pkCandidate = candidate
}

// PkCandidateExist implements pkoracle.Candidate — reports whether PK data has been loaded.
func (s *baseSegment) PkCandidateExist() bool {
	return s.pkCandidate != nil && s.pkCandidate.PkCandidateExist()
}

// UpdatePkCandidate feeds new primary keys into the PK candidate.
func (s *baseSegment) UpdatePkCandidate(pks []storage.PrimaryKey) {
	if s.skipGrowingBF {
		return
	}
	if s.pkCandidate != nil {
		s.pkCandidate.UpdatePkCandidate(pks)
	}
}

// Stats implements pkoracle.Candidate — returns PK statistics (min/max PK).
func (s *baseSegment) Stats() *storage.PkStatistics {
	if s.pkCandidate != nil {
		return s.pkCandidate.Stats()
	}
	return nil
}

// Charge implements pkoracle.Candidate — charges memory resources.
func (s *baseSegment) Charge() {
	if s.pkCandidate != nil {
		s.pkCandidate.Charge()
	}
}

// Refund implements pkoracle.Candidate — releases memory resources.
func (s *baseSegment) Refund() {
	if s.pkCandidate != nil {
		s.pkCandidate.Refund()
	}
}

func (s *baseSegment) UpdateBM25Stats(stats map[int64]*storage.BM25Stats) {
	for fieldID, new := range stats {
		if current, ok := s.bm25Stats[fieldID]; ok {
			current.Merge(new)
		} else {
			s.bm25Stats[fieldID] = new
		}
	}
}

func (s *baseSegment) GetBM25Stats() map[int64]*storage.BM25Stats {
	return s.bm25Stats
}

// MayPkExist returns true if the given PK exists in the PK range and being positive through the bloom filter,
// false otherwise,
// may returns true even the PK doesn't exist actually
func (s *baseSegment) MayPkExist(pk *storage.LocationsCache) bool {
	if s.skipGrowingBF {
		return true
	}
	if s.pkCandidate == nil {
		return true // No candidate, assume PK might exist
	}
	return s.pkCandidate.MayPkExist(pk)
}

func (s *baseSegment) GetMinPk() *storage.PrimaryKey {
	if s.pkCandidate != nil {
		if stats := s.pkCandidate.Stats(); stats != nil {
			return &stats.MinPK
		}
	}
	return nil
}

func (s *baseSegment) GetMaxPk() *storage.PrimaryKey {
	if s.pkCandidate != nil {
		if stats := s.pkCandidate.Stats(); stats != nil {
			return &stats.MaxPK
		}
	}
	return nil
}

func (s *baseSegment) BatchPkExist(lc *storage.BatchLocationsCache) []bool {
	if s.skipGrowingBF {
		allPositive := make([]bool, lc.Size())
		for i := 0; i < lc.Size(); i++ {
			allPositive[i] = true
		}
		return allPositive
	}
	if s.pkCandidate == nil {
		allPositive := make([]bool, lc.Size())
		for i := 0; i < lc.Size(); i++ {
			allPositive[i] = true
		}
		return allPositive
	}
	return s.pkCandidate.BatchPkExist(lc)
}

// ResourceUsageEstimate returns the final estimated resource usage of the segment.
func (s *baseSegment) ResourceUsageEstimate() ResourceUsage {
	if s.segmentType == SegmentTypeGrowing {
		// Growing segment cannot do resource usage estimate.
		return ResourceUsage{}
	}
	cache := s.resourceUsageCache.Load()
	if cache != nil {
		return *cache
	}

	usage, err := estimateLogicalResourceUsageOfSegment(s.collection.Schema(), s.LoadInfo(), resourceEstimateFactor{
		deltaDataExpansionFactor:        paramtable.Get().QueryNodeCfg.DeltaDataExpansionRate.GetAsFloat(),
		TieredEvictionEnabled:           paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool(),
		TieredEvictableMemoryCacheRatio: paramtable.Get().QueryNodeCfg.TieredEvictableMemoryCacheRatio.GetAsFloat(),
		TieredEvictableDiskCacheRatio:   paramtable.Get().QueryNodeCfg.TieredEvictableDiskCacheRatio.GetAsFloat(),
	})
	if err != nil {
		// Should never failure, if failed, segment should never be loaded.
		log.Warn("unreachable: failed to get resource usage estimate of segment", zap.Error(err), zap.Int64("collectionID", s.Collection()), zap.Int64("segmentID", s.ID()))
		return ResourceUsage{}
	}
	s.resourceUsageCache.Store(usage)
	return *usage
}

func (s *baseSegment) NeedUpdatedVersion() int64 {
	return s.needUpdatedVersion.Load()
}

func (s *baseSegment) SetLoadInfo(loadInfo *querypb.SegmentLoadInfo) {
	s.resourceUsageCache.Store(nil)
	s.loadInfo.Store(loadInfo)
}

func (s *baseSegment) SetNeedUpdatedVersion(version int64) {
	s.needUpdatedVersion.Store(version)
}

type FieldInfo struct {
	*datapb.FieldBinlog
	RowCount int64
}

var _ Segment = (*LocalSegment)(nil)

// Segment is a wrapper of the underlying C-structure segment.
type LocalSegment struct {
	baseSegment
	manager SegmentManager
	ptrLock *state.LoadStateLock
	ptr     C.CSegmentInterface // TODO: Remove in future, after move load index into segcore package.
	// always keep same with csegment.RawPtr(), for eaiser to access,
	csegment segcore.CSegment

	// cached results, to avoid too many CGO calls
	memSize     *atomic.Int64
	binlogSize  *atomic.Int64
	rowNum      *atomic.Int64
	insertCount *atomic.Int64

	deltaMut           sync.Mutex
	lastDeltaTimestamp *atomic.Uint64
	fields             *typeutil.ConcurrentMap[int64, *FieldInfo]
	fieldIndexes       *typeutil.ConcurrentMap[int64, *IndexedFieldInfo] // indexID -> IndexedFieldInfo
	fieldJSONStats     map[int64]*querypb.JsonStatsInfo
	fieldJSONStatsMu   sync.RWMutex
}

func NewSegment(ctx context.Context,
	collection *Collection,
	manager SegmentManager,
	segmentType SegmentType,
	version int64,
	loadInfo *querypb.SegmentLoadInfo,
) (Segment, error) {
	log := log.Ctx(ctx)
	segmentStart := time.Now()
	var baseDur time.Duration
	var createSubmitDur time.Duration
	var createCgoDur time.Duration
	var structDur time.Duration
	var initializeDur time.Duration
	defer func() {
		newSegmentTiming.record(baseDur, createSubmitDur, createCgoDur, structDur, initializeDur, time.Since(segmentStart))
	}()
	/*
		CStatus
		NewSegment(CCollection collection, uint64_t segment_id, SegmentType seg_type, CSegmentInterface* newSegment);
	*/
	if loadInfo.GetLevel() == datapb.SegmentLevel_L0 {
		return NewL0Segment(collection, segmentType, version, loadInfo)
	}

	stageStart := time.Now()
	base, err := newBaseSegment(collection, segmentType, version, loadInfo)
	baseDur = time.Since(stageStart)
	if err != nil {
		return nil, err
	}

	var locker *state.LoadStateLock
	switch segmentType {
	case SegmentTypeSealed:
		locker = state.NewLoadStateLock(state.LoadStateOnlyMeta)
	case SegmentTypeGrowing:
		locker = state.NewLoadStateLock(state.LoadStateDataLoaded)
	default:
		return nil, fmt.Errorf("illegal segment type %d when create segment %d", segmentType, loadInfo.GetSegmentID())
	}

	logger := log.With(
		zap.Int64("collectionID", loadInfo.GetCollectionID()),
		zap.Int64("partitionID", loadInfo.GetPartitionID()),
		zap.Int64("segmentID", loadInfo.GetSegmentID()),
		zap.String("segmentType", segmentType.String()),
		zap.String("level", loadInfo.GetLevel().String()),
	)

	var csegment segcore.CSegment
	stageStart = time.Now()
	if _, err := GetDynamicPool().Submit(func() (any, error) {
		var err error
		cgoStart := time.Now()
		csegment, err = segcore.CreateCSegment(&segcore.CreateCSegmentRequest{
			Collection:  collection.ccollection,
			SegmentID:   loadInfo.GetSegmentID(),
			SegmentType: segmentType,
			IsSorted:    loadInfo.GetIsSorted(),
			LoadInfo:    loadInfo,
		})
		createCgoDur = time.Since(cgoStart)
		return nil, err
	}).Await(); err != nil {
		createSubmitDur = time.Since(stageStart)
		logger.Warn("create segment failed", zap.Error(err))
		return nil, err
	}
	createSubmitDur = time.Since(stageStart)
	logger.Info("create segment done")

	stageStart = time.Now()
	segment := &LocalSegment{
		baseSegment:        base,
		manager:            manager,
		ptrLock:            locker,
		ptr:                C.CSegmentInterface(csegment.RawPointer()),
		csegment:           csegment,
		lastDeltaTimestamp: atomic.NewUint64(0),
		fields:             typeutil.NewConcurrentMap[int64, *FieldInfo](),
		fieldIndexes:       typeutil.NewConcurrentMap[int64, *IndexedFieldInfo](),
		fieldJSONStats:     make(map[int64]*querypb.JsonStatsInfo),

		memSize:     atomic.NewInt64(-1),
		binlogSize:  atomic.NewInt64(0),
		rowNum:      atomic.NewInt64(-1),
		insertCount: atomic.NewInt64(0),
	}
	structDur = time.Since(stageStart)

	stageStart = time.Now()
	if err := segment.initializeSegment(); err != nil {
		initializeDur = time.Since(stageStart)
		csegment.Release()
		return nil, err
	}
	initializeDur = time.Since(stageStart)
	return segment, nil
}

func (s *LocalSegment) initializeSegment() error {
	initializeStart := time.Now()
	var separateDur time.Duration
	var schemaHelperDur time.Duration
	var indexLoopDur time.Duration
	var schemaGetDur time.Duration
	var insertIndexDur time.Duration
	var hasRawDataDur time.Duration
	var insertFieldDur time.Duration
	var fieldBinlogLoopDur time.Duration
	var storeInsertDur time.Duration
	loadInfo := s.loadInfo.Load()
	stageStart := time.Now()
	indexedFieldInfos, fieldBinlogs := separateIndexAndBinlog(loadInfo)
	separateDur = time.Since(stageStart)
	stageStart = time.Now()
	schemaHelper, _ := typeutil.CreateSchemaHelper(s.collection.Schema())
	schemaHelperDur = time.Since(stageStart)

	indexLoopStart := time.Now()
	for _, info := range indexedFieldInfos {
		fieldID := info.IndexInfo.FieldID
		stageStart = time.Now()
		field, err := schemaHelper.GetFieldFromID(fieldID)
		schemaGetDur += time.Since(stageStart)
		if err != nil {
			return err
		}
		indexInfo := info.IndexInfo
		stageStart = time.Now()
		s.fieldIndexes.Insert(indexInfo.GetIndexID(), &IndexedFieldInfo{
			FieldBinlog: &datapb.FieldBinlog{
				FieldID: indexInfo.GetFieldID(),
			},
			IndexInfo: indexInfo,
			IsLoaded:  false,
		})
		insertIndexDur += time.Since(stageStart)
		hasRawData := false
		if !typeutil.IsVectorType(field.GetDataType()) {
			stageStart = time.Now()
			hasRawData = s.HasRawData(fieldID)
			hasRawDataDur += time.Since(stageStart)
		}
		if !typeutil.IsVectorType(field.GetDataType()) && !hasRawData {
			stageStart = time.Now()
			s.fields.Insert(fieldID, &FieldInfo{
				FieldBinlog: info.FieldBinlog,
				RowCount:    loadInfo.GetNumOfRows(),
			})
			insertFieldDur += time.Since(stageStart)
		}
	}
	indexLoopDur = time.Since(indexLoopStart)

	stageStart = time.Now()
	for _, binlogs := range fieldBinlogs {
		s.fields.Insert(binlogs.FieldID, &FieldInfo{
			FieldBinlog: binlogs,
			RowCount:    loadInfo.GetNumOfRows(),
		})
	}
	fieldBinlogLoopDur = time.Since(stageStart)

	// Update the insert count when initialize the segment and update the metrics.
	stageStart = time.Now()
	s.insertCount.Store(loadInfo.GetNumOfRows())
	storeInsertDur = time.Since(stageStart)
	initializeSegmentTiming.record(len(indexedFieldInfos), len(fieldBinlogs), separateDur, schemaHelperDur, indexLoopDur, schemaGetDur, insertIndexDur, hasRawDataDur, insertFieldDur, fieldBinlogLoopDur, storeInsertDur, time.Since(initializeStart))
	return nil
}

// PinIfNotReleased acquires the `ptrLock` and returns true if the pointer is valid
// Provide ONLY the read lock operations,
// don't make `ptrLock` public to avoid abusing of the mutex.
func (s *LocalSegment) PinIfNotReleased() error {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	return nil
}

func (s *LocalSegment) Unpin() {
	s.ptrLock.Unpin()
}

func (s *LocalSegment) InsertCount() int64 {
	return s.insertCount.Load()
}

func (s *LocalSegment) RowNum() int64 {
	// if segment is not loaded, return 0 (maybe not loaded or release by lru)
	if !s.ptrLock.PinIf(state.IsDataLoaded) {
		return 0
	}
	defer s.ptrLock.Unpin()

	rowNum := s.rowNum.Load()
	if rowNum < 0 {
		GetDynamicPool().Submit(func() (any, error) {
			rowNum = s.csegment.RowNum()
			s.rowNum.Store(rowNum)
			return nil, nil
		}).Await()
	}
	return rowNum
}

func (s *LocalSegment) MemSize() int64 {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return 0
	}
	defer s.ptrLock.Unpin()

	memSize := s.memSize.Load()
	if memSize < 0 {
		GetDynamicPool().Submit(func() (any, error) {
			memSize = s.csegment.MemSize()
			s.memSize.Store(memSize)
			return nil, nil
		}).Await()
	}
	return memSize
}

func (s *LocalSegment) LastDeltaTimestamp() uint64 {
	return s.lastDeltaTimestamp.Load()
}

// advanceLastDeltaTimestamp moves lastDeltaTimestamp forward to max(current, max(tss)).
// Consumers (file-level skip in segment_loader.LoadDeltaLogs, dist_handler reporting to
// QueryCoord) treat this field as a high-water-mark. Using tss[last] on unsorted batches
// underestimates the watermark, so we scan for the true max.
func (s *LocalSegment) advanceLastDeltaTimestamp(tss []typeutil.Timestamp) {
	if len(tss) == 0 {
		return
	}
	maxTs := tss[0]
	for _, t := range tss[1:] {
		if t > maxTs {
			maxTs = t
		}
	}
	for {
		cur := s.lastDeltaTimestamp.Load()
		if maxTs <= cur {
			return
		}
		if s.lastDeltaTimestamp.CompareAndSwap(cur, maxTs) {
			return
		}
	}
}

// UpdatePkCandidate updates the PK candidate with provided pks and charges resource.
// Overrides baseSegment.UpdatePkCandidate to handle resource charging for growing segments.
func (s *LocalSegment) UpdatePkCandidate(pks []storage.PrimaryKey) {
	if s.skipGrowingBF {
		return
	}

	s.pkCandidate.UpdatePkCandidate(pks)

	// Charge resource (safe to call multiple times - only charges once)
	s.pkCandidate.Charge()
}

func (s *LocalSegment) GetIndexByID(indexID int64) *IndexedFieldInfo {
	info, _ := s.fieldIndexes.Get(indexID)
	return info
}

func (s *LocalSegment) GetIndex(fieldID int64) []*IndexedFieldInfo {
	var info []*IndexedFieldInfo
	s.fieldIndexes.Range(func(key int64, value *IndexedFieldInfo) bool {
		if value.IndexInfo.FieldID == fieldID {
			info = append(info, value)
		}
		return true
	})
	return info
}

func (s *LocalSegment) ExistIndex(fieldID int64) bool {
	contain := false
	s.fieldIndexes.Range(func(key int64, value *IndexedFieldInfo) bool {
		if value.IndexInfo.FieldID == fieldID {
			contain = true
		}
		return !contain
	})

	return contain
}

func (s *LocalSegment) HasRawData(fieldID int64) bool {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return false
	}
	defer s.ptrLock.Unpin()

	return s.csegment.HasRawData(fieldID)
}

func (s *LocalSegment) HasFieldData(fieldID int64) bool {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return false
	}
	defer s.ptrLock.Unpin()
	return s.csegment.HasFieldData(fieldID)
}

func (s *LocalSegment) DropIndex(ctx context.Context, indexID int64) error {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	if indexInfo, ok := s.fieldIndexes.Get(indexID); ok {
		field := typeutil.GetField(s.collection.Schema(), indexInfo.IndexInfo.FieldID)
		if typeutil.IsJSONType(field.GetDataType()) {
			nestedPath, err := funcutil.GetAttrByKeyFromRepeatedKV(common.JSONPathKey, indexInfo.IndexInfo.GetIndexParams())
			if err != nil {
				return err
			}
			err = s.csegment.DropJSONIndex(ctx, indexInfo.IndexInfo.FieldID, nestedPath)
			if err != nil {
				return err
			}
		} else {
			err := s.csegment.DropIndex(ctx, indexInfo.IndexInfo.FieldID)
			if err != nil {
				return err
			}
		}

		s.fieldIndexes.Remove(indexID)
	}
	return nil
}

func (s *LocalSegment) Indexes() []*IndexedFieldInfo {
	var result []*IndexedFieldInfo
	s.fieldIndexes.Range(func(key int64, value *IndexedFieldInfo) bool {
		result = append(result, value)
		return true
	})
	return result
}

func (s *LocalSegment) IsLazyLoad() bool {
	for _, indexInfo := range s.Indexes() {
		if !indexInfo.IsLoaded {
			return true
		}
	}
	return false
}

func (s *LocalSegment) ResetIndexesLazyLoad(lazyState bool) {
	for _, indexInfo := range s.Indexes() {
		indexInfo.IsLoaded = lazyState
	}
}

// Search executes a search on the segment.
// If searchReq.FilterOnly() is true, only executes the filter and returns valid_count (Stage 1 of two-stage search).
func (s *LocalSegment) Search(ctx context.Context, searchReq *segcore.SearchRequest) (*segcore.SearchResult, error) {
	filterOnly := searchReq.FilterOnly()
	log := log.Ctx(ctx).WithLazy(
		zap.Uint64("mvcc", searchReq.MVCC()),
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("segmentID", s.ID()),
		zap.String("segmentType", s.segmentType.String()),
		zap.Bool("filterOnly", filterOnly),
	)

	if !s.ptrLock.PinIf(state.IsNotReleased) {
		// TODO: check if the segment is readable but not released. too many related logic need to be refactor.
		return nil, merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	hasIndex := s.ExistIndex(searchReq.SearchFieldID())
	log = log.With(zap.Bool("withIndex", hasIndex))
	log.Debug("search segment...")

	tr := timerecord.NewTimeRecorder("cgoSearch")
	result, err := s.csegment.Search(ctx, searchReq)
	if err != nil {
		log.Warn("Search failed")
		return nil, err
	}
	metrics.QueryNodeSQSegmentLatencyInCore.WithLabelValues(paramtable.GetStringNodeID(), metrics.SearchLabel).Observe(float64(tr.ElapseSpan().Microseconds()) / 1000.0)
	if filterOnly {
		log.Debug("search filter only segment done", zap.Int64("validCount", result.ValidCount()))
	} else {
		log.Debug("search segment done")
	}
	return result, nil
}

func (s *LocalSegment) retrieve(ctx context.Context, plan *segcore.RetrievePlan, log *zap.Logger) (*segcore.RetrieveResult, error) {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		// TODO: check if the segment is readable but not released. too many related logic need to be refactor.
		return nil, merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	log.Debug("begin to retrieve")

	tr := timerecord.NewTimeRecorder("cgoRetrieve")
	result, err := s.csegment.Retrieve(ctx, plan)
	if err != nil {
		log.Warn("Retrieve failed")
		return nil, err
	}
	metrics.QueryNodeSQSegmentLatencyInCore.WithLabelValues(paramtable.GetStringNodeID(),
		contextutil.GetQueryLabel(ctx)).Observe(float64(tr.ElapseSpan().Microseconds()) / 1000.0)
	return result, nil
}

func (s *LocalSegment) Retrieve(ctx context.Context, plan *segcore.RetrievePlan) (*segcorepb.RetrieveResults, error) {
	log := log.Ctx(ctx).WithLazy(
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Uint64("mvcc", plan.Timestamp),
		zap.String("segmentType", s.segmentType.String()),
	)

	result, err := s.retrieve(ctx, plan, log)
	if err != nil {
		return nil, err
	}
	defer result.Release()

	_, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "partial-segcore-results-deserialization")
	defer span.End()

	retrieveResult, err := result.GetResult()
	if err != nil {
		log.Warn("unmarshal retrieve result failed", zap.Error(err))
		return nil, err
	}
	log.Debug("retrieve segment done", zap.Int("resultNum", len(retrieveResult.Offset)))
	return retrieveResult, nil
}

func (s *LocalSegment) retrieveByOffsets(ctx context.Context, plan *segcore.RetrievePlanWithOffsets, log *zap.Logger) (*segcore.RetrieveResult, error) {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		// TODO: check if the segment is readable but not released. too many related logic need to be refactor.
		return nil, merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	log.Debug("begin to retrieve by offsets")
	tr := timerecord.NewTimeRecorder("cgoRetrieveByOffsets")
	result, err := s.csegment.RetrieveByOffsets(ctx, plan)
	if err != nil {
		log.Warn("RetrieveByOffsets failed")
		return nil, err
	}
	metrics.QueryNodeSQSegmentLatencyInCore.WithLabelValues(paramtable.GetStringNodeID(),
		contextutil.GetQueryLabel(ctx)).Observe(float64(tr.ElapseSpan().Microseconds()) / 1000.0)
	return result, nil
}

func (s *LocalSegment) RetrieveByOffsets(ctx context.Context, plan *segcore.RetrievePlanWithOffsets) (*segcorepb.RetrieveResults, error) {
	log := log.Ctx(ctx).WithLazy(zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Int64("msgID", plan.MsgID()),
		zap.String("segmentType", s.segmentType.String()),
		zap.Int("resultNum", len(plan.Offsets)),
	)

	result, err := s.retrieveByOffsets(ctx, plan, log)
	if err != nil {
		return nil, err
	}
	defer result.Release()

	_, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "reduced-segcore-results-deserialization")
	defer span.End()

	retrieveResult, err := result.GetResult()
	if err != nil {
		log.Warn("unmarshal retrieve by offsets result failed", zap.Error(err))
		return nil, err
	}
	log.Debug("retrieve by segment offsets done", zap.Int("resultNum", len(retrieveResult.Offset)))
	return retrieveResult, nil
}

func (s *LocalSegment) Insert(ctx context.Context, rowIDs []int64, timestamps []typeutil.Timestamp, record *segcorepb.InsertRecord) error {
	if s.Type() != SegmentTypeGrowing {
		return fmt.Errorf("unexpected segmentType when segmentInsert, segmentType = %s", s.segmentType.String())
	}
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	var result *segcore.InsertResult
	var err error
	GetDynamicPool().Submit(func() (any, error) {
		start := time.Now()
		defer func() {
			metrics.QueryNodeCGOCallLatency.WithLabelValues(
				paramtable.GetStringNodeID(),
				"Insert",
				"Sync",
			).Observe(float64(time.Since(start).Milliseconds()))
		}()

		result, err = s.csegment.Insert(ctx, &segcore.InsertRequest{
			RowIDs:     rowIDs,
			Timestamps: timestamps,
			Record:     record,
		})
		return nil, nil
	}).Await()

	if err != nil {
		return err
	}
	s.insertCount.Add(result.InsertedRows)
	s.rowNum.Store(-1)
	s.memSize.Store(-1)
	return nil
}

func (s *LocalSegment) Delete(ctx context.Context, primaryKeys storage.PrimaryKeys, timestamps []typeutil.Timestamp) error {
	/*
		CStatus
		Delete(CSegmentInterface c_segment,
		           long int reserved_offset,
		           long size,
		           const long* primary_keys,
		           const unsigned long* timestamps);
	*/

	if primaryKeys.Len() == 0 {
		return nil
	}
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	s.deltaMut.Lock()
	defer s.deltaMut.Unlock()

	// segcore DeletedRecord::InternalPush is idempotent on (PK, ts):
	// duplicate deletes against already-deleted rows are discarded internally.
	// Do NOT add a ts-watermark skip here. In L0-forward + partial-L0-compaction
	// scenarios, batches contain ts values below the watermark that have NOT yet
	// been applied to this segment, and skipping them causes silent data loss.

	var err error
	GetDynamicPool().Submit(func() (any, error) {
		start := time.Now()
		defer func() {
			metrics.QueryNodeCGOCallLatency.WithLabelValues(
				paramtable.GetStringNodeID(),
				"Delete",
				"Sync",
			).Observe(float64(time.Since(start).Milliseconds()))
		}()
		_, err = s.csegment.Delete(ctx, &segcore.DeleteRequest{
			PrimaryKeys: primaryKeys,
			Timestamps:  timestamps,
		})
		return nil, nil
	}).Await()

	if err != nil {
		return err
	}

	s.rowNum.Store(-1)
	// Track max ts as a high-water-mark (consumed by file-level skip in
	// LoadDeltaLogs and by dist_handler for QueryCoord reporting). Using
	// tss[last] on unsorted batches underestimates the watermark.
	s.advanceLastDeltaTimestamp(timestamps)
	return nil
}

// -------------------------------------------------------------------------------------- interfaces for sealed segment
func (s *LocalSegment) LoadFieldData(ctx context.Context, fieldID int64, rowCount int64, field *datapb.FieldBinlog) error {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, fmt.Sprintf("LoadFieldData-%d-%d", s.ID(), fieldID))
	defer sp.End()

	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Int64("fieldID", fieldID),
		zap.Int64("rowCount", rowCount),
	)
	log.Info("start loading field data for field")

	// TODO retrieve_enable should be considered
	collection := s.collection
	fieldSchema, err := getFieldSchema(collection.Schema(), fieldID)
	if err != nil {
		return err
	}
	mmapEnabled := isDataMmapEnable(fieldSchema)
	fieldWarmupPolicy := getFieldWarmupPolicy(fieldSchema)

	req := &segcore.LoadFieldDataRequest{
		Fields: []segcore.LoadFieldDataInfo{{
			Field:        field,
			EnableMMap:   mmapEnabled,
			WarmupPolicy: fieldWarmupPolicy,
		}},
		RowCount:       rowCount,
		StorageVersion: s.LoadInfo().GetStorageVersion(),
	}

	GetLoadPool().Submit(func() (any, error) {
		start := time.Now()
		defer func() {
			metrics.QueryNodeCGOCallLatency.WithLabelValues(
				paramtable.GetStringNodeID(),
				"LoadFieldData",
				"Sync",
			).Observe(float64(time.Since(start).Milliseconds()))
		}()
		_, err = s.csegment.LoadFieldData(ctx, req)
		log.Info("submitted loadFieldData task to load pool")
		return nil, nil
	}).Await()

	if err != nil {
		log.Warn("LoadFieldData failed", zap.Error(err))
		return err
	}
	log.Info("load field done")
	return nil
}

func (s *LocalSegment) LoadDeltaData(ctx context.Context, deltaData *storage.DeltaData) error {
	if deltaData.DeleteRowCount() == 0 {
		return nil
	}

	pks, tss := deltaData.DeletePks(), deltaData.DeleteTimestamps()
	rowNum := deltaData.DeleteRowCount()

	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
	)

	s.deltaMut.Lock()
	defer s.deltaMut.Unlock()

	// See comment in Delete(): segcore dedups at (PK, ts) level, and tss is
	// NOT sorted across L0 segments (BufferForwarder appends in iteration
	// order), so comparing against tss[last] is both unnecessary and incorrect.

	ids, err := storage.ParsePrimaryKeysBatch2IDs(pks)
	if err != nil {
		return err
	}

	idsBlob, err := proto.Marshal(ids)
	if err != nil {
		return err
	}

	loadInfo := C.CLoadDeletedRecordInfo{
		timestamps:        unsafe.Pointer(&tss[0]),
		primary_keys:      (*C.uint8_t)(unsafe.Pointer(&idsBlob[0])),
		primary_keys_size: C.uint64_t(len(idsBlob)),
		row_count:         C.int64_t(rowNum),
	}
	/*
		CStatus
		LoadDeletedRecord(CSegmentInterface c_segment, CLoadDeletedRecordInfo deleted_record_info)
	*/
	var status C.CStatus
	GetDynamicPool().Submit(func() (any, error) {
		start := time.Now()
		defer func() {
			metrics.QueryNodeCGOCallLatency.WithLabelValues(
				paramtable.GetStringNodeID(),
				"LoadDeletedRecord",
				"Sync",
			).Observe(float64(time.Since(start).Milliseconds()))
		}()
		status = C.LoadDeletedRecord(s.ptr, loadInfo)
		return nil, nil
	}).Await()

	if err := HandleCStatus(ctx, &status, "LoadDeletedRecord failed",
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID())); err != nil {
		return err
	}

	s.rowNum.Store(-1)
	s.advanceLastDeltaTimestamp(tss)

	log.Info("load deleted record done",
		zap.Int64("rowNum", rowNum),
		zap.String("segmentType", s.Type().String()))
	return nil
}

type cLoadInfoTiming struct {
	newInfoDur    time.Duration
	appendInfoDur time.Duration
	callbackDur   time.Duration
	deleteInfoDur time.Duration
}

const scalarIndexEngineVersionKey = "scalar_index_engine_version"

func GetCLoadInfoWithFunc(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
	f func(c *LoadIndexInfo) error,
) error {
	return getCLoadInfoWithFunc(ctx, fieldSchema, loadInfo, indexInfo, f, nil)
}

func prepareLoadIndexParams(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
) (map[string]string, bool, error) {
	indexParams := funcutil.KeyValuePair2Map(indexInfo.IndexParams)
	// as Knowhere reports error if encounter an unknown param, we need to delete it
	delete(indexParams, common.MmapEnabledKey)

	// some build params also exist in indexParams, which are useless during loading process
	if vecindexmgr.GetVecIndexMgrInstance().IsDiskANN(indexParams["index_type"]) {
		if err := indexparams.SetDiskIndexLoadParams(paramtable.Get(), indexParams, indexInfo.GetNumRows()); err != nil {
			return nil, false, err
		}
	}

	// set whether enable offset cache for bitmap index
	if indexParams["index_type"] == indexparamcheck.IndexBitmap {
		indexparams.SetBitmapIndexLoadParams(paramtable.Get(), indexParams)
	}

	if err := indexparams.AppendPrepareLoadParams(paramtable.Get(), indexParams); err != nil {
		return nil, false, err
	}

	enableMmap := isIndexMmapEnable(fieldSchema, indexInfo)
	// Add warmup policy to index_params if not already present
	// C++ will pass it to Knowhere for index loading
	if existingWarmup, exists := indexParams[common.WarmupKey]; exists {
		log.Ctx(ctx).Info("warmup policy already in index params (from QueryCoord)",
			zap.Int64("segmentID", loadInfo.GetSegmentID()),
			zap.Int64("fieldID", indexInfo.GetFieldID()),
			zap.String("warmup", existingWarmup))
	} else {
		warmupPolicy := getIndexWarmupPolicy(fieldSchema, indexInfo)
		log.Ctx(ctx).Info("warmup policy from getIndexWarmupPolicy",
			zap.Int64("segmentID", loadInfo.GetSegmentID()),
			zap.Int64("fieldID", indexInfo.GetFieldID()),
			zap.String("warmup", warmupPolicy))
		if warmupPolicy != "" {
			indexParams[common.WarmupKey] = warmupPolicy
		}
	}

	return indexParams, enableMmap, nil
}

func getLoadIndexInfoProto(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
) (*cgopb.LoadIndexInfo, error) {
	indexParams, enableMmap, err := prepareLoadIndexParams(ctx, fieldSchema, loadInfo, indexInfo)
	if err != nil {
		return nil, err
	}

	// Pass DataCoord-built index file paths through; QueryNode should not
	// attach v0/v1 path layout semantics to the read path.
	return &cgopb.LoadIndexInfo{
		CollectionID:              loadInfo.GetCollectionID(),
		PartitionID:               loadInfo.GetPartitionID(),
		SegmentID:                 loadInfo.GetSegmentID(),
		Field:                     fieldSchema,
		EnableMmap:                enableMmap,
		IndexID:                   indexInfo.GetIndexID(),
		IndexBuildID:              indexInfo.GetBuildID(),
		IndexVersion:              indexInfo.GetIndexVersion(),
		IndexParams:               indexParams,
		IndexFiles:                indexInfo.GetIndexFilePaths(),
		IndexEngineVersion:        indexInfo.GetCurrentIndexVersion(),
		IndexFileSize:             indexInfo.GetIndexSize(),
		NumRows:                   indexInfo.GetNumRows(),
		CurrentScalarIndexVersion: indexInfo.GetCurrentScalarIndexVersion(),
	}, nil
}

type loadIndexResourceEstimateInfo struct {
	fieldType          int32
	elementType        int32
	indexEngineVersion int32
	indexSize          int64
	enableMmap         bool
	numRows            int64
	dim                int64
	indexParams        map[string]string
}

type cIndexParamArrays struct {
	keys     **C.char
	values   **C.char
	keyMem   unsafe.Pointer
	valueMem unsafe.Pointer
	strings  []*C.char
	count    C.uint64_t
}

func newCIndexParamArrays(params map[string]string) *cIndexParamArrays {
	if len(params) == 0 {
		return &cIndexParamArrays{}
	}

	count := len(params)
	keyMem := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(uintptr(0))))
	valueMem := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(uintptr(0))))
	keys := unsafe.Slice((**C.char)(keyMem), count)
	values := unsafe.Slice((**C.char)(valueMem), count)
	strings := make([]*C.char, 0, count*2)

	idx := 0
	for key, value := range params {
		cKey := C.CString(key)
		cValue := C.CString(value)
		keys[idx] = cKey
		values[idx] = cValue
		strings = append(strings, cKey, cValue)
		idx++
	}

	return &cIndexParamArrays{
		keys:     (**C.char)(keyMem),
		values:   (**C.char)(valueMem),
		keyMem:   keyMem,
		valueMem: valueMem,
		strings:  strings,
		count:    C.uint64_t(count),
	}
}

func (a *cIndexParamArrays) free() {
	for _, s := range a.strings {
		C.free(unsafe.Pointer(s))
	}
	if a.keyMem != nil {
		C.free(a.keyMem)
	}
	if a.valueMem != nil {
		C.free(a.valueMem)
	}
}

func getLoadIndexResourceEstimateInfo(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
) (*loadIndexResourceEstimateInfo, error) {
	indexParams, enableMmap, err := prepareLoadIndexParams(ctx, fieldSchema, loadInfo, indexInfo)
	if err != nil {
		return nil, err
	}
	if scalarVersion := indexInfo.GetCurrentScalarIndexVersion(); scalarVersion > 0 {
		indexParams[scalarIndexEngineVersionKey] = fmt.Sprintf("%d", scalarVersion)
	}

	dim := int64(1)
	if typeutil.IsVectorType(fieldSchema.GetDataType()) && !typeutil.IsSparseFloatVectorType(fieldSchema.GetDataType()) {
		dim, err = typeutil.GetDim(fieldSchema)
		if err != nil {
			return nil, err
		}
	}

	return &loadIndexResourceEstimateInfo{
		fieldType:          int32(fieldSchema.GetDataType()),
		elementType:        int32(fieldSchema.GetElementType()),
		indexEngineVersion: indexInfo.GetCurrentIndexVersion(),
		indexSize:          indexInfo.GetIndexSize(),
		enableMmap:         enableMmap,
		numRows:            indexInfo.GetNumRows(),
		dim:                dim,
		indexParams:        indexParams,
	}, nil
}

func estimateLoadIndexResourceWithTiming(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
	timing *cLoadInfoTiming,
) (ResourceEstimate, time.Duration, error) {
	estimateInfo, err := getLoadIndexResourceEstimateInfo(ctx, fieldSchema, loadInfo, indexInfo)
	if err != nil {
		return ResourceEstimate{}, 0, err
	}
	cIndexParams := newCIndexParamArrays(estimateInfo.indexParams)
	defer cIndexParams.free()

	var request C.LoadResourceRequest
	var estimateDur time.Duration
	_, _ = GetDynamicPool().Submit(func() (any, error) {
		cInfo := C.CLoadIndexResourceInfo{
			field_type:           C.int32_t(estimateInfo.fieldType),
			element_type:         C.int32_t(estimateInfo.elementType),
			index_engine_version: C.int32_t(estimateInfo.indexEngineVersion),
			index_size:           C.int64_t(estimateInfo.indexSize),
			enable_mmap:          C.bool(estimateInfo.enableMmap),
			num_rows:             C.int64_t(estimateInfo.numRows),
			dim:                  C.int64_t(estimateInfo.dim),
			index_param_keys:     cIndexParams.keys,
			index_param_values:   cIndexParams.values,
			index_param_count:    cIndexParams.count,
		}

		callbackStart := time.Now()
		estimateStart := time.Now()
		request = C.EstimateLoadIndexResourceFromInfo(cInfo)
		estimateDur = time.Since(estimateStart)
		if timing != nil {
			timing.callbackDur += time.Since(callbackStart)
		}
		return nil, nil
	}).Await()

	return GetResourceEstimate(&request), estimateDur, nil
}

func getCLoadInfoWithFunc(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	loadInfo *querypb.SegmentLoadInfo,
	indexInfo *querypb.FieldIndexInfo,
	f func(c *LoadIndexInfo) error,
	timing *cLoadInfoTiming,
) error {
	// 1.
	newInfoStart := time.Now()
	loadIndexInfo, err := newLoadIndexInfo(ctx)
	if timing != nil {
		timing.newInfoDur += time.Since(newInfoStart)
	}
	if err != nil {
		return err
	}
	defer func() {
		deleteInfoStart := time.Now()
		deleteLoadIndexInfo(loadIndexInfo)
		if timing != nil {
			timing.deleteInfoDur += time.Since(deleteInfoStart)
		}
	}()

	indexInfoProto, err := getLoadIndexInfoProto(ctx, fieldSchema, loadInfo, indexInfo)
	if err != nil {
		return err
	}

	// 2.
	appendInfoStart := time.Now()
	err = loadIndexInfo.appendLoadIndexInfo(ctx, indexInfoProto)
	if timing != nil {
		timing.appendInfoDur += time.Since(appendInfoStart)
	}
	if err != nil {
		log.Warn("fail to append load index info", zap.Error(err))
		return err
	}
	callbackStart := time.Now()
	err = f(loadIndexInfo)
	if timing != nil {
		timing.callbackDur += time.Since(callbackStart)
	}
	return err
}

func (s *LocalSegment) LoadIndex(ctx context.Context, indexInfo *querypb.FieldIndexInfo, fieldType schemapb.DataType) error {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Int64("fieldID", indexInfo.GetFieldID()),
		zap.Int64("indexID", indexInfo.GetIndexID()),
	)

	old := s.GetIndexByID(indexInfo.GetIndexID())
	// the index loaded
	if old != nil && old.IsLoaded {
		log.Warn("index already loaded")
		return nil
	}

	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, fmt.Sprintf("LoadIndex-%d-%d", s.ID(), indexInfo.GetFieldID()))
	defer sp.End()

	tr := timerecord.NewTimeRecorder("loadIndex")

	schemaHelper, err := typeutil.CreateSchemaHelper(s.GetCollection().Schema())
	if err != nil {
		return err
	}
	fieldSchema, err := schemaHelper.GetFieldFromID(indexInfo.GetFieldID())
	if err != nil {
		return err
	}

	// // if segment is pk sorted, user created indexes bring no performance gain but extra memory usage
	if s.IsSorted() && fieldSchema.GetIsPrimaryKey() {
		log.Info("skip loading index for pk field in sorted segment")
		// set field index, preventing repeated loading index task
		s.fieldIndexes.Insert(indexInfo.GetFieldID(), &IndexedFieldInfo{
			FieldBinlog: &datapb.FieldBinlog{
				FieldID: indexInfo.GetFieldID(),
			},
			IndexInfo: indexInfo,
			IsLoaded:  true,
		})
		return nil
	}

	return s.innerLoadIndex(ctx, fieldSchema, indexInfo, tr, fieldType)
}

func (s *LocalSegment) innerLoadIndex(ctx context.Context,
	fieldSchema *schemapb.FieldSchema,
	indexInfo *querypb.FieldIndexInfo,
	tr *timerecord.TimeRecorder,
	fieldType schemapb.DataType,
) error {
	err := GetCLoadInfoWithFunc(ctx, fieldSchema,
		s.LoadInfo(), indexInfo, func(loadIndexInfo *LoadIndexInfo) error {
			newLoadIndexInfoSpan := tr.RecordSpan()

			if err := loadIndexInfo.loadIndex(ctx); err != nil {
				if loadIndexInfo.cleanLocalData(ctx) != nil {
					log.Warn("failed to clean cached data on disk after append index failed",
						zap.Int64("buildID", indexInfo.BuildID),
						zap.Int64("index version", indexInfo.IndexVersion))
				}
				return err
			}
			if s.Type() != SegmentTypeSealed {
				errMsg := fmt.Sprintln("updateSegmentIndex failed, illegal segment type ", s.segmentType, "segmentID = ", s.ID())
				return errors.New(errMsg)
			}
			appendLoadIndexInfoSpan := tr.RecordSpan()

			// 3.
			err := s.UpdateIndexInfo(ctx, indexInfo, loadIndexInfo)
			if err != nil {
				return err
			}
			updateIndexInfoSpan := tr.RecordSpan()

			log.Info("Finish loading index",
				zap.Duration("newLoadIndexInfoSpan", newLoadIndexInfoSpan),
				zap.Duration("appendLoadIndexInfoSpan", appendLoadIndexInfoSpan),
				zap.Duration("updateIndexInfoSpan", updateIndexInfoSpan),
			)
			return nil
		})
	if err != nil {
		log.Warn("load index failed", zap.Error(err))
	}
	return err
}

func (s *LocalSegment) LoadJSONKeyIndex(ctx context.Context, jsonKeyStats *datapb.JsonKeyStats, schemaHelper *typeutil.SchemaHelper, basePath string) error {
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	if !paramtable.Get().CommonCfg.EnabledJSONKeyStats.GetAsBool() {
		log.Ctx(ctx).Warn("load json key index failed, json key stats is not enabled")
		return nil
	}

	// for compatibility, we only support load data format version equal to the current data format version
	// if the data format version is less than the current version, wait for trigger a stats task again
	if jsonKeyStats.GetJsonKeyStatsDataFormat() != common.JSONStatsDataFormatVersion {
		log.Ctx(ctx).Warn("load json key index failed dataformat invalid", zap.Int64("dataformat", jsonKeyStats.GetJsonKeyStatsDataFormat()), zap.Int64("field id", jsonKeyStats.GetFieldID()), zap.Any("json key logs", jsonKeyStats))
		return nil
	}

	log.Ctx(ctx).Info("load json key index", zap.Int64("field id", jsonKeyStats.GetFieldID()), zap.Any("json key logs", jsonKeyStats))
	s.fieldJSONStatsMu.RLock()
	_, loaded := s.fieldJSONStats[jsonKeyStats.GetFieldID()]
	s.fieldJSONStatsMu.RUnlock()
	if loaded {
		log.Warn("JsonKeyIndexStats already loaded", zap.Int64("field id", jsonKeyStats.GetFieldID()), zap.Any("json key logs", jsonKeyStats))
		return nil
	}

	f, err := schemaHelper.GetFieldFromID(jsonKeyStats.GetFieldID())
	if err != nil {
		return err
	}

	// JSON key stats should based on scala field's warmup policy
	warmupPolicy := getScalarDataWarmupPolicy(f)

	cgoProto := &indexcgopb.LoadJsonKeyIndexInfo{
		FieldID:      jsonKeyStats.GetFieldID(),
		Version:      jsonKeyStats.GetVersion(),
		BuildID:      jsonKeyStats.GetBuildID(),
		Files:        jsonKeyStats.GetFiles(),
		Schema:       f,
		CollectionID: s.Collection(),
		PartitionID:  s.Partition(),
		LoadPriority: s.loadInfo.Load().GetPriority(),
		EnableMmap:   paramtable.Get().QueryNodeCfg.MmapJSONStats.GetAsBool(),
		MmapDirPath:  paramtable.Get().QueryNodeCfg.MmapDirPath.GetValue(),
		StatsSize:    jsonKeyStats.GetLogSize(),
		WarmupPolicy: warmupPolicy,
		BasePath:     basePath,
	}

	marshaled, err := proto.Marshal(cgoProto)
	if err != nil {
		return err
	}

	guard := segcore.NewCancellationGuard(ctx)
	defer guard.Close()

	var status C.CStatus
	_, _ = GetLoadPool().Submit(func() (any, error) {
		traceCtx := ParseCTraceContext(ctx)
		status = C.LoadJsonKeyIndex(traceCtx.ctx, s.ptr, (*C.uint8_t)(unsafe.Pointer(&marshaled[0])), (C.uint64_t)(len(marshaled)), (C.CLoadCancellationSource)(guard.Source()))
		return nil, nil
	}).Await()

	if err := HandleCStatus(ctx, &status, "Load JsonKeyStats failed"); err != nil {
		return err
	}
	s.fieldJSONStatsMu.Lock()
	s.fieldJSONStats[jsonKeyStats.GetFieldID()] = &querypb.JsonStatsInfo{
		FieldID:           jsonKeyStats.GetFieldID(),
		DataFormatVersion: jsonKeyStats.GetJsonKeyStatsDataFormat(),
		BuildID:           jsonKeyStats.GetBuildID(),
		VersionID:         jsonKeyStats.GetVersion(),
	}
	s.fieldJSONStatsMu.Unlock()
	return nil
}

func (s *LocalSegment) UpdateIndexInfo(ctx context.Context, indexInfo *querypb.FieldIndexInfo, info *LoadIndexInfo) error {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Int64("fieldID", indexInfo.FieldID),
	)
	if !s.ptrLock.PinIf(state.IsNotReleased) {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released")
	}
	defer s.ptrLock.Unpin()

	var status C.CStatus
	GetDynamicPool().Submit(func() (any, error) {
		status = C.UpdateSealedSegmentIndex(s.ptr, info.cLoadIndexInfo)
		return nil, nil
	}).Await()

	if err := HandleCStatus(ctx, &status, "UpdateSealedSegmentIndex failed",
		zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.Int64("fieldID", indexInfo.FieldID)); err != nil {
		return err
	}

	s.fieldIndexes.Insert(indexInfo.GetIndexID(), &IndexedFieldInfo{
		FieldBinlog: &datapb.FieldBinlog{
			FieldID: indexInfo.GetFieldID(),
		},
		IndexInfo: indexInfo,
		IsLoaded:  true,
	})
	log.Info("updateSegmentIndex done")
	return nil
}

func (s *LocalSegment) UpdateFieldRawDataSize(ctx context.Context, numRows int64, fieldBinlog *datapb.FieldBinlog) error {
	var status C.CStatus
	fieldID := fieldBinlog.FieldID
	fieldDataSize := int64(0)
	for _, binlog := range fieldBinlog.GetBinlogs() {
		fieldDataSize += binlog.GetMemorySize()
	}
	GetDynamicPool().Submit(func() (any, error) {
		status = C.UpdateFieldRawDataSize(s.ptr, C.int64_t(fieldID), C.int64_t(numRows), C.int64_t(fieldDataSize))
		return nil, nil
	}).Await()

	if err := HandleCStatus(ctx, &status, "updateFieldRawDataSize failed"); err != nil {
		return err
	}

	log.Ctx(ctx).Info("updateFieldRawDataSize done", zap.Int64("segmentID", s.ID()))

	return nil
}

func (s *LocalSegment) syncFieldJSONStatsFromLoadInfo(ctx context.Context, loadInfo *querypb.SegmentLoadInfo) {
	jsonStatsInfo := make(map[int64]*querypb.JsonStatsInfo)
	if !paramtable.Get().CommonCfg.EnabledJSONKeyStats.GetAsBool() {
		log.Ctx(ctx).Warn("skip sync json key stats, json key stats is not enabled", zap.Int64("segmentID", s.ID()))
		s.fieldJSONStatsMu.Lock()
		s.fieldJSONStats = jsonStatsInfo
		s.fieldJSONStatsMu.Unlock()
		return
	}

	if !s.hasJSONKeyStatsField() {
		s.fieldJSONStatsMu.Lock()
		s.fieldJSONStats = jsonStatsInfo
		s.fieldJSONStatsMu.Unlock()
		return
	}

	statsResult := packed.NewStatsResolverFromLoadInfo(loadInfo).TextAndJSONIndexStatsWithBasePaths()
	jsonKeyStats := statsResult.JSONKeyStats
	if statsResult.Err() != nil {
		log.Ctx(ctx).Warn("failed to resolve json key stats from manifest",
			zap.Int64("segmentID", loadInfo.GetSegmentID()),
			zap.String("manifestPath", loadInfo.GetManifestPath()),
			zap.Error(statsResult.Err()))
		jsonKeyStats = loadInfo.GetJsonKeyStatsLogs()
	}

	for fieldID, stats := range jsonKeyStats {
		if stats == nil {
			continue
		}
		if stats.GetJsonKeyStatsDataFormat() != common.JSONStatsDataFormatVersion {
			log.Ctx(ctx).Warn("skip sync json key stats, data format invalid",
				zap.Int64("segmentID", loadInfo.GetSegmentID()),
				zap.Int64("fieldID", fieldID),
				zap.Int64("buildID", stats.GetBuildID()),
				zap.Int64("version", stats.GetVersion()),
				zap.Int64("dataFormat", stats.GetJsonKeyStatsDataFormat()),
				zap.Int64("expectedDataFormat", common.JSONStatsDataFormatVersion))
			continue
		}
		jsonStatsInfo[fieldID] = &querypb.JsonStatsInfo{
			FieldID:           stats.GetFieldID(),
			DataFormatVersion: stats.GetJsonKeyStatsDataFormat(),
			BuildID:           stats.GetBuildID(),
			VersionID:         stats.GetVersion(),
		}
	}

	s.fieldJSONStatsMu.Lock()
	s.fieldJSONStats = jsonStatsInfo
	s.fieldJSONStatsMu.Unlock()
}

func (s *LocalSegment) hasJSONKeyStatsField() bool {
	for _, field := range s.collection.Schema().GetFields() {
		if typeutil.CreateFieldSchemaHelper(field).EnableJSONKeyStatsIndex() {
			return true
		}
	}
	return false
}

func (s *LocalSegment) Load(ctx context.Context) error {
	if err := s.csegment.Load(ctx); err != nil {
		return err
	}
	s.syncFieldJSONStatsFromLoadInfo(ctx, s.LoadInfo())
	return nil
}

func (s *LocalSegment) Reopen(ctx context.Context, newLoadInfo *querypb.SegmentLoadInfo) error {
	if !s.ptrLock.PinIfNotReleased() {
		return merr.WrapErrSegmentNotLoaded(s.ID(), "segment released during reopen")
	}
	defer s.ptrLock.Unpin()

	schema, schemaVersion := s.collection.SchemaAndVersion()
	err := s.csegment.Reopen(ctx, &segcore.ReopenRequest{
		LoadInfo:      newLoadInfo,
		Schema:        schema,
		SchemaVersion: schemaVersion,
	})
	if err != nil {
		return err
	}
	s.loadInfo.Store(newLoadInfo)
	s.syncFieldJSONStatsFromLoadInfo(ctx, newLoadInfo)
	s.compactLoadInfoForRuntime()
	return nil
}

type ReleaseScope int

const (
	ReleaseScopeAll ReleaseScope = iota
	ReleaseScopeData
)

type releaseOptions struct {
	Scope ReleaseScope
}

func newReleaseOptions() *releaseOptions {
	return &releaseOptions{
		Scope: ReleaseScopeAll,
	}
}

type releaseOption func(*releaseOptions)

func WithReleaseScope(scope ReleaseScope) releaseOption {
	return func(options *releaseOptions) {
		options.Scope = scope
	}
}

func (s *LocalSegment) Release(ctx context.Context, opts ...releaseOption) {
	options := newReleaseOptions()
	for _, opt := range opts {
		opt(options)
	}
	stateLockGuard := s.startRelease(options.Scope)
	if stateLockGuard == nil { // release is already done.
		return
	}
	// release will never fail
	defer stateLockGuard.Done(nil)

	log := log.Ctx(ctx).With(zap.Int64("collectionID", s.Collection()),
		zap.Int64("partitionID", s.Partition()),
		zap.Int64("segmentID", s.ID()),
		zap.String("segmentType", s.segmentType.String()),
		zap.Int64("insertCount", s.InsertCount()),
	)

	// wait all read ops finished
	ptr := s.ptr
	if options.Scope == ReleaseScopeData {
		s.ReleaseSegmentData()
		log.Info("release segment data done and the field indexes info has been set lazy load=true")
		return
	}

	if paramtable.Get().QueryNodeCfg.ExprResCacheEnabled.GetAsBool() {
		// erase expr-cache for this segment before deleting C segment
		C.ExprResCacheEraseSegment(C.int64_t(s.ID()))
	}

	GetDynamicPool().Submit(func() (any, error) {
		C.DeleteSegment(ptr)
		return nil, nil
	}).Await()

	// TODO: disable logical resource handling for now
	// usage := s.ResourceUsageEstimate()
	// s.manager.SubLogicalResource(usage)

	// Refund PK candidate resource
	s.pkCandidate.Refund()

	binlogSize := s.binlogSize.Load()
	if binlogSize > 0 {
		// no concurrent change to s.binlogSize, so the subtraction is safe
		s.manager.SubLoadedBinlogSize(binlogSize)
		s.binlogSize.Store(0)
	}

	log.Info("delete segment from memory")
}

// ReleaseSegmentData releases the segment data.
func (s *LocalSegment) ReleaseSegmentData() {
	GetDynamicPool().Submit(func() (any, error) {
		C.ClearSegmentData(s.ptr)
		return nil, nil
	}).Await()
	for _, indexInfo := range s.Indexes() {
		indexInfo.IsLoaded = false
	}
}

// StartLoadData starts the loading process of the segment.
func (s *LocalSegment) StartLoadData() (state.LoadStateLockGuard, error) {
	return s.ptrLock.StartLoadData()
}

// startRelease starts the releasing process of the segment.
func (s *LocalSegment) startRelease(scope ReleaseScope) state.LoadStateLockGuard {
	switch scope {
	case ReleaseScopeData:
		return s.ptrLock.StartReleaseData()
	case ReleaseScopeAll:
		return s.ptrLock.StartReleaseAll()
	default:
		panic(fmt.Sprintf("unexpected release scope %d", scope))
	}
}

func (s *LocalSegment) RemoveFieldFile(fieldId int64) {
	GetDynamicPool().Submit(func() (any, error) {
		C.RemoveFieldFile(s.ptr, C.int64_t(fieldId))
		return nil, nil
	}).Await()
}

func (s *LocalSegment) RemoveUnusedFieldFiles() error {
	schema := s.collection.Schema()
	indexInfos, _ := separateIndexAndBinlog(s.LoadInfo())
	for _, indexInfo := range indexInfos {
		need, err := s.indexNeedLoadRawData(schema, indexInfo)
		if err != nil {
			return err
		}
		if !need {
			s.RemoveFieldFile(indexInfo.IndexInfo.FieldID)
		}
	}
	return nil
}

func (s *LocalSegment) indexNeedLoadRawData(schema *schemapb.CollectionSchema, indexInfo *IndexedFieldInfo) (bool, error) {
	schemaHelper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		return false, err
	}
	fieldSchema, err := schemaHelper.GetFieldFromID(indexInfo.IndexInfo.FieldID)
	if err != nil {
		return false, err
	}
	return !typeutil.IsVectorType(fieldSchema.DataType) && s.HasRawData(indexInfo.IndexInfo.FieldID), nil
}

func (s *LocalSegment) GetFieldJSONIndexStats() map[int64]*querypb.JsonStatsInfo {
	s.fieldJSONStatsMu.RLock()
	defer s.fieldJSONStatsMu.RUnlock()

	stats := make(map[int64]*querypb.JsonStatsInfo, len(s.fieldJSONStats))
	for fieldID, info := range s.fieldJSONStats {
		if info == nil {
			continue
		}
		stats[fieldID] = proto.Clone(info).(*querypb.JsonStatsInfo)
	}
	return stats
}

// FlushData flushes data from the growing segment directly to storage via C++ milvus-storage.
// This is a unified interface that combines data extraction from segcore and writing to storage.
// The C++ side handles: extracting raw field data from ConcurrentVector, converting to Arrow,
// and writing to storage via milvus-storage with TEXT column LOB handling.
func (s *LocalSegment) FlushData(ctx context.Context, startOffset, endOffset int64, config *FlushConfig) (*FlushResult, error) {
	// currently only growing segments support FlushData
	if s.Type() != SegmentTypeGrowing {
		return nil, errors.Errorf("FlushData is only supported for growing segments, got %s", s.Type().String())
	}

	// validate offsets
	if startOffset < 0 || endOffset < startOffset {
		return nil, errors.Errorf("invalid offsets: start=%d, end=%d", startOffset, endOffset)
	}

	// no data to flush
	if startOffset == endOffset {
		return nil, nil
	}

	// build C flush config
	var cConfig C.CFlushConfig
	cSegmentPath := C.CString(config.SegmentBasePath)
	defer C.free(unsafe.Pointer(cSegmentPath))
	cConfig.segment_path = cSegmentPath

	cConfig.read_version = C.int64_t(config.ReadVersion)
	cConfig.retry_limit = C.uint32_t(3)

	// populate TEXT column configs
	// All arrays must be C-allocated to avoid "Go pointer to unpinned Go pointer" panic
	numTextCols := len(config.TextFieldIDs)
	if numTextCols > 0 {
		// allocate C arrays via C.malloc (not Go slices) to satisfy CGO pointer rules
		cFieldIDs := (*C.int64_t)(C.malloc(C.size_t(numTextCols) * C.size_t(unsafe.Sizeof(C.int64_t(0)))))
		defer C.free(unsafe.Pointer(cFieldIDs))

		cLobPathPtrs := (**C.char)(C.malloc(C.size_t(numTextCols) * C.size_t(unsafe.Sizeof((*C.char)(nil)))))
		defer C.free(unsafe.Pointer(cLobPathPtrs))

		fieldIDSlice := unsafe.Slice(cFieldIDs, numTextCols)
		lobPathSlice := unsafe.Slice(cLobPathPtrs, numTextCols)
		for i := 0; i < numTextCols; i++ {
			fieldIDSlice[i] = C.int64_t(config.TextFieldIDs[i])
			lobPathSlice[i] = C.CString(config.TextLobPaths[i])
			defer C.free(unsafe.Pointer(lobPathSlice[i]))
		}

		cConfig.text_field_ids = cFieldIDs
		cConfig.text_lob_paths = cLobPathPtrs
		cConfig.num_text_columns = C.size_t(numTextCols)
	} else {
		cConfig.text_field_ids = nil
		cConfig.text_lob_paths = nil
		cConfig.num_text_columns = 0
	}

	// call C FFI
	var cResult C.CFlushResult
	status := C.FlushGrowingSegmentData(
		s.ptr,
		C.int64_t(startOffset),
		C.int64_t(endOffset),
		&cConfig,
		&cResult,
	)
	defer C.FreeFlushResult(&cResult)

	if err := HandleCStatus(ctx, &status, "FlushGrowingSegmentData"); err != nil {
		return nil, err
	}

	// no data flushed
	if cResult.manifest_path == nil {
		return nil, nil
	}

	// C++ SegmentWriter returns a raw file path like:
	//   "files/insert_log/{coll}/{part}/{seg}/_metadata/manifest-{ver}.avro"
	// But GetLoonManifest() expects a JSON-encoded manifest path like:
	//   {"ver":{ver},"base_path":"files/insert_log/{coll}/{part}/{seg}"}
	// Convert using the committed_version and base path extraction.
	rawPath := C.GoString(cResult.manifest_path)
	committedVersion := int64(cResult.committed_version)

	// Extract base path: strip "/_metadata/manifest-{ver}.avro" suffix
	basePath := rawPath
	if idx := strings.Index(rawPath, "/_metadata/"); idx >= 0 {
		basePath = rawPath[:idx]
	}
	manifestPath := packed.MarshalManifestPath(basePath, committedVersion)

	return &FlushResult{
		ManifestPath: manifestPath,
		NumRows:      int64(cResult.num_rows),
	}, nil
}
