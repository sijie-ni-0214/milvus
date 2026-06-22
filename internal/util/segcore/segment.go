package segcore

/*
#cgo pkg-config: milvus_core

#include "common/type_c.h"
#include "futures/future_c.h"
#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/cgo"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/segcorepb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	SegmentTypeGrowing SegmentType = commonpb.SegmentState_Growing
	SegmentTypeSealed  SegmentType = commonpb.SegmentState_Sealed

	createCSegmentTimingLogInterval = 5 * time.Second
)

type (
	SegmentType       = commonpb.SegmentState
	CSegmentInterface C.CSegmentInterface
)

var createCSegmentTiming = newCreateCSegmentTimingStats()

type segmentLoadInfoConversionTiming struct {
	binlogFieldCount   int
	statslogFieldCount int
	deltalogFieldCount int
	indexInfoCount     int
	textStatsCount     int
	bm25FieldCount     int
	jsonStatsCount     int

	resolveStatsDur time.Duration
	binlogsDur      time.Duration
	statslogsDur    time.Duration
	deltalogsDur    time.Duration
	indexInfosDur   time.Duration
	textStatsDur    time.Duration
	bm25Dur         time.Duration
	jsonStatsDur    time.Duration
	totalDur        time.Duration
}

type createCSegmentTimingStats struct {
	count          *atomic.Int64
	loadInfoCount  *atomic.Int64
	storageV2Count *atomic.Int64
	storageV3Count *atomic.Int64
	storageOther   *atomic.Int64

	totalBinlogField   *atomic.Int64
	totalStatslogField *atomic.Int64
	totalDeltalogField *atomic.Int64
	totalIndexInfo     *atomic.Int64
	totalTextStats     *atomic.Int64
	totalBM25Field     *atomic.Int64
	totalJSONStats     *atomic.Int64
	totalBlobBytes     *atomic.Int64

	totalResolveStats *atomic.Int64
	totalBinlogs      *atomic.Int64
	totalStatslogs    *atomic.Int64
	totalDeltalogs    *atomic.Int64
	totalIndexInfos   *atomic.Int64
	totalTextStatsDur *atomic.Int64
	totalBM25         *atomic.Int64
	totalJSONStatsDur *atomic.Int64
	totalConvertOther *atomic.Int64
	totalConvert      *atomic.Int64
	totalMarshal      *atomic.Int64
	totalCNew         *atomic.Int64
	totalCommit       *atomic.Int64
	totalCreate       *atomic.Int64

	maxConvert *atomic.Int64
	maxMarshal *atomic.Int64
	maxCNew    *atomic.Int64
	maxCreate  *atomic.Int64

	lastLogUnixNano *atomic.Int64
}

func newCreateCSegmentTimingStats() *createCSegmentTimingStats {
	return &createCSegmentTimingStats{
		count:          atomic.NewInt64(0),
		loadInfoCount:  atomic.NewInt64(0),
		storageV2Count: atomic.NewInt64(0),
		storageV3Count: atomic.NewInt64(0),
		storageOther:   atomic.NewInt64(0),

		totalBinlogField:   atomic.NewInt64(0),
		totalStatslogField: atomic.NewInt64(0),
		totalDeltalogField: atomic.NewInt64(0),
		totalIndexInfo:     atomic.NewInt64(0),
		totalTextStats:     atomic.NewInt64(0),
		totalBM25Field:     atomic.NewInt64(0),
		totalJSONStats:     atomic.NewInt64(0),
		totalBlobBytes:     atomic.NewInt64(0),

		totalResolveStats: atomic.NewInt64(0),
		totalBinlogs:      atomic.NewInt64(0),
		totalStatslogs:    atomic.NewInt64(0),
		totalDeltalogs:    atomic.NewInt64(0),
		totalIndexInfos:   atomic.NewInt64(0),
		totalTextStatsDur: atomic.NewInt64(0),
		totalBM25:         atomic.NewInt64(0),
		totalJSONStatsDur: atomic.NewInt64(0),
		totalConvertOther: atomic.NewInt64(0),
		totalConvert:      atomic.NewInt64(0),
		totalMarshal:      atomic.NewInt64(0),
		totalCNew:         atomic.NewInt64(0),
		totalCommit:       atomic.NewInt64(0),
		totalCreate:       atomic.NewInt64(0),

		maxConvert: atomic.NewInt64(0),
		maxMarshal: atomic.NewInt64(0),
		maxCNew:    atomic.NewInt64(0),
		maxCreate:  atomic.NewInt64(0),

		lastLogUnixNano: atomic.NewInt64(time.Now().UnixNano()),
	}
}

func updateCreateCSegmentMaxDuration(max *atomic.Int64, value time.Duration) {
	for {
		old := max.Load()
		if int64(value) <= old {
			return
		}
		if max.CompareAndSwap(old, int64(value)) {
			return
		}
	}
}

func avgCreateCSegmentDuration(total, count int64) time.Duration {
	if count == 0 {
		return 0
	}
	return time.Duration(total / count)
}

func (s *createCSegmentTimingStats) record(hasLoadInfo bool, storageVersion int64, blobBytes int, conversion segmentLoadInfoConversionTiming, marshalDur, cNewDur, commitDur, totalDur time.Duration) {
	s.count.Inc()
	if hasLoadInfo {
		s.loadInfoCount.Inc()
		switch storageVersion {
		case storage.StorageV2:
			s.storageV2Count.Inc()
		case storage.StorageV3:
			s.storageV3Count.Inc()
		default:
			s.storageOther.Inc()
		}
	}

	s.totalBinlogField.Add(int64(conversion.binlogFieldCount))
	s.totalStatslogField.Add(int64(conversion.statslogFieldCount))
	s.totalDeltalogField.Add(int64(conversion.deltalogFieldCount))
	s.totalIndexInfo.Add(int64(conversion.indexInfoCount))
	s.totalTextStats.Add(int64(conversion.textStatsCount))
	s.totalBM25Field.Add(int64(conversion.bm25FieldCount))
	s.totalJSONStats.Add(int64(conversion.jsonStatsCount))
	s.totalBlobBytes.Add(int64(blobBytes))

	convertAccounted := conversion.resolveStatsDur + conversion.binlogsDur + conversion.statslogsDur +
		conversion.deltalogsDur + conversion.indexInfosDur + conversion.textStatsDur +
		conversion.bm25Dur + conversion.jsonStatsDur
	convertOther := conversion.totalDur - convertAccounted
	if convertOther < 0 {
		convertOther = 0
	}

	s.totalResolveStats.Add(int64(conversion.resolveStatsDur))
	s.totalBinlogs.Add(int64(conversion.binlogsDur))
	s.totalStatslogs.Add(int64(conversion.statslogsDur))
	s.totalDeltalogs.Add(int64(conversion.deltalogsDur))
	s.totalIndexInfos.Add(int64(conversion.indexInfosDur))
	s.totalTextStatsDur.Add(int64(conversion.textStatsDur))
	s.totalBM25.Add(int64(conversion.bm25Dur))
	s.totalJSONStatsDur.Add(int64(conversion.jsonStatsDur))
	s.totalConvertOther.Add(int64(convertOther))
	s.totalConvert.Add(int64(conversion.totalDur))
	s.totalMarshal.Add(int64(marshalDur))
	s.totalCNew.Add(int64(cNewDur))
	s.totalCommit.Add(int64(commitDur))
	s.totalCreate.Add(int64(totalDur))

	updateCreateCSegmentMaxDuration(s.maxConvert, conversion.totalDur)
	updateCreateCSegmentMaxDuration(s.maxMarshal, marshalDur)
	updateCreateCSegmentMaxDuration(s.maxCNew, cNewDur)
	updateCreateCSegmentMaxDuration(s.maxCreate, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(createCSegmentTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}
	windowDur := time.Duration(now.UnixNano() - last)

	count := s.count.Swap(0)
	loadInfoCount := s.loadInfoCount.Swap(0)
	storageV2Count := s.storageV2Count.Swap(0)
	storageV3Count := s.storageV3Count.Swap(0)
	storageOther := s.storageOther.Swap(0)
	totalBinlogField := s.totalBinlogField.Swap(0)
	totalStatslogField := s.totalStatslogField.Swap(0)
	totalDeltalogField := s.totalDeltalogField.Swap(0)
	totalIndexInfo := s.totalIndexInfo.Swap(0)
	totalTextStats := s.totalTextStats.Swap(0)
	totalBM25Field := s.totalBM25Field.Swap(0)
	totalJSONStats := s.totalJSONStats.Swap(0)
	totalBlobBytes := s.totalBlobBytes.Swap(0)
	totalResolveStats := s.totalResolveStats.Swap(0)
	totalBinlogs := s.totalBinlogs.Swap(0)
	totalStatslogs := s.totalStatslogs.Swap(0)
	totalDeltalogs := s.totalDeltalogs.Swap(0)
	totalIndexInfos := s.totalIndexInfos.Swap(0)
	totalTextStatsDur := s.totalTextStatsDur.Swap(0)
	totalBM25 := s.totalBM25.Swap(0)
	totalJSONStatsDur := s.totalJSONStatsDur.Swap(0)
	totalConvertOther := s.totalConvertOther.Swap(0)
	totalConvert := s.totalConvert.Swap(0)
	totalMarshal := s.totalMarshal.Swap(0)
	totalCNew := s.totalCNew.Swap(0)
	totalCommit := s.totalCommit.Swap(0)
	totalCreate := s.totalCreate.Swap(0)
	maxConvert := s.maxConvert.Swap(0)
	maxMarshal := s.maxMarshal.Swap(0)
	maxCNew := s.maxCNew.Swap(0)
	maxCreate := s.maxCreate.Swap(0)
	if count == 0 {
		return
	}

	log.Warn("[SN recovery] load timing stats",
		zap.String("phase", "segcore.create_segment"),
		zap.Duration("windowDur", windowDur),
		zap.Int64("count", count),
		zap.Int64("loadInfoCount", loadInfoCount),
		zap.Int64("storageV2Count", storageV2Count),
		zap.Int64("storageV3Count", storageV3Count),
		zap.Int64("storageOtherCount", storageOther),
		zap.Int64("avgBinlogFieldCount", totalBinlogField/count),
		zap.Int64("avgStatslogFieldCount", totalStatslogField/count),
		zap.Int64("avgDeltalogFieldCount", totalDeltalogField/count),
		zap.Int64("avgIndexInfoCount", totalIndexInfo/count),
		zap.Int64("avgTextStatsCount", totalTextStats/count),
		zap.Int64("avgBM25FieldCount", totalBM25Field/count),
		zap.Int64("avgJSONStatsCount", totalJSONStats/count),
		zap.Int64("avgLoadInfoBlobBytes", totalBlobBytes/count),
		zap.Duration("avgResolveStatsDur", avgCreateCSegmentDuration(totalResolveStats, count)),
		zap.Duration("avgConvertBinlogsDur", avgCreateCSegmentDuration(totalBinlogs, count)),
		zap.Duration("avgConvertStatslogsDur", avgCreateCSegmentDuration(totalStatslogs, count)),
		zap.Duration("avgConvertDeltalogsDur", avgCreateCSegmentDuration(totalDeltalogs, count)),
		zap.Duration("avgConvertIndexInfosDur", avgCreateCSegmentDuration(totalIndexInfos, count)),
		zap.Duration("avgConvertTextStatsDur", avgCreateCSegmentDuration(totalTextStatsDur, count)),
		zap.Duration("avgConvertBM25Dur", avgCreateCSegmentDuration(totalBM25, count)),
		zap.Duration("avgConvertJSONStatsDur", avgCreateCSegmentDuration(totalJSONStatsDur, count)),
		zap.Duration("avgConvertOtherDur", avgCreateCSegmentDuration(totalConvertOther, count)),
		zap.Duration("avgConvertTotalDur", avgCreateCSegmentDuration(totalConvert, count)),
		zap.Duration("avgMarshalDur", avgCreateCSegmentDuration(totalMarshal, count)),
		zap.Duration("avgCNewDur", avgCreateCSegmentDuration(totalCNew, count)),
		zap.Duration("avgCommitDur", avgCreateCSegmentDuration(totalCommit, count)),
		zap.Duration("avgTotalDur", avgCreateCSegmentDuration(totalCreate, count)),
		zap.Duration("maxConvertTotalDur", time.Duration(maxConvert)),
		zap.Duration("maxMarshalDur", time.Duration(maxMarshal)),
		zap.Duration("maxCNewDur", time.Duration(maxCNew)),
		zap.Duration("maxTotalDur", time.Duration(maxCreate)),
	)
}

// CreateCSegmentRequest is a request to create a segment.
type CreateCSegmentRequest struct {
	Collection  *CCollection
	SegmentID   int64
	SegmentType SegmentType
	IsSorted    bool
	LoadInfo    *querypb.SegmentLoadInfo
}

func (req *CreateCSegmentRequest) getCSegmentType() C.SegmentType {
	var segmentType C.SegmentType
	switch req.SegmentType {
	case SegmentTypeGrowing:
		segmentType = C.Growing
	case SegmentTypeSealed:
		segmentType = C.Sealed
	default:
		panic(fmt.Sprintf("invalid segment type: %d", req.SegmentType))
	}
	return segmentType
}

// CreateCSegment creates a segment from a CreateCSegmentRequest.
func CreateCSegment(req *CreateCSegmentRequest) (seg CSegment, err error) {
	start := time.Now()
	var ptr C.CSegmentInterface
	var status C.CStatus
	var conversionTiming segmentLoadInfoConversionTiming
	var marshalDur time.Duration
	var cNewDur time.Duration
	var commitDur time.Duration
	var blobBytes int
	defer func() {
		storageVersion := int64(0)
		if req.LoadInfo != nil {
			storageVersion = req.LoadInfo.GetStorageVersion()
		}
		createCSegmentTiming.record(req.LoadInfo != nil, storageVersion, blobBytes, conversionTiming, marshalDur, cNewDur, commitDur, time.Since(start))
	}()

	if req.LoadInfo != nil {
		resolveManifestStats := true
		if req.Collection != nil {
			resolveManifestStats = schemaNeedsManifestStats(req.Collection.Schema())
		}
		if forceSkipNewSegmentStatsExperiment() {
			resolveManifestStats = false
		}
		segLoadInfo, timing := convertToSegcoreSegmentLoadInfoWithTiming(req.LoadInfo, resolveManifestStats)
		conversionTiming = timing
		marshalStart := time.Now()
		loadInfoBlob, err := proto.Marshal(segLoadInfo)
		marshalDur = time.Since(marshalStart)
		if err != nil {
			return nil, err
		}
		blobBytes = len(loadInfoBlob)

		cNewStart := time.Now()
		status = C.NewSegmentWithLoadInfo(req.Collection.rawPointer(), req.getCSegmentType(), C.int64_t(req.SegmentID), &ptr, C.bool(req.IsSorted), (*C.uint8_t)(unsafe.Pointer(&loadInfoBlob[0])), C.int64_t(len(loadInfoBlob)))
		cNewDur = time.Since(cNewStart)
	} else {
		cNewStart := time.Now()
		status = C.NewSegment(req.Collection.rawPointer(), req.getCSegmentType(), C.int64_t(req.SegmentID), &ptr, C.bool(req.IsSorted))
		cNewDur = time.Since(cNewStart)
	}
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, err
	}
	cseg := &cSegmentImpl{id: req.SegmentID, ptr: ptr}
	seg = cseg
	if req.LoadInfo != nil {
		if commitTs := req.LoadInfo.GetCommitTimestamp(); commitTs != 0 {
			commitStart := time.Now()
			if err := cseg.SetCommitTimestamp(commitTs); err != nil {
				commitDur = time.Since(commitStart)
				C.DeleteSegment(ptr)
				return nil, errors.Wrap(err, "failed to set commit timestamp on segment")
			}
			commitDur = time.Since(commitStart)
		}
	}
	return seg, nil
}

// cSegmentImpl is a wrapper for cSegmentImplInterface.
type cSegmentImpl struct {
	id  int64
	ptr C.CSegmentInterface
}

// ID returns the ID of the segment.
func (s *cSegmentImpl) ID() int64 {
	return s.id
}

// RawPointer returns the raw pointer of the segment.
func (s *cSegmentImpl) RawPointer() CSegmentInterface {
	return CSegmentInterface(s.ptr)
}

// RowNum returns the number of rows in the segment.
func (s *cSegmentImpl) RowNum() int64 {
	rowCount := C.GetRealCount(s.ptr)
	return int64(rowCount)
}

// MemSize returns the memory size of the segment.
func (s *cSegmentImpl) MemSize() int64 {
	cMemSize := C.GetMemoryUsageInBytes(s.ptr)
	return int64(cMemSize)
}

// HasRawData checks if the segment has raw data.
func (s *cSegmentImpl) HasRawData(fieldID int64) bool {
	ret := C.HasRawData(s.ptr, C.int64_t(fieldID))
	return bool(ret)
}

// HasFieldData checks if the segment has field data.
func (s *cSegmentImpl) HasFieldData(fieldID int64) bool {
	ret := C.HasFieldData(s.ptr, C.int64_t(fieldID))
	return bool(ret)
}

// Search requests a search on the segment.
// If searchReq.FilterOnly() is true, only executes the filter and returns valid_count (Stage 1 of two-stage search).
func (s *cSegmentImpl) Search(ctx context.Context, searchReq *SearchRequest) (*SearchResult, error) {
	traceCtx := ParseCTraceContext(ctx)
	defer runtime.KeepAlive(traceCtx)
	defer runtime.KeepAlive(searchReq)

	// Use physical time for entity-level TTL (issue #47413)
	physicalTimeUs := int64(searchReq.entityTTLPhysicalTime)
	if physicalTimeUs == 0 {
		physicalTimeMs, _ := tsoutil.ParseHybridTs(searchReq.mvccTimestamp)
		physicalTimeUs = physicalTimeMs * 1000
	}

	future := cgo.Async(ctx,
		func() cgo.CFuturePtr {
			return (cgo.CFuturePtr)(C.AsyncSearch(
				traceCtx.ctx,
				s.ptr,
				searchReq.plan.cSearchPlan,
				searchReq.cPlaceholderGroup,
				C.uint64_t(searchReq.mvccTimestamp),
				C.int32_t(searchReq.consistencyLevel),
				C.uint64_t(searchReq.collectionTTL),
				C.uint64_t(physicalTimeUs),
				C.bool(searchReq.filterOnly),
				C.bool(searchReq.enableExprCache),
			))
		},
		cgo.WithName("search"),
	)
	defer future.Release()

	result, err := future.BlockAndLeakyGet()
	if err != nil {
		return nil, err
	}
	return &SearchResult{cSearchResult: (C.CSearchResult)(result)}, nil
}

// Retrieve retrieves entities from the segment.
func (s *cSegmentImpl) Retrieve(ctx context.Context, plan *RetrievePlan) (*RetrieveResult, error) {
	traceCtx := ParseCTraceContext(ctx)
	defer runtime.KeepAlive(traceCtx)
	defer runtime.KeepAlive(plan)

	// Use physical time for entity-level TTL (issue #47413)
	physicalTimeUs := int64(plan.entityTTLPhysicalTime)
	if physicalTimeUs == 0 {
		physicalTimeMs, _ := tsoutil.ParseHybridTs(plan.Timestamp)
		physicalTimeUs = physicalTimeMs * 1000
	}

	future := cgo.Async(
		ctx,
		func() cgo.CFuturePtr {
			return (cgo.CFuturePtr)(C.AsyncRetrieve(
				traceCtx.ctx,
				s.ptr,
				plan.cRetrievePlan,
				C.uint64_t(plan.Timestamp),
				C.int64_t(plan.maxLimitSize),
				C.bool(plan.ignoreNonPk),
				C.int32_t(plan.consistencyLevel),
				C.uint64_t(plan.collectionTTL),
				C.uint64_t(physicalTimeUs),
			))
		},
		cgo.WithName("retrieve"),
	)
	defer future.Release()
	result, err := future.BlockAndLeakyGet()
	if err != nil {
		return nil, err
	}
	return &RetrieveResult{cRetrieveResult: (*C.CRetrieveResult)(result)}, nil
}

// RetrieveByOffsets retrieves entities from the segment by offsets.
func (s *cSegmentImpl) RetrieveByOffsets(ctx context.Context, plan *RetrievePlanWithOffsets) (*RetrieveResult, error) {
	if len(plan.Offsets) == 0 {
		return nil, merr.WrapErrParameterInvalid("segment offsets", "empty offsets")
	}

	traceCtx := ParseCTraceContext(ctx)
	defer runtime.KeepAlive(traceCtx)
	defer runtime.KeepAlive(plan)
	defer runtime.KeepAlive(plan.Offsets)

	future := cgo.Async(
		ctx,
		func() cgo.CFuturePtr {
			return (cgo.CFuturePtr)(C.AsyncRetrieveByOffsets(
				traceCtx.ctx,
				s.ptr,
				plan.cRetrievePlan,
				(*C.int64_t)(unsafe.Pointer(&plan.Offsets[0])),
				C.int64_t(len(plan.Offsets)),
			))
		},
		cgo.WithName("retrieve-by-offsets"),
	)
	defer future.Release()
	result, err := future.BlockAndLeakyGet()
	if err != nil {
		return nil, err
	}
	return &RetrieveResult{cRetrieveResult: (*C.CRetrieveResult)(result)}, nil
}

// Insert inserts entities into the segment.
func (s *cSegmentImpl) Insert(ctx context.Context, request *InsertRequest) (*InsertResult, error) {
	offset, err := s.preInsert(len(request.RowIDs))
	if err != nil {
		return nil, err
	}

	insertRecordBlob, err := proto.Marshal(request.Record)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal insert record: %s", err)
	}

	numOfRow := len(request.RowIDs)
	cOffset := C.int64_t(offset)
	cNumOfRows := C.int64_t(numOfRow)
	cEntityIDsPtr := (*C.int64_t)(&(request.RowIDs)[0])
	cTimestampsPtr := (*C.uint64_t)(&(request.Timestamps)[0])

	status := C.Insert(s.ptr,
		cOffset,
		cNumOfRows,
		cEntityIDsPtr,
		cTimestampsPtr,
		(*C.uint8_t)(unsafe.Pointer(&insertRecordBlob[0])),
		(C.uint64_t)(len(insertRecordBlob)),
	)
	return &InsertResult{InsertedRows: int64(numOfRow)}, ConsumeCStatusIntoError(&status)
}

func (s *cSegmentImpl) preInsert(numOfRecords int) (int64, error) {
	var offset int64
	cOffset := (*C.int64_t)(&offset)
	status := C.PreInsert(s.ptr, C.int64_t(int64(numOfRecords)), cOffset)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return 0, err
	}
	return offset, nil
}

// Delete deletes entities from the segment.
func (s *cSegmentImpl) Delete(ctx context.Context, request *DeleteRequest) (*DeleteResult, error) {
	cSize := C.int64_t(request.PrimaryKeys.Len())
	cTimestampsPtr := (*C.uint64_t)(&(request.Timestamps)[0])

	ids, err := storage.ParsePrimaryKeysBatch2IDs(request.PrimaryKeys)
	if err != nil {
		return nil, err
	}

	dataBlob, err := proto.Marshal(ids)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ids: %s", err)
	}
	status := C.Delete(s.ptr,
		cSize,
		(*C.uint8_t)(unsafe.Pointer(&dataBlob[0])),
		(C.uint64_t)(len(dataBlob)),
		cTimestampsPtr,
	)
	return &DeleteResult{}, ConsumeCStatusIntoError(&status)
}

// LoadFieldData loads field data into the segment.
func (s *cSegmentImpl) LoadFieldData(ctx context.Context, request *LoadFieldDataRequest) (*LoadFieldDataResult, error) {
	creq, err := request.getCLoadFieldDataRequest()
	if err != nil {
		return nil, err
	}
	defer creq.Release()

	status := C.LoadFieldData(s.ptr, creq.cLoadFieldDataInfo)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "failed to load field data")
	}
	return &LoadFieldDataResult{}, nil
}

func (s *cSegmentImpl) Load(ctx context.Context) error {
	traceCtx := ParseCTraceContext(ctx)
	defer runtime.KeepAlive(traceCtx)

	future := cgo.Async(ctx,
		func() cgo.CFuturePtr {
			return (cgo.CFuturePtr)(C.AsyncSegmentLoad(
				traceCtx.ctx,
				s.ptr,
			))
		},
		cgo.WithName("segment-load"),
	)
	defer future.Release()
	_, err := future.BlockAndLeakyGet()
	return err
}

func (s *cSegmentImpl) Reopen(ctx context.Context, req *ReopenRequest) error {
	if req == nil {
		return errors.New("reopen request is nil")
	}
	if req.LoadInfo == nil {
		return errors.New("reopen load info is nil")
	}
	if req.Schema == nil {
		return errors.New("reopen schema is nil")
	}

	traceCtx := ParseCTraceContext(ctx)
	defer runtime.KeepAlive(traceCtx)
	defer runtime.KeepAlive(req)

	segLoadInfo := ConvertToSegcoreSegmentLoadInfo(req.LoadInfo)
	loadInfoBlob, err := proto.Marshal(segLoadInfo)
	if err != nil {
		return err
	}
	if len(loadInfoBlob) == 0 {
		return errors.New("reopen load info blob is empty")
	}

	schemaBlob, err := proto.Marshal(req.Schema)
	if err != nil {
		return err
	}
	if len(schemaBlob) == 0 {
		return errors.New("reopen schema blob is empty")
	}
	defer runtime.KeepAlive(schemaBlob)

	future := cgo.Async(ctx,
		func() cgo.CFuturePtr {
			return (cgo.CFuturePtr)(C.AsyncReopenSegment(
				traceCtx.ctx,
				s.ptr,
				(*C.uint8_t)(unsafe.Pointer(&loadInfoBlob[0])),
				C.int64_t(len(loadInfoBlob)),
				unsafe.Pointer(&schemaBlob[0]),
				C.int64_t(len(schemaBlob)),
				C.uint64_t(req.SchemaVersion),
			))
		},
		cgo.WithName("segment-reopen"),
	)
	defer future.Release()
	_, err = future.BlockAndLeakyGet()
	return err
}

func (s *cSegmentImpl) DropIndex(ctx context.Context, fieldID int64) error {
	status := C.DropSealedSegmentIndex(s.ptr, C.int64_t(fieldID))
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return errors.Wrap(err, "failed to drop index")
	}
	return nil
}

func (s *cSegmentImpl) DropJSONIndex(ctx context.Context, fieldID int64, nestedPath string) error {
	status := C.DropSealedSegmentJSONIndex(s.ptr, C.int64_t(fieldID), C.CString(nestedPath))
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return errors.Wrap(err, "failed to drop json index")
	}
	return nil
}

// Release releases the segment.
func (s *cSegmentImpl) Release() {
	C.DeleteSegment(s.ptr)
}

// SetCommitTimestamp sets the commit timestamp for the segment.
// Import segments use this to ensure rows with old historical timestamps are
// not visible to queries dispatched before T_commit.
func (s *cSegmentImpl) SetCommitTimestamp(ts uint64) error {
	status := C.SegmentSetCommitTimestamp(s.ptr, C.uint64_t(ts))
	return ConsumeCStatusIntoError(&status)
}

// ConvertToSegcoreSegmentLoadInfo converts querypb.SegmentLoadInfo to segcorepb.SegmentLoadInfo.
// This function is needed because segcorepb.SegmentLoadInfo is a simplified version that doesn't
// depend on data_coord.proto and excludes fields like start_position, delta_position, and level.
func ConvertToSegcoreSegmentLoadInfo(src *querypb.SegmentLoadInfo) *segcorepb.SegmentLoadInfo {
	info, _ := convertToSegcoreSegmentLoadInfoWithTiming(src, true)
	return info
}

func forceSkipNewSegmentStatsExperiment() bool {
	switch strings.ToLower(os.Getenv("MILVUS_EXPERIMENT_SKIP_NEW_SEGMENT_STATS")) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func schemaNeedsManifestStats(schema *schemapb.CollectionSchema) bool {
	if schema == nil {
		return true
	}
	for _, field := range typeutil.GetAllFieldSchemas(schema) {
		helper := typeutil.CreateFieldSchemaHelper(field)
		if helper.EnableMatch() {
			return true
		}
		if paramtable.Get().CommonCfg.EnabledJSONKeyStats.GetAsBool() && helper.EnableJSONKeyStatsIndex() {
			return true
		}
	}
	return false
}

func convertToSegcoreSegmentLoadInfoWithTiming(src *querypb.SegmentLoadInfo, resolveManifestStats bool) (info *segcorepb.SegmentLoadInfo, timing segmentLoadInfoConversionTiming) {
	start := time.Now()
	defer func() {
		timing.totalDur = time.Since(start)
	}()

	if src == nil {
		return nil, timing
	}

	// Resolve text/json stats with basePaths.
	// V2: stats come from src proto fields, basePaths computed from metadata + rootPath.
	// V3: stats resolved from manifest (src proto fields are empty), basePaths from manifest paths.
	stageStart := time.Now()
	textStats, jsonStats, textBasePaths, jsonBasePaths := resolveStatsWithBasePathsOption(src, resolveManifestStats)
	timing.resolveStatsDur = time.Since(stageStart)

	timing.binlogFieldCount = len(src.GetBinlogPaths())
	timing.statslogFieldCount = len(src.GetStatslogs())
	timing.deltalogFieldCount = len(src.GetDeltalogs())
	timing.indexInfoCount = len(src.GetIndexInfos())
	timing.textStatsCount = len(textStats)
	timing.bm25FieldCount = len(src.GetBm25Logs())
	timing.jsonStatsCount = len(jsonStats)

	stageStart = time.Now()
	binlogPaths := convertFieldBinlogs(src.GetBinlogPaths())
	timing.binlogsDur = time.Since(stageStart)

	stageStart = time.Now()
	statslogs := convertFieldBinlogs(src.GetStatslogs())
	timing.statslogsDur = time.Since(stageStart)

	stageStart = time.Now()
	deltalogs := convertFieldBinlogs(src.GetDeltalogs())
	timing.deltalogsDur = time.Since(stageStart)

	stageStart = time.Now()
	indexInfos := convertFieldIndexInfos(src.GetIndexInfos())
	timing.indexInfosDur = time.Since(stageStart)

	stageStart = time.Now()
	textStatsLogs := convertTextIndexStats(textStats, textBasePaths)
	timing.textStatsDur = time.Since(stageStart)

	stageStart = time.Now()
	bm25Logs := convertFieldBinlogs(src.GetBm25Logs())
	timing.bm25Dur = time.Since(stageStart)

	stageStart = time.Now()
	jsonKeyStatsLogs := convertJSONKeyStats(jsonStats, jsonBasePaths)
	timing.jsonStatsDur = time.Since(stageStart)

	info = &segcorepb.SegmentLoadInfo{
		SegmentID:            src.GetSegmentID(),
		PartitionID:          src.GetPartitionID(),
		CollectionID:         src.GetCollectionID(),
		DbID:                 src.GetDbID(),
		FlushTime:            src.GetFlushTime(),
		BinlogPaths:          binlogPaths,
		NumOfRows:            src.GetNumOfRows(),
		Statslogs:            statslogs,
		Deltalogs:            deltalogs,
		CompactionFrom:       src.GetCompactionFrom(),
		IndexInfos:           indexInfos,
		SegmentSize:          src.GetSegmentSize(),
		InsertChannel:        src.GetInsertChannel(),
		ReadableVersion:      src.GetReadableVersion(),
		StorageVersion:       src.GetStorageVersion(),
		IsSorted:             src.GetIsSorted(),
		TextStatsLogs:        textStatsLogs,
		Bm25Logs:             bm25Logs,
		JsonKeyStatsLogs:     jsonKeyStatsLogs,
		Priority:             src.GetPriority(),
		ManifestPath:         src.GetManifestPath(),
		UseTakeForOutput:     paramtable.Get().QueryNodeCfg.ExternalCollectionUseTakeForOutput.GetAsBool(),
		EstimatedBytesPerRow: src.GetEstimatedBytesPerRow(),
		CommitTimestamp:      src.GetCommitTimestamp(),
	}
	return info, timing
}

// resolveStatsWithBasePaths resolves text/json stats and computes basePaths.
// V2: stats from src proto fields, basePaths computed from rootPath + metadata.
// V3: stats resolved from manifest via StatsResolver, basePaths extracted from manifest paths.
func resolveStatsWithBasePaths(src *querypb.SegmentLoadInfo) (
	map[int64]*datapb.TextIndexStats,
	map[int64]*datapb.JsonKeyStats,
	map[int64]string, // textBasePaths
	map[int64]string, // jsonBasePaths
) {
	return resolveStatsWithBasePathsOption(src, true)
}

func resolveStatsWithBasePathsOption(src *querypb.SegmentLoadInfo, resolveManifestStats bool) (
	map[int64]*datapb.TextIndexStats,
	map[int64]*datapb.JsonKeyStats,
	map[int64]string, // textBasePaths
	map[int64]string, // jsonBasePaths
) {
	textStats := src.GetTextStatsLogs()
	jsonStats := src.GetJsonKeyStatsLogs()

	// For V3 (manifest-based): resolve stats from manifest if proto fields are empty.
	if src.GetStorageVersion() == storage.StorageV3 && resolveManifestStats {
		result := packed.NewStatsResolverFromLoadInfo(src).TextAndJSONIndexStatsWithBasePaths()
		if result.Err() != nil {
			log.Warn("failed to resolve stats from manifest for segcore load info",
				zap.Int64("segmentID", src.GetSegmentID()),
				zap.String("manifestPath", src.GetManifestPath()),
				zap.Error(result.Err()))
		} else {
			return result.TextIndexStats, result.JSONKeyStats, result.TextBasePaths, result.JSONBasePaths
		}
	}

	// V2: compute basePaths from rootPath + stats metadata.
	rootPath := paramtable.Get().MinioCfg.RootPath.GetValue()

	textBasePaths := make(map[int64]string, len(textStats))
	for fieldID, stats := range textStats {
		textBasePaths[fieldID] = metautil.BuildTextIndexPrefix(rootPath,
			stats.GetBuildID(), stats.GetVersion(),
			src.GetCollectionID(), src.GetPartitionID(), src.GetSegmentID(), fieldID)
	}

	jsonBasePaths := make(map[int64]string, len(jsonStats))
	for fieldID, stats := range jsonStats {
		jsonBasePaths[fieldID] = metautil.BuildJSONKeyStatsPrefix(rootPath, stats.GetJsonKeyStatsDataFormat(),
			stats.GetBuildID(), stats.GetVersion(),
			src.GetCollectionID(), src.GetPartitionID(), src.GetSegmentID(), fieldID)
	}

	return textStats, jsonStats, textBasePaths, jsonBasePaths
}

// convertFieldBinlogs converts datapb.FieldBinlog to segcorepb.FieldBinlog.
func convertFieldBinlogs(src []*datapb.FieldBinlog) []*segcorepb.FieldBinlog {
	if src == nil {
		return nil
	}

	result := make([]*segcorepb.FieldBinlog, 0, len(src))
	for _, fb := range src {
		if fb == nil {
			continue
		}

		result = append(result, &segcorepb.FieldBinlog{
			FieldID:     fb.GetFieldID(),
			Binlogs:     convertBinlogs(fb.GetBinlogs()),
			ChildFields: fb.GetChildFields(),
		})
	}
	return result
}

// convertBinlogs converts datapb.Binlog to segcorepb.Binlog.
func convertBinlogs(src []*datapb.Binlog) []*segcorepb.Binlog {
	if src == nil {
		return nil
	}

	result := make([]*segcorepb.Binlog, 0, len(src))
	for _, b := range src {
		if b == nil {
			continue
		}

		result = append(result, &segcorepb.Binlog{
			EntriesNum:    b.GetEntriesNum(),
			TimestampFrom: b.GetTimestampFrom(),
			TimestampTo:   b.GetTimestampTo(),
			LogPath:       b.GetLogPath(),
			LogSize:       b.GetLogSize(),
			LogID:         b.GetLogID(),
			MemorySize:    b.GetMemorySize(),
		})
	}
	return result
}

// convertFieldIndexInfos converts querypb.FieldIndexInfo to segcorepb.FieldIndexInfo.
func convertFieldIndexInfos(src []*querypb.FieldIndexInfo) []*segcorepb.FieldIndexInfo {
	if src == nil {
		return nil
	}

	result := make([]*segcorepb.FieldIndexInfo, 0, len(src))
	for _, fii := range src {
		if fii == nil {
			continue
		}

		result = append(result, &segcorepb.FieldIndexInfo{
			FieldID:                   fii.GetFieldID(),
			EnableIndex:               fii.GetEnableIndex(),
			IndexName:                 fii.GetIndexName(),
			IndexID:                   fii.GetIndexID(),
			BuildID:                   fii.GetBuildID(),
			IndexParams:               fii.GetIndexParams(),
			IndexFilePaths:            fii.GetIndexFilePaths(),
			IndexSize:                 fii.GetIndexSize(),
			IndexVersion:              fii.GetIndexVersion(),
			NumRows:                   fii.GetNumRows(),
			CurrentIndexVersion:       fii.GetCurrentIndexVersion(),
			CurrentScalarIndexVersion: fii.GetCurrentScalarIndexVersion(),
		})
	}
	return result
}

// convertTextIndexStats converts datapb.TextIndexStats to segcorepb.TextIndexStats.
func convertTextIndexStats(src map[int64]*datapb.TextIndexStats, basePaths map[int64]string) map[int64]*segcorepb.TextIndexStats {
	if src == nil {
		return nil
	}

	result := make(map[int64]*segcorepb.TextIndexStats, len(src))
	for k, v := range src {
		if v == nil {
			continue
		}

		files := v.GetFiles()
		basePath := basePaths[k]
		// V2 legacy segments may carry full paths in Files (reconstructed by
		// metautil.BuildTextLogPaths on etcd load). The C++ loader expects
		// relative filenames and prepends BasePath itself, so strip any
		// basePath prefix here to honor the contract.
		if basePath != "" {
			prefix := basePath + "/"
			stripped := make([]string, len(files))
			for i, f := range files {
				stripped[i] = strings.TrimPrefix(f, prefix)
			}
			files = stripped
		}
		log.Info("convertTextIndexStats",
			zap.Int64("fieldID", v.GetFieldID()),
			zap.Int64("buildID", v.GetBuildID()),
			zap.Int64("version", v.GetVersion()),
			zap.String("basePath", basePath),
			zap.Int("fileCount", len(files)),
			zap.Strings("files", files),
		)

		result[k] = &segcorepb.TextIndexStats{
			FieldID:                   v.GetFieldID(),
			Version:                   v.GetVersion(),
			Files:                     files,
			LogSize:                   v.GetLogSize(),
			MemorySize:                v.GetMemorySize(),
			BuildID:                   v.GetBuildID(),
			CurrentScalarIndexVersion: v.GetCurrentScalarIndexVersion(),
			BasePath:                  basePath,
		}
	}
	return result
}

// convertJSONKeyStats converts datapb.JsonKeyStats to segcorepb.JsonKeyStats.
func convertJSONKeyStats(src map[int64]*datapb.JsonKeyStats, basePaths map[int64]string) map[int64]*segcorepb.JsonKeyStats {
	if src == nil {
		return nil
	}

	result := make(map[int64]*segcorepb.JsonKeyStats, len(src))
	for k, v := range src {
		if v == nil {
			continue
		}

		files := v.GetFiles()
		basePath := basePaths[k]
		// V2 legacy segments may carry full paths in Files; strip basePath
		// prefix so the C++ loader (which prepends BasePath) sees relative
		// filenames. See convertTextIndexStats for details.
		if basePath != "" {
			prefix := basePath + "/"
			stripped := make([]string, len(files))
			for i, f := range files {
				stripped[i] = strings.TrimPrefix(f, prefix)
			}
			files = stripped
		}
		log.Info("convertJSONKeyStats",
			zap.Int64("fieldID", v.GetFieldID()),
			zap.Int64("buildID", v.GetBuildID()),
			zap.Int64("version", v.GetVersion()),
			zap.String("basePath", basePath),
			zap.Int("fileCount", len(files)),
			zap.Strings("files", files),
		)

		result[k] = &segcorepb.JsonKeyStats{
			FieldID:                v.GetFieldID(),
			Version:                v.GetVersion(),
			Files:                  files,
			LogSize:                v.GetLogSize(),
			MemorySize:             v.GetMemorySize(),
			BuildID:                v.GetBuildID(),
			JsonKeyStatsDataFormat: v.GetJsonKeyStatsDataFormat(),
			BasePath:               basePath,
		}
	}
	return result
}
