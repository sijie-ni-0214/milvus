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

#include "segcore/load_index_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"io"
	"math"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/atomic"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/pkoracle"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/indexparams"
	"github.com/milvus-io/milvus/pkg/v3/util/logutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/syncutil"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	UsedDiskMemoryRatio      = 4
	UsedDiskMemoryRatioAisaq = 64
)

var errRetryTimerNotified = errors.New("retry timer notified")

const requestResourceTimingLogInterval = 5 * time.Second

var (
	requestResourceTiming         = newRequestResourceTimingStats()
	estimateSegmentResourceTiming = newEstimateSegmentResourceTimingStats()
	segmentLoadTiming             = newSegmentLoadTimingStats()
	sealedLoadTiming              = newSealedLoadTimingStats()
	bloomFilterLoadTiming         = newBloomFilterLoadTimingStats()
)

type requestResourceTimingStats struct {
	count           *atomic.Int64
	totalEstimate   *atomic.Int64
	totalLockWait   *atomic.Int64
	totalLockHold   *atomic.Int64
	totalRequest    *atomic.Int64
	maxEstimate     *atomic.Int64
	maxLockWait     *atomic.Int64
	maxLockHold     *atomic.Int64
	maxRequest      *atomic.Int64
	lastLogUnixNano *atomic.Int64
}

func newRequestResourceTimingStats() *requestResourceTimingStats {
	return &requestResourceTimingStats{
		count:           atomic.NewInt64(0),
		totalEstimate:   atomic.NewInt64(0),
		totalLockWait:   atomic.NewInt64(0),
		totalLockHold:   atomic.NewInt64(0),
		totalRequest:    atomic.NewInt64(0),
		maxEstimate:     atomic.NewInt64(0),
		maxLockWait:     atomic.NewInt64(0),
		maxLockHold:     atomic.NewInt64(0),
		maxRequest:      atomic.NewInt64(0),
		lastLogUnixNano: atomic.NewInt64(time.Now().UnixNano()),
	}
}

func updateMaxDuration(max *atomic.Int64, value time.Duration) {
	v := int64(value)
	for {
		old := max.Load()
		if v <= old || max.CompareAndSwap(old, v) {
			return
		}
	}
}

func avgDuration(total int64, count int64) time.Duration {
	if count == 0 {
		return 0
	}
	return time.Duration(total / count)
}

type segmentLoadTimingStats struct {
	count            *atomic.Int64
	totalLoadData    *atomic.Int64
	totalDeltaLog    *atomic.Int64
	totalPKCandidate *atomic.Int64
	totalPut         *atomic.Int64
	totalNotify      *atomic.Int64
	totalSegment     *atomic.Int64
	maxLoadData      *atomic.Int64
	maxDeltaLog      *atomic.Int64
	maxPKCandidate   *atomic.Int64
	maxPut           *atomic.Int64
	maxNotify        *atomic.Int64
	maxSegment       *atomic.Int64
	lastLogUnixNano  *atomic.Int64
}

type sealedLoadTimingStats struct {
	count              *atomic.Int64
	totalStateLock     *atomic.Int64
	totalPoolWait      *atomic.Int64
	totalCgoLoad       *atomic.Int64
	totalSyncJSONStats *atomic.Int64
	totalPatchEntry    *atomic.Int64
	totalSealedLoad    *atomic.Int64
	maxStateLock       *atomic.Int64
	maxPoolWait        *atomic.Int64
	maxCgoLoad         *atomic.Int64
	maxSyncJSONStats   *atomic.Int64
	maxPatchEntry      *atomic.Int64
	maxSealedLoad      *atomic.Int64
	lastLogUnixNano    *atomic.Int64
}

type bloomFilterLoadTimingStats struct {
	count                *atomic.Int64
	segmentCount         *atomic.Int64
	totalStub            *atomic.Int64
	totalMetadata        *atomic.Int64
	totalMemoryEstimate  *atomic.Int64
	totalReserve         *atomic.Int64
	totalRemoteLoad      *atomic.Int64
	totalCharge          *atomic.Int64
	totalBloomFilterLoad *atomic.Int64
	maxStub              *atomic.Int64
	maxMetadata          *atomic.Int64
	maxMemoryEstimate    *atomic.Int64
	maxReserve           *atomic.Int64
	maxRemoteLoad        *atomic.Int64
	maxCharge            *atomic.Int64
	maxBloomFilterLoad   *atomic.Int64
	lastLogUnixNano      *atomic.Int64
}

type estimateSegmentResourceTimingStats struct {
	count           *atomic.Int64
	totalIndex      *atomic.Int64
	totalBinlog     *atomic.Int64
	totalSchema     *atomic.Int64
	totalIndexLoop  *atomic.Int64
	totalCLoadInfo  *atomic.Int64
	totalCEstimate  *atomic.Int64
	totalBinlogLoop *atomic.Int64
	totalStats      *atomic.Int64
	totalDelete     *atomic.Int64
	totalJSONStats  *atomic.Int64
	totalTextStats  *atomic.Int64
	totalSegment    *atomic.Int64
	maxSegment      *atomic.Int64
	lastLogUnixNano *atomic.Int64
}

func newEstimateSegmentResourceTimingStats() *estimateSegmentResourceTimingStats {
	return &estimateSegmentResourceTimingStats{
		count:           atomic.NewInt64(0),
		totalIndex:      atomic.NewInt64(0),
		totalBinlog:     atomic.NewInt64(0),
		totalSchema:     atomic.NewInt64(0),
		totalIndexLoop:  atomic.NewInt64(0),
		totalCLoadInfo:  atomic.NewInt64(0),
		totalCEstimate:  atomic.NewInt64(0),
		totalBinlogLoop: atomic.NewInt64(0),
		totalStats:      atomic.NewInt64(0),
		totalDelete:     atomic.NewInt64(0),
		totalJSONStats:  atomic.NewInt64(0),
		totalTextStats:  atomic.NewInt64(0),
		totalSegment:    atomic.NewInt64(0),
		maxSegment:      atomic.NewInt64(0),
		lastLogUnixNano: atomic.NewInt64(time.Now().UnixNano()),
	}
}

func newSealedLoadTimingStats() *sealedLoadTimingStats {
	return &sealedLoadTimingStats{
		count:              atomic.NewInt64(0),
		totalStateLock:     atomic.NewInt64(0),
		totalPoolWait:      atomic.NewInt64(0),
		totalCgoLoad:       atomic.NewInt64(0),
		totalSyncJSONStats: atomic.NewInt64(0),
		totalPatchEntry:    atomic.NewInt64(0),
		totalSealedLoad:    atomic.NewInt64(0),
		maxStateLock:       atomic.NewInt64(0),
		maxPoolWait:        atomic.NewInt64(0),
		maxCgoLoad:         atomic.NewInt64(0),
		maxSyncJSONStats:   atomic.NewInt64(0),
		maxPatchEntry:      atomic.NewInt64(0),
		maxSealedLoad:      atomic.NewInt64(0),
		lastLogUnixNano:    atomic.NewInt64(time.Now().UnixNano()),
	}
}

func newSegmentLoadTimingStats() *segmentLoadTimingStats {
	return &segmentLoadTimingStats{
		count:            atomic.NewInt64(0),
		totalLoadData:    atomic.NewInt64(0),
		totalDeltaLog:    atomic.NewInt64(0),
		totalPKCandidate: atomic.NewInt64(0),
		totalPut:         atomic.NewInt64(0),
		totalNotify:      atomic.NewInt64(0),
		totalSegment:     atomic.NewInt64(0),
		maxLoadData:      atomic.NewInt64(0),
		maxDeltaLog:      atomic.NewInt64(0),
		maxPKCandidate:   atomic.NewInt64(0),
		maxPut:           atomic.NewInt64(0),
		maxNotify:        atomic.NewInt64(0),
		maxSegment:       atomic.NewInt64(0),
		lastLogUnixNano:  atomic.NewInt64(time.Now().UnixNano()),
	}
}

func newBloomFilterLoadTimingStats() *bloomFilterLoadTimingStats {
	return &bloomFilterLoadTimingStats{
		count:                atomic.NewInt64(0),
		segmentCount:         atomic.NewInt64(0),
		totalStub:            atomic.NewInt64(0),
		totalMetadata:        atomic.NewInt64(0),
		totalMemoryEstimate:  atomic.NewInt64(0),
		totalReserve:         atomic.NewInt64(0),
		totalRemoteLoad:      atomic.NewInt64(0),
		totalCharge:          atomic.NewInt64(0),
		totalBloomFilterLoad: atomic.NewInt64(0),
		maxStub:              atomic.NewInt64(0),
		maxMetadata:          atomic.NewInt64(0),
		maxMemoryEstimate:    atomic.NewInt64(0),
		maxReserve:           atomic.NewInt64(0),
		maxRemoteLoad:        atomic.NewInt64(0),
		maxCharge:            atomic.NewInt64(0),
		maxBloomFilterLoad:   atomic.NewInt64(0),
		lastLogUnixNano:      atomic.NewInt64(time.Now().UnixNano()),
	}
}

func (s *sealedLoadTimingStats) record(stateLockDur, poolWaitDur, cgoLoadDur, syncJSONStatsDur, patchEntryDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalStateLock.Add(int64(stateLockDur))
	s.totalPoolWait.Add(int64(poolWaitDur))
	s.totalCgoLoad.Add(int64(cgoLoadDur))
	s.totalSyncJSONStats.Add(int64(syncJSONStatsDur))
	s.totalPatchEntry.Add(int64(patchEntryDur))
	s.totalSealedLoad.Add(int64(totalDur))
	updateMaxDuration(s.maxStateLock, stateLockDur)
	updateMaxDuration(s.maxPoolWait, poolWaitDur)
	updateMaxDuration(s.maxCgoLoad, cgoLoadDur)
	updateMaxDuration(s.maxSyncJSONStats, syncJSONStatsDur)
	updateMaxDuration(s.maxPatchEntry, patchEntryDur)
	updateMaxDuration(s.maxSealedLoad, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}

	count := s.count.Swap(0)
	totalStateLock := s.totalStateLock.Swap(0)
	totalPoolWait := s.totalPoolWait.Swap(0)
	totalCgoLoad := s.totalCgoLoad.Swap(0)
	totalSyncJSONStats := s.totalSyncJSONStats.Swap(0)
	totalPatchEntry := s.totalPatchEntry.Swap(0)
	totalSealedLoad := s.totalSealedLoad.Swap(0)
	maxStateLock := s.maxStateLock.Swap(0)
	maxPoolWait := s.maxPoolWait.Swap(0)
	maxCgoLoad := s.maxCgoLoad.Swap(0)
	maxSyncJSONStats := s.maxSyncJSONStats.Swap(0)
	maxPatchEntry := s.maxPatchEntry.Swap(0)
	maxSealedLoad := s.maxSealedLoad.Swap(0)
	if count == 0 {
		return
	}

	pool := GetLoadPool()
	log.Warn("sealed segment load timing stats",
		zap.Int64("count", count),
		zap.Duration("avgStateLockDur", avgDuration(totalStateLock, count)),
		zap.Duration("avgPoolWaitDur", avgDuration(totalPoolWait, count)),
		zap.Duration("avgCgoLoadDur", avgDuration(totalCgoLoad, count)),
		zap.Duration("avgSyncJSONStatsDur", avgDuration(totalSyncJSONStats, count)),
		zap.Duration("avgPatchEntryDur", avgDuration(totalPatchEntry, count)),
		zap.Duration("avgTotalDur", avgDuration(totalSealedLoad, count)),
		zap.Duration("maxStateLockDur", time.Duration(maxStateLock)),
		zap.Duration("maxPoolWaitDur", time.Duration(maxPoolWait)),
		zap.Duration("maxCgoLoadDur", time.Duration(maxCgoLoad)),
		zap.Duration("maxSyncJSONStatsDur", time.Duration(maxSyncJSONStats)),
		zap.Duration("maxPatchEntryDur", time.Duration(maxPatchEntry)),
		zap.Duration("maxTotalDur", time.Duration(maxSealedLoad)),
		zap.Int("loadPoolCap", pool.Cap()),
		zap.Int("loadPoolRunning", pool.Running()),
		zap.Int("loadPoolWaiting", pool.Waiting()),
	)
}

func (s *segmentLoadTimingStats) record(loadDataDur, deltaLogDur, pkCandidateDur, putDur, notifyDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalLoadData.Add(int64(loadDataDur))
	s.totalDeltaLog.Add(int64(deltaLogDur))
	s.totalPKCandidate.Add(int64(pkCandidateDur))
	s.totalPut.Add(int64(putDur))
	s.totalNotify.Add(int64(notifyDur))
	s.totalSegment.Add(int64(totalDur))
	updateMaxDuration(s.maxLoadData, loadDataDur)
	updateMaxDuration(s.maxDeltaLog, deltaLogDur)
	updateMaxDuration(s.maxPKCandidate, pkCandidateDur)
	updateMaxDuration(s.maxPut, putDur)
	updateMaxDuration(s.maxNotify, notifyDur)
	updateMaxDuration(s.maxSegment, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}

	count := s.count.Swap(0)
	totalLoadData := s.totalLoadData.Swap(0)
	totalDeltaLog := s.totalDeltaLog.Swap(0)
	totalPKCandidate := s.totalPKCandidate.Swap(0)
	totalPut := s.totalPut.Swap(0)
	totalNotify := s.totalNotify.Swap(0)
	totalSegment := s.totalSegment.Swap(0)
	maxLoadData := s.maxLoadData.Swap(0)
	maxDeltaLog := s.maxDeltaLog.Swap(0)
	maxPKCandidate := s.maxPKCandidate.Swap(0)
	maxPut := s.maxPut.Swap(0)
	maxNotify := s.maxNotify.Swap(0)
	maxSegment := s.maxSegment.Swap(0)
	if count == 0 {
		return
	}

	pool := GetLoadPool()
	log.Warn("segment load timing stats",
		zap.Int64("count", count),
		zap.Duration("avgLoadDataDur", avgDuration(totalLoadData, count)),
		zap.Duration("avgDeltaLogDur", avgDuration(totalDeltaLog, count)),
		zap.Duration("avgPKCandidateDur", avgDuration(totalPKCandidate, count)),
		zap.Duration("avgPutDur", avgDuration(totalPut, count)),
		zap.Duration("avgNotifyDur", avgDuration(totalNotify, count)),
		zap.Duration("avgTotalDur", avgDuration(totalSegment, count)),
		zap.Duration("maxLoadDataDur", time.Duration(maxLoadData)),
		zap.Duration("maxDeltaLogDur", time.Duration(maxDeltaLog)),
		zap.Duration("maxPKCandidateDur", time.Duration(maxPKCandidate)),
		zap.Duration("maxPutDur", time.Duration(maxPut)),
		zap.Duration("maxNotifyDur", time.Duration(maxNotify)),
		zap.Duration("maxTotalDur", time.Duration(maxSegment)),
		zap.Int("loadPoolCap", pool.Cap()),
		zap.Int("loadPoolRunning", pool.Running()),
		zap.Int("loadPoolWaiting", pool.Waiting()),
	)
}

func (s *bloomFilterLoadTimingStats) record(segmentNum int, stubDur, metadataDur, memoryEstimateDur, reserveDur, remoteLoadDur, chargeDur, totalDur time.Duration) {
	s.count.Inc()
	s.segmentCount.Add(int64(segmentNum))
	s.totalStub.Add(int64(stubDur))
	s.totalMetadata.Add(int64(metadataDur))
	s.totalMemoryEstimate.Add(int64(memoryEstimateDur))
	s.totalReserve.Add(int64(reserveDur))
	s.totalRemoteLoad.Add(int64(remoteLoadDur))
	s.totalCharge.Add(int64(chargeDur))
	s.totalBloomFilterLoad.Add(int64(totalDur))
	updateMaxDuration(s.maxStub, stubDur)
	updateMaxDuration(s.maxMetadata, metadataDur)
	updateMaxDuration(s.maxMemoryEstimate, memoryEstimateDur)
	updateMaxDuration(s.maxReserve, reserveDur)
	updateMaxDuration(s.maxRemoteLoad, remoteLoadDur)
	updateMaxDuration(s.maxCharge, chargeDur)
	updateMaxDuration(s.maxBloomFilterLoad, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}

	count := s.count.Swap(0)
	segmentCount := s.segmentCount.Swap(0)
	totalStub := s.totalStub.Swap(0)
	totalMetadata := s.totalMetadata.Swap(0)
	totalMemoryEstimate := s.totalMemoryEstimate.Swap(0)
	totalReserve := s.totalReserve.Swap(0)
	totalRemoteLoad := s.totalRemoteLoad.Swap(0)
	totalCharge := s.totalCharge.Swap(0)
	totalBloomFilterLoad := s.totalBloomFilterLoad.Swap(0)
	maxStub := s.maxStub.Swap(0)
	maxMetadata := s.maxMetadata.Swap(0)
	maxMemoryEstimate := s.maxMemoryEstimate.Swap(0)
	maxReserve := s.maxReserve.Swap(0)
	maxRemoteLoad := s.maxRemoteLoad.Swap(0)
	maxCharge := s.maxCharge.Swap(0)
	maxBloomFilterLoad := s.maxBloomFilterLoad.Swap(0)
	if count == 0 {
		return
	}

	log.Warn("bloom filter load timing stats",
		zap.Int64("requestCount", count),
		zap.Int64("segmentCount", segmentCount),
		zap.Duration("avgStubDur", avgDuration(totalStub, count)),
		zap.Duration("avgMetadataDur", avgDuration(totalMetadata, count)),
		zap.Duration("avgMemoryEstimateDur", avgDuration(totalMemoryEstimate, count)),
		zap.Duration("avgReserveDur", avgDuration(totalReserve, count)),
		zap.Duration("avgRemoteLoadDur", avgDuration(totalRemoteLoad, count)),
		zap.Duration("avgChargeDur", avgDuration(totalCharge, count)),
		zap.Duration("avgTotalDur", avgDuration(totalBloomFilterLoad, count)),
		zap.Duration("maxStubDur", time.Duration(maxStub)),
		zap.Duration("maxMetadataDur", time.Duration(maxMetadata)),
		zap.Duration("maxMemoryEstimateDur", time.Duration(maxMemoryEstimate)),
		zap.Duration("maxReserveDur", time.Duration(maxReserve)),
		zap.Duration("maxRemoteLoadDur", time.Duration(maxRemoteLoad)),
		zap.Duration("maxChargeDur", time.Duration(maxCharge)),
		zap.Duration("maxTotalDur", time.Duration(maxBloomFilterLoad)),
	)
}

func (s *requestResourceTimingStats) record(estimateDur, lockWaitDur, lockHoldDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalEstimate.Add(int64(estimateDur))
	s.totalLockWait.Add(int64(lockWaitDur))
	s.totalLockHold.Add(int64(lockHoldDur))
	s.totalRequest.Add(int64(totalDur))
	updateMaxDuration(s.maxEstimate, estimateDur)
	updateMaxDuration(s.maxLockWait, lockWaitDur)
	updateMaxDuration(s.maxLockHold, lockHoldDur)
	updateMaxDuration(s.maxRequest, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}

	count := s.count.Swap(0)
	totalEstimate := s.totalEstimate.Swap(0)
	totalLockWait := s.totalLockWait.Swap(0)
	totalLockHold := s.totalLockHold.Swap(0)
	totalRequest := s.totalRequest.Swap(0)
	maxEstimate := s.maxEstimate.Swap(0)
	maxLockWait := s.maxLockWait.Swap(0)
	maxLockHold := s.maxLockHold.Swap(0)
	maxRequest := s.maxRequest.Swap(0)
	if count == 0 {
		return
	}

	log.Warn("request resource timing stats",
		zap.Int64("count", count),
		zap.Duration("avgEstimateDur", avgDuration(totalEstimate, count)),
		zap.Duration("avgLockWaitDur", avgDuration(totalLockWait, count)),
		zap.Duration("avgLockHoldDur", avgDuration(totalLockHold, count)),
		zap.Duration("avgTotalDur", avgDuration(totalRequest, count)),
		zap.Duration("maxEstimateDur", time.Duration(maxEstimate)),
		zap.Duration("maxLockWaitDur", time.Duration(maxLockWait)),
		zap.Duration("maxLockHoldDur", time.Duration(maxLockHold)),
		zap.Duration("maxTotalDur", time.Duration(maxRequest)),
	)
}

func (s *estimateSegmentResourceTimingStats) record(indexCount, binlogCount int, schemaDur, indexLoopDur, cLoadInfoDur, cEstimateDur, binlogLoopDur, statsDur, deleteDur, jsonStatsDur, textStatsDur, totalDur time.Duration) {
	s.count.Inc()
	s.totalIndex.Add(int64(indexCount))
	s.totalBinlog.Add(int64(binlogCount))
	s.totalSchema.Add(int64(schemaDur))
	s.totalIndexLoop.Add(int64(indexLoopDur))
	s.totalCLoadInfo.Add(int64(cLoadInfoDur))
	s.totalCEstimate.Add(int64(cEstimateDur))
	s.totalBinlogLoop.Add(int64(binlogLoopDur))
	s.totalStats.Add(int64(statsDur))
	s.totalDelete.Add(int64(deleteDur))
	s.totalJSONStats.Add(int64(jsonStatsDur))
	s.totalTextStats.Add(int64(textStatsDur))
	s.totalSegment.Add(int64(totalDur))
	updateMaxDuration(s.maxSegment, totalDur)

	now := time.Now()
	last := s.lastLogUnixNano.Load()
	if now.UnixNano()-last < int64(requestResourceTimingLogInterval) {
		return
	}
	if !s.lastLogUnixNano.CompareAndSwap(last, now.UnixNano()) {
		return
	}

	count := s.count.Swap(0)
	totalIndex := s.totalIndex.Swap(0)
	totalBinlog := s.totalBinlog.Swap(0)
	totalSchema := s.totalSchema.Swap(0)
	totalIndexLoop := s.totalIndexLoop.Swap(0)
	totalCLoadInfo := s.totalCLoadInfo.Swap(0)
	totalCEstimate := s.totalCEstimate.Swap(0)
	totalBinlogLoop := s.totalBinlogLoop.Swap(0)
	totalStats := s.totalStats.Swap(0)
	totalDelete := s.totalDelete.Swap(0)
	totalJSONStats := s.totalJSONStats.Swap(0)
	totalTextStats := s.totalTextStats.Swap(0)
	totalSegment := s.totalSegment.Swap(0)
	maxSegment := s.maxSegment.Swap(0)
	if count == 0 {
		return
	}

	log.Warn("estimate loading resource usage timing stats",
		zap.Int64("segmentCount", count),
		zap.Int64("avgIndexCount", totalIndex/count),
		zap.Int64("avgBinlogCount", totalBinlog/count),
		zap.Duration("avgSchemaDur", avgDuration(totalSchema, count)),
		zap.Duration("avgIndexLoopDur", avgDuration(totalIndexLoop, count)),
		zap.Duration("avgCLoadInfoDur", avgDuration(totalCLoadInfo, count)),
		zap.Duration("avgCEstimateDur", avgDuration(totalCEstimate, count)),
		zap.Duration("avgBinlogLoopDur", avgDuration(totalBinlogLoop, count)),
		zap.Duration("avgStatsDur", avgDuration(totalStats, count)),
		zap.Duration("avgDeleteDur", avgDuration(totalDelete, count)),
		zap.Duration("avgJSONStatsDur", avgDuration(totalJSONStats, count)),
		zap.Duration("avgTextStatsDur", avgDuration(totalTextStats, count)),
		zap.Duration("avgTotalDur", avgDuration(totalSegment, count)),
		zap.Duration("maxTotalDur", time.Duration(maxSegment)),
	)
}

type Loader interface {
	// Load loads binlogs, and spawn segments,
	// NOTE: make sure the ref count of the corresponding collection will never go down to 0 during this
	Load(ctx context.Context, collectionID int64, segmentType SegmentType, version int64, segments ...*querypb.SegmentLoadInfo) ([]Segment, error)

	// LoadDeltaLogs load deltalog and write delta data into provided segment.
	// it also executes resource protection logic in case of OOM.
	LoadDeltaLogs(ctx context.Context, segment Segment, loadInfo *querypb.SegmentLoadInfo) error

	// LoadBloomFilterSet loads needed statslog for RemoteSegment.
	LoadBloomFilterSet(ctx context.Context, collectionID int64, infos ...*querypb.SegmentLoadInfo) ([]*pkoracle.BloomFilterSet, error)

	// GetChunkManager returns the chunk manager for remote storage access.
	GetChunkManager() storage.ChunkManager

	// LoadIndex append index for segment and remove vector binlogs.
	LoadIndex(ctx context.Context,
		segment Segment,
		info *querypb.SegmentLoadInfo,
		version int64) error

	// ReopenSegments update segment data according to new load info.
	ReopenSegments(ctx context.Context,
		loadInfos []*querypb.SegmentLoadInfo,
	) error
}

type ResourceEstimate struct {
	MaxMemoryCost   uint64
	MaxDiskCost     uint64
	FinalMemoryCost uint64
	FinalDiskCost   uint64
	HasRawData      bool
}

func GetResourceEstimate(estimate *C.LoadResourceRequest) ResourceEstimate {
	return ResourceEstimate{
		MaxMemoryCost:   uint64(estimate.max_memory_cost),
		MaxDiskCost:     uint64(estimate.max_disk_cost),
		FinalMemoryCost: uint64(estimate.final_memory_cost),
		FinalDiskCost:   uint64(estimate.final_disk_cost),
		HasRawData:      bool(estimate.has_raw_data),
	}
}

type requestResourceResult struct {
	Resource          LoadResource
	LogicalResource   LoadResource
	CommittedResource LoadResource
	ConcurrencyLevel  int
}

type LoadResource struct {
	MemorySize uint64
	DiskSize   uint64
}

func (r *LoadResource) Add(resource LoadResource) {
	r.MemorySize += resource.MemorySize
	r.DiskSize += resource.DiskSize
}

func (r *LoadResource) Sub(resource LoadResource) {
	r.MemorySize -= resource.MemorySize
	r.DiskSize -= resource.DiskSize
}

func (r *LoadResource) IsZero() bool {
	return r.MemorySize == 0 && r.DiskSize == 0
}

type resourceEstimateFactor struct {
	memoryUsageFactor               float64
	memoryIndexUsageFactor          float64
	EnableInterminSegmentIndex      bool
	tempSegmentIndexFactor          float64
	deltaDataExpansionFactor        float64
	jsonKeyStatsExpansionFactor     float64
	textIndexExpansionFactor        float64
	TieredEvictionEnabled           bool
	TieredEvictableMemoryCacheRatio float64
	TieredEvictableDiskCacheRatio   float64
	// externalRawDataFactor is the peak-memory safety factor for external
	// segments. External tables always download, decompress and deserialize
	// row groups into Arrow buffers regardless of mmap / TieredEviction
	// settings, so peak transient memory = rawDataSize * factor. Defaults
	// to 2.0 via paramtable queryNode.externalCollection.rawDataFactor.
	externalRawDataFactor float64
}

func NewLoader(
	ctx context.Context,
	manager *Manager,
	cm storage.ChunkManager,
) *segmentLoader {
	duf := NewDiskUsageFetcher(ctx)
	go duf.Start()

	loader := &segmentLoader{
		manager:                   manager,
		cm:                        cm,
		loadingSegments:           typeutil.NewConcurrentMap[int64, *loadResult](),
		committedResourceNotifier: syncutil.NewVersionedNotifier(),
		duf:                       duf,
		totalMemory:               hardware.GetMemoryCount(),
	}

	return loader
}

type loadStatus = int32

const (
	loading loadStatus = iota + 1
	success
	failure
)

type loadResult struct {
	status *atomic.Int32
	cond   *sync.Cond
}

func newLoadResult() *loadResult {
	return &loadResult{
		status: atomic.NewInt32(loading),
		cond:   sync.NewCond(&sync.Mutex{}),
	}
}

func (r *loadResult) SetResult(status loadStatus) {
	r.status.CompareAndSwap(loading, status)
	r.cond.Broadcast()
}

// segmentLoader is only responsible for loading the field data from binlog
type segmentLoader struct {
	manager *Manager
	cm      storage.ChunkManager

	// The channel will be closed as the segment loaded
	loadingSegments *typeutil.ConcurrentMap[int64, *loadResult]

	mut                       sync.Mutex // guards committedResource
	committedResource         LoadResource
	committedLogicalResource  LoadResource
	committedResourceNotifier *syncutil.VersionedNotifier

	duf *diskUsageFetcher

	totalMemory uint64
}

var _ Loader = (*segmentLoader)(nil)

func (loader *segmentLoader) Load(ctx context.Context,
	collectionID int64,
	segmentType SegmentType,
	version int64,
	segments ...*querypb.SegmentLoadInfo,
) ([]Segment, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", collectionID),
		zap.String("segmentType", segmentType.String()),
	)

	if len(segments) == 0 {
		log.Info("no segment to load")
		return nil, nil
	}

	collection := loader.manager.Collection.Get(collectionID)
	if collection == nil {
		err := merr.WrapErrCollectionNotFound(collectionID)
		log.Warn("failed to get collection", zap.Error(err))
		return nil, err
	}
	// Filter out loaded & loading segments
	infos := loader.prepare(ctx, segmentType, segments...)
	defer loader.unregister(infos...)

	// continue to wait other task done
	log.Info("start loading...", zap.Int("segmentNum", len(segments)), zap.Int("afterFilter", len(infos)))

	var err error
	var requestResourceResult requestResourceResult

	// Check memory & storage limit
	// no need to check resource for lazy load here
	requestResourceResult, err = loader.requestResource(ctx, infos...)
	if err != nil {
		log.Warn("request resource failed", zap.Error(err))
		return nil, err
	}
	defer loader.freeRequestResource(requestResourceResult)

	newSegments := typeutil.NewConcurrentMap[int64, Segment]()
	loaded := typeutil.NewConcurrentMap[int64, Segment]()
	defer func() {
		newSegments.Range(func(segmentID int64, s Segment) bool {
			log.Warn("release new segment created due to load failure",
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			s.Release(context.Background())
			return true
		})
	}()

	for _, info := range infos {
		loadInfo := info

		for _, indexInfo := range loadInfo.IndexInfos {
			indexParams := funcutil.KeyValuePair2Map(indexInfo.IndexParams)

			// some build params also exist in indexParams, which are useless during loading process
			if vecindexmgr.GetVecIndexMgrInstance().IsDiskANN(indexParams["index_type"]) {
				if err := indexparams.SetDiskIndexLoadParams(paramtable.Get(), indexParams, indexInfo.GetNumRows()); err != nil {
					return nil, err
				}
			}

			// set whether enable offset cache for bitmap index
			if indexParams["index_type"] == indexparamcheck.IndexBitmap {
				indexparams.SetBitmapIndexLoadParams(paramtable.Get(), indexParams)
			}

			if err := indexparams.AppendPrepareLoadParams(paramtable.Get(), indexParams); err != nil {
				return nil, err
			}

			indexInfo.IndexParams = funcutil.Map2KeyValuePair(indexParams)
		}

		segment, err := NewSegment(
			ctx,
			collection,
			loader.manager.Segment,
			segmentType,
			version,
			loadInfo,
		)
		if err != nil {
			log.Warn("load segment failed when create new segment",
				zap.Int64("partitionID", loadInfo.GetPartitionID()),
				zap.Int64("segmentID", loadInfo.GetSegmentID()),
				zap.Error(err),
			)
			return nil, err
		}

		newSegments.Insert(loadInfo.GetSegmentID(), segment)
	}

	loadSegmentFunc := func(idx int) (err error) {
		loadInfo := infos[idx]
		partitionID := loadInfo.PartitionID
		segmentID := loadInfo.SegmentID
		segment, _ := newSegments.Get(segmentID)
		segmentLoadStart := time.Now()
		var loadDataDur time.Duration
		var deltaLogDur time.Duration
		var pkCandidateDur time.Duration
		var putDur time.Duration
		var notifyDur time.Duration

		logger := log.With(zap.Int64("partitionID", partitionID),
			zap.Int64("segmentID", segmentID),
			zap.String("segmentType", loadInfo.GetLevel().String()))
		metrics.QueryNodeLoadSegmentConcurrency.WithLabelValues(paramtable.GetStringNodeID(), "LoadSegment").Inc()
		defer func() {
			metrics.QueryNodeLoadSegmentConcurrency.WithLabelValues(paramtable.GetStringNodeID(), "LoadSegment").Dec()
			segmentLoadTiming.record(loadDataDur, deltaLogDur, pkCandidateDur, putDur, notifyDur, time.Since(segmentLoadStart))
			if err != nil {
				logger.Warn("load segment failed when load data into memory", zap.Error(err))
			}
			logger.Info("load segment done")
		}()
		tr := timerecord.NewTimeRecorder("loadDurationPerSegment")
		logger.Info("load segment...")

		// L0 segment has no index or data to be load.
		if loadInfo.GetLevel() != datapb.SegmentLevel_L0 {
			// lazy load segment do not load segment at first time.
			stageStart := time.Now()
			if err = loader.LoadSegment(ctx, segment, loadInfo); err != nil {
				loadDataDur = time.Since(stageStart)
				return errors.Wrap(err, "At LoadSegment")
			}
			loadDataDur = time.Since(stageStart)
		}
		// Skip delta logs for external collections (they are read-only, no deletions)
		if !typeutil.IsExternalCollection(collection.Schema()) {
			stageStart := time.Now()
			if err = loader.loadDeltalogs(ctx, segment, loadInfo); err != nil {
				deltaLogDur = time.Since(stageStart)
				return errors.Wrap(err, "At LoadDeltaLogs")
			}
			deltaLogDur = time.Since(stageStart)
		}

		stageStart := time.Now()
		if !segment.PkCandidateExist() {
			log.Debug("loading PK candidate for segment", zap.Int64("segmentID", segment.ID()))
			// For external collections, use ExternalSegmentCandidate instead of BloomFilterSet
			if typeutil.IsExternalCollection(collection.Schema()) {
				candidate := pkoracle.NewExternalSegmentCandidate(
					loadInfo.GetSegmentID(),
					loadInfo.GetPartitionID(),
					segment.Type(),
				)
				segment.SetPKCandidate(candidate)
				log.Info("using ExternalSegmentCandidate for external collection",
					zap.Int64("segmentID", loadInfo.GetSegmentID()))

				// Check for truncated segment ID collision with other segments being loaded.
				collisions := detectVirtualPKCollisions(loadInfo.GetSegmentID(), infos)
				for _, collidingID := range collisions {
					log.Warn("virtual PK collision detected: two segments share truncated segment ID",
						zap.Int64("segmentID1", loadInfo.GetSegmentID()),
						zap.Int64("segmentID2", collidingID),
						zap.Int64("truncatedID", loadInfo.GetSegmentID()&0xFFFFFFFF))
				}
			} else if paramtable.Get().CommonCfg.BloomFilterEnabled.GetAsBool() {
				bfs, err := loader.loadSingleBloomFilterSet(ctx, loadInfo.GetCollectionID(), loadInfo, segment.Type())
				if err != nil {
					return errors.Wrap(err, "At LoadBloomFilter")
				}
				segment.SetPKCandidate(bfs)
				// Charge bloom filter resource
				bfs.Charge()
			}
		}
		pkCandidateDur = time.Since(stageStart)

		stageStart = time.Now()
		if segment.Level() != datapb.SegmentLevel_L0 {
			loader.manager.Segment.Put(ctx, segmentType, segment)
		}
		putDur = time.Since(stageStart)
		newSegments.GetAndRemove(segmentID)
		loaded.Insert(segmentID, segment)
		stageStart = time.Now()
		loader.notifyLoadFinish(loadInfo)
		notifyDur = time.Since(stageStart)

		metrics.QueryNodeLoadSegmentLatency.WithLabelValues(paramtable.GetStringNodeID()).Observe(float64(tr.ElapseSpan().Milliseconds()))
		return nil
	}

	// Start to load,
	// Make sure we can always benefit from concurrency, and not spawn too many idle goroutines
	log.Info("start to load segments in parallel",
		zap.Int("segmentNum", len(infos)),
		zap.Int("concurrencyLevel", requestResourceResult.ConcurrencyLevel))

	err = funcutil.ProcessFuncParallel(len(infos),
		requestResourceResult.ConcurrencyLevel, loadSegmentFunc, "loadSegmentFunc")
	if err != nil {
		log.Warn("failed to load some segments", zap.Error(err))
		return nil, err
	}

	// Wait for all segments loaded
	segmentIDs := lo.Map(segments, func(info *querypb.SegmentLoadInfo, _ int) int64 { return info.GetSegmentID() })
	if err := loader.waitSegmentLoadDone(ctx, segmentType, segmentIDs, version); err != nil {
		log.Warn("failed to wait the filtered out segments load done", zap.Error(err))
		return nil, err
	}

	log.Info("all segment load done")
	var result []Segment
	loaded.Range(func(_ int64, s Segment) bool {
		result = append(result, s)
		return true
	})
	return result, nil
}

func (loader *segmentLoader) prepare(ctx context.Context, segmentType SegmentType, segments ...*querypb.SegmentLoadInfo) []*querypb.SegmentLoadInfo {
	log := log.Ctx(ctx).With(
		zap.Stringer("segmentType", segmentType),
	)

	// filter out loaded & loading segments
	infos := make([]*querypb.SegmentLoadInfo, 0, len(segments))
	for _, segment := range segments {
		// Not loaded & loading & releasing.
		if !loader.manager.Segment.Exist(segment.GetSegmentID(), segmentType) &&
			!loader.loadingSegments.Contain(segment.GetSegmentID()) {
			infos = append(infos, segment)
			loader.loadingSegments.Insert(segment.GetSegmentID(), newLoadResult())
		} else {
			log.Info("skip loaded/loading segment",
				zap.Int64("segmentID", segment.GetSegmentID()),
				zap.Bool("isLoaded", len(loader.manager.Segment.GetBy(WithType(segmentType), WithID(segment.GetSegmentID()))) > 0),
				zap.Bool("isLoading", loader.loadingSegments.Contain(segment.GetSegmentID())),
			)
		}
	}

	return infos
}

func (loader *segmentLoader) unregister(segments ...*querypb.SegmentLoadInfo) {
	for i := range segments {
		result, ok := loader.loadingSegments.GetAndRemove(segments[i].GetSegmentID())
		if ok {
			result.SetResult(failure)
		}
	}
}

func (loader *segmentLoader) notifyLoadFinish(segments ...*querypb.SegmentLoadInfo) {
	for _, loadInfo := range segments {
		result, ok := loader.loadingSegments.Get(loadInfo.GetSegmentID())
		if ok {
			result.SetResult(success)
		}
	}
}

// requestResource requests memory & storage to load segments,
// returns the memory usage, disk usage and concurrency with the gained memory.
func (loader *segmentLoader) requestResource(ctx context.Context, infos ...*querypb.SegmentLoadInfo) (requestResourceResult, error) {
	requestStart := time.Now()
	// we need to deal with empty infos case separately,
	// because the following judgement for requested resources are based on current status and static config
	// which may block empty-load operations by accident
	if len(infos) == 0 {
		return requestResourceResult{}, nil
	}

	segmentIDs := lo.Map(infos, func(info *querypb.SegmentLoadInfo, _ int) int64 {
		return info.GetSegmentID()
	})
	log := log.Ctx(ctx).With(
		zap.Int64s("segmentIDs", segmentIDs),
	)

	physicalMemoryUsage := hardware.GetUsedMemoryCount()
	totalMemory := loader.totalMemory
	if totalMemory == 0 {
		totalMemory = hardware.GetMemoryCount()
	}

	physicalDiskUsage, err := loader.duf.GetDiskUsage()
	if err != nil {
		return requestResourceResult{}, errors.Wrap(err, "get local used size failed")
	}
	diskCap := paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsUint64()

	estimateStart := time.Now()
	loadingUsage, maxSegmentSize, err := loader.estimateSegmentLoadingResourceUsage(ctx, infos...)
	estimateDur := time.Since(estimateStart)
	if err != nil {
		log.Warn("no sufficient physical resource to load segments", zap.Error(err))
		return requestResourceResult{}, err
	}

	lockWaitStart := time.Now()
	loader.mut.Lock()
	lockWaitDur := time.Since(lockWaitStart)
	lockHoldStart := time.Now()
	defer func() {
		lockHoldDur := time.Since(lockHoldStart)
		loader.mut.Unlock()
		requestResourceTiming.record(estimateDur, lockWaitDur, lockHoldDur, time.Since(requestStart))
	}()

	result := requestResourceResult{
		CommittedResource: loader.committedResource,
	}

	if loader.committedResource.MemorySize+physicalMemoryUsage >= totalMemory {
		return result, merr.WrapErrServiceMemoryLimitExceeded(float32(loader.committedResource.MemorySize+physicalMemoryUsage), float32(totalMemory))
	} else if loader.committedResource.DiskSize+uint64(physicalDiskUsage) >= diskCap {
		return result, merr.WrapErrServiceDiskLimitExceeded(float32(loader.committedResource.DiskSize+uint64(physicalDiskUsage)), float32(diskCap))
	}

	result.ConcurrencyLevel = funcutil.Min(hardware.GetCPUNum(), len(infos))

	// TODO: disable logical resource checking for now
	// lmu, ldu, err := loader.checkLogicalSegmentSize(ctx, infos, totalMemory)
	// if err != nil {
	// 	log.Warn("no sufficient logical resource to load segments", zap.Error(err))
	// 	return result, err
	// }

	// then get physical resource usage for loading segments
	memUsage := physicalMemoryUsage + loader.committedResource.MemorySize
	if memUsage == 0 || totalMemory == 0 {
		return result, errors.New("get memory failed when checkSegmentSize")
	}
	diskUsage := uint64(physicalDiskUsage) + loader.committedResource.DiskSize
	predictMemUsage := memUsage + loadingUsage.MemorySize
	predictDiskUsage := diskUsage + loadingUsage.DiskSize

	log.Info("predict memory and disk usage while loading (in MiB)",
		zap.Float64("maxSegmentSize(MB)", logutil.ToMB(float64(maxSegmentSize))),
		zap.Float64("committedMemSize(MB)", logutil.ToMB(float64(loader.committedResource.MemorySize))),
		zap.Float64("memLimit(MB)", logutil.ToMB(float64(totalMemory))),
		zap.Float64("memUsage(MB)", logutil.ToMB(float64(memUsage))),
		zap.Float64("committedDiskSize(MB)", logutil.ToMB(float64(loader.committedResource.DiskSize))),
		zap.Float64("diskUsage(MB)", logutil.ToMB(float64(diskUsage))),
		zap.Float64("predictMemUsage(MB)", logutil.ToMB(float64(predictMemUsage))),
		zap.Float64("predictDiskUsage(MB)", logutil.ToMB(float64(predictDiskUsage))),
		zap.Int("mmapFieldCount", loadingUsage.MmapFieldCount),
	)

	if paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool() {
		// try to reserve loading resource from caching layer
		if ok := C.TryReserveLoadingResourceWithTimeout(C.CResourceUsage{
			memory_bytes: C.int64_t(loadingUsage.MemorySize),
			disk_bytes:   C.int64_t(loadingUsage.DiskSize),
		}, 1000); !ok {
			return result, fmt.Errorf("failed to reserve loading resource from caching layer, predictMemUsage = %v MB, predictDiskUsage = %v MB, memUsage = %v MB, diskUsage = %v MB, memoryThresholdFactor = %f, diskThresholdFactor = %f",
				logutil.ToMB(float64(predictMemUsage)),
				logutil.ToMB(float64(predictDiskUsage)),
				logutil.ToMB(float64(memUsage)),
				logutil.ToMB(float64(diskUsage)),
				paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(),
				paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat(),
			)
		}
	} else {
		// fallback to original segment loading logic
		if predictMemUsage > uint64(float64(totalMemory)*paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat()) {
			log.Warn("load segment failed, OOM if load",
				zap.String("resourceType", "Memory"),
				zap.Float64("maxSegmentSizeMB", logutil.ToMB(float64(maxSegmentSize))),
				zap.Float64("memUsageMB", logutil.ToMB(float64(memUsage))),
				zap.Float64("predictMemUsageMB", logutil.ToMB(float64(predictMemUsage))),
				zap.Float64("totalMemMB", logutil.ToMB(float64(totalMemory))),
				zap.Float64("thresholdFactor", paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat()),
			)
			return result, merr.WrapErrSegmentRequestResourceFailed("Memory")
		}

		if predictDiskUsage > uint64(float64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64())*paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat()) {
			log.Warn("load segment failed, disk space is not enough",
				zap.String("resourceType", "Disk"),
				zap.Float64("diskUsageMB", logutil.ToMB(float64(diskUsage))),
				zap.Float64("predictDiskUsageMB", logutil.ToMB(float64(predictDiskUsage))),
				zap.Float64("totalDiskMB", logutil.ToMB(float64(uint64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64())))),
				zap.Float64("thresholdFactor", paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat()),
			)
			return result, merr.WrapErrSegmentRequestResourceFailed("Disk")
		}
	}

	err = checkSegmentGpuMemSize(loadingUsage.FieldGpuMemorySize, float32(paramtable.Get().GpuConfig.OverloadedMemoryThresholdPercentage.GetAsFloat()))
	if err != nil {
		return result, err
	}

	result.Resource.MemorySize = loadingUsage.MemorySize
	result.Resource.DiskSize = loadingUsage.DiskSize
	// result.LogicalResource.MemorySize = lmu
	// result.LogicalResource.DiskSize = ldu

	loader.committedResource.Add(result.Resource)
	// loader.committedLogicalResource.Add(result.LogicalResource)
	log.Info("request resource for loading segments (unit in MiB)",
		zap.Float64("memory", logutil.ToMB(float64(result.Resource.MemorySize))),
		zap.Float64("committedMemory", logutil.ToMB(float64(loader.committedResource.MemorySize))),
		zap.Float64("disk", logutil.ToMB(float64(result.Resource.DiskSize))),
		zap.Float64("committedDisk", logutil.ToMB(float64(loader.committedResource.DiskSize))),
	)

	return result, nil
}

// freeRequestResource returns request memory & storage usage request.
func (loader *segmentLoader) freeRequestResource(requestResourceResult requestResourceResult) {
	loader.mut.Lock()
	defer loader.mut.Unlock()

	resource := requestResourceResult.Resource
	// logicalResource := requestResourceResult.LogicalResource

	if paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool() {
		C.ReleaseLoadingResource(C.CResourceUsage{
			memory_bytes: C.int64_t(resource.MemorySize),
			disk_bytes:   C.int64_t(resource.DiskSize),
		})
	}

	loader.committedResource.Sub(resource)
	// loader.committedLogicalResource.Sub(logicalResource)
	loader.committedResourceNotifier.NotifyAll()
}

func (loader *segmentLoader) waitSegmentLoadDone(ctx context.Context, segmentType SegmentType, segmentIDs []int64, version int64) error {
	log := log.Ctx(ctx).With(
		zap.String("segmentType", segmentType.String()),
		zap.Int64s("segmentIDs", segmentIDs),
	)
	for _, segmentID := range segmentIDs {
		if loader.manager.Segment.GetWithType(segmentID, segmentType) != nil {
			continue
		}

		result, ok := loader.loadingSegments.Get(segmentID)
		if !ok {
			log.Warn("segment was removed from the loading map early", zap.Int64("segmentID", segmentID))
			return errors.New("segment was removed from the loading map early")
		}

		log.Info("wait segment loaded...", zap.Int64("segmentID", segmentID))

		signal := make(chan struct{})
		go func() {
			select {
			case <-signal:
			case <-ctx.Done():
				result.cond.Broadcast()
			}
		}()
		result.cond.L.Lock()
		for result.status.Load() == loading && ctx.Err() == nil {
			result.cond.Wait()
		}
		result.cond.L.Unlock()
		close(signal)

		if ctx.Err() != nil {
			log.Warn("failed to wait segment loaded due to context done", zap.Int64("segmentID", segmentID))
			return ctx.Err()
		}

		if result.status.Load() == failure {
			log.Warn("failed to wait segment loaded", zap.Int64("segmentID", segmentID))
			return merr.WrapErrSegmentLack(segmentID, "failed to wait segment loaded")
		}

		// try to update segment version after wait segment loaded
		loader.manager.Segment.UpdateBy(IncreaseVersion(version), WithType(segmentType), WithID(segmentID))

		log.Info("segment loaded...", zap.Int64("segmentID", segmentID))
	}
	return nil
}

func (loader *segmentLoader) GetChunkManager() storage.ChunkManager {
	return loader.cm
}

// load single bloom filter
func (loader *segmentLoader) loadSingleBloomFilterSet(ctx context.Context, collectionID int64, loadInfo *querypb.SegmentLoadInfo, segtype SegmentType) (*pkoracle.BloomFilterSet, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", collectionID),
		zap.Int64("segmentIDs", loadInfo.GetSegmentID()))

	partitionID := loadInfo.PartitionID
	segmentID := loadInfo.SegmentID

	if !paramtable.Get().CommonCfg.BloomFilterEnabled.GetAsBool() {
		bfs := pkoracle.NewBloomFilterSet(segmentID, partitionID, segtype)
		log.Info("skip loading bloom filter for remote segment because bloom filter is disabled")
		return bfs, nil
	}

	collection := loader.manager.Collection.Get(collectionID)
	if collection == nil {
		err := merr.WrapErrCollectionNotFound(collectionID)
		log.Warn("failed to get collection while loading segment", zap.Error(err))
		return nil, err
	}
	pkField := GetPkField(collection.Schema())

	log.Info("start loading remote...", zap.Int("segmentNum", 1))

	// For external collections, return empty bloom filter set.
	// External collections use ExternalSegmentCandidate for PK checking (set on segment)
	// and don't have stats logs, so we skip loading bloom filters.
	// NOTE: This is a defensive guard. Normal external collection load path uses
	// ExternalSegmentCandidate directly and should not reach here.
	if typeutil.IsExternalCollection(collection.Schema()) {
		bfs := pkoracle.NewBloomFilterSet(segmentID, partitionID, segtype)
		log.Debug("external collection: returning empty bloom filter set (defensive path)")
		return bfs, nil
	}

	lazyCtx := context.WithoutCancel(ctx)
	pkFieldID := pkField.GetFieldID()
	return pkoracle.NewLazyBloomFilterSet(segmentID, partitionID, segtype, func(bfs *pkoracle.BloomFilterSet) error {
		start := time.Now()
		log.Info("lazy loading bloom filter for remote segment")

		stageStart := time.Now()
		pkStatsBinlogs, err := packed.NewStatsResolverFromLoadInfo(loadInfo).BloomFilterPaths(pkFieldID)
		resolveDur := time.Since(stageStart)
		if err != nil {
			return err
		}

		stageStart = time.Now()
		err = loader.loadBloomFilter(lazyCtx, segmentID, bfs, pkStatsBinlogs)
		loadDur := time.Since(stageStart)
		if err != nil {
			log.Warn("load remote segment bloom filter failed",
				zap.Int64("partitionID", partitionID),
				zap.Int64("segmentID", segmentID),
				zap.Error(err),
			)
			return err
		}
		log.Info("lazy loaded bloom filter for remote segment",
			zap.Int("pathNum", len(pkStatsBinlogs)),
			zap.Duration("resolvePathsDur", resolveDur),
			zap.Duration("loadBloomFilterDur", loadDur),
			zap.Duration("totalDur", time.Since(start)))
		return nil
	}), nil
}

func (loader *segmentLoader) LoadBloomFilterSet(ctx context.Context, collectionID int64, infos ...*querypb.SegmentLoadInfo) ([]*pkoracle.BloomFilterSet, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", collectionID),
		zap.Int64s("segmentIDs", lo.Map(infos, func(info *querypb.SegmentLoadInfo, _ int) int64 {
			return info.GetSegmentID()
		})),
	)

	segmentNum := len(infos)
	if segmentNum == 0 {
		log.Info("no segment to load")
		return nil, nil
	}
	totalStart := time.Now()
	var (
		stubDur           time.Duration
		metadataDur       time.Duration
		memoryEstimateDur time.Duration
		reserveDur        time.Duration
		remoteLoadDur     time.Duration
		chargeDur         time.Duration
	)
	defer func() {
		bloomFilterLoadTiming.record(segmentNum, stubDur, metadataDur, memoryEstimateDur, reserveDur, remoteLoadDur, chargeDur, time.Since(totalStart))
	}()

	// Phase 1: always create metadata-only stubs (segmentID / partitionID / type).
	// This gives callers valid candidates even when BF data is not loaded,
	// so partition filtering and type-based delete-scope logic never need nil guards.
	stageStart := time.Now()
	bfSets := make([]*pkoracle.BloomFilterSet, segmentNum)
	for i, info := range infos {
		bfSets[i] = pkoracle.NewBloomFilterSet(info.GetSegmentID(), info.GetPartitionID(), commonpb.SegmentState_Sealed)
	}
	stubDur = time.Since(stageStart)

	// Phase 2: load BF stats into the stubs (skip when disabled or external collection).
	stageStart = time.Now()
	if !paramtable.Get().CommonCfg.BloomFilterEnabled.GetAsBool() {
		metadataDur = time.Since(stageStart)
		log.Info("bloom filter disabled: returning metadata-only stubs")
		return bfSets, nil
	}

	collection := loader.manager.Collection.Get(collectionID)
	if collection == nil {
		metadataDur = time.Since(stageStart)
		err := merr.WrapErrCollectionNotFound(collectionID)
		log.Warn("failed to get collection while loading segment", zap.Error(err))
		return nil, err
	}
	pkField := GetPkField(collection.Schema())
	pkFieldID := pkField.GetFieldID()

	// External collections use ExternalSegmentCandidate for PK checking and have no stats logs.
	if typeutil.IsExternalCollection(collection.Schema()) {
		metadataDur = time.Since(stageStart)
		return bfSets, nil
	}
	metadataDur = time.Since(stageStart)

	stageStart = time.Now()
	lazyCtx := context.WithoutCancel(ctx)
	for i, info := range infos {
		info := info
		segmentID := info.GetSegmentID()
		partitionID := info.GetPartitionID()
		bfSets[i] = pkoracle.NewLazyBloomFilterSet(segmentID, partitionID, commonpb.SegmentState_Sealed, func(bfs *pkoracle.BloomFilterSet) error {
			start := time.Now()
			stageStart := time.Now()
			pkStatsBinlogs, err := packed.NewStatsResolverFromLoadInfo(info).BloomFilterPaths(pkFieldID)
			resolveDur := time.Since(stageStart)
			if err != nil {
				return err
			}

			stageStart = time.Now()
			err = loader.loadBloomFilter(lazyCtx, bfs.ID(), bfs, pkStatsBinlogs)
			loadDur := time.Since(stageStart)
			if err != nil {
				log.Warn("load remote segment bloom filter failed",
					zap.Int64("partitionID", bfs.Partition()),
					zap.Int64("segmentID", bfs.ID()),
					zap.Error(err),
				)
				return err
			}
			log.Info("lazy loaded bloom filter for remote segment",
				zap.Int64("segmentID", bfs.ID()),
				zap.Int("pathNum", len(pkStatsBinlogs)),
				zap.Duration("resolvePathsDur", resolveDur),
				zap.Duration("loadBloomFilterDur", loadDur),
				zap.Duration("totalDur", time.Since(start)))
			return nil
		})
	}
	stubDur += time.Since(stageStart)

	return bfSets, nil
}

func separateIndexAndBinlog(loadInfo *querypb.SegmentLoadInfo) (map[int64]*IndexedFieldInfo, []*datapb.FieldBinlog) {
	fieldID2IndexInfo := make(map[int64][]*querypb.FieldIndexInfo)
	for _, indexInfo := range loadInfo.IndexInfos {
		if len(indexInfo.GetIndexFilePaths()) > 0 {
			fieldID := indexInfo.FieldID
			fieldID2IndexInfo[fieldID] = append(fieldID2IndexInfo[fieldID], indexInfo)
		}
	}

	preferFieldData := paramtable.Get().QueryNodeCfg.PreferFieldDataWhenIndexHasRawData.GetAsBool()

	indexedFieldInfos := make(map[int64]*IndexedFieldInfo)
	fieldBinlogs := make([]*datapb.FieldBinlog, 0, len(loadInfo.BinlogPaths))

	for _, fieldBinlog := range loadInfo.BinlogPaths {
		fieldID := fieldBinlog.FieldID
		// check num rows of data meta and index meta are consistent
		if indexInfo, ok := fieldID2IndexInfo[fieldID]; ok {
			for _, index := range indexInfo {
				fieldInfo := &IndexedFieldInfo{
					FieldBinlog: fieldBinlog,
					IndexInfo:   index,
				}
				indexedFieldInfos[index.IndexID] = fieldInfo
			}
			if preferFieldData {
				fieldBinlogs = append(fieldBinlogs, fieldBinlog)
			}
		} else {
			fieldBinlogs = append(fieldBinlogs, fieldBinlog)
		}
	}

	return indexedFieldInfos, fieldBinlogs
}

// detectVirtualPKCollisions checks if any segments in infos share the same
// truncated (lower 32 bits) segment ID as segmentID. A collision means two
// segments produce overlapping virtual PK spaces.
func detectVirtualPKCollisions(segmentID int64, infos []*querypb.SegmentLoadInfo) []int64 {
	truncatedID := segmentID & 0xFFFFFFFF
	var collisions []int64
	for _, info := range infos {
		if info.GetSegmentID() != segmentID &&
			(info.GetSegmentID()&0xFFFFFFFF) == truncatedID {
			collisions = append(collisions, info.GetSegmentID())
		}
	}
	return collisions
}

func separateLoadInfoV2(loadInfo *querypb.SegmentLoadInfo, schema *schemapb.CollectionSchema) (
	map[int64]*IndexedFieldInfo, // indexed info
	[]*datapb.FieldBinlog, // fields info
	map[int64]*datapb.TextIndexStats, // text indexed info
	map[int64]struct{}, // unindexed text fields
	map[int64]*datapb.JsonKeyStats, // json key stats info
	map[int64]string, // text index base paths
	map[int64]string, // json key stats base paths
) {
	storageVersion := loadInfo.GetStorageVersion()

	// Build a map of external field IDs for quick lookup
	// External fields are skipped during loading (lazy loaded on demand)
	externalFieldIDs := make(map[int64]bool)
	isExternalColl := typeutil.IsExternalCollection(schema)
	if isExternalColl {
		for _, field := range schema.GetFields() {
			if IsExternalField(field) {
				externalFieldIDs[field.GetFieldID()] = true
			}
		}
	}

	fieldID2IndexInfo := make(map[int64][]*querypb.FieldIndexInfo)
	for _, indexInfo := range loadInfo.IndexInfos {
		if len(indexInfo.GetIndexFilePaths()) > 0 {
			fieldID := indexInfo.FieldID
			fieldID2IndexInfo[fieldID] = append(fieldID2IndexInfo[fieldID], indexInfo)
		}
	}

	preferFieldData := paramtable.Get().QueryNodeCfg.PreferFieldDataWhenIndexHasRawData.GetAsBool()

	indexedFieldInfos := make(map[int64]*IndexedFieldInfo)
	fieldBinlogs := make([]*datapb.FieldBinlog, 0, len(loadInfo.BinlogPaths))

	if storageVersion == storage.StorageV2 || storageVersion == storage.StorageV3 {
		for _, fieldBinlog := range loadInfo.BinlogPaths {
			fieldID := fieldBinlog.FieldID

			// Skip external fields - they are lazy loaded on demand
			if externalFieldIDs[fieldID] {
				continue
			}

			if fieldID == storagecommon.DefaultShortColumnGroupID {
				allFields := typeutil.GetAllFieldSchemas(schema)
				// for short column group, we need to load all fields in the group
				for _, field := range allFields {
					// Skip external fields in short column group
					if externalFieldIDs[field.GetFieldID()] {
						continue
					}
					if infos, ok := fieldID2IndexInfo[field.GetFieldID()]; ok {
						for _, indexInfo := range infos {
							fieldInfo := &IndexedFieldInfo{
								FieldBinlog: fieldBinlog,
								IndexInfo:   indexInfo,
							}
							indexedFieldInfos[indexInfo.IndexID] = fieldInfo
						}
					}
				}
				fieldBinlogs = append(fieldBinlogs, fieldBinlog)
			} else {
				// for single file field, such as vector field, text field
				if infos, ok := fieldID2IndexInfo[fieldID]; ok {
					for _, indexInfo := range infos {
						fieldInfo := &IndexedFieldInfo{
							FieldBinlog: fieldBinlog,
							IndexInfo:   indexInfo,
						}
						indexedFieldInfos[indexInfo.IndexID] = fieldInfo
					}
					if preferFieldData {
						fieldBinlogs = append(fieldBinlogs, fieldBinlog)
					}
				} else {
					fieldBinlogs = append(fieldBinlogs, fieldBinlog)
				}
			}
		}
	} else {
		for _, fieldBinlog := range loadInfo.BinlogPaths {
			fieldID := fieldBinlog.FieldID

			// Skip external fields - they are lazy loaded on demand
			if externalFieldIDs[fieldID] {
				continue
			}

			if infos, ok := fieldID2IndexInfo[fieldID]; ok {
				for _, indexInfo := range infos {
					fieldInfo := &IndexedFieldInfo{
						FieldBinlog: fieldBinlog,
						IndexInfo:   indexInfo,
					}
					indexedFieldInfos[indexInfo.IndexID] = fieldInfo
				}
				if preferFieldData {
					fieldBinlogs = append(fieldBinlogs, fieldBinlog)
				}
			} else {
				fieldBinlogs = append(fieldBinlogs, fieldBinlog)
			}
		}
	}

	// For external table segments (ManifestPath set, BinlogPaths empty), extract
	// indexes directly from fieldID2IndexInfo without a corresponding FieldBinlog,
	// because the segment data lives in the external store rather than Milvus binlogs.
	if loadInfo.GetManifestPath() != "" {
		for _, infos := range fieldID2IndexInfo {
			for _, indexInfo := range infos {
				if _, exists := indexedFieldInfos[indexInfo.IndexID]; !exists {
					indexedFieldInfos[indexInfo.IndexID] = &IndexedFieldInfo{
						FieldBinlog: &datapb.FieldBinlog{},
						IndexInfo:   indexInfo,
					}
				}
			}
		}
	}

	statsResult := packed.NewStatsResolverFromLoadInfo(loadInfo).TextAndJSONIndexStatsWithBasePaths()
	textIndexedInfo := statsResult.TextIndexStats
	jsonKeyIndexInfo := statsResult.JSONKeyStats
	textBasePaths := statsResult.TextBasePaths
	jsonBasePaths := statsResult.JSONBasePaths
	if statsResult.Err() != nil {
		log.Warn("failed to load text/json stats from manifest",
			zap.String("manifestPath", loadInfo.GetManifestPath()), zap.Error(statsResult.Err()))
		textIndexedInfo = make(map[int64]*datapb.TextIndexStats)
		jsonKeyIndexInfo = make(map[int64]*datapb.JsonKeyStats)
		textBasePaths = make(map[int64]string)
		jsonBasePaths = make(map[int64]string)
	}

	if textBasePaths == nil {
		textBasePaths = make(map[int64]string)
	}
	if jsonBasePaths == nil {
		jsonBasePaths = make(map[int64]string)
	}

	// For V2 (non-manifest) segments, compute basePaths from metadata.
	// The resolver returns empty basePaths for V2; we compute them here.
	rootPath := paramtable.Get().MinioCfg.RootPath.GetValue()
	for fieldID, stats := range textIndexedInfo {
		if _, ok := textBasePaths[fieldID]; !ok {
			textBasePaths[fieldID] = metautil.BuildTextIndexPrefix(rootPath,
				stats.GetBuildID(), stats.GetVersion(),
				loadInfo.GetCollectionID(), loadInfo.GetPartitionID(), loadInfo.GetSegmentID(), fieldID)
		}
	}
	for fieldID, stats := range jsonKeyIndexInfo {
		if _, ok := jsonBasePaths[fieldID]; !ok {
			jsonBasePaths[fieldID] = metautil.BuildJSONKeyStatsPrefix(rootPath, stats.GetJsonKeyStatsDataFormat(),
				stats.GetBuildID(), stats.GetVersion(),
				loadInfo.GetCollectionID(), loadInfo.GetPartitionID(), loadInfo.GetSegmentID(), fieldID)
		}
	}

	unindexedTextFields := make(map[int64]struct{})
	// todo(SpadeA): consider struct fields when index is ready
	for _, field := range schema.GetFields() {
		h := typeutil.CreateFieldSchemaHelper(field)
		_, textIndexExist := textIndexedInfo[field.GetFieldID()]
		if h.EnableMatch() && !textIndexExist {
			unindexedTextFields[field.GetFieldID()] = struct{}{}
		}
	}

	return indexedFieldInfos, fieldBinlogs, textIndexedInfo, unindexedTextFields, jsonKeyIndexInfo, textBasePaths, jsonBasePaths
}

func (loader *segmentLoader) loadSealedSegment(ctx context.Context, loadInfo *querypb.SegmentLoadInfo, segment *LocalSegment) (err error) {
	// TODO: we should create a transaction-like api to load segment for segment interface,
	// but not do many things in segment loader.
	loadStart := time.Now()
	stateLockGuard, err := segment.StartLoadData()
	stateLockDur := time.Since(loadStart)
	// segment can not do load now.
	if err != nil {
		return err
	}
	if stateLockGuard == nil {
		return nil
	}
	var poolWaitDur time.Duration
	var cgoLoadDur time.Duration
	var syncJSONStatsDur time.Duration
	var patchEntryNumberSpan time.Duration
	defer func() {
		sealedLoadTiming.record(stateLockDur, poolWaitDur, cgoLoadDur, syncJSONStatsDur, patchEntryNumberSpan, time.Since(loadStart))
		if err != nil {
			// Release partial loaded segment data if load failed.
			segment.ReleaseSegmentData()
		}
		stateLockGuard.Done(err)
	}()

	collection := segment.GetCollection()
	indexedFieldInfos, _, textIndexes, unindexedTextFields, jsonKeyStats, _, _ := separateLoadInfoV2(loadInfo, collection.Schema())

	log := log.Ctx(ctx).With(zap.Int64("segmentID", segment.ID()))
	tr := timerecord.NewTimeRecorder("segmentLoader.loadSealedSegment")
	log.Info("Start loading fields...",
		zap.Int("indexedFields count", len(indexedFieldInfos)),
		zap.Int64s("indexed text fields", lo.Keys(textIndexes)),
		zap.Int64s("unindexed text fields", lo.Keys(unindexedTextFields)),
		zap.Int64s("indexed json key fields", lo.Keys(jsonKeyStats)),
	)
	submitStart := time.Now()
	_, err = GetLoadPool().Submit(func() (any, error) {
		poolWaitDur = time.Since(submitStart)
		cgoLoadStart := time.Now()
		if err = segment.csegment.Load(ctx); err != nil {
			cgoLoadDur = time.Since(cgoLoadStart)
			return struct{}{}, errors.Wrap(err, "At Load")
		}
		cgoLoadDur = time.Since(cgoLoadStart)

		syncJSONStatsStart := time.Now()
		segment.syncFieldJSONStatsFromLoadInfo(ctx, segment.LoadInfo())
		syncJSONStatsDur = time.Since(syncJSONStatsStart)

		return struct{}{}, nil
	}).Await()
	if err != nil {
		return err
	}

	for _, indexInfo := range loadInfo.IndexInfos {
		segment.fieldIndexes.Insert(indexInfo.GetIndexID(), &IndexedFieldInfo{
			FieldBinlog: &datapb.FieldBinlog{
				FieldID: indexInfo.GetFieldID(),
			},
			IndexInfo: indexInfo,
			IsLoaded:  true,
		})
	}

	// 4. rectify entries number for binlog in very rare cases
	// https://github.com/milvus-io/milvus/23654
	// legacy entry num = 0
	patchEntryStart := time.Now()
	if err := loader.patchEntryNumber(ctx, segment, loadInfo); err != nil {
		return err
	}
	patchEntryNumberSpan = time.Since(patchEntryStart)
	sealedLoadSpan := tr.RecordSpan()
	log.Info("Finish loading segment",
		zap.Duration("patchEntryNumberSpan", patchEntryNumberSpan),
		zap.Duration("sealedLoadSpan", sealedLoadSpan),
	)
	return nil
}

func (loader *segmentLoader) LoadSegment(ctx context.Context,
	seg Segment,
	loadInfo *querypb.SegmentLoadInfo,
) (err error) {
	segment, ok := seg.(*LocalSegment)
	if !ok {
		return merr.WrapErrParameterInvalid("LocalSegment", fmt.Sprintf("%T", seg))
	}
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", segment.Collection()),
		zap.Int64("partitionID", segment.Partition()),
		zap.String("shard", segment.Shard().VirtualName()),
		zap.Int64("segmentID", segment.ID()),
	)

	log.Info("start loading segment files",
		zap.Int64("rowNum", loadInfo.GetNumOfRows()),
		zap.String("segmentType", segment.Type().String()),
		zap.Int32("priority", int32(loadInfo.GetPriority())))

	collection := loader.manager.Collection.Get(segment.Collection())
	if collection == nil {
		err := merr.WrapErrCollectionNotFound(segment.Collection())
		log.Warn("failed to get collection while loading segment", zap.Error(err))
		return err
	}
	pkField := GetPkField(collection.Schema())

	if segment.Type() == SegmentTypeSealed {
		if err := loader.loadSealedSegment(ctx, loadInfo, segment); err != nil {
			return err
		}
	} else {
		if err := segment.Load(ctx); err != nil {
			return err
		}
	}

	binlogSize := calculateSegmentMemorySize(segment.LoadInfo())
	segment.manager.AddLoadedBinlogSize(binlogSize)
	segment.binlogSize.Store(binlogSize)

	// load statslog if it's growing segment
	if segment.segmentType == SegmentTypeGrowing {
		if bf, ok := segment.pkCandidate.(*pkoracle.BloomFilterSet); ok {
			log.Info("loading statslog...")
			resolver := packed.NewStatsResolverFromLoadInfo(loadInfo)
			bfPaths, err := resolver.BloomFilterPaths(pkField.GetFieldID())
			if err != nil {
				return err
			}
			if err := loader.loadBloomFilter(ctx, segment.ID(), bf, bfPaths); err != nil {
				return err
			}

			bm25Paths, err := resolver.BM25StatsPaths()
			if err != nil {
				return err
			}
			if err := loader.loadBm25Stats(ctx, segment.ID(), segment.bm25Stats, bm25Paths); err != nil {
				return err
			}
		}
	}
	return nil
}

func loadSealedSegmentFields(ctx context.Context, collection *Collection, segment *LocalSegment, fields []*datapb.FieldBinlog, rowCount int64) error {
	runningGroup, _ := errgroup.WithContext(ctx)
	for _, field := range fields {
		fieldBinLog := field
		fieldID := field.FieldID
		runningGroup.Go(func() error {
			return segment.LoadFieldData(ctx, fieldID, rowCount, fieldBinLog)
		})
	}
	err := runningGroup.Wait()
	if err != nil {
		return err
	}

	log.Ctx(ctx).Info("load field binlogs done for sealed segment",
		zap.Int64("collection", segment.Collection()),
		zap.Int64("segment", segment.ID()),
		zap.Int("len(field)", len(fields)),
		zap.String("segmentType", segment.Type().String()))

	return nil
}

func (loader *segmentLoader) loadFieldsIndex(ctx context.Context,
	schemaHelper *typeutil.SchemaHelper,
	segment *LocalSegment,
	numRows int64,
	indexedFieldInfos map[int64]*IndexedFieldInfo,
) error {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", segment.Collection()),
		zap.Int64("partitionID", segment.Partition()),
		zap.Int64("segmentID", segment.ID()),
		zap.Int64("rowCount", numRows),
	)

	for _, fieldInfo := range indexedFieldInfos {
		fieldID := fieldInfo.IndexInfo.FieldID
		indexInfo := fieldInfo.IndexInfo
		tr := timerecord.NewTimeRecorder("loadFieldIndex")
		err := loader.loadFieldIndex(ctx, segment, indexInfo)
		loadFieldIndexSpan := tr.RecordSpan()
		if err != nil {
			return err
		}

		log.Info("load field binlogs done for sealed segment with index",
			zap.Int64("fieldID", fieldID),
			zap.Any("binlog", fieldInfo.FieldBinlog.Binlogs),
			zap.Int32("current_index_version", fieldInfo.IndexInfo.GetCurrentIndexVersion()),
			zap.Duration("load_duration", loadFieldIndexSpan),
		)

		// set average row data size of variable field
		field, err := schemaHelper.GetFieldFromID(fieldID)
		if err != nil {
			return err
		}
		if typeutil.IsVariableDataType(field.GetDataType()) {
			err = segment.UpdateFieldRawDataSize(ctx, numRows, fieldInfo.FieldBinlog)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (loader *segmentLoader) loadBm25Stats(ctx context.Context, segmentID int64, stats map[int64]*storage.BM25Stats, binlogPaths map[int64][]string) error {
	log := log.Ctx(ctx).With(
		zap.Int64("segmentID", segmentID),
	)
	if len(binlogPaths) == 0 {
		log.Info("there are no bm25 stats logs saved with segment")
		return nil
	}

	pathList := []string{}
	fieldList := []int64{}
	fieldOffset := []int{}
	for fieldId, logpaths := range binlogPaths {
		pathList = append(pathList, logpaths...)
		fieldList = append(fieldList, fieldId)
		fieldOffset = append(fieldOffset, len(logpaths))
	}

	startTs := time.Now()
	values, err := loader.cm.MultiRead(ctx, pathList)
	if err != nil {
		return err
	}

	cnt := 0
	for i, fieldID := range fieldList {
		newStats, ok := stats[fieldID]
		if !ok {
			newStats = storage.NewBM25Stats()
			stats[fieldID] = newStats
		}

		for j := 0; j < fieldOffset[i]; j++ {
			err := newStats.Deserialize(values[cnt+j])
			if err != nil {
				return err
			}
		}
		cnt += fieldOffset[i]
		log.Info("Successfully load bm25 stats", zap.Duration("time", time.Since(startTs)), zap.Int64("numRow", newStats.NumRow()), zap.Int64("fieldID", fieldID))
	}

	return nil
}

func (loader *segmentLoader) loadFieldIndex(ctx context.Context, segment *LocalSegment, indexInfo *querypb.FieldIndexInfo) error {
	filteredPaths := make([]string, 0, len(indexInfo.IndexFilePaths))

	for _, indexPath := range indexInfo.IndexFilePaths {
		if path.Base(indexPath) != storage.IndexParamsKey {
			filteredPaths = append(filteredPaths, indexPath)
		}
	}

	indexInfo.IndexFilePaths = filteredPaths
	fieldType, err := loader.getFieldType(segment.Collection(), indexInfo.FieldID)
	if err != nil {
		return err
	}

	collection := loader.manager.Collection.Get(segment.Collection())
	if collection == nil {
		return merr.WrapErrCollectionNotLoaded(segment.Collection(), "failed to load field index")
	}

	return segment.LoadIndex(ctx, indexInfo, fieldType)
}

func (loader *segmentLoader) loadBloomFilter(ctx context.Context, segmentID int64, bfs *pkoracle.BloomFilterSet,
	binlogPaths []string,
) error {
	log := log.Ctx(ctx).With(
		zap.Int64("segmentID", segmentID),
	)
	if len(binlogPaths) == 0 {
		log.Info("there are no stats logs saved with segment")
		return nil
	}

	startTs := time.Now()
	values, err := loader.cm.MultiRead(ctx, binlogPaths)
	if err != nil {
		return err
	}
	blobs := make([]*storage.Blob, len(values))
	for i := range values {
		blobs[i] = &storage.Blob{Value: values[i]}
	}

	stats, err := storage.DeserializeBloomFilterStats(binlogPaths, blobs)
	if err != nil {
		log.Warn("failed to deserialize bloom filter stats", zap.Error(err))
		return err
	}

	var size uint
	for _, stat := range stats {
		pkStat := &storage.PkStatistics{
			PkFilter: stat.BF,
			MinPK:    stat.MinPk,
			MaxPK:    stat.MaxPk,
		}
		size += stat.BF.Cap()
		bfs.AddHistoricalStats(pkStat)
	}
	log.Info("Successfully load pk stats", zap.Duration("time", time.Since(startTs)), zap.Uint("size", size))
	return nil
}

// loadDeltalogs performs the internal actions of `LoadDeltaLogs`
// this function does not perform resource check and is meant be used among other load APIs.
func (loader *segmentLoader) loadDeltalogs(ctx context.Context, segment Segment, loadInfo *querypb.SegmentLoadInfo) error {
	deltaLogs := loadInfo.GetDeltalogs()
	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, fmt.Sprintf("LoadDeltalogs-%d", segment.ID()))
	defer sp.End()
	log := log.Ctx(ctx).With(
		zap.Int64("segmentID", segment.ID()),
		zap.Int("deltaNum", len(deltaLogs)),
	)
	log.Info("loading delta...")

	var rowNums int64
	valid := func(binlog *datapb.Binlog, _ int) bool {
		// the segment has applied the delta logs, skip it
		if binlog.GetTimestampTo() > 0 && // this field may be missed in legacy versions
			binlog.GetTimestampTo() < segment.LastDeltaTimestamp() {
			return false
		}
		return true
	}
	for _, deltaLog := range deltaLogs {
		rowNums += lo.SumBy(lo.Filter(deltaLog.GetBinlogs(), valid), func(binlog *datapb.Binlog) int64 {
			return binlog.GetEntriesNum()
		})
	}

	collection := loader.manager.Collection.Get(segment.Collection())

	helper, _ := typeutil.CreateSchemaHelper(collection.Schema())
	pkField, _ := helper.GetPrimaryKeyField()
	deltaData, err := storage.NewDeltaDataWithPkType(rowNums, pkField.DataType)
	if err != nil {
		return err
	}

	readDeltaRecords := func(reader storage.RecordReader) error {
		defer reader.Close()
		for {
			dl, err := reader.Next()
			if err != nil {
				if err == io.EOF {
					break
				}
				return err
			}

			for i := 0; i < dl.Len(); i++ {
				var pk storage.PrimaryKey
				switch pkField.DataType {
				case schemapb.DataType_Int64:
					pk = storage.NewInt64PrimaryKey(dl.Column(0).(*array.Int64).Value(i))
				case schemapb.DataType_VarChar:
					pk = storage.NewVarCharPrimaryKey(dl.Column(0).(*array.String).Value(i))
				}
				ts := typeutil.Timestamp(dl.Column(1).(*array.Int64).Value(i))
				err = deltaData.Append(pk, ts)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}

	// Collect delta paths and reader options based on storage version.
	var paths []string
	var opts []storage.RwOption
	if manifestPath := loadInfo.GetManifestPath(); manifestPath != "" {
		// V3: delta data lives in manifest
		paths, err = packed.GetDeltaLogPathsFromManifest(manifestPath, createStorageConfig())
		if err != nil {
			return err
		}
		opts = []storage.RwOption{
			storage.WithStorageConfig(createStorageConfig()),
			storage.WithVersion(storage.StorageV3),
		}
	} else {
		// V1: delta data referenced by Deltalogs entries
		for _, deltalog := range deltaLogs {
			for _, binlog := range lo.Filter(deltalog.Binlogs, valid) {
				if p := binlog.GetLogPath(); p != "" {
					paths = append(paths, p)
				}
			}
		}
		opts = []storage.RwOption{
			storage.WithDownloader(func(ctx context.Context, paths []string) ([][]byte, error) {
				return loader.cm.MultiRead(ctx, paths)
			}),
		}
	}

	if len(paths) > 0 {
		reader, err := storage.NewDeltalogReader(pkField.DataType, paths, opts...)
		if err != nil {
			return err
		}
		if err := readDeltaRecords(reader); err != nil {
			return err
		}
	}

	err = segment.LoadDeltaData(ctx, deltaData)
	if err != nil {
		return err
	}

	log.Info("load delta logs done", zap.Int64("deleteCount", deltaData.DeleteRowCount()))
	return nil
}

// LoadDeltaLogs load deltalog and write delta data into provided segment.
// it also executes resource protection logic in case of OOM.
func (loader *segmentLoader) LoadDeltaLogs(ctx context.Context, segment Segment, loadInfo *querypb.SegmentLoadInfo) error {
	// Check memory & storage limit
	requestResourceResult, err := loader.requestResource(ctx, loadInfo)
	if err != nil {
		log.Warn("request resource failed", zap.Error(err))
		return err
	}
	defer loader.freeRequestResource(requestResourceResult)
	return loader.loadDeltalogs(ctx, segment, loadInfo)
}

func createStorageConfig() *indexpb.StorageConfig {
	params := paramtable.Get()
	if params.CommonCfg.StorageType.GetValue() == "local" {
		return &indexpb.StorageConfig{
			RootPath:    params.LocalStorageCfg.Path.GetValue(),
			StorageType: params.CommonCfg.StorageType.GetValue(),
		}
	}
	return &indexpb.StorageConfig{
		Address:           params.MinioCfg.Address.GetValue(),
		AccessKeyID:       params.MinioCfg.AccessKeyID.GetValue(),
		SecretAccessKey:   params.MinioCfg.SecretAccessKey.GetValue(),
		UseSSL:            params.MinioCfg.UseSSL.GetAsBool(),
		SslCACert:         params.MinioCfg.SslCACert.GetValue(),
		BucketName:        params.MinioCfg.BucketName.GetValue(),
		RootPath:          params.MinioCfg.RootPath.GetValue(),
		UseIAM:            params.MinioCfg.UseIAM.GetAsBool(),
		IAMEndpoint:       params.MinioCfg.IAMEndpoint.GetValue(),
		StorageType:       params.CommonCfg.StorageType.GetValue(),
		Region:            params.MinioCfg.Region.GetValue(),
		UseVirtualHost:    params.MinioCfg.UseVirtualHost.GetAsBool(),
		CloudProvider:     params.MinioCfg.CloudProvider.GetValue(),
		RequestTimeoutMs:  params.MinioCfg.RequestTimeoutMs.GetAsInt64(),
		GcpCredentialJSON: params.MinioCfg.GcpCredentialJSON.GetValue(),
		SslTlsMinVersion:  params.MinioCfg.SslTLSMinVersion.GetValue(),
	}
}

func (loader *segmentLoader) patchEntryNumber(ctx context.Context, segment *LocalSegment, loadInfo *querypb.SegmentLoadInfo) error {
	var needReset bool

	segment.fieldIndexes.Range(func(indexID int64, info *IndexedFieldInfo) bool {
		for _, info := range info.FieldBinlog.GetBinlogs() {
			if info.GetEntriesNum() == 0 {
				needReset = true
				return false
			}
		}
		return true
	})
	if !needReset {
		return nil
	}

	log.Warn("legacy segment binlog found, start to patch entry num", zap.Int64("segmentID", segment.ID()))
	rowIDField := lo.FindOrElse(loadInfo.BinlogPaths, nil, func(binlog *datapb.FieldBinlog) bool {
		return binlog.GetFieldID() == common.RowIDField
	})

	if rowIDField == nil {
		return errors.New("rowID field binlog not found")
	}

	counts := make([]int64, 0, len(rowIDField.GetBinlogs()))
	for _, binlog := range rowIDField.GetBinlogs() {
		// binlog.LogPath has already been filled
		bs, err := loader.cm.Read(ctx, binlog.LogPath)
		if err != nil {
			return err
		}

		// get binlog entry num from rowID field
		// since header does not store entry numb, we have to read all data here

		reader, err := storage.NewBinlogReader(bs)
		if err != nil {
			return err
		}
		er, err := reader.NextEventReader()
		if err != nil {
			return err
		}

		rowIDs, _, err := er.GetInt64FromPayload()
		if err != nil {
			return err
		}
		counts = append(counts, int64(len(rowIDs)))
	}

	var err error
	segment.fieldIndexes.Range(func(indexID int64, info *IndexedFieldInfo) bool {
		if len(info.FieldBinlog.GetBinlogs()) != len(counts) {
			err = errors.New("rowID & index binlog number not matched")
			return false
		}
		for i, binlog := range info.FieldBinlog.GetBinlogs() {
			binlog.EntriesNum = counts[i]
		}
		return true
	})
	return err
}

// JoinIDPath joins ids to path format.
func JoinIDPath(ids ...int64) string {
	idStr := make([]string, 0, len(ids))
	for _, id := range ids {
		idStr = append(idStr, strconv.FormatInt(id, 10))
	}
	return path.Join(idStr...)
}

// After introducing the caching layer's lazy loading and eviction mechanisms, most parts of a segment won't be
// loaded into memory or disk immediately, even if the segment is marked as LOADED. This means physical resource
// usage may be very low.
// However, we still need to reserve enough resources for the segments marked as LOADED. The reserved resource is
// treated as the logical resource usage. Logical resource usage is based on the segment final resource usage.
// checkLogicalSegmentSize checks whether the memory & disk is sufficient to load the segments,
// returns the memory & disk logical usage while loading if possible to load, otherwise, returns error
func (loader *segmentLoader) checkLogicalSegmentSize(ctx context.Context, segmentLoadInfos []*querypb.SegmentLoadInfo, totalMem uint64) (uint64, uint64, error) {
	if !paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool() {
		return 0, 0, nil
	}

	if len(segmentLoadInfos) == 0 {
		return 0, 0, nil
	}

	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", segmentLoadInfos[0].GetCollectionID()),
	)

	logicalMemUsage := loader.manager.Segment.GetLogicalResource().MemorySize
	logicalDiskUsage := loader.manager.Segment.GetLogicalResource().DiskSize

	logicalMemUsage += loader.committedLogicalResource.MemorySize
	logicalDiskUsage += loader.committedLogicalResource.DiskSize

	// logical resource usage is based on the segment final resource usage,
	// so we need to estimate the final resource usage of the segments
	finalFactor := resourceEstimateFactor{
		deltaDataExpansionFactor:        paramtable.Get().QueryNodeCfg.DeltaDataExpansionRate.GetAsFloat(),
		textIndexExpansionFactor:        paramtable.Get().QueryNodeCfg.TextIndexExpansionFactor.GetAsFloat(),
		TieredEvictionEnabled:           paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool(),
		TieredEvictableMemoryCacheRatio: paramtable.Get().QueryNodeCfg.TieredEvictableMemoryCacheRatio.GetAsFloat(),
		TieredEvictableDiskCacheRatio:   paramtable.Get().QueryNodeCfg.TieredEvictableDiskCacheRatio.GetAsFloat(),
	}
	predictLogicalMemUsage := logicalMemUsage
	predictLogicalDiskUsage := logicalDiskUsage
	for _, loadInfo := range segmentLoadInfos {
		collection := loader.manager.Collection.Get(loadInfo.GetCollectionID())
		finalUsage, err := estimateLogicalResourceUsageOfSegment(collection.Schema(), loadInfo, finalFactor)
		if err != nil {
			log.Warn(
				"failed to estimate final resource usage of segment",
				zap.Int64("collectionID", loadInfo.GetCollectionID()),
				zap.Int64("segmentID", loadInfo.GetSegmentID()),
				zap.Error(err))
			return 0, 0, err
		}

		log.Debug("segment logical resource for loading",
			zap.Int64("segmentID", loadInfo.GetSegmentID()),
			zap.Float64("memoryUsage(MB)", logutil.ToMB(float64(finalUsage.MemorySize))),
			zap.Float64("diskUsage(MB)", logutil.ToMB(float64(finalUsage.DiskSize))),
		)
		predictLogicalDiskUsage += finalUsage.DiskSize
		predictLogicalMemUsage += finalUsage.MemorySize
	}

	log.Info("predict memory and disk logical usage after loaded (in MiB)",
		zap.Float64("predictLogicalMemUsage(MB)", logutil.ToMB(float64(predictLogicalMemUsage))),
		zap.Float64("predictLogicalDiskUsage(MB)", logutil.ToMB(float64(predictLogicalDiskUsage))),
	)

	logicalMemUsageLimit := uint64(float64(totalMem) * paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat())
	logicalDiskUsageLimit := uint64(float64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64()) * paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat())

	if predictLogicalMemUsage > logicalMemUsageLimit {
		log.Warn("logical memory usage checking for segment loading failed",
			zap.String("resourceType", "Memory"),
			zap.Float64("predictLogicalMemUsageMB", logutil.ToMB(float64(predictLogicalMemUsage))),
			zap.Float64("logicalMemUsageLimitMB", logutil.ToMB(float64(logicalMemUsageLimit))),
			zap.Float64("evictableMemoryCacheRatio", paramtable.Get().QueryNodeCfg.TieredEvictableMemoryCacheRatio.GetAsFloat()),
		)
		return 0, 0, merr.WrapErrSegmentRequestResourceFailed("Memory")
	}

	if predictLogicalDiskUsage > logicalDiskUsageLimit {
		log.Warn(fmt.Sprintf("Logical disk usage checking for segment loading failed, predictLogicalDiskUsage = %v MB, LogicalDiskUsageLimit = %v MB, decrease the evictableDiskCacheRatio (current: %v) if you want to load more segments",
			logutil.ToMB(float64(predictLogicalDiskUsage)),
			logutil.ToMB(float64(logicalDiskUsageLimit)),
			paramtable.Get().QueryNodeCfg.TieredEvictableDiskCacheRatio.GetAsFloat(),
		))
		return 0, 0, merr.WrapErrSegmentRequestResourceFailed("Disk")
	}

	return predictLogicalMemUsage - logicalMemUsage, predictLogicalDiskUsage - logicalDiskUsage, nil
}

func (loader *segmentLoader) estimateSegmentLoadingResourceUsage(ctx context.Context, segmentLoadInfos ...*querypb.SegmentLoadInfo) (*ResourceUsage, uint64, error) {
	if len(segmentLoadInfos) == 0 {
		return &ResourceUsage{}, 0, nil
	}

	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", segmentLoadInfos[0].GetCollectionID()),
	)

	maxFactor := resourceEstimateFactor{
		memoryUsageFactor:           paramtable.Get().QueryNodeCfg.LoadMemoryUsageFactor.GetAsFloat(),
		memoryIndexUsageFactor:      paramtable.Get().QueryNodeCfg.MemoryIndexLoadPredictMemoryUsageFactor.GetAsFloat(),
		EnableInterminSegmentIndex:  paramtable.Get().QueryNodeCfg.EnableInterminSegmentIndex.GetAsBool(),
		tempSegmentIndexFactor:      paramtable.Get().QueryNodeCfg.InterimIndexMemExpandRate.GetAsFloat(),
		deltaDataExpansionFactor:    paramtable.Get().QueryNodeCfg.DeltaDataExpansionRate.GetAsFloat(),
		jsonKeyStatsExpansionFactor: paramtable.Get().QueryNodeCfg.JSONKeyStatsExpansionFactor.GetAsFloat(),
		textIndexExpansionFactor:    paramtable.Get().QueryNodeCfg.TextIndexExpansionFactor.GetAsFloat(),
		TieredEvictionEnabled:       paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool(),
		externalRawDataFactor:       paramtable.Get().QueryNodeCfg.ExternalCollectionRawDataFactor.GetAsFloat(),
	}
	maxSegmentSize := uint64(0)
	loadingUsage := &ResourceUsage{}
	var predictGpuMemUsage []uint64
	for _, loadInfo := range segmentLoadInfos {
		collection := loader.manager.Collection.Get(loadInfo.GetCollectionID())
		segmentUsage, err := estimateLoadingResourceUsageOfSegment(collection.Schema(), loadInfo, maxFactor)
		if err != nil {
			log.Warn(
				"failed to estimate max resource usage of segment",
				zap.Int64("collectionID", loadInfo.GetCollectionID()),
				zap.Int64("segmentID", loadInfo.GetSegmentID()),
				zap.Error(err))
			return nil, 0, err
		}

		log.Debug("segment resource for loading",
			zap.Int64("segmentID", loadInfo.GetSegmentID()),
			zap.Float64("loadingMemoryUsage(MB)", logutil.ToMB(float64(segmentUsage.MemorySize))),
			zap.Float64("loadingDiskUsage(MB)", logutil.ToMB(float64(segmentUsage.DiskSize))),
			zap.Float64("memoryLoadFactor", maxFactor.memoryUsageFactor),
		)
		loadingUsage.MmapFieldCount += segmentUsage.MmapFieldCount
		loadingUsage.DiskSize += segmentUsage.DiskSize
		loadingUsage.MemorySize += segmentUsage.MemorySize
		predictGpuMemUsage = append(predictGpuMemUsage, segmentUsage.FieldGpuMemorySize...)
		if segmentUsage.MemorySize > maxSegmentSize {
			maxSegmentSize = segmentUsage.MemorySize
		}
	}
	loadingUsage.FieldGpuMemorySize = predictGpuMemUsage

	return loadingUsage, maxSegmentSize, nil
}

// checkSegmentSize checks whether the memory & disk is sufficient to load the segments
// returns the memory & disk usage while loading if possible to load,
// otherwise, returns error
func (loader *segmentLoader) checkSegmentSize(ctx context.Context, segmentLoadInfos []*querypb.SegmentLoadInfo, totalMem, memUsage uint64, localDiskUsage int64) (uint64, uint64, error) {
	if len(segmentLoadInfos) == 0 {
		return 0, 0, nil
	}

	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", segmentLoadInfos[0].GetCollectionID()),
	)

	memUsage = memUsage + loader.committedResource.MemorySize
	if memUsage == 0 || totalMem == 0 {
		return 0, 0, errors.New("get memory failed when checkSegmentSize")
	}

	diskUsage := uint64(localDiskUsage) + loader.committedResource.DiskSize

	loadingUsage, maxSegmentSize, err := loader.estimateSegmentLoadingResourceUsage(ctx, segmentLoadInfos...)
	if err != nil {
		return 0, 0, err
	}
	predictMemUsage := memUsage + loadingUsage.MemorySize
	predictDiskUsage := diskUsage + loadingUsage.DiskSize

	log.Info("predict memory and disk usage while loading (in MiB)",
		zap.Float64("maxSegmentSize(MB)", logutil.ToMB(float64(maxSegmentSize))),
		zap.Float64("committedMemSize(MB)", logutil.ToMB(float64(loader.committedResource.MemorySize))),
		zap.Float64("memLimit(MB)", logutil.ToMB(float64(totalMem))),
		zap.Float64("memUsage(MB)", logutil.ToMB(float64(memUsage))),
		zap.Float64("committedDiskSize(MB)", logutil.ToMB(float64(loader.committedResource.DiskSize))),
		zap.Float64("diskUsage(MB)", logutil.ToMB(float64(diskUsage))),
		zap.Float64("predictMemUsage(MB)", logutil.ToMB(float64(predictMemUsage))),
		zap.Float64("predictDiskUsage(MB)", logutil.ToMB(float64(predictDiskUsage))),
		zap.Int("mmapFieldCount", loadingUsage.MmapFieldCount),
	)

	if paramtable.Get().QueryNodeCfg.TieredEvictionEnabled.GetAsBool() {
		// try to reserve loading resource from caching layer
		if ok := C.TryReserveLoadingResourceWithTimeout(C.CResourceUsage{
			memory_bytes: C.int64_t(loadingUsage.MemorySize),
			disk_bytes:   C.int64_t(loadingUsage.DiskSize),
		}, 1000); !ok {
			return 0, 0, fmt.Errorf("failed to reserve loading resource from caching layer, predictMemUsage = %v MB, predictDiskUsage = %v MB, memUsage = %v MB, diskUsage = %v MB, memoryThresholdFactor = %f, diskThresholdFactor = %f",
				logutil.ToMB(float64(predictMemUsage)),
				logutil.ToMB(float64(predictDiskUsage)),
				logutil.ToMB(float64(memUsage)),
				logutil.ToMB(float64(diskUsage)),
				paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat(),
				paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat(),
			)
		}
	} else {
		// fallback to original segment loading logic
		if predictMemUsage > uint64(float64(totalMem)*paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat()) {
			log.Warn("load segment failed, OOM if load",
				zap.String("resourceType", "Memory"),
				zap.Float64("maxSegmentSizeMB", logutil.ToMB(float64(maxSegmentSize))),
				zap.Float64("memUsageMB", logutil.ToMB(float64(memUsage))),
				zap.Float64("predictMemUsageMB", logutil.ToMB(float64(predictMemUsage))),
				zap.Float64("totalMemMB", logutil.ToMB(float64(totalMem))),
				zap.Float64("thresholdFactor", paramtable.Get().QueryNodeCfg.OverloadedMemoryThresholdPercentage.GetAsFloat()),
			)
			return 0, 0, merr.WrapErrSegmentRequestResourceFailed("Memory")
		}

		if predictDiskUsage > uint64(float64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64())*paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat()) {
			log.Warn("load segment failed, disk space is not enough",
				zap.String("resourceType", "Disk"),
				zap.Float64("diskUsageMB", logutil.ToMB(float64(diskUsage))),
				zap.Float64("predictDiskUsageMB", logutil.ToMB(float64(predictDiskUsage))),
				zap.Float64("totalDiskMB", logutil.ToMB(float64(uint64(paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64())))),
				zap.Float64("thresholdFactor", paramtable.Get().QueryNodeCfg.MaxDiskUsagePercentage.GetAsFloat()),
			)
			return 0, 0, merr.WrapErrSegmentRequestResourceFailed("Disk")
		}
	}

	err = checkSegmentGpuMemSize(loadingUsage.FieldGpuMemorySize, float32(paramtable.Get().GpuConfig.OverloadedMemoryThresholdPercentage.GetAsFloat()))
	if err != nil {
		return 0, 0, err
	}

	return loadingUsage.MemorySize, loadingUsage.DiskSize, nil
}

// this function is used to estimate the logical resource usage of a segment, which should only be used when tiered eviction is enabled
// the result is the final resource usage of the segment inevictable part plus the final usage of evictable part with cache ratio applied
// TODO: the inevictable part is not correct, since we cannot know the final resource usage of interim index and default-value column before loading,
// current they are ignored, but we should consider them in the future
func estimateLogicalResourceUsageOfSegment(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo, multiplyFactor resourceEstimateFactor) (usage *ResourceUsage, err error) {
	var segmentInevictableMemorySize, segmentInevictableDiskSize uint64
	var segmentEvictableMemorySize, segmentEvictableDiskSize uint64

	id2Binlogs := lo.SliceToMap(loadInfo.BinlogPaths, func(fieldBinlog *datapb.FieldBinlog) (int64, *datapb.FieldBinlog) {
		return fieldBinlog.GetFieldID(), fieldBinlog
	})

	schemaHelper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		log.Warn("failed to create schema helper", zap.String("name", schema.GetName()), zap.Error(err))
		return nil, err
	}
	ctx := context.Background()

	// PART 1: calculate logical resource usage of indexes
	for _, fieldIndexInfo := range loadInfo.IndexInfos {
		fieldID := fieldIndexInfo.GetFieldID()
		if len(fieldIndexInfo.GetIndexFilePaths()) > 0 {
			fieldSchema, err := schemaHelper.GetFieldFromID(fieldID)
			if err != nil {
				return nil, err
			}
			isVectorType := typeutil.IsVectorType(fieldSchema.GetDataType())

			var estimateResult ResourceEstimate
			err = GetCLoadInfoWithFunc(ctx, fieldSchema, loadInfo, fieldIndexInfo, func(c *LoadIndexInfo) error {
				GetDynamicPool().Submit(func() (any, error) {
					loadResourceRequest := C.EstimateLoadIndexResource(c.cLoadIndexInfo)
					estimateResult = GetResourceEstimate(&loadResourceRequest)
					return nil, nil
				}).Await()
				return nil
			})
			if err != nil {
				return nil, errors.Wrapf(err, "failed to estimate logical resource usage of index, collection %d, segment %d, indexBuildID %d",
					loadInfo.GetCollectionID(),
					loadInfo.GetSegmentID(),
					fieldIndexInfo.GetBuildID())
			}
			segmentEvictableMemorySize += estimateResult.FinalMemoryCost
			segmentEvictableDiskSize += estimateResult.FinalDiskCost

			// could skip binlog or
			// could be missing for new field or storage v2 group 0
			if estimateResult.HasRawData &&
				!paramtable.Get().QueryNodeCfg.PreferFieldDataWhenIndexHasRawData.GetAsBool() {
				delete(id2Binlogs, fieldID)
				continue
			}

			// BM25 only checks vector datatype
			// scalar index does not have metrics type key
			if !isVectorType {
				continue
			}

			metricType, err := funcutil.GetAttrByKeyFromRepeatedKV(common.MetricTypeKey, fieldIndexInfo.IndexParams)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to estimate logical resource usage of index, metric type not found, collection %d, segment %d, indexBuildID %d",
					loadInfo.GetCollectionID(),
					loadInfo.GetSegmentID(),
					fieldIndexInfo.GetBuildID())
			}
			// skip raw data for BM25 index
			if metricType == metric.BM25 {
				delete(id2Binlogs, fieldID)
			}
		}
	}

	// PART 2: calculate logical resource usage of binlogs
	for fieldID, fieldBinlog := range id2Binlogs {
		fieldIDs := fieldBinlog.GetChildFields()
		// legacy default split
		if len(fieldIDs) == 0 {
			fieldIDs = []int64{fieldID}
		}
		binlogSize := uint64(getBinlogDataMemorySize(fieldBinlog))

		var supportInterimIndexDataType bool
		var containsTimestampField bool
		var doubleMemoryDataField bool
		var legacyNilSchema bool
		mmapEnabled := true
		isVectorType := true

		for _, fieldID := range fieldIDs {
			// get field schema from fieldID
			fieldSchema, err := schemaHelper.GetFieldFromID(fieldID)
			if err != nil {
				log.Warn("failed to get field schema", zap.Int64("fieldID", fieldID), zap.String("name", schema.GetName()), zap.Error(err))
				return nil, err
			}

			// missing mapping, shall be "0" group for storage v2
			if fieldSchema == nil {
				legacyNilSchema = true
				break
			}

			supportInterimIndexDataType = supportInterimIndexDataType || SupportInterimIndexDataType(fieldSchema.GetDataType())
			isVectorType = isVectorType && typeutil.IsVectorType(fieldSchema.GetDataType())
			// constainSystemField = constainSystemField || common.IsSystemField(fieldSchema.GetFieldID())
			mmapEnabled = mmapEnabled && isDataMmapEnable(fieldSchema)
			containsTimestampField = containsTimestampField || DoubleMemorySystemField(fieldSchema.GetFieldID())
			doubleMemoryDataField = doubleMemoryDataField || DoubleMemoryDataType(fieldSchema.GetDataType())
		}

		// TODO: add default-value column's resource usage to inevictable part
		// TODO: add interim index's resource usage to inevictable part

		if legacyNilSchema {
			segmentEvictableMemorySize += binlogSize
			continue
		}

		// timestamp field double in InsertRecord & TimestampIndex
		if containsTimestampField {
			timestampSize := lo.SumBy(fieldBinlog.GetBinlogs(), func(binlog *datapb.Binlog) int64 {
				return binlog.GetEntriesNum() * 4
			})
			segmentInevictableMemorySize += 2 * uint64(timestampSize)
		}

		if isVectorType {
			mmapVectorField := paramtable.Get().QueryNodeCfg.MmapVectorField.GetAsBool()
			if mmapVectorField {
				segmentEvictableDiskSize += binlogSize
			} else {
				segmentEvictableMemorySize += binlogSize
			}
		} else if !mmapEnabled {
			segmentEvictableMemorySize += binlogSize
			if doubleMemoryDataField {
				segmentEvictableMemorySize += binlogSize
			}
		} else {
			segmentEvictableDiskSize += binlogSize
		}
	}

	// PART 3: calculate logical resource usage of stats data
	for _, fieldBinlog := range loadInfo.Statslogs {
		segmentInevictableMemorySize += uint64(getBinlogDataMemorySize(fieldBinlog))
	}

	// PART 4: calculate logical resource usage of delete data
	for _, fieldBinlog := range loadInfo.Deltalogs {
		// MemorySize of filedBinlog is the actual size in memory, so the expansionFactor
		//   should be 1, in most cases.
		expansionFactor := float64(1)
		memSize := getBinlogDataMemorySize(fieldBinlog)

		// Note: If MemorySize == DiskSize, it means the segment comes from Milvus 2.3,
		//   MemorySize is actually compressed DiskSize of deltalog, so we'll fallback to use
		//   deltaExpansionFactor to compromise the compression ratio.
		if memSize == getBinlogDataDiskSize(fieldBinlog) {
			expansionFactor = multiplyFactor.deltaDataExpansionFactor
		}
		segmentInevictableMemorySize += uint64(float64(memSize) * expansionFactor)
	}

	// PART 5: calculate logical resource usage of text index stats data
	// Text match indexes are evictable (support_eviction=true in caching layer).
	// Text match index mmap is driven by scalar_field_enable_mmap (same as raw scalar data).
	textIndexMmapEnable := paramtable.Get().QueryNodeCfg.MmapScalarField.GetAsBool()
	for _, textStats := range loadInfo.GetTextStatsLogs() {
		if textIndexMmapEnable {
			segmentEvictableDiskSize += uint64(float64(textStats.GetMemorySize()) * multiplyFactor.textIndexExpansionFactor)
		} else {
			segmentEvictableMemorySize += uint64(float64(textStats.GetMemorySize()) * multiplyFactor.textIndexExpansionFactor)
		}
	}

	log.Debug("estimate logical resoure usage result",
		zap.Int64("segmentID", loadInfo.GetSegmentID()),
		zap.Uint64("segmentInevictableMemorySize", segmentInevictableMemorySize),
		zap.Uint64("segmentEvictableMemorySize", segmentEvictableMemorySize),
		zap.Uint64("segmentInevictableDiskSize", segmentInevictableDiskSize),
		zap.Uint64("segmentEvictableDiskSize", segmentEvictableDiskSize),
	)

	return &ResourceUsage{
		MemorySize: segmentInevictableMemorySize + uint64(float64(segmentEvictableMemorySize)*multiplyFactor.TieredEvictableMemoryCacheRatio),
		DiskSize:   segmentInevictableDiskSize + uint64(float64(segmentEvictableDiskSize)*multiplyFactor.TieredEvictableDiskCacheRatio),
	}, nil
}

// estimateLoadingResourceUsageOfSegment estimates the resource usage of the segment when loading,
// it will return two different results, depending on the value of tiered eviction parameter:
//   - when tiered eviction is enabled, the result is the max resource usage of the segment that cannot be managed by caching layer,
//     which should be a subset of the segment inevictable part
//   - when tiered eviction is disabled, the result is the max resource usage of both the segment evictable and inevictable part
func estimateLoadingResourceUsageOfSegment(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo, multiplyFactor resourceEstimateFactor) (usage *ResourceUsage, err error) {
	var segMemoryLoadingSize, segDiskLoadingSize uint64
	var indexMemorySize uint64
	var mmapFieldCount int
	var fieldGpuMemorySize []uint64

	segmentEstimateStart := time.Now()
	var schemaDur time.Duration
	var indexLoopDur time.Duration
	var cLoadInfoDur time.Duration
	var cEstimateDur time.Duration
	var binlogLoopDur time.Duration
	var statsDur time.Duration
	var deleteDur time.Duration
	var jsonStatsDur time.Duration
	var textStatsDur time.Duration
	var estimatedIndexCount int
	defer func() {
		estimateSegmentResourceTiming.record(estimatedIndexCount, len(loadInfo.GetBinlogPaths()), schemaDur, indexLoopDur, cLoadInfoDur, cEstimateDur, binlogLoopDur, statsDur, deleteDur, jsonStatsDur, textStatsDur, time.Since(segmentEstimateStart))
	}()

	schemaStart := time.Now()

	id2Binlogs := lo.SliceToMap(loadInfo.BinlogPaths, func(fieldBinlog *datapb.FieldBinlog) (int64, *datapb.FieldBinlog) {
		return fieldBinlog.GetFieldID(), fieldBinlog
	})

	schemaHelper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		log.Warn("failed to create schema helper", zap.String("name", schema.GetName()), zap.Error(err))
		return nil, err
	}
	indexedFields := make(map[int64]struct{})
	schemaDur = time.Since(schemaStart)
	ctx := context.Background()

	// PART 1: calculate size of indexes
	indexLoopStart := time.Now()
	for _, fieldIndexInfo := range loadInfo.IndexInfos {
		fieldID := fieldIndexInfo.GetFieldID()
		if len(fieldIndexInfo.GetIndexFilePaths()) > 0 {
			estimatedIndexCount++
			fieldSchema, err := schemaHelper.GetFieldFromID(fieldID)
			if err != nil {
				return nil, err
			}
			indexedFields[fieldID] = struct{}{}

			isVectorType := typeutil.IsVectorType(fieldSchema.GetDataType())

			var estimateResult ResourceEstimate
			cLoadInfoStart := time.Now()
			err = GetCLoadInfoWithFunc(ctx, fieldSchema, loadInfo, fieldIndexInfo, func(c *LoadIndexInfo) error {
				var estimateDur time.Duration
				GetDynamicPool().Submit(func() (any, error) {
					estimateStart := time.Now()
					loadResourceRequest := C.EstimateLoadIndexResource(c.cLoadIndexInfo)
					estimateDur = time.Since(estimateStart)
					estimateResult = GetResourceEstimate(&loadResourceRequest)
					return nil, nil
				}).Await()
				cEstimateDur += estimateDur
				return nil
			})
			cLoadInfoDur += time.Since(cLoadInfoStart)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to estimate loading resource usage of index, collection %d, segment %d, indexBuildID %d",
					loadInfo.GetCollectionID(),
					loadInfo.GetSegmentID(),
					fieldIndexInfo.GetBuildID())
			}

			if !multiplyFactor.TieredEvictionEnabled {
				indexMemorySize += estimateResult.MaxMemoryCost
				segDiskLoadingSize += estimateResult.MaxDiskCost
			}

			if vecindexmgr.GetVecIndexMgrInstance().IsGPUVecIndex(common.GetIndexType(fieldIndexInfo.IndexParams)) {
				fieldGpuMemorySize = append(fieldGpuMemorySize, estimateResult.MaxMemoryCost)
			}

			// could skip binlog or
			// could be missing for new field or storage v2 group 0
			if estimateResult.HasRawData &&
				!paramtable.Get().QueryNodeCfg.PreferFieldDataWhenIndexHasRawData.GetAsBool() {
				delete(id2Binlogs, fieldID)
				continue
			}

			// BM25 only checks vector datatype
			// scalar index does not have metrics type key
			if !isVectorType {
				continue
			}

			metricType, err := funcutil.GetAttrByKeyFromRepeatedKV(common.MetricTypeKey, fieldIndexInfo.IndexParams)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to estimate loading resource usage of index, metric type not found, collection %d, segment %d, indexBuildID %d",
					loadInfo.GetCollectionID(),
					loadInfo.GetSegmentID(),
					fieldIndexInfo.GetBuildID())
			}
			// skip raw data for BM25 index
			if metricType == metric.BM25 {
				delete(id2Binlogs, fieldID)
			}
		}
	}
	indexLoopDur = time.Since(indexLoopStart)
	binlogLoopStart := time.Now()

	// PART 2: calculate size of binlogs
	for fieldID, fieldBinlog := range id2Binlogs {
		fieldIDs := fieldBinlog.GetChildFields()
		// legacy default split
		if len(fieldIDs) == 0 {
			fieldIDs = []int64{fieldID}
		}
		binlogSize := uint64(getBinlogDataMemorySize(fieldBinlog))

		var supportInterimIndexDataType bool
		var containsTimestampField bool
		var doubleMomoryDataField bool
		var legacyNilSchema bool
		mmapEnabled := true
		isVectorType := true
		hasIndex := true

		for _, fieldID := range fieldIDs {
			// get field schema from fieldID
			fieldSchema, err := schemaHelper.GetFieldFromID(fieldID)
			if err != nil {
				log.Warn("failed to get field schema", zap.Int64("fieldID", fieldID), zap.String("name", schema.GetName()), zap.Error(err))
				return nil, err
			}
			if _, ok := indexedFields[fieldID]; !ok {
				hasIndex = false
			}

			// missing mapping, shall be "0" group for storage v2
			if fieldSchema == nil {
				if !multiplyFactor.TieredEvictionEnabled {
					segMemoryLoadingSize += binlogSize
				}
				legacyNilSchema = true
				break
			}

			supportInterimIndexDataType = supportInterimIndexDataType || SupportInterimIndexDataType(fieldSchema.GetDataType())
			isVectorType = isVectorType && typeutil.IsVectorType(fieldSchema.GetDataType())
			mmapEnabled = mmapEnabled && isDataMmapEnable(fieldSchema)
			containsTimestampField = containsTimestampField || DoubleMemorySystemField(fieldSchema.GetFieldID())
			doubleMomoryDataField = doubleMomoryDataField || DoubleMemoryDataType(fieldSchema.GetDataType())
		}
		// legacy v2 segment without children
		if legacyNilSchema {
			continue
		}

		if !hasIndex {
			if !multiplyFactor.TieredEvictionEnabled {
				interimIndexEnable := multiplyFactor.EnableInterminSegmentIndex && !isGrowingMmapEnable() && supportInterimIndexDataType
				if interimIndexEnable {
					segMemoryLoadingSize += uint64(float64(binlogSize) * multiplyFactor.tempSegmentIndexFactor)
				}
			}
		}

		if isVectorType {
			mmapVectorField := paramtable.Get().QueryNodeCfg.MmapVectorField.GetAsBool()
			if mmapVectorField {
				if !multiplyFactor.TieredEvictionEnabled {
					segDiskLoadingSize += binlogSize
				}
			} else {
				if !multiplyFactor.TieredEvictionEnabled {
					segMemoryLoadingSize += binlogSize
				}
			}
			continue
		}

		// timestamp field double in InsertRecord & TimestampIndex
		if containsTimestampField {
			timestampSize := lo.SumBy(fieldBinlog.GetBinlogs(), func(binlog *datapb.Binlog) int64 {
				return binlog.GetEntriesNum() * 4
			})
			segMemoryLoadingSize += 2 * uint64(timestampSize)
		}

		if !mmapEnabled {
			if !multiplyFactor.TieredEvictionEnabled {
				segMemoryLoadingSize += binlogSize
				if doubleMomoryDataField {
					segMemoryLoadingSize += binlogSize
				}
			}
		} else {
			if !multiplyFactor.TieredEvictionEnabled {
				segDiskLoadingSize += uint64(getBinlogDataMemorySize(fieldBinlog))
			}
		}
	}
	binlogLoopDur = time.Since(binlogLoopStart)

	// PART 2.5: external segment adjustments
	//
	// External segments carry pre-computed MemorySize in fake binlogs (from
	// DataNode Take sampling). Adjust the memory estimate for two external-
	// specific behaviors:
	//   1. Non-lazy path: apply externalRawDataFactor to cover the peak
	//      transient memory during download + decompress + Arrow deserialize
	//      (normal packed segments do not have this peak because their
	//      binlogs are already in Arrow IPC format).
	//   2. Full-lazy path (all external fields warmup=disable): no eager
	//      load, so subtract the raw data size that PART 2 added.
	// Also propagate EstimatedBytesPerRow to the C++ ManifestGroupTranslator
	// so the tiered-cache layer sizes chunks correctly.
	if typeutil.IsExternalCollection(schema) && loadInfo.GetNumOfRows() > 0 {
		var fakeBinlogMemSize int64
		for _, fb := range loadInfo.BinlogPaths {
			fakeBinlogMemSize += getBinlogDataMemorySize(fb)
		}
		loadInfo.EstimatedBytesPerRow = fakeBinlogMemSize / loadInfo.GetNumOfRows()

		if isExternalCollectionLazyLoad(schema) {
			// Full-lazy → zero eager load. Undo PART 2's rawSize addition.
			// Safety factor does not apply: no peak to cover.
			if segMemoryLoadingSize >= uint64(fakeBinlogMemSize) {
				segMemoryLoadingSize -= uint64(fakeBinlogMemSize)
			} else {
				segMemoryLoadingSize = 0
			}
		} else if factor := multiplyFactor.externalRawDataFactor; factor > 1.0 {
			// Non-lazy → add peak margin on top of rawSize that PART 2 added.
			segMemoryLoadingSize += uint64(float64(fakeBinlogMemSize) * (factor - 1.0))
		}
	}

	// PART 3: calculate size of stats data
	// stats data isn't managed by the caching layer, so its size should always be included,
	// regardless of the tiered eviction value
	statsStart := time.Now()
	for _, fieldBinlog := range loadInfo.Statslogs {
		segMemoryLoadingSize += uint64(getBinlogDataMemorySize(fieldBinlog))
	}
	statsDur = time.Since(statsStart)

	// PART 4: calculate size of delete data
	// delete data isn't managed by the caching layer, so its size should always be included,
	// regardless of the tiered eviction value
	deleteStart := time.Now()
	for _, fieldBinlog := range loadInfo.Deltalogs {
		// MemorySize of filedBinlog is the actual size in memory, but we should also consider
		// the memcpy from golang to cpp side, so the expansionFactor is set to 2.
		expansionFactor := float64(2)
		memSize := getBinlogDataMemorySize(fieldBinlog)

		// Note: If MemorySize == DiskSize, it means the segment comes from Milvus 2.3,
		//   MemorySize is actually compressed DiskSize of deltalog, so we'll fallback to use
		//   deltaExpansionFactor to compromise the compression ratio.
		if memSize == getBinlogDataDiskSize(fieldBinlog) {
			expansionFactor = multiplyFactor.deltaDataExpansionFactor
		}
		segMemoryLoadingSize += uint64(float64(memSize) * expansionFactor)
	}
	deleteDur = time.Since(deleteStart)

	// PART 5: calculate size of json key stats data
	jsonStatsMmapEnable := paramtable.Get().QueryNodeCfg.MmapJSONStats.GetAsBool()
	jsonStatsStart := time.Now()
	for _, jsonKeyStats := range loadInfo.GetJsonKeyStatsLogs() {
		if jsonStatsMmapEnable {
			if !multiplyFactor.TieredEvictionEnabled {
				segDiskLoadingSize += uint64(float64(jsonKeyStats.GetMemorySize()) * multiplyFactor.jsonKeyStatsExpansionFactor)
			}
		} else {
			if !multiplyFactor.TieredEvictionEnabled {
				segMemoryLoadingSize += uint64(float64(jsonKeyStats.GetMemorySize()) * multiplyFactor.jsonKeyStatsExpansionFactor)
			}
		}
	}
	jsonStatsDur = time.Since(jsonStatsStart)

	// per struct memory size, used to keep mapping between row id and element id
	var structArrayOffsetsSize uint64
	// PART 6: calculate size of struct array offsets
	// The memory size is 4 * row_count + 4 * total_element_count
	// We cannot easily get the element count, so we estimate it by the row count * 10
	rowCount := uint64(loadInfo.GetNumOfRows())
	for range len(schema.GetStructArrayFields()) {
		structArrayOffsetsSize += 4*rowCount + 4*rowCount*10
	}

	// PART 7: calculate size of text index stats data
	// text index data is managed by the caching layer when tiered eviction is enabled,
	// so it only needs to be included when tiered eviction is disabled.
	// Text match index mmap is driven by scalar_field_enable_mmap (same as raw scalar data).
	// memory_size = sum of Tantivy index file sizes (same value as C++ ByteSize() after load),
	// so 1.0x is the baseline; textIndexExpansionFactor allows tuning if needed.
	textIndexMmapEnable := paramtable.Get().QueryNodeCfg.MmapScalarField.GetAsBool()
	textStatsStart := time.Now()
	for _, textStats := range loadInfo.GetTextStatsLogs() {
		if textIndexMmapEnable {
			if !multiplyFactor.TieredEvictionEnabled {
				segDiskLoadingSize += uint64(float64(textStats.GetMemorySize()) * multiplyFactor.textIndexExpansionFactor)
			}
		} else {
			if !multiplyFactor.TieredEvictionEnabled {
				segMemoryLoadingSize += uint64(float64(textStats.GetMemorySize()) * multiplyFactor.textIndexExpansionFactor)
			}
		}
	}
	textStatsDur = time.Since(textStatsStart)

	return &ResourceUsage{
		MemorySize:         segMemoryLoadingSize + indexMemorySize + structArrayOffsetsSize,
		DiskSize:           segDiskLoadingSize,
		MmapFieldCount:     mmapFieldCount,
		FieldGpuMemorySize: fieldGpuMemorySize,
	}, nil
}

func DoubleMemoryDataType(dataType schemapb.DataType) bool {
	return dataType == schemapb.DataType_String ||
		dataType == schemapb.DataType_VarChar ||
		dataType == schemapb.DataType_JSON
}

func DoubleMemorySystemField(fieldID int64) bool {
	return fieldID == common.TimeStampField
}

func SupportInterimIndexDataType(dataType schemapb.DataType) bool {
	return dataType == schemapb.DataType_FloatVector ||
		dataType == schemapb.DataType_SparseFloatVector ||
		dataType == schemapb.DataType_Float16Vector ||
		dataType == schemapb.DataType_BFloat16Vector
}

func (loader *segmentLoader) getFieldType(collectionID, fieldID int64) (schemapb.DataType, error) {
	collection := loader.manager.Collection.Get(collectionID)
	if collection == nil {
		return 0, merr.WrapErrCollectionNotFound(collectionID)
	}

	for _, field := range collection.Schema().GetFields() {
		if field.GetFieldID() == fieldID {
			return field.GetDataType(), nil
		}
	}

	for _, structField := range collection.Schema().GetStructArrayFields() {
		if structField.GetFieldID() == fieldID {
			return schemapb.DataType_ArrayOfStruct, nil
		}
		for _, subField := range structField.GetFields() {
			if subField.GetFieldID() == fieldID {
				return subField.GetDataType(), nil
			}
		}
	}

	return 0, merr.WrapErrFieldNotFound(fieldID)
}

func (loader *segmentLoader) LoadIndex(ctx context.Context,
	seg Segment,
	loadInfo *querypb.SegmentLoadInfo,
	version int64,
) error {
	segment, ok := seg.(*LocalSegment)
	if !ok {
		return merr.WrapErrParameterInvalid("LocalSegment", fmt.Sprintf("%T", seg))
	}
	log := log.Ctx(ctx).With(
		zap.Int64("collection", segment.Collection()),
		zap.Int64("segment", segment.ID()),
	)

	// Filter out LOADING segments only
	// use None to avoid loaded check
	infos := loader.prepare(ctx, commonpb.SegmentState_SegmentStateNone, loadInfo)
	defer loader.unregister(infos...)

	indexInfo := lo.Map(infos, func(info *querypb.SegmentLoadInfo, _ int) *querypb.SegmentLoadInfo {
		info = typeutil.Clone(info)
		// remain binlog paths whose field id is in index infos to estimate resource usage correctly
		indexFields := typeutil.NewSet(lo.Map(info.GetIndexInfos(), func(indexInfo *querypb.FieldIndexInfo, _ int) int64 { return indexInfo.GetFieldID() })...)
		var binlogPaths []*datapb.FieldBinlog
		for _, binlog := range info.GetBinlogPaths() {
			if indexFields.Contain(binlog.GetFieldID()) {
				binlogPaths = append(binlogPaths, binlog)
			}
		}
		info.BinlogPaths = binlogPaths
		info.Deltalogs = nil
		info.Statslogs = nil
		return info
	})
	requestResourceResult, err := loader.requestResource(ctx, indexInfo...)
	if err != nil {
		return err
	}
	defer loader.freeRequestResource(requestResourceResult)

	log.Info("segment loader start to load index", zap.Int("segmentNumAfterFilter", len(infos)))
	metrics.QueryNodeLoadSegmentConcurrency.WithLabelValues(paramtable.GetStringNodeID(), "LoadIndex").Inc()
	defer metrics.QueryNodeLoadSegmentConcurrency.WithLabelValues(paramtable.GetStringNodeID(), "LoadIndex").Dec()

	tr := timerecord.NewTimeRecorder("segmentLoader.LoadIndex")
	defer metrics.QueryNodeLoadIndexLatency.WithLabelValues(paramtable.GetStringNodeID()).Observe(float64(tr.ElapseSpan().Milliseconds()))
	for _, loadInfo := range infos {
		for _, info := range loadInfo.GetIndexInfos() {
			if len(info.GetIndexFilePaths()) == 0 {
				log.Warn("failed to add index for segment, index file list is empty, the segment may be too small")
				return merr.WrapErrIndexNotFound("index file list empty")
			}

			err := loader.loadFieldIndex(ctx, segment, info)
			if err != nil {
				log.Warn("failed to load index for segment", zap.Error(err))
				return err
			}
		}
		loader.notifyLoadFinish(loadInfo)
	}

	return loader.waitSegmentLoadDone(ctx, commonpb.SegmentState_SegmentStateNone, []int64{loadInfo.GetSegmentID()}, version)
}

func (loader *segmentLoader) ReopenSegments(ctx context.Context,
	loadInfos []*querypb.SegmentLoadInfo,
) error {
	// Filter out LOADING segments only
	// use None to avoid loaded check
	infos := loader.prepare(ctx, commonpb.SegmentState_SegmentStateNone, loadInfos...)
	defer loader.unregister(infos...)

	// use full resource in case of whole segment reopen
	// TODO use calculated resource from segcore after supported
	requestResourceResult, err := loader.requestResource(ctx, infos...)
	if err != nil {
		log.Warn("reopen segment request resource failed", zap.Error(err))
		return err
	}
	defer loader.freeRequestResource(requestResourceResult)

	for _, info := range infos {
		segment := loader.manager.Segment.GetSealed(info.GetSegmentID())
		if segment == nil {
			log.Warn("failed to reopen segment, segment not loaded", zap.Int64("segmentID", info.GetSegmentID()))
			continue
		}

		err := segment.Reopen(ctx, info)
		if err != nil {
			log.Warn("failed to reopen segment", zap.Int64("segmentID", info.GetSegmentID()), zap.Error(err))
			return err
		}
	}

	return nil
}

func getBinlogDataDiskSize(fieldBinlog *datapb.FieldBinlog) int64 {
	fieldSize := int64(0)
	for _, binlog := range fieldBinlog.Binlogs {
		fieldSize += binlog.GetLogSize()
	}

	return fieldSize
}

func getBinlogDataMemorySize(fieldBinlog *datapb.FieldBinlog) int64 {
	fieldSize := int64(0)
	for _, binlog := range fieldBinlog.Binlogs {
		fieldSize += binlog.GetMemorySize()
	}

	return fieldSize
}

func checkSegmentGpuMemSize(fieldGpuMemSizeList []uint64, OverloadedMemoryThresholdPercentage float32) error {
	gpuInfos, err := hardware.GetAllGPUMemoryInfo()
	if err != nil {
		if len(fieldGpuMemSizeList) == 0 {
			return nil
		}
		return err
	}
	var usedGpuMem []uint64
	var maxGpuMemSize []uint64
	for _, gpuInfo := range gpuInfos {
		usedGpuMem = append(usedGpuMem, gpuInfo.TotalMemory-gpuInfo.FreeMemory)
		maxGpuMemSize = append(maxGpuMemSize, uint64(float32(gpuInfo.TotalMemory)*OverloadedMemoryThresholdPercentage))
	}
	currentGpuMem := usedGpuMem
	for _, fieldGpuMem := range fieldGpuMemSizeList {
		var minId int = -1
		var minGpuMem uint64 = math.MaxUint64
		for i := int(0); i < len(gpuInfos); i++ {
			GpuiMem := currentGpuMem[i] + fieldGpuMem
			if GpuiMem < maxGpuMemSize[i] && GpuiMem < minGpuMem {
				minId = i
				minGpuMem = GpuiMem
			}
		}
		if minId == -1 {
			log.Warn("load segment failed, GPU OOM if loaded",
				zap.String("resourceType", "GPU"),
				zap.Uint64("gpuMemUsageBytes", fieldGpuMem),
				zap.Any("usedGpuMemBytes", usedGpuMem),
				zap.Any("maxGpuMemBytes", maxGpuMemSize),
			)
			return merr.WrapErrSegmentRequestResourceFailed("GPU")
		}
		currentGpuMem[minId] += minGpuMem
	}
	return nil
}
