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

// Package datacoord contains core functions in datacoord
package datacoord

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/hashicorp/golang-lru/v2/expirable"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/samber/lo"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/indexparams"
	"github.com/milvus-io/milvus/pkg/v3/util/lock"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var errIndexOperationIgnored = errors.New("index operation ignored")

func validateIndexStorePathVersionForFinish(buildID int64, requested, actual indexpb.IndexStorePathVersion) error {
	if metautil.IsSegmentRooted(requested) && actual != requested {
		return merr.WrapErrServiceInternalMsg("index worker changed segment-rooted path layout, buildID=%d, requested=%s, actual=%s",
			buildID, requested.String(), actual.String())
	}
	if actual > requested {
		return merr.WrapErrParameterInvalidMsg(
			"index store path version returned by worker is newer than requested, buildID=%d, requested=%s, actual=%s",
			buildID,
			requested.String(),
			actual.String(),
		)
	}
	return nil
}

type indexMeta struct {
	ctx     context.Context
	catalog metastore.DataCoordCatalog

	// fieldIndexLock protects collection index definitions and serializes the
	// resident publication of segment index records observed together by
	// getQueryViewIndexSnapshot. It must not span catalog I/O.
	// collectionLock orders index DDL and stored-size updates for the same collection.
	// indexes: collID -> indexID -> index
	fieldIndexLock  sync.RWMutex
	indexes         map[UniqueID]map[UniqueID]*model.Index
	taskCounts      indexTaskCounts
	storedIndexSize storedIndexSizeTracker
	collectionOnce  sync.Once
	collectionLock  *lock.KeyLock[UniqueID]

	// buildID2Meta records building index meta information of the segment
	segmentBuildInfo *segmentBuildInfo

	// buildID -> lock
	keyLock *lock.KeyLock[UniqueID]
	// segmentID -> indexID -> segmentIndex
	segmentIndexes *typeutil.ConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]]
}

func newIndexTaskStats(s *model.SegmentIndex) *metricsinfo.IndexTaskStats {
	return &metricsinfo.IndexTaskStats{
		IndexID:         s.IndexID,
		CollectionID:    s.CollectionID,
		SegmentID:       s.SegmentID,
		BuildID:         s.BuildID,
		IndexState:      s.IndexState.String(),
		FailReason:      s.FailReason,
		IndexSize:       s.IndexMemSize,
		IndexVersion:    s.IndexVersion,
		CreatedUTCTime:  typeutil.TimestampToString(s.CreatedUTCTime * 1000),
		FinishedUTCTime: typeutil.TimestampToString(s.FinishedUTCTime * 1000),
		NodeID:          s.NodeID,
	}
}

// buildIDShards splits the build table into maps of their own: recovery
// installs every record from parallel workers, which would otherwise all
// update the length counter of one map. shardOf gives a recovery worker whose
// count divides it whole shards of its own.
const buildIDShards = 64

type segmentBuildInfo struct {
	// buildID2Meta records the meta information of the segment
	// buildID -> segmentIndex, split by shardOf(buildID, buildIDShards)
	buildID2SegmentIndex [buildIDShards]*typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]
	// taskStats records the task stats of the segment
	taskStats *expirable.LRU[UniqueID, *metricsinfo.IndexTaskStats]
}

const taskStatsLRUCapacity = 1024

func newSegmentIndexBuildInfo() *segmentBuildInfo {
	info := &segmentBuildInfo{
		// build ID -> task stats
		taskStats: expirable.NewLRU[UniqueID, *metricsinfo.IndexTaskStats](taskStatsLRUCapacity, nil, time.Minute*30),
	}
	// build ID -> segment index
	for i := range info.buildID2SegmentIndex {
		info.buildID2SegmentIndex[i] = typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	}
	return info
}

// byBuildID returns the shard of the build table that holds buildID.
func (m *segmentBuildInfo) byBuildID(buildID UniqueID) *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex] {
	return m.buildID2SegmentIndex[shardOf(buildID, buildIDShards)]
}

func (m *segmentBuildInfo) Add(segIdx *model.SegmentIndex) {
	m.byBuildID(segIdx.BuildID).Insert(segIdx.BuildID, segIdx)
	m.recordTaskStats(segIdx)
}

// AddForRecovery inserts segment index during recovery. It skips the LRU write
// only when the LRU is already full and the index task is finished, since
// finished tasks are unlikely to be queried and the LRU write is expensive
// during bulk recovery.
func (m *segmentBuildInfo) AddForRecovery(segIdx *model.SegmentIndex) {
	m.byBuildID(segIdx.BuildID).Insert(segIdx.BuildID, segIdx)
	if m.taskStats.Len() >= taskStatsLRUCapacity && segIdx.IndexState == commonpb.IndexState_Finished {
		return
	}
	m.recordTaskStats(segIdx)
}

// insertIfAbsent inserts segIdx unless its build ID is already resident, and
// reports whether it did. It records no task stats.
func (m *segmentBuildInfo) insertIfAbsent(segIdx *model.SegmentIndex) bool {
	_, loaded := m.byBuildID(segIdx.BuildID).GetOrInsert(segIdx.BuildID, segIdx)
	return !loaded
}

func (m *segmentBuildInfo) recordTaskStats(segIdx *model.SegmentIndex) {
	m.taskStats.Add(segIdx.BuildID, newIndexTaskStats(segIdx))
}

// replayRecoveryTaskStats fills the task-stats LRU with the recovered segment
// indexes, in scan order, that AddForRecovery adds over one pass in that
// order: every one until the LRU is full, then the unfinished ones.
// unfinished holds the positions of the unfinished ones, in any order.
func (m *segmentBuildInfo) replayRecoveryTaskStats(segmentIndexes []*model.SegmentIndex, unfinished []int) {
	next := 0
	for ; next < len(segmentIndexes) && m.taskStats.Len() < taskStatsLRUCapacity; next++ {
		m.taskStats.Add(segmentIndexes[next].BuildID, newIndexTaskStats(segmentIndexes[next]))
	}
	slices.Sort(unfinished)
	for _, i := range unfinished {
		if i >= next {
			m.taskStats.Add(segmentIndexes[i].BuildID, newIndexTaskStats(segmentIndexes[i]))
		}
	}
}

func (m *segmentBuildInfo) Get(key UniqueID) (*model.SegmentIndex, bool) {
	value, exists := m.byBuildID(key).Get(key)
	return value, exists
}

func (m *segmentBuildInfo) Remove(key UniqueID) {
	m.byBuildID(key).Remove(key)
}

func (m *segmentBuildInfo) List() []*model.SegmentIndex {
	n := 0
	for _, shard := range m.buildID2SegmentIndex {
		n += shard.Len()
	}
	indexes := make([]*model.SegmentIndex, 0, n)
	m.Range(func(segmentIndex *model.SegmentIndex) bool {
		indexes = append(indexes, segmentIndex)
		return true
	})
	return indexes
}

// Range visits resident SegmentIndex entries without first materializing a
// full slice. The callback observes sync.Map traversal semantics and may miss
// concurrent updates; GC revalidates every candidate by buildID before acting.
func (m *segmentBuildInfo) Range(f func(*model.SegmentIndex) bool) {
	next := true
	for _, shard := range m.buildID2SegmentIndex {
		shard.Range(func(_ UniqueID, segmentIndex *model.SegmentIndex) bool {
			next = f(segmentIndex)
			return next
		})
		if !next {
			return
		}
	}
}

func (m *segmentBuildInfo) GetTaskStats() []*metricsinfo.IndexTaskStats {
	return m.taskStats.Values()
}

// NewMeta creates meta from provided `kv.TxnKV`
func newIndexMeta(ctx context.Context, catalog metastore.DataCoordCatalog) (*indexMeta, error) {
	return recoverIndexMeta(ctx, catalog, nil)
}

// recoverIndexMeta creates the index meta like newIndexMeta. collectionIDs, the
// collections recovery knows, split the scans of a catalog that reads key
// ranges into parallel ranges; the indexes of other collections are read all
// the same.
func recoverIndexMeta(ctx context.Context, catalog metastore.DataCoordCatalog, collectionIDs []int64) (*indexMeta, error) {
	mt := &indexMeta{
		ctx:              ctx,
		catalog:          catalog,
		indexes:          make(map[UniqueID]map[UniqueID]*model.Index),
		keyLock:          lock.NewKeyLock[UniqueID](),
		segmentBuildInfo: newSegmentIndexBuildInfo(),
		segmentIndexes:   typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
	}
	err := mt.reloadFromKV(collectionIDs)
	if err != nil {
		return nil, err
	}
	return mt, nil
}

const (
	segmentIndexCacheRecoveryProgressLogInterval = 100000
)

type allSegmentIndexRecoveryCatalog interface {
	ListAllSegmentIndexes(ctx context.Context) ([]*model.SegmentIndex, error)
}

// rangedIndexRecoveryCatalog reads the index prefixes as parallel key ranges
// split at the given collections, returning what ListIndexes and
// ListAllSegmentIndexes return.
type rangedIndexRecoveryCatalog interface {
	ListIndexesForRecovery(ctx context.Context, collectionIDs []int64) ([]*model.Index, error)
	ListAllSegmentIndexesForRecovery(ctx context.Context, collectionIDs []int64) ([]*model.SegmentIndex, error)
}

func (m *indexMeta) listIndexes(collectionIDs []int64) ([]*model.Index, error) {
	if catalog, ok := m.catalog.(rangedIndexRecoveryCatalog); ok {
		return catalog.ListIndexesForRecovery(m.ctx, collectionIDs)
	}
	return m.catalog.ListIndexes(m.ctx)
}

func (m *indexMeta) listAllSegmentIndexes(collectionIDs []int64) ([]*model.SegmentIndex, error) {
	if catalog, ok := m.catalog.(rangedIndexRecoveryCatalog); ok {
		return catalog.ListAllSegmentIndexesForRecovery(m.ctx, collectionIDs)
	}
	if catalog, ok := m.catalog.(allSegmentIndexRecoveryCatalog); ok {
		return catalog.ListAllSegmentIndexes(m.ctx)
	}
	// Compatibility path for tests and catalog implementations that predate
	// the recovery-only global scan method.
	return m.catalog.ListSegmentIndexes(m.ctx, 0)
}

// reloadFromKV loads meta from KV storage
func (m *indexMeta) reloadFromKV(collectionIDs []int64) error {
	record := timerecord.NewTimeRecorder("indexMeta-reloadFromKV")

	var segmentIndexes []*model.SegmentIndex
	var recoveredIndexSizes map[UniqueID]map[UniqueID]uint64
	g, _ := errgroup.WithContext(m.ctx)
	g.Go(func() error {
		fieldIndexScanStart := time.Now()
		mlog.Info(m.ctx, "indexMeta field index catalog scan started")
		fieldIndexes, err := m.listIndexes(collectionIDs)
		if err != nil {
			mlog.Error(m.ctx, "indexMeta reloadFromKV load field indexes fail",
				mlog.Duration("duration", time.Since(fieldIndexScanStart)),
				mlog.Err(err))
			return err
		}
		mlog.Info(m.ctx, "indexMeta field index catalog scan done",
			mlog.Int("numFieldIndexes", len(fieldIndexes)),
			mlog.Duration("duration", time.Since(fieldIndexScanStart)))
		fieldIndexCacheStart := time.Now()
		for _, fieldIndex := range fieldIndexes {
			m.updateCollectionIndex(fieldIndex)
		}
		mlog.Info(m.ctx, "indexMeta field index cache rebuild done",
			mlog.Int("numFieldIndexes", len(fieldIndexes)),
			mlog.Duration("duration", time.Since(fieldIndexCacheStart)))
		return nil
	})
	g.Go(func() error {
		scanStart := time.Now()
		mlog.Info(m.ctx, "indexMeta global segment index catalog scan started")
		var err error
		segmentIndexes, err = m.listAllSegmentIndexes(collectionIDs)
		if err != nil {
			mlog.Warn(m.ctx, "indexMeta global segment index catalog scan failed", mlog.Err(err))
			return err
		}
		mlog.Info(m.ctx, "indexMeta global segment index catalog scan done",
			mlog.Int("numSegmentIndexes", len(segmentIndexes)),
			mlog.Duration("duration", time.Since(scanStart)))

		cacheBuildStart := time.Now()
		recoveredIndexSizes = m.rebuildSegmentIndexCache(segmentIndexes)
		mlog.Info(m.ctx, "indexMeta segment index cache rebuild done",
			mlog.Int("numSegmentIndexes", len(segmentIndexes)),
			mlog.Duration("duration", time.Since(cacheBuildStart)))
		return nil
	})
	if err := g.Wait(); err != nil {
		return err
	}

	// Rebuild the active index-size aggregate after both independent walkers
	// have completed. DropIndex then updates the gauge without scanning all
	// segment indexes.
	m.storedIndexSize.recover(m.indexes, recoveredIndexSizes)

	// Update the index file-count histogram asynchronously. The stored-size
	// gauge is initialized synchronously above so DDL callbacks can safely use
	// the aggregate as soon as recovery returns.
	go func() {
		for _, segIdx := range segmentIndexes {
			metrics.FlushedSegmentFileNum.WithLabelValues(metrics.IndexFileLabel).Observe(float64(len(segIdx.IndexFileKeys)))
		}
	}()

	mlog.Info(m.ctx, "indexMeta reloadFromKV done", mlog.Duration("duration", record.ElapseSpan()))
	return nil
}

// rebuildSegmentIndexCache installs the recovered segment indexes, in scan
// order, the way one pass over them in that order does, and returns their
// stored sizes by collection and index. It runs on recovery workers: sharded
// by segment ID for the per-segment index maps and the sizes, then by build ID
// for the build table, so the records of one key are installed in scan order,
// then by index for the task counts. The task-stats LRU, whose content depends
// on the order of every record, is then filled with the records such a pass
// adds to it, in that order.
func (m *indexMeta) rebuildSegmentIndexCache(segmentIndexes []*model.SegmentIndex) map[UniqueID]map[UniqueID]uint64 {
	start := time.Now()
	workers := recoveryWorkers()
	memSizeMultiplier := paramtable.Get().DataCoordCfg.IndexMemSizeEstimateMultiplier.GetAsUint64()

	sizes := make([]map[UniqueID]map[UniqueID]uint64, workers)
	for worker := range sizes {
		sizes[worker] = make(map[UniqueID]map[UniqueID]uint64)
	}
	_ = forEachInShards(len(segmentIndexes), workers, func(i int) int64 { return segmentIndexes[i].SegmentID }, func(worker, i int) error {
		segIdx := segmentIndexes[i]
		indexSizes, ok := sizes[worker][segIdx.CollectionID]
		if !ok {
			indexSizes = make(map[UniqueID]uint64)
			sizes[worker][segIdx.CollectionID] = indexSizes
		}
		indexSizes[segIdx.IndexID] += segIdx.IndexSerializedSize

		if segIdx.IndexMemSize == 0 {
			segIdx.IndexMemSize = segIdx.IndexSerializedSize * memSizeMultiplier
		}
		indexes, ok := m.segmentIndexes.Get(segIdx.SegmentID)
		if !ok {
			indexes = typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
			m.segmentIndexes.Insert(segIdx.SegmentID, indexes)
		}
		indexes.Insert(segIdx.IndexID, segIdx)
		return nil
	})

	unfinished := make([][]int, workers)
	rebuilt := newRecoveryProgress(workers, segmentIndexCacheRecoveryProgressLogInterval)
	_ = forEachInShards(len(segmentIndexes), workers, func(i int) int64 { return segmentIndexes[i].BuildID }, func(worker, i int) error {
		segIdx := segmentIndexes[i]
		m.segmentBuildInfo.byBuildID(segIdx.BuildID).Insert(segIdx.BuildID, segIdx)
		if segIdx.IndexState != commonpb.IndexState_Finished {
			unfinished[worker] = append(unfinished[worker], i)
		}
		if completed, report := rebuilt.add(worker); report && completed < len(segmentIndexes) {
			mlog.Info(m.ctx, "indexMeta segment index cache rebuild progress",
				mlog.Int("completedSegmentIndexes", completed),
				mlog.Int("totalSegmentIndexes", len(segmentIndexes)),
				mlog.Duration("duration", time.Since(start)))
		}
		return nil
	})

	// Replacing the task of a build by each of its records in turn
	// (replaceTask) leaves the count of its last record, the one the build
	// table holds: count those, sharded by index so the workers count
	// disjoint indexes.
	counts := make([]map[[2]int64]indexTaskStateCounts, workers)
	for worker := range counts {
		counts[worker] = make(map[[2]int64]indexTaskStateCounts)
	}
	_ = forEachInShards(len(segmentIndexes), workers, func(i int) int64 {
		return segmentIndexes[i].CollectionID*1000003 + segmentIndexes[i].IndexID
	}, func(worker, i int) error {
		segIdx := segmentIndexes[i]
		if last, _ := m.segmentBuildInfo.Get(segIdx.BuildID); last != segIdx {
			return nil
		}
		if key, slot, ok := taskCountSlot(segIdx); ok {
			taskCounts := counts[worker][key]
			taskCounts[slot]++
			counts[worker][key] = taskCounts
		}
		return nil
	})
	for _, workerCounts := range counts {
		m.taskCounts.addTasks(workerCounts)
	}
	m.segmentBuildInfo.replayRecoveryTaskStats(segmentIndexes, slices.Concat(unfinished...))

	recoveredIndexSizes := sizes[0]
	for _, workerSizes := range sizes[1:] {
		for collectionID, indexSizes := range workerSizes {
			merged, ok := recoveredIndexSizes[collectionID]
			if !ok {
				recoveredIndexSizes[collectionID] = indexSizes
				continue
			}
			for indexID, size := range indexSizes {
				merged[indexID] += size
			}
		}
	}
	return recoveredIndexSizes
}

func (m *indexMeta) updateCollectionIndex(index *model.Index) {
	m.taskCounts.replaceIndex(m.indexes[index.CollectionID][index.IndexID], index)
	if _, ok := m.indexes[index.CollectionID]; !ok {
		m.indexes[index.CollectionID] = make(map[UniqueID]*model.Index)
	}
	m.indexes[index.CollectionID][index.IndexID] = index
}

func (m *indexMeta) getCollectionLock() *lock.KeyLock[UniqueID] {
	m.collectionOnce.Do(func() {
		if m.collectionLock == nil {
			m.collectionLock = lock.NewKeyLock[UniqueID]()
		}
	})
	return m.collectionLock
}

// lockCollections serializes index DDL for the same collection while allowing
// unrelated collections to persist metadata concurrently. IDs are sorted so
// AlterIndex can safely lock a batch spanning more than one collection.
func (m *indexMeta) lockCollections(collectionIDs ...UniqueID) func() {
	if len(collectionIDs) == 0 {
		return func() {}
	}

	collectionLock := m.getCollectionLock()
	if len(collectionIDs) == 1 {
		collectionID := collectionIDs[0]
		collectionLock.Lock(collectionID)
		return func() { collectionLock.Unlock(collectionID) }
	}

	uniqueIDs := make(map[UniqueID]struct{}, len(collectionIDs))
	ids := make([]UniqueID, 0, len(collectionIDs))
	for _, collectionID := range collectionIDs {
		if _, ok := uniqueIDs[collectionID]; ok {
			continue
		}
		uniqueIDs[collectionID] = struct{}{}
		ids = append(ids, collectionID)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	for _, collectionID := range ids {
		collectionLock.Lock(collectionID)
	}
	return func() {
		for i := len(ids) - 1; i >= 0; i-- {
			collectionLock.Unlock(ids[i])
		}
	}
}

func (m *indexMeta) updateSegmentIndex(segIdx *model.SegmentIndex) {
	old, _ := m.segmentBuildInfo.Get(segIdx.BuildID)
	m.taskCounts.replaceTask(old, segIdx)
	m.putSegmentIndex(segIdx)
	m.segmentBuildInfo.Add(segIdx)
}

// putSegmentIndex makes segIdx its segment's record for its index ID.
func (m *indexMeta) putSegmentIndex(segIdx *model.SegmentIndex) {
	indexes, ok := m.segmentIndexes.Get(segIdx.SegmentID)
	if ok {
		indexes.Insert(segIdx.IndexID, segIdx)
	} else {
		indexes := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
		indexes.Insert(segIdx.IndexID, segIdx)
		m.segmentIndexes.Insert(segIdx.SegmentID, indexes)
	}
}

// manifestIndexRecovery installs the SegmentIndex records startup recovers from
// segment manifests, millions of them on a large cluster. install updates the
// resident maps exactly as updateSegmentIndex does. The task counts, the
// task-stats LRU and the stored-size aggregate, which updateSegmentIndex and
// updateStoredIndexSizeMetric would update per record, are updated once by
// finish into the state the per-record updates leave:
//   - The task counts receive each index's summed records by state, as
//     adjustTask adds them one at a time.
//   - The stored-size tracker and gauge receive each collection's summed record
//     sizes. A record only adds its size to an active index's total, and index
//     DDL cannot run before the recovered meta is published.
//   - The task-stats LRU receives the last taskStatsLRUCapacity records in
//     install order. The fresh indexMeta's LRU holds only resident build IDs,
//     which install skips, so every installed build ID is new to the LRU and
//     adding them all would have evicted each earlier one.
//
// Recoveries installing the records of disjoint segments on parallel
// goroutines merge into one before it finishes.
type manifestIndexRecovery struct {
	meta *indexMeta
	// sizes sums installed sizes by collectionID and indexID.
	sizes map[UniqueID]map[UniqueID]uint64
	// counts sums installed records by index and state slot (see
	// taskCountSlot).
	counts map[[2]int64]indexTaskStateCounts
	// latest holds the last records of the latestCount it was given.
	latest      [taskStatsLRUCapacity]*model.SegmentIndex
	latestCount int
	installed   int
	finished    bool
}

func (m *indexMeta) newManifestIndexRecovery() *manifestIndexRecovery {
	return &manifestIndexRecovery{meta: m, sizes: make(map[UniqueID]map[UniqueID]uint64), counts: make(map[[2]int64]indexTaskStateCounts)}
}

// install adds segIdx unless a catalog row or an earlier manifest already holds
// its build ID, which then stays authoritative.
func (r *manifestIndexRecovery) install(segIdx *model.SegmentIndex) {
	m := r.meta
	if !m.segmentBuildInfo.insertIfAbsent(segIdx) {
		return
	}
	m.putSegmentIndex(segIdx)
	if key, slot, ok := taskCountSlot(segIdx); ok {
		counts := r.counts[key]
		counts[slot]++
		r.counts[key] = counts
	}
	sizes, ok := r.sizes[segIdx.CollectionID]
	if !ok {
		sizes = make(map[UniqueID]uint64)
		r.sizes[segIdx.CollectionID] = sizes
	}
	sizes[segIdx.IndexID] += segIdx.IndexSerializedSize
	r.remember(segIdx)
	r.installed++
}

func (r *manifestIndexRecovery) remember(segIdx *model.SegmentIndex) {
	r.latest[r.latestCount%taskStatsLRUCapacity] = segIdx
	r.latestCount++
}

// merge adds what other installed, after what r installed, to r.
func (r *manifestIndexRecovery) merge(other *manifestIndexRecovery) {
	for collectionID, sizes := range other.sizes {
		merged, ok := r.sizes[collectionID]
		if !ok {
			r.sizes[collectionID] = sizes
			continue
		}
		for indexID, size := range sizes {
			merged[indexID] += size
		}
	}
	for key, counts := range other.counts {
		merged := r.counts[key]
		for slot := range merged {
			merged[slot] += counts[slot]
		}
		r.counts[key] = merged
	}
	for i := max(0, other.latestCount-taskStatsLRUCapacity); i < other.latestCount; i++ {
		r.remember(other.latest[i%taskStatsLRUCapacity])
	}
	r.installed += other.installed
}

// finish applies the aggregates of the records installed so far. Later calls
// do nothing.
func (r *manifestIndexRecovery) finish() {
	if r.finished {
		return
	}
	r.finished = true
	m := r.meta
	m.taskCounts.addTasks(r.counts)
	collectionLock := m.getCollectionLock()
	for collectionID, sizes := range r.sizes {
		// updateStoredIndexSizeMetric's locking, once per collection.
		collectionLock.RLock(collectionID)
		m.fieldIndexLock.RLock()
		for indexID := range sizes {
			if !m.storedIndexSizeCounted(collectionID, indexID) {
				delete(sizes, indexID)
			}
		}
		m.fieldIndexLock.RUnlock()
		if len(sizes) > 0 {
			m.storedIndexSize.add(collectionID, sizes)
		}
		collectionLock.RUnlock(collectionID)
	}
	for i := max(0, r.latestCount-taskStatsLRUCapacity); i < r.latestCount; i++ {
		m.segmentBuildInfo.recordTaskStats(r.latest[i%taskStatsLRUCapacity])
	}
}

// writeSegmentIndexToManifest selects manifest publication for new StorageV3
// tasks. Existing index builds follow their persisted path version. SegmentIndex
// remains the durable task record while a build is active or failed. Only a successful
// publication retires its Finished etcd row.
func writeSegmentIndexToManifest() bool {
	return paramtable.Get().DataCoordCfg.WriteSegmentIndexToManifest.GetAsBool()
}

func (m *indexMeta) alterSegmentIndexes(segIdxes []*model.SegmentIndex) error {
	err := m.catalog.AlterSegmentIndexes(m.ctx, segIdxes)
	if err != nil {
		mlog.Error(context.TODO(), "failed to alter segments index in meta store", mlog.Int("segment indexes num", len(segIdxes)),
			mlog.Err(err))
		return err
	}
	m.fieldIndexLock.Lock()
	defer m.fieldIndexLock.Unlock()
	for _, segIdx := range segIdxes {
		m.updateSegmentIndex(segIdx)
	}
	return nil
}

func (m *indexMeta) updateIndexMeta(index *model.Index, updateFunc func(clonedIndex *model.Index) error) error {
	return updateFunc(model.CloneIndex(index))
}

func (m *indexMeta) updateSegIndexMeta(segIdx *model.SegmentIndex, updateFunc func(clonedSegIdx *model.SegmentIndex) error) error {
	return updateFunc(model.CloneSegmentIndex(segIdx))
}

// buildFinishedSegmentIndex projects a worker completion result onto a fresh
// copy of the task's metadata without persisting it or touching the in-memory
// record. Callers that must compose this update with another metadata write
// stage the returned value in the same catalog transaction; the second return
// value is the previously recorded serialized size, needed for the stored-size
// gauge delta.
//
// Precondition: the caller holds keyLock for taskInfo.BuildID.
func (m *indexMeta) buildFinishedSegmentIndex(segIdx *model.SegmentIndex, taskInfo *workerpb.IndexTaskInfo) (*model.SegmentIndex, uint64, error) {
	actualPathVersion := taskInfo.GetIndexStorePathVersion()
	// Failed/empty results may be synthesized by the scheduler without a
	// worker payload. They carry no artifact layout; retain the dispatched
	// version so terminal tasks can retire and GC can reclaim their uploads.
	if taskInfo.GetState() != commonpb.IndexState_Finished || len(taskInfo.GetIndexFileKeys()) == 0 {
		actualPathVersion = segIdx.IndexStorePathVersion
	}
	if err := validateIndexStorePathVersionForFinish(taskInfo.GetBuildID(), segIdx.IndexStorePathVersion, actualPathVersion); err != nil {
		mlog.Warn(m.ctx, "invalid index store path version returned by worker",
			mlog.Int64("buildID", taskInfo.GetBuildID()),
			mlog.String("requested", segIdx.IndexStorePathVersion.String()),
			mlog.String("actual", actualPathVersion.String()),
			mlog.Err(err))
		return nil, 0, err
	}
	if actualPathVersion < segIdx.IndexStorePathVersion {
		mlog.Info(m.ctx, "worker downgraded index store path version",
			mlog.Int64("buildID", taskInfo.GetBuildID()),
			mlog.String("requested", segIdx.IndexStorePathVersion.String()),
			mlog.String("actual", actualPathVersion.String()))
	}

	updated := model.CloneSegmentIndex(segIdx)
	updated.IndexState = taskInfo.GetState()
	updated.IndexFileKeys = common.CloneStringList(taskInfo.GetIndexFileKeys())
	updated.FailReason = taskInfo.GetFailReason()
	updated.IndexSerializedSize = taskInfo.GetSerializedSize()
	updated.IndexMemSize = taskInfo.GetMemSize()
	updated.CurrentIndexVersion = taskInfo.GetCurrentIndexVersion()
	updated.FinishedUTCTime = uint64(time.Now().Unix())
	updated.CurrentScalarIndexVersion = taskInfo.GetCurrentScalarIndexVersion()
	updated.IndexStorePathVersion = actualPathVersion
	return updated, segIdx.IndexSerializedSize, nil
}

// recordFinishedTask updates process-local observability only. The caller must
// have already persisted updated and installed it in the in-memory metadata.
func (m *indexMeta) recordFinishedTask(old, updated *model.SegmentIndex, oldSize uint64, taskInfo *workerpb.IndexTaskInfo) {
	mlog.Info(m.ctx, "finish index task success", mlog.Int64("buildID", taskInfo.GetBuildID()),
		mlog.String("state", taskInfo.GetState().String()), mlog.String("fail reason", taskInfo.GetFailReason()),
		mlog.Int32("current_index_version", taskInfo.GetCurrentIndexVersion()))

	// The collection read lock inside this update serializes it with index DDL.
	m.updateStoredIndexSizeMetric(m.ctx, old.CollectionID, old.IndexID, oldSize, updated.IndexSerializedSize)

	metrics.FlushedSegmentFileNum.WithLabelValues(metrics.IndexFileLabel).Observe(float64(len(taskInfo.GetIndexFileKeys())))
}

func (m *indexMeta) indexTaskCountsSnapshot() indexTaskStateCounts {
	return m.taskCounts.snapshot()
}

func (m *indexMeta) updateIndexTasksMetrics() {
	counts := m.indexTaskCountsSnapshot()
	jobType := indexpb.JobType_JobTypeIndexJob.String()
	for state, count := range counts {
		metrics.IndexStatsTaskNum.WithLabelValues(jobType, indexTaskMetricStates[state].String()).Set(float64(count))
	}
}

func checkIdenticalJSON(index *model.Index, req *indexpb.CreateIndexRequest) bool {
	// Skip error handling since json path existence is guaranteed in CreateIndex
	jsonPath1, _ := getIndexParam(index.IndexParams, common.JSONPathKey)
	jsonPath2, _ := getIndexParam(req.GetIndexParams(), common.JSONPathKey)

	if jsonPath1 != jsonPath2 {
		return false
	}
	castType1, _ := getIndexParam(index.IndexParams, common.JSONCastTypeKey)
	castType2, _ := getIndexParam(req.GetIndexParams(), common.JSONCastTypeKey)
	return castType1 == castType2
}

func checkParams(fieldIndex *model.Index, req *indexpb.CreateIndexRequest) bool {
	metaTypeParams := DeleteParams(fieldIndex.TypeParams, []string{common.MmapEnabledKey, common.WarmupKey})
	reqTypeParams := DeleteParams(req.TypeParams, []string{common.MmapEnabledKey, common.WarmupKey})
	if len(metaTypeParams) != len(reqTypeParams) {
		return false
	}
	notEq := false
	for _, param1 := range metaTypeParams {
		exist := false
		for _, param2 := range reqTypeParams {
			if param2.Key == param1.Key && param2.Value == param1.Value {
				exist = true
			}
		}
		if !exist {
			notEq = true
			break
		}
	}
	if notEq {
		return false
	}

	useAutoIndex := false
	userIndexParamsWithoutConfigableKey := make([]*commonpb.KeyValuePair, 0)
	for _, param := range fieldIndex.UserIndexParams {
		if indexparams.IsConfigableIndexParam(param.Key) {
			continue
		}
		if param.Key == common.IndexTypeKey && param.Value == common.AutoIndexName {
			useAutoIndex = true
		}
		userIndexParamsWithoutConfigableKey = append(userIndexParamsWithoutConfigableKey, param)
	}

	if len(userIndexParamsWithoutConfigableKey) != len(req.GetUserIndexParams()) {
		return false
	}
	for _, param1 := range userIndexParamsWithoutConfigableKey {
		exist := false
		for i, param2 := range req.GetUserIndexParams() {
			if param2.Key == param1.Key && param2.Value == param1.Value {
				exist = true
				break
			} else if param1.Key == common.MetricTypeKey && param2.Key == param1.Key && useAutoIndex && !req.GetUserAutoindexMetricTypeSpecified() {
				// when users use autoindex, metric type is the only thing they can specify
				// if they do not specify metric type, will use autoindex default metric type
				// when autoindex default config upgraded, remain the old metric type at the very first time for compatibility
				// warn! replace request metric type
				mlog.Warn(context.TODO(), "user not specify autoindex metric type, autoindex config has changed, use old metric for compatibility",
					mlog.String("old metric type", param1.Value), mlog.String("new metric type", param2.Value))
				req.GetUserIndexParams()[i].Value = param1.Value
				for j, param := range req.GetIndexParams() {
					if param.Key == common.MetricTypeKey {
						req.GetIndexParams()[j].Value = param1.Value
						break
					}
				}
				exist = true
				break
			}
		}
		if !exist {
			notEq = true
			break
		}
	}
	// Check whether new index type match old, if not, only
	// allow autoindex config changed when upgraded to new config
	// using store meta config to rewrite new config
	if !notEq && req.GetIsAutoIndex() && useAutoIndex {
		for _, param1 := range fieldIndex.IndexParams {
			if param1.Key == common.IndexTypeKey &&
				indexparamcheck.IsScalarIndexType(param1.Value) {
				for _, param2 := range req.GetIndexParams() {
					if param1.Key == param2.Key && param1.Value != param2.Value {
						req.IndexParams = make([]*commonpb.KeyValuePair, len(fieldIndex.IndexParams))
						copy(req.IndexParams, fieldIndex.IndexParams)
						break
					}
				}
			}
		}
	}
	mlog.Info(context.TODO(), "final request", mlog.Any("create index request", req.String()))
	return !notEq
}

// CanCreateIndex currently is used in Unittest
func (m *indexMeta) CanCreateIndex(req *indexpb.CreateIndexRequest, isJSON bool) (UniqueID, error) {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()
	indexID, err := m.canCreateIndex(req, isJSON)
	if err != nil {
		return 0, err
	}

	return indexID, nil
}

func (m *indexMeta) canCreateIndex(req *indexpb.CreateIndexRequest, isJSON bool) (UniqueID, error) {
	indexes, ok := m.indexes[req.CollectionID]
	if !ok {
		return 0, nil
	}
	for _, index := range indexes {
		if index.IsDeleted {
			continue
		}
		if req.IndexName == index.IndexName {
			if req.FieldID == index.FieldID && checkParams(index, req) &&
				/*only check json params when it is json index*/ (!isJSON || checkIdenticalJSON(index, req)) {
				return index.IndexID, errIndexOperationIgnored
			}
			errMsg := "at most one distinct index is allowed per field"
			mlog.Warn(context.TODO(), errMsg,
				mlog.String("source index", fmt.Sprintf("{index_name: %s, field_id: %d, index_params: %v, user_params: %v, type_params: %v}",
					index.IndexName, index.FieldID, index.IndexParams, index.UserIndexParams, index.TypeParams)),
				mlog.String("current index", fmt.Sprintf("{index_name: %s, field_id: %d, index_params: %v, user_params: %v, type_params: %v}",
					req.GetIndexName(), req.GetFieldID(), req.GetIndexParams(), req.GetUserIndexParams(), req.GetTypeParams())))
			return 0, merr.WrapErrParameterInvalidMsg("%s", errMsg)
		}
		if req.FieldID == index.FieldID {
			if isJSON {
				// Skip error handling since json path existence is guaranteed in CreateIndex
				jsonPath1, _ := getIndexParam(index.IndexParams, common.JSONPathKey)
				jsonPath2, _ := getIndexParam(req.GetIndexParams(), common.JSONPathKey)

				if jsonPath1 != jsonPath2 {
					// if json path is not same, create index is allowed
					continue
				}
			}
			// creating multiple indexes on same field is not supported
			errMsg := "CreateIndex failed: creating multiple indexes on same field is not supported"
			mlog.Warn(context.TODO(), errMsg)
			// Client-caused conflict, same family as the "one distinct index per
			// field" case above: return an input-class error rather than
			// ServiceInternal (code 5, "never return out of Milvus") so the
			// caller is not blamed with a system error / counted as a system-caused failure.
			return 0, merr.WrapErrParameterInvalidMsg("%s", errMsg)
		}
	}
	return 0, nil
}

// HasSameReq determine whether there are same indexing tasks.
func (m *indexMeta) HasSameReq(req *indexpb.CreateIndexRequest) (bool, UniqueID) {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	for _, fieldIndex := range m.indexes[req.CollectionID] {
		if fieldIndex.IsDeleted {
			continue
		}
		if fieldIndex.FieldID != req.FieldID || fieldIndex.IndexName != req.IndexName {
			continue
		}
		if !checkParams(fieldIndex, req) {
			continue
		}
		mlog.Debug(context.TODO(), "has same index", mlog.Int64("collectionID", req.CollectionID),
			mlog.Int64("fieldID", req.FieldID), mlog.String("indexName", req.IndexName),
			mlog.Int64("indexID", fieldIndex.IndexID))
		return true, fieldIndex.IndexID
	}

	return false, 0
}

func (m *indexMeta) CreateIndex(ctx context.Context, index *model.Index) error {
	unlockCollections := m.lockCollections(index.CollectionID)
	defer unlockCollections()

	mlog.Info(ctx, "meta update: CreateIndex", mlog.Int64("collectionID", index.CollectionID),
		mlog.Int64("fieldID", index.FieldID), mlog.Int64("indexID", index.IndexID), mlog.String("indexName", index.IndexName))

	if err := m.catalog.CreateIndex(ctx, index); err != nil {
		mlog.Error(ctx, "meta update: CreateIndex save meta fail", mlog.Int64("collectionID", index.CollectionID),
			mlog.Int64("fieldID", index.FieldID), mlog.Int64("indexID", index.IndexID),
			mlog.String("indexName", index.IndexName), mlog.Err(err))
		return err
	}

	m.fieldIndexLock.Lock()
	m.updateCollectionIndex(index)
	m.fieldIndexLock.Unlock()
	m.storedIndexSize.activate(index.CollectionID, index.IndexID)
	mlog.Info(ctx, "meta update: CreateIndex success", mlog.Int64("collectionID", index.CollectionID),
		mlog.Int64("fieldID", index.FieldID), mlog.Int64("indexID", index.IndexID), mlog.String("indexName", index.IndexName))
	return nil
}

func (m *indexMeta) AlterIndex(ctx context.Context, indexes ...*model.Index) error {
	collectionIDs := make([]UniqueID, 0, len(indexes))
	for _, index := range indexes {
		collectionIDs = append(collectionIDs, index.CollectionID)
	}
	unlockCollections := m.lockCollections(collectionIDs...)
	defer unlockCollections()

	err := m.catalog.AlterIndexes(ctx, indexes)
	if err != nil {
		return err
	}

	m.fieldIndexLock.Lock()
	for _, index := range indexes {
		m.updateCollectionIndex(index)
	}
	m.fieldIndexLock.Unlock()

	return nil
}

// updateStoredIndexSizeMetric holds a collection read lock across the active
// index check and aggregate update. Index DDL takes the corresponding write
// lock, so catalog I/O for another collection never blocks this path.
func (m *indexMeta) updateStoredIndexSizeMetric(ctx context.Context, collID, indexID UniqueID, oldSize, newSize uint64) {
	collectionLock := m.getCollectionLock()
	collectionLock.RLock(collID)
	defer collectionLock.RUnlock(collID)

	m.fieldIndexLock.RLock()
	isActive := m.storedIndexSizeCounted(collID, indexID)
	m.fieldIndexLock.RUnlock()
	if isActive {
		m.storedIndexSize.update(ctx, collID, indexID, oldSize, newSize)
	}
}

// storedIndexSizeCounted reports whether an index's records count toward the
// stored-size gauge: only those of a live index do. The caller holds
// fieldIndexLock.
func (m *indexMeta) storedIndexSizeCounted(collID, indexID UniqueID) bool {
	idx, ok := m.indexes[collID][indexID]
	return ok && !idx.IsDeleted
}

// AddSegmentIndex adds the index meta corresponding the indexBuildID to meta table.
func (m *indexMeta) AddSegmentIndex(ctx context.Context, segIndex *model.SegmentIndex) error {
	return m.addSegmentIndex(ctx, segIndex, true)
}

// AddSegmentIndexFromManifest installs a completed index already made durable
// by a copied segment's published manifest. Copy targets use fresh build IDs,
// so there is no prior etcd task row to retire.
func (m *indexMeta) AddSegmentIndexFromManifest(ctx context.Context, segIndex *model.SegmentIndex) error {
	if segIndex == nil {
		return merr.WrapErrServiceInternalMsg("manifest-resident segment index is nil")
	}
	if segIndex.IndexState != commonpb.IndexState_Finished || len(segIndex.IndexFileKeys) == 0 {
		return merr.WrapErrServiceInternalMsg(
			"manifest-resident segment index must be finished and carry artifact files, buildID=%d", segIndex.BuildID)
	}
	return m.addSegmentIndex(ctx, segIndex, false)
}

func (m *indexMeta) addSegmentIndex(ctx context.Context, segIndex *model.SegmentIndex, persist bool) error {
	buildID := segIndex.BuildID

	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	mlog.Info(ctx, "meta update: adding segment index", mlog.Int64("collectionID", segIndex.CollectionID),
		mlog.Int64("segmentID", segIndex.SegmentID), mlog.Int64("indexID", segIndex.IndexID),
		mlog.Int64("buildID", buildID), mlog.String("indexType", segIndex.IndexType))

	// Only override to Unissued if the caller hasn't set a specific state.
	// Copy segment sets Finished because index files are already copied.
	if segIndex.IndexState == commonpb.IndexState_IndexStateNone {
		segIndex.IndexState = commonpb.IndexState_Unissued
	}
	if persist {
		if err := m.catalog.CreateSegmentIndex(ctx, segIndex); err != nil {
			mlog.Warn(ctx, "meta update: adding segment index failed",
				mlog.Int64("segmentID", segIndex.SegmentID), mlog.Int64("indexID", segIndex.IndexID),
				mlog.Int64("buildID", segIndex.BuildID), mlog.String("indexType", segIndex.IndexType), mlog.Err(err))
			return err
		}
	}

	// Resident publication must be serialized with collection-index mutation
	// and QueryView index snapshots. The size tracker has its own collection
	// ordering and therefore updates after releasing this lock.
	m.fieldIndexLock.Lock()
	m.updateSegmentIndex(segIndex)
	m.fieldIndexLock.Unlock()

	if segIndex.IndexState == commonpb.IndexState_Finished {
		m.updateStoredIndexSizeMetric(ctx, segIndex.CollectionID, segIndex.IndexID,
			0, segIndex.IndexSerializedSize)
	}

	mlog.Info(ctx, "meta update: adding segment index success", mlog.Int64("collectionID", segIndex.CollectionID),
		mlog.Int64("segmentID", segIndex.SegmentID), mlog.Int64("indexID", segIndex.IndexID),
		mlog.Int64("buildID", buildID), mlog.String("indexType", segIndex.IndexType))
	return nil
}

func (m *indexMeta) GetIndexIDByName(collID int64, indexName string) map[int64]uint64 {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	indexID2CreateTs := make(map[int64]uint64)

	fieldIndexes, ok := m.indexes[collID]
	if !ok {
		return indexID2CreateTs
	}

	for _, index := range fieldIndexes {
		if !index.IsDeleted && (indexName == "" || index.IndexName == indexName) {
			indexID2CreateTs[index.IndexID] = index.CreateTime
		}
	}
	return indexID2CreateTs
}

func (m *indexMeta) GetSegmentIndexState(collID, segmentID UniqueID, indexID UniqueID) *indexpb.SegmentIndexState {
	state := &indexpb.SegmentIndexState{
		SegmentID:  segmentID,
		State:      commonpb.IndexState_IndexStateNone,
		FailReason: "",
	}

	m.fieldIndexLock.RLock()
	fieldIndexesMap, ok := m.indexes[collID]
	if !ok {
		state.FailReason = fmt.Sprintf("collection not exist with ID: %d", collID)
		m.fieldIndexLock.RUnlock()
		return state
	}
	// Copy the map to avoid data race after releasing the lock
	fieldIndexes := make(map[UniqueID]*model.Index, len(fieldIndexesMap))
	for id, index := range fieldIndexesMap {
		fieldIndexes[id] = index
	}
	m.fieldIndexLock.RUnlock()

	indexes, ok := m.segmentIndexes.Get(segmentID)
	if !ok {
		state.State = commonpb.IndexState_Unissued
		state.FailReason = fmt.Sprintf("segment index not exist with ID: %d", segmentID)
		return state
	}

	if index, ok := fieldIndexes[indexID]; ok && !index.IsDeleted {
		if segIdx, ok := indexes.Get(indexID); ok && !segIdx.IsDeleted {
			state.IndexName = index.IndexName
			state.State = segIdx.IndexState
			state.FailReason = segIdx.FailReason
			return state
		}
		state.State = commonpb.IndexState_Unissued
		return state
	}

	state.FailReason = fmt.Sprintf("there is no index on indexID: %d", indexID)
	return state
}

func (m *indexMeta) GetIndexedSegments(collectionID int64, segmentIDs, fieldIDs []UniqueID) []int64 {
	m.fieldIndexLock.RLock()
	fieldIndexesMap, ok := m.indexes[collectionID]
	if !ok {
		m.fieldIndexLock.RUnlock()
		return nil
	}
	// Copy the map to avoid data race after releasing the lock
	fieldIndexes := make(map[UniqueID]*model.Index, len(fieldIndexesMap))
	for id, index := range fieldIndexesMap {
		fieldIndexes[id] = index
	}
	m.fieldIndexLock.RUnlock()

	fieldIDSet := typeutil.NewUniqueSet(fieldIDs...)
	matchedFields := typeutil.NewUniqueSet()

	targetIndices := lo.Filter(lo.Values(fieldIndexes), func(index *model.Index, _ int) bool {
		return fieldIDSet.Contain(index.FieldID) && !index.IsDeleted
	})
	for _, index := range targetIndices {
		matchedFields.Insert(index.FieldID)
	}

	// some field has no index on it
	if len(matchedFields) != len(fieldIDSet) {
		return nil
	}

	checkSegmentState := func(indexes *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]) bool {
		indexedFields := 0
		for _, index := range targetIndices {
			if segIdx, ok := indexes.Get(index.IndexID); ok && !segIdx.IsDeleted && segIdx.IndexState == commonpb.IndexState_Finished {
				indexedFields += 1
			}
		}

		return indexedFields == len(targetIndices)
	}

	ret := make([]int64, 0)
	for _, sid := range segmentIDs {
		if indexes, ok := m.segmentIndexes.Get(sid); ok {
			if checkSegmentState(indexes) {
				ret = append(ret, sid)
			}
		}
	}

	return ret
}

// GetIndexesForCollection gets all indexes info with the specified collection.
func (m *indexMeta) GetIndexesForCollection(collID UniqueID, indexName string) []*model.Index {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	indexInfos := make([]*model.Index, 0)
	for _, index := range m.indexes[collID] {
		if index.IsDeleted {
			continue
		}
		if indexName == "" || indexName == index.IndexName {
			indexInfos = append(indexInfos, model.CloneIndex(index))
		}
	}
	return indexInfos
}

// GetIndexesByID resolves exact index references, including definitions retained
// after logical deletion. It does not substitute a newer index on the same field.
func (m *indexMeta) GetIndexesByID(collectionID int64, ids []int64) []*model.Index {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()
	result := make([]*model.Index, 0, len(ids))
	for _, id := range ids {
		if index := m.indexes[collectionID][id]; index != nil {
			result = append(result, model.CloneIndex(index))
		}
	}
	return result
}

// ListAllIndexes returns a clone of every index that is not deleted, across all
// collections. Used by the feature usage report.
func (m *indexMeta) ListAllIndexes() []*model.Index {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	out := make([]*model.Index, 0)
	for _, indexes := range m.indexes {
		for _, index := range indexes {
			if index.IsDeleted {
				continue
			}
			out = append(out, model.CloneIndex(index))
		}
	}
	return out
}

func (m *indexMeta) GetFieldIndexes(collID, fieldID UniqueID, indexName string) []*model.Index {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	indexInfos := make([]*model.Index, 0)
	for _, index := range m.indexes[collID] {
		if index.IsDeleted || index.FieldID != fieldID {
			continue
		}
		if indexName == "" || indexName == index.IndexName {
			indexInfos = append(indexInfos, model.CloneIndex(index))
		}
	}
	return indexInfos
}

// MarkIndexAsDeleted marks indexes as deleted. File cleanup is split by path layout:
// recycleUnusedIndexFilesV0 handles legacy v0 index_files objects, and
// recycleUnusedIndexFilesV1 handles v1 index_v1 objects.
func (m *indexMeta) MarkIndexAsDeleted(ctx context.Context, collID UniqueID, indexIDs []UniqueID) error {
	mlog.Info(ctx, "IndexCoord metaTable MarkIndexAsDeleted", mlog.Int64("collectionID", collID),
		mlog.Int64s("indexIDs", indexIDs))

	unlockCollections := m.lockCollections(collID)
	defer unlockCollections()

	m.fieldIndexLock.RLock()
	if len(indexIDs) == 0 {
		// drop all indexes if indexIDs is empty.
		indexIDs = make([]UniqueID, 0, len(m.indexes[collID]))
		for indexID, index := range m.indexes[collID] {
			if index.IsDeleted {
				continue
			}
			indexIDs = append(indexIDs, indexID)
		}
	}

	fieldIndexes, ok := m.indexes[collID]
	if !ok {
		m.fieldIndexLock.RUnlock()
		return nil
	}
	indexes := make([]*model.Index, 0)
	for _, indexID := range indexIDs {
		index, ok := fieldIndexes[indexID]
		if !ok || index.IsDeleted {
			continue
		}
		clonedIndex := model.CloneIndex(index)
		clonedIndex.IsDeleted = true
		indexes = append(indexes, clonedIndex)
	}
	m.fieldIndexLock.RUnlock()
	if len(indexes) == 0 {
		return nil
	}
	err := m.catalog.AlterIndexes(ctx, indexes)
	if err != nil {
		mlog.Error(ctx, "failed to alter index meta in meta store", mlog.Int("indexes num", len(indexes)), mlog.Err(err))
		return err
	}

	deletedIndexIDs := make([]UniqueID, 0, len(indexes))
	m.fieldIndexLock.Lock()
	for _, index := range indexes {
		m.updateCollectionIndex(index)
		deletedIndexIDs = append(deletedIndexIDs, index.IndexID)
	}
	m.fieldIndexLock.Unlock()

	// Subtract immediately — gauge tracks active indexes only. The aggregate
	// makes this O(number of deleted indexes) instead of scanning every segment
	// index. The collection lock keeps this commit ordered with size updates.
	m.storedIndexSize.deactivate(ctx, collID, deletedIndexIDs)

	mlog.Info(ctx, "IndexCoord metaTable MarkIndexAsDeleted success", mlog.Int64("collectionID", collID), mlog.Int64s("indexIDs", indexIDs))
	return nil
}

func (m *indexMeta) IsUnIndexedSegment(collectionID UniqueID, segID UniqueID) bool {
	return m.GetUnIndexedSegmentIDsForIndexTask(collectionID, []UniqueID{segID}).Contain(segID)
}

func (m *indexMeta) GetUnIndexedSegmentIDsForIndexTask(collectionID UniqueID, segIDs []UniqueID) typeutil.UniqueSet {
	unindexed := typeutil.NewUniqueSet()
	if len(segIDs) == 0 {
		return unindexed
	}

	m.fieldIndexLock.RLock()
	fieldIndexesMap, ok := m.indexes[collectionID]
	if !ok {
		m.fieldIndexLock.RUnlock()
		return unindexed
	}
	activeIndexIDs := make([]UniqueID, 0, len(fieldIndexesMap))
	for _, index := range fieldIndexesMap {
		if !index.IsDeleted {
			activeIndexIDs = append(activeIndexIDs, index.IndexID)
		}
	}
	m.fieldIndexLock.RUnlock()
	if len(activeIndexIDs) == 0 {
		return unindexed
	}

	for _, segID := range segIDs {
		// the segment should be unindexed status if the fieldIndexes is not nil
		segIndexInfos, ok := m.segmentIndexes.Get(segID)
		if !ok || segIndexInfos.Len() == 0 {
			unindexed.Insert(segID)
			continue
		}
		for _, indexID := range activeIndexIDs {
			if _, ok := segIndexInfos.Get(indexID); !ok {
				// the segment should be unindexed status if the segment index is not found within field indexes
				unindexed.Insert(segID)
				break
			}
		}
	}

	return unindexed
}

func (m *indexMeta) GetSegmentsIndexes(collectionID UniqueID, segIDs []UniqueID) map[int64]map[UniqueID]*model.SegmentIndex {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()
	segmentsIndexes := make(map[int64]map[UniqueID]*model.SegmentIndex)
	for _, segmentID := range segIDs {
		segmentsIndexes[segmentID] = m.getSegmentIndexes(collectionID, segmentID)
	}
	return segmentsIndexes
}

// getQueryViewIndexSnapshot returns collection index definitions and segment
// index records from the same field-index metadata snapshot. Callers must use
// the returned collection definitions to build FieldIndexInfo instead of
// looking the definitions up again after this method returns.
func (m *indexMeta) getQueryViewIndexSnapshot(collectionID UniqueID, segmentIDs []UniqueID) ([]*model.Index, map[int64]map[UniqueID]*model.SegmentIndex) {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	indexes := make([]*model.Index, 0, len(m.indexes[collectionID]))
	for _, index := range m.indexes[collectionID] {
		if !index.IsDeleted {
			indexes = append(indexes, model.CloneIndex(index))
		}
	}
	segmentIndexes := make(map[int64]map[UniqueID]*model.SegmentIndex, len(segmentIDs))
	for _, segmentID := range segmentIDs {
		segmentIndexes[segmentID] = m.getSegmentIndexes(collectionID, segmentID)
	}
	return indexes, segmentIndexes
}

func (m *indexMeta) GetSegmentIndexes(collectionID UniqueID, segID UniqueID) map[UniqueID]*model.SegmentIndex {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()
	return m.getSegmentIndexes(collectionID, segID)
}

// GetAllSegmentIndexes returns a deep-clone of every SegmentIndex recorded
// for segID, including deleted tasks and records whose parent field index has
// been deleted. GC needs their file keys until cleanup completes.
//
// It only reads the self-synchronized segmentIndexes map and does not acquire
// fieldIndexLock or consult field-index definitions.
func (m *indexMeta) GetAllSegmentIndexes(segID UniqueID) []*model.SegmentIndex {
	if m.segmentIndexes == nil {
		return nil
	}
	segIndexInfos, ok := m.segmentIndexes.Get(segID)
	if !ok || segIndexInfos.Len() == 0 {
		return nil
	}

	segIndexes := segIndexInfos.Values()
	ret := make([]*model.SegmentIndex, 0, len(segIndexes))
	for _, segIdx := range segIndexes {
		ret = append(ret, model.CloneSegmentIndex(segIdx))
	}
	return ret
}

// Note: thread-unsafe, don't call it outside indexMeta
func (m *indexMeta) getSegmentIndexes(collectionID UniqueID, segID UniqueID) map[UniqueID]*model.SegmentIndex {
	ret := make(map[UniqueID]*model.SegmentIndex, 0)
	segIndexInfos, ok := m.segmentIndexes.Get(segID)
	if !ok || segIndexInfos.Len() == 0 {
		return ret
	}

	fieldIndexes, ok := m.indexes[collectionID]
	if !ok {
		return ret
	}

	for _, segIdx := range segIndexInfos.Values() {
		if index, ok := fieldIndexes[segIdx.IndexID]; ok && !index.IsDeleted && !segIdx.IsDeleted {
			ret[segIdx.IndexID] = model.CloneSegmentIndex(segIdx)
		}
	}
	return ret
}

// rangeSegmentIndexesBySegment visits, without cloning, the SegmentIndex record
// of each index of each segment: the records getSegmentIndexes filters.
// RangeSegmentIndexes ranges builds instead, so it also visits a build that a
// newer build of the same segment index replaced. Writers replace a resident
// record instead of mutating it, so fn may read it but must neither mutate nor
// keep it; fn returns false to stop. Like any sync.Map traversal, it may miss
// concurrent updates.
func (m *indexMeta) rangeSegmentIndexesBySegment(fn func(segIdx *model.SegmentIndex) bool) {
	m.segmentIndexes.Range(func(_ UniqueID, segIndexes *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]) bool {
		next := true
		segIndexes.Range(func(_ UniqueID, segIdx *model.SegmentIndex) bool {
			next = fn(segIdx)
			return next
		})
		return next
	})
}

func (m *indexMeta) GetFieldIDByIndexID(collID, indexID UniqueID) UniqueID {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	if fieldIndexes, ok := m.indexes[collID]; ok {
		if index, ok := fieldIndexes[indexID]; ok {
			return index.FieldID
		}
	}
	return 0
}

func (m *indexMeta) GetIndexNameByID(collID, indexID UniqueID) string {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	if fieldIndexes, ok := m.indexes[collID]; ok {
		if index, ok := fieldIndexes[indexID]; ok {
			return index.IndexName
		}
	}
	return ""
}

func (m *indexMeta) GetIndexParams(collID, indexID UniqueID) []*commonpb.KeyValuePair {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	fieldIndexes, ok := m.indexes[collID]
	if !ok {
		return nil
	}
	index, ok := fieldIndexes[indexID]
	if !ok {
		return nil
	}
	indexParams := make([]*commonpb.KeyValuePair, 0, len(index.IndexParams))

	for _, param := range index.IndexParams {
		indexParams = append(indexParams, proto.Clone(param).(*commonpb.KeyValuePair))
	}

	return indexParams
}

func (m *indexMeta) GetTypeParams(collID, indexID UniqueID) []*commonpb.KeyValuePair {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	fieldIndexes, ok := m.indexes[collID]
	if !ok {
		return nil
	}
	index, ok := fieldIndexes[indexID]
	if !ok {
		return nil
	}
	typeParams := make([]*commonpb.KeyValuePair, 0, len(index.TypeParams))

	for _, param := range index.TypeParams {
		typeParams = append(typeParams, proto.Clone(param).(*commonpb.KeyValuePair))
	}

	return typeParams
}

func (m *indexMeta) GetIndexJob(buildID UniqueID) (*model.SegmentIndex, bool) {
	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if ok {
		return model.CloneSegmentIndex(segIdx), true
	}

	return nil, false
}

func (m *indexMeta) IsIndexExist(collID, indexID UniqueID) bool {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	fieldIndexes, ok := m.indexes[collID]
	if !ok {
		return false
	}
	if index, ok := fieldIndexes[indexID]; !ok || index.IsDeleted {
		return false
	}

	return true
}

// UpdateVersion updates the version and nodeID of the index meta, whenever the task is built once, the version will be updated once.
func (m *indexMeta) UpdateVersion(buildID, nodeID UniqueID) error {
	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	mlog.Info(m.ctx, "IndexCoord metaTable UpdateVersion receive", mlog.Int64("buildID", buildID), mlog.Int64("nodeID", nodeID))
	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if !ok {
		return merr.WrapErrServiceInternalMsg("there is no index with buildID: %d", buildID)
	}

	updateFunc := func(segIdx *model.SegmentIndex) error {
		segIdx.IndexVersion++
		segIdx.NodeID = nodeID
		return m.alterSegmentIndexes([]*model.SegmentIndex{segIdx})
	}

	return m.updateSegIndexMeta(segIdx, updateFunc)
}

func (m *indexMeta) UpdateIndexState(buildID UniqueID, state commonpb.IndexState, failReason string) error {
	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if !ok {
		mlog.Warn(m.ctx, "there is no index with buildID", mlog.Int64("buildID", buildID))
		return merr.WrapErrServiceInternalMsg("there is no index with buildID: %d", buildID)
	}

	updateFunc := func(segIdx *model.SegmentIndex) error {
		segIdx.IndexState = state
		segIdx.FailReason = failReason
		return m.alterSegmentIndexes([]*model.SegmentIndex{segIdx})
	}

	if err := m.updateSegIndexMeta(segIdx, updateFunc); err != nil {
		mlog.Warn(m.ctx, "failed to update index meta", mlog.Int64("buildID", buildID), mlog.Err(err))
		return err
	}

	mlog.Info(m.ctx, "update index task state success", mlog.Int64("buildID", buildID),
		mlog.String("state", state.String()), mlog.String("fail reason", failReason))
	return nil
}

func (m *indexMeta) FinishTask(taskInfo *workerpb.IndexTaskInfo) error {
	m.keyLock.Lock(taskInfo.GetBuildID())
	defer m.keyLock.Unlock(taskInfo.GetBuildID())

	segIdx, ok := m.segmentBuildInfo.Get(taskInfo.GetBuildID())
	if !ok {
		mlog.Warn(m.ctx, "there is no index with buildID", mlog.Int64("buildID", taskInfo.GetBuildID()))
		return nil
	}

	updated, oldSize, err := m.buildFinishedSegmentIndex(segIdx, taskInfo)
	if err != nil {
		return err
	}
	if err := m.alterSegmentIndexes([]*model.SegmentIndex{updated}); err != nil {
		return err
	}
	m.recordFinishedTask(segIdx, updated, oldSize, taskInfo)
	return nil
}

func (m *indexMeta) DeleteTask(buildID int64) error {
	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if !ok {
		mlog.Warn(m.ctx, "there is no index with buildID", mlog.Int64("buildID", buildID))
		return nil
	}

	if segIdx.IsDeleted {
		return nil
	}

	updateFunc := func(segIdx *model.SegmentIndex) error {
		segIdx.IsDeleted = true
		return m.alterSegmentIndexes([]*model.SegmentIndex{segIdx})
	}

	if err := m.updateSegIndexMeta(segIdx, updateFunc); err != nil {
		return err
	}

	mlog.Info(m.ctx, "delete index task success", mlog.Int64("buildID", buildID))
	return nil
}

// BuildIndex set the index state to be InProgress. It means IndexNode is building the index.
func (m *indexMeta) BuildIndex(buildID UniqueID) error {
	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if !ok {
		return merr.WrapErrServiceInternalMsg("there is no index with buildID: %d", buildID)
	}

	updateFunc := func(segIdx *model.SegmentIndex) error {
		segIdx.IndexState = commonpb.IndexState_InProgress

		err := m.alterSegmentIndexes([]*model.SegmentIndex{segIdx})
		if err != nil {
			mlog.Error(m.ctx, "meta Update: segment index in progress fail", mlog.Int64("buildID", segIdx.BuildID), mlog.Err(err))
			return err
		}
		return nil
	}
	if err := m.updateSegIndexMeta(segIdx, updateFunc); err != nil {
		return err
	}
	mlog.Info(m.ctx, "meta update: segment index in progress success", mlog.Int64("buildID", segIdx.BuildID),
		mlog.Int64("segmentID", segIdx.SegmentID))
	return nil
}

func (m *indexMeta) GetAllSegIndexes() map[int64]*model.SegmentIndex {
	tasks := m.segmentBuildInfo.List()
	segIndexes := make(map[int64]*model.SegmentIndex, len(tasks))
	for _, segIndex := range tasks {
		segIndexes[segIndex.BuildID] = segIndex
	}
	return segIndexes
}

func (m *indexMeta) RangeSegmentIndexes(f func(*model.SegmentIndex) bool) {
	m.segmentBuildInfo.Range(f)
}

func sameSegmentIndexGCVersion(candidate, current *model.SegmentIndex) bool {
	if candidate == nil || current == nil {
		return false
	}
	return candidate.BuildID == current.BuildID &&
		candidate.CollectionID == current.CollectionID &&
		candidate.PartitionID == current.PartitionID &&
		candidate.SegmentID == current.SegmentID &&
		candidate.IndexID == current.IndexID &&
		candidate.IndexVersion == current.IndexVersion &&
		candidate.IndexStorePathVersion == current.IndexStorePathVersion &&
		candidate.IndexState == current.IndexState &&
		slices.Equal(candidate.IndexFileKeys, current.IndexFileKeys)
}

// RemoveSegmentIndexes persists an exact batch removal before publishing any
// resident deletion. Candidates changed since file GC are skipped. If a
// catalog transaction fails, all resident entries remain for an idempotent
// retry. The input is capped at the metastore transaction limit so this method
// never spans multiple persistence transactions.
func (m *indexMeta) RemoveSegmentIndexes(ctx context.Context, candidates []*model.SegmentIndex) (int, error) {
	maxBatchSize := Params.MetaStoreCfg.MaxEtcdTxnNum.GetAsInt()
	if maxBatchSize <= 0 {
		maxBatchSize = 64
	}
	if len(candidates) > maxBatchSize {
		return 0, merr.WrapErrServiceInternalMsg(
			"segment index metadata batch size %d exceeds metastore transaction limit %d",
			len(candidates), maxBatchSize)
	}

	candidateByBuildID := make(map[UniqueID]*model.SegmentIndex, len(candidates))
	buildIDs := make([]UniqueID, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}
		if _, exists := candidateByBuildID[candidate.BuildID]; !exists {
			buildIDs = append(buildIDs, candidate.BuildID)
		}
		candidateByBuildID[candidate.BuildID] = candidate
	}
	lock.LockManyOrdered(m.keyLock, buildIDs)
	defer m.keyLock.UnlockMany(buildIDs)

	currentIndexes := make([]*model.SegmentIndex, 0, len(buildIDs))
	for _, buildID := range buildIDs {
		current, exists := m.segmentBuildInfo.Get(buildID)
		if !exists || !sameSegmentIndexGCVersion(candidateByBuildID[buildID], current) {
			continue
		}
		currentIndexes = append(currentIndexes, current)
	}
	if len(currentIndexes) == 0 {
		return 0, nil
	}

	if err := m.catalog.DropSegmentIndexes(ctx, currentIndexes); err != nil {
		return 0, err
	}
	for _, segIdx := range currentIndexes {
		m.removeSegmentIndexInMemory(segIdx)
	}
	return len(currentIndexes), nil
}

func (m *indexMeta) RemoveSegmentIndex(ctx context.Context, buildID UniqueID) error {
	m.keyLock.Lock(buildID)
	defer m.keyLock.Unlock(buildID)

	segIdx, ok := m.segmentBuildInfo.Get(buildID)
	if !ok {
		return nil
	}

	err := m.catalog.DropSegmentIndex(ctx, segIdx.CollectionID, segIdx.PartitionID, segIdx.SegmentID, buildID)
	if err != nil {
		return err
	}

	m.removeSegmentIndexInMemory(segIdx)
	return nil
}

// removeSegmentIndexInMemory drops a SegmentIndex from the in-memory indexes
// and reconciles its metrics, after its catalog record has been removed. The
// caller must hold keyLock for segIdx.BuildID. RemoveSegmentIndex, which owns
// the catalog removal itself, uses this combined form; the manifest-commit
// staging below uses the same two halves but runs the record removal at
// publication and defers the gauge until the publication is done, so the
// two paths cannot drift.
func (m *indexMeta) removeSegmentIndexInMemory(segIdx *model.SegmentIndex) {
	m.subtractStoredIndexSizeOnRemoval(segIdx)
	m.removeSegmentIndexRecordInMemory(segIdx)
}

// subtractStoredIndexSizeOnRemoval is the metric half of
// removeSegmentIndexInMemory. It is split out because the manifest-commit
// install defers it until the publication is done (see
// stagedSegmentIndexMutation): it takes the read side of the per-collection
// index DDL lock, whose write side every index DDL holds across an etcd round
// trip, so blocking on it would hold the publication behind that round trip.
func (m *indexMeta) subtractStoredIndexSizeOnRemoval(segIdx *model.SegmentIndex) {
	// Reclaim size only when the index is still alive (segment-drop case);
	// MarkIndexAsDeleted already subtracted otherwise.
	m.updateStoredIndexSizeMetric(m.ctx, segIdx.CollectionID, segIdx.IndexID,
		segIdx.IndexSerializedSize, 0)
}

// removeSegmentIndexRecordInMemory is the record half: it mutates only the
// in-memory segment-index maps and the memory-only task counters. It never
// acquires fieldIndexLock, so it is safe to run during the publication.
func (m *indexMeta) removeSegmentIndexRecordInMemory(segIdx *model.SegmentIndex) {
	if segIndexes, ok := m.segmentIndexes.Get(segIdx.SegmentID); ok {
		// A newer build of the same index may have replaced this record.
		if resident, exists := segIndexes.Get(segIdx.IndexID); exists && resident.BuildID == segIdx.BuildID {
			segIndexes.Remove(segIdx.IndexID)
			if segIndexes.Len() == 0 {
				m.segmentIndexes.Remove(segIdx.SegmentID)
			} else {
				m.segmentIndexes.Insert(segIdx.SegmentID, segIndexes)
			}
		}
	}

	m.taskCounts.replaceTask(segIdx, nil)
	m.segmentBuildInfo.Remove(segIdx.BuildID)
}

// errSegmentIndexRecordGone reports that the SegmentIndex record a staged
// upsert must be projected from no longer exists. The commit that staged the
// mutation is abandoned before creating a manifest revision; publishing would
// strand an index entry that no SegmentIndex record can ever drive GC for.
var errSegmentIndexRecordGone = errors.New("segment index record no longer exists")

// stagedSegmentIndexMutation carries a SegmentIndex change from its projection
// under keyLock to the in-memory install that follows a successful catalog
// write. action is nil when the mutation resolves to no catalog write at all;
// record is the locked task projection used to validate manifest contents.
//
// install performs in-memory changes only — it never acquires fieldIndexLock, so
// the manifest commit may run it during its publication. It returns
// the observability update it deferred (never nil): the stored-index-size gauge
// must serialize against MarkIndexAsDeleted via the per-collection index DDL
// lock, whose write side every index DDL (CreateIndex, AlterIndex,
// MarkIndexAsDeleted, RemoveIndex) holds across an etcd round trip. The caller
// MUST run the returned closure only after the publication — acquiring that
// lock during it would let one slow DDL hold the publication — and before
// releasing keyLock(BuildID), so the install-then-metric sequence stays ordered
// against the next writer of the same build.
type stagedSegmentIndexMutation struct {
	action  *metastore.UpdateAction
	record  *model.SegmentIndex
	install func() (deferredMetric func())
}

// stageSegmentIndexMutation projects mut against the SegmentIndex record
// current under its per-buildID key lock, and returns the catalog action to
// compose into the caller's transaction plus the in-memory install to run once
// that transaction commits. The lock is held from projection through install,
// so a persisted value can never be built from a read that predates it.
//
// A missing record is terminal for an upsert (errSegmentIndexRecordGone) and
// benign for a removal - the record already being gone is that mutation's
// intended end state, so the caller's manifest revision still publishes.
//
// Precondition: the caller holds keyLock(mut.BuildID) across this projection,
// the returned install, and the deferred metric closure install returns. It is
// acquired by the caller rather than here so that it is held across the whole commit,
// matching every other SegmentIndex writer; see CommitSegmentManifest's
// lock-order note.
func (m *indexMeta) stageSegmentIndexMutation(mut SegmentIndexMutation) (*stagedSegmentIndexMutation, error) {
	if mut.BuildID == 0 {
		return nil, merr.WrapErrServiceInternalMsg("segment index mutation requires a build ID")
	}
	switch mut.Type {
	case SegmentIndexUpsert:
		if mut.FinishedTask == nil {
			return nil, merr.WrapErrServiceInternalMsg("segment index upsert requires a finished task, buildID=%d", mut.BuildID)
		}
		if mut.FinishedTask.GetBuildID() != mut.BuildID {
			return nil, merr.WrapErrServiceInternalMsg("finished task buildID %d does not match segment index mutation buildID %d",
				mut.FinishedTask.GetBuildID(), mut.BuildID)
		}
	case SegmentIndexRemove:
		if mut.FinishedTask != nil {
			return nil, merr.WrapErrServiceInternalMsg("segment index removal cannot carry a finished task, buildID=%d", mut.BuildID)
		}
	default:
		return nil, merr.WrapErrServiceInternalMsg("unknown segment index mutation type %d, buildID=%d", mut.Type, mut.BuildID)
	}

	staged := &stagedSegmentIndexMutation{install: func() func() { return func() {} }}

	previous, ok := m.segmentBuildInfo.Get(mut.BuildID)
	if !ok {
		if mut.Type == SegmentIndexUpsert {
			return staged, errSegmentIndexRecordGone
		}
		return staged, nil
	}

	if mut.Type == SegmentIndexRemove {
		segIdx := previous
		action := metastore.DropSegmentIndex(segIdx)
		staged.action = &action
		staged.record = segIdx
		staged.install = func() func() {
			m.removeSegmentIndexRecordInMemory(segIdx)
			return func() { m.subtractStoredIndexSizeOnRemoval(segIdx) }
		}
		return staged, nil
	}

	finished, previousSize, err := m.buildFinishedSegmentIndex(previous, mut.FinishedTask)
	if err != nil {
		return staged, err
	}
	// The task row remains durable throughout Unissued/InProgress/Failed. A
	// successful manifest publication is the single transition that retires
	// it, in the same catalog transaction that advances manifest_path.
	action := metastore.DropSegmentIndex(finished)
	staged.action = &action
	staged.record = finished
	staged.install = func() func() {
		m.updateSegmentIndex(finished)
		// recordFinishedTask is observability only, but its stored-size gauge
		// takes the collection index DDL lock, so the whole step is deferred
		// until the publication is done.
		return func() { m.recordFinishedTask(previous, finished, previousSize, mut.FinishedTask) }
	}
	return staged, nil
}

func (m *indexMeta) GetDeletedIndexes() []*model.Index {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	deletedIndexes := make([]*model.Index, 0)
	for _, fieldIndexes := range m.indexes {
		for _, index := range fieldIndexes {
			if index.IsDeleted {
				deletedIndexes = append(deletedIndexes, index)
			}
		}
	}
	return deletedIndexes
}

func (m *indexMeta) RemoveIndex(ctx context.Context, collID, indexID UniqueID) error {
	unlockCollections := m.lockCollections(collID)
	defer unlockCollections()

	mlog.Info(ctx, "IndexCoord meta table remove index", mlog.Int64("collectionID", collID), mlog.Int64("indexID", indexID))
	err := m.catalog.DropIndex(ctx, collID, indexID)
	if err != nil {
		mlog.Info(ctx, "IndexCoord meta table remove index fail", mlog.Int64("collectionID", collID),
			mlog.Int64("indexID", indexID), mlog.Err(err))
		return err
	}

	m.fieldIndexLock.Lock()
	collectionRemoved := m.removeIndexLocked(collID, indexID)
	m.fieldIndexLock.Unlock()

	if collectionRemoved {
		deleteCollectionIndexTaskMetrics(collID)
	}
	mlog.Info(ctx, "IndexCoord meta table remove index success", mlog.Int64("collectionID", collID), mlog.Int64("indexID", indexID))
	return nil
}

// removeIndexLocked drops a field index whose catalog record is gone and
// reports whether it was the last index of its collection. The caller holds
// fieldIndexLock.
func (m *indexMeta) removeIndexLocked(collID, indexID UniqueID) bool {
	m.taskCounts.replaceIndex(m.indexes[collID][indexID], nil)
	delete(m.indexes[collID], indexID)
	if len(m.indexes[collID]) > 0 {
		return false
	}
	delete(m.indexes, collID)
	return true
}

func deleteCollectionIndexTaskMetrics(collID UniqueID) {
	metrics.IndexTaskNum.Delete(prometheus.Labels{"collection_id": strconv.FormatInt(collID, 10), "index_task_status": metrics.UnissuedIndexTaskLabel})
	metrics.IndexTaskNum.Delete(prometheus.Labels{"collection_id": strconv.FormatInt(collID, 10), "index_task_status": metrics.InProgressIndexTaskLabel})
	metrics.IndexTaskNum.Delete(prometheus.Labels{"collection_id": strconv.FormatInt(collID, 10), "index_task_status": metrics.FinishedIndexTaskLabel})
	metrics.IndexTaskNum.Delete(prometheus.Labels{"collection_id": strconv.FormatInt(collID, 10), "index_task_status": metrics.FailedIndexTaskLabel})
}

// RemoveIndexes persists a bounded exact-key batch before publishing resident
// field-index removals. Collection locks keep the candidate revalidation and
// publication ordered with concurrent index DDL.
func (m *indexMeta) RemoveIndexes(ctx context.Context, candidates []*model.Index) (int, error) {
	maxBatchSize := Params.MetaStoreCfg.MaxEtcdTxnNum.GetAsInt()
	if maxBatchSize <= 0 {
		maxBatchSize = 64
	}
	if len(candidates) > maxBatchSize {
		return 0, merr.WrapErrServiceInternalMsg(
			"field index metadata batch size %d exceeds metastore transaction limit %d",
			len(candidates), maxBatchSize)
	}

	collectionIDs := make([]UniqueID, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate != nil {
			collectionIDs = append(collectionIDs, candidate.CollectionID)
		}
	}
	unlockCollections := m.lockCollections(collectionIDs...)
	defer unlockCollections()

	currentIndexes := make([]*model.Index, 0, len(candidates))
	seen := make(map[[2]UniqueID]struct{}, len(candidates))
	m.fieldIndexLock.RLock()
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}
		key := [2]UniqueID{candidate.CollectionID, candidate.IndexID}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		current, exists := m.indexes[candidate.CollectionID][candidate.IndexID]
		if !exists || current != candidate || !current.IsDeleted {
			continue
		}
		currentIndexes = append(currentIndexes, current)
	}
	m.fieldIndexLock.RUnlock()
	if len(currentIndexes) == 0 {
		return 0, nil
	}

	if err := m.catalog.DropIndexes(ctx, currentIndexes); err != nil {
		return 0, err
	}

	removedCollections := make([]UniqueID, 0)
	m.fieldIndexLock.Lock()
	for _, index := range currentIndexes {
		if m.removeIndexLocked(index.CollectionID, index.IndexID) {
			removedCollections = append(removedCollections, index.CollectionID)
		}
	}
	m.fieldIndexLock.Unlock()

	for _, collID := range removedCollections {
		deleteCollectionIndexTaskMetrics(collID)
	}
	return len(currentIndexes), nil
}

func (m *indexMeta) CheckCleanSegmentIndex(buildID UniqueID) (bool, *model.SegmentIndex) {
	if segIndex, ok := m.segmentBuildInfo.Get(buildID); ok {
		if segIndex.IndexState == commonpb.IndexState_Finished {
			return true, model.CloneSegmentIndex(segIndex)
		}
		return false, model.CloneSegmentIndex(segIndex)
	}
	return true, nil
}

func (m *indexMeta) getSegmentsIndexStates(collectionID UniqueID, segmentIDs []UniqueID) map[int64]map[int64]*indexpb.SegmentIndexState {
	ret := make(map[int64]map[int64]*indexpb.SegmentIndexState, 0)
	m.fieldIndexLock.RLock()
	fieldIndexesMap, ok := m.indexes[collectionID]
	if !ok {
		m.fieldIndexLock.RUnlock()
		return ret
	}
	fieldIndexes := make(map[UniqueID]*model.Index, len(fieldIndexesMap))
	for id, index := range fieldIndexesMap {
		fieldIndexes[id] = index
	}
	m.fieldIndexLock.RUnlock()

	for _, segID := range segmentIDs {
		segIndexInfos, ok := m.segmentIndexes.Get(segID)
		if !ok || segIndexInfos.Len() == 0 {
			// Leave the entry out rather than storing an empty map. Both
			// callers read it as `states[indexID]`, which behaves identically
			// on a missing key and on an empty map - and a caller that scans
			// every flushed segment of a large collection on a timer would
			// otherwise allocate one map per segment per pass.
			continue
		}
		ret[segID] = make(map[int64]*indexpb.SegmentIndexState)

		for _, segIdx := range segIndexInfos.Values() {
			if index, ok := fieldIndexes[segIdx.IndexID]; ok && !index.IsDeleted && !segIdx.IsDeleted {
				indexVersion := segIdx.CurrentIndexVersion
				if indexparamcheck.IsScalarIndexType(segIdx.IndexType) {
					indexVersion = segIdx.CurrentScalarIndexVersion
				}
				ret[segID][segIdx.IndexID] = &indexpb.SegmentIndexState{
					SegmentID:    segID,
					State:        segIdx.IndexState,
					FailReason:   segIdx.FailReason,
					IndexName:    index.IndexName,
					IndexVersion: indexVersion,
				}
			}
		}
	}

	return ret
}

func (m *indexMeta) GetUnindexedSegments(collectionID int64, segmentIDs []int64) []int64 {
	indexes := m.GetIndexesForCollection(collectionID, "")
	if len(indexes) == 0 {
		// doesn't have index
		return nil
	}
	indexed := make([]int64, 0, len(segmentIDs))
	segIndexStates := m.getSegmentsIndexStates(collectionID, segmentIDs)
	for segmentID, states := range segIndexStates {
		indexStates := lo.Filter(lo.Values(states), func(state *indexpb.SegmentIndexState, _ int) bool {
			return state.GetState() == commonpb.IndexState_Finished
		})
		if len(indexStates) == len(indexes) {
			indexed = append(indexed, segmentID)
		}
	}
	return lo.Without(segmentIDs, indexed...)
}

func (m *indexMeta) AllDenseWithDiskIndex(collectionID int64, schema *schemapb.CollectionSchema) bool {
	indexInfos := m.GetIndexesForCollection(collectionID, "")

	vectorFields := typeutil.GetDenseVectorFieldSchemas(schema)
	fieldIndexTypes := lo.SliceToMap(indexInfos, func(t *model.Index) (int64, indexparamcheck.IndexType) {
		return t.FieldID, GetIndexType(t.IndexParams)
	})
	vectorFieldsWithDiskIndex := lo.Filter(vectorFields, func(field *schemapb.FieldSchema, _ int) bool {
		if indexType, ok := fieldIndexTypes[field.FieldID]; ok {
			return vecindexmgr.GetVecIndexMgrInstance().IsDiskVecIndex(indexType)
		}
		return false
	})

	return len(vectorFields) == len(vectorFieldsWithDiskIndex)
}

func (m *indexMeta) HasIndex(collectionID int64) bool {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	indexes, ok := m.indexes[collectionID]
	if ok {
		for _, index := range indexes {
			if !index.IsDeleted {
				return true
			}
		}
	}
	return false
}

func (m *indexMeta) TaskStatsJSON() string {
	tasks := m.segmentBuildInfo.GetTaskStats()
	ret, err := json.Marshal(tasks)
	if err != nil {
		return ""
	}
	return string(ret)
}

func (m *indexMeta) GetIndexJSON(collectionID int64) string {
	m.fieldIndexLock.RLock()
	defer m.fieldIndexLock.RUnlock()

	var indexMetrics []*metricsinfo.Index
	for collID, indexes := range m.indexes {
		for _, index := range indexes {
			if collectionID == 0 || collID == collectionID {
				im := &metricsinfo.Index{
					CollectionID:    collID,
					IndexID:         index.IndexID,
					FieldID:         index.FieldID,
					Name:            index.IndexName,
					IsDeleted:       index.IsDeleted,
					CreateTime:      tsoutil.PhysicalTimeFormat(index.CreateTime),
					IndexParams:     funcutil.KeyValuePair2Map(index.IndexParams),
					IsAutoIndex:     index.IsAutoIndex,
					UserIndexParams: funcutil.KeyValuePair2Map(index.UserIndexParams),
				}
				indexMetrics = append(indexMetrics, im)
			}
		}
	}

	ret, err := json.Marshal(indexMetrics)
	if err != nil {
		return ""
	}
	return string(ret)
}

func (m *indexMeta) GetSegmentIndexedFields(collectionID UniqueID, segmentID UniqueID) (bool, []*metricsinfo.IndexedField) {
	m.fieldIndexLock.RLock()
	fieldIndexesMap, ok := m.indexes[collectionID]
	if !ok {
		// the segment should be unindexed status if the collection has no indexes
		m.fieldIndexLock.RUnlock()
		return false, []*metricsinfo.IndexedField{}
	}
	// Copy the map to avoid data race after releasing the lock
	fieldIndexes := make(map[UniqueID]*model.Index, len(fieldIndexesMap))
	for id, index := range fieldIndexesMap {
		fieldIndexes[id] = index
	}
	m.fieldIndexLock.RUnlock()

	// the segment should be unindexed status if the segment indexes is not found
	segIndexInfos, ok := m.segmentIndexes.Get(segmentID)
	if !ok || segIndexInfos.Len() == 0 {
		return false, []*metricsinfo.IndexedField{}
	}

	isIndexed := true
	var segmentIndexes []*metricsinfo.IndexedField
	for _, index := range fieldIndexes {
		if si, ok := segIndexInfos.Get(index.IndexID); !index.IsDeleted {
			buildID := int64(-1)
			serializedSize := int64(0)
			if !ok || si == nil {
				// the segment should be unindexed status if the segment index is not found within field indexes
				isIndexed = false
			} else {
				buildID = si.BuildID
				serializedSize = int64(si.IndexSerializedSize)
			}

			segmentIndexes = append(segmentIndexes, &metricsinfo.IndexedField{
				IndexFieldID: index.FieldID,
				IndexID:      index.IndexID,
				BuildID:      buildID,
				IndexSize:    serializedSize,
			})
		}
	}
	return isIndexed, segmentIndexes
}
