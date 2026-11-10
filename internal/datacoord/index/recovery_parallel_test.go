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
package index

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/lock"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func newRecoveryTestIndexMeta() *IndexMeta {
	return &IndexMeta{
		ctx:              context.Background(),
		Indexes:          make(map[UniqueID]map[UniqueID]*model.Index),
		KeyLock:          lock.NewKeyLock[UniqueID](),
		SegmentBuildInfo: NewSegmentIndexBuildInfo(),
		SegmentIndexes:   typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
	}
}

// rebuildSegmentIndexCacheInOrder is the recovery rebuild as one pass over the
// records in order, the reference of rebuildSegmentIndexCache.
func rebuildSegmentIndexCacheInOrder(m *IndexMeta, segmentIndexes []*model.SegmentIndex) map[UniqueID]map[UniqueID]uint64 {
	recoveredIndexSizes := make(map[UniqueID]map[UniqueID]uint64)
	for _, segIdx := range segmentIndexes {
		indexSizes, ok := recoveredIndexSizes[segIdx.CollectionID]
		if !ok {
			indexSizes = make(map[UniqueID]uint64)
			recoveredIndexSizes[segIdx.CollectionID] = indexSizes
		}
		indexSizes[segIdx.IndexID] += segIdx.IndexSerializedSize
		if segIdx.IndexMemSize == 0 {
			segIdx.IndexMemSize = segIdx.IndexSerializedSize * paramtable.Get().DataCoordCfg.IndexMemSizeEstimateMultiplier.GetAsUint64()
		}
		indexes, ok := m.SegmentIndexes.Get(segIdx.SegmentID)
		if ok {
			indexes.Insert(segIdx.IndexID, segIdx)
		} else {
			indexes = typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
			indexes.Insert(segIdx.IndexID, segIdx)
			m.SegmentIndexes.Insert(segIdx.SegmentID, indexes)
		}
		old, _ := m.SegmentBuildInfo.Get(segIdx.BuildID)
		m.taskCounts.replaceTask(old, segIdx)
		m.SegmentBuildInfo.AddForRecovery(segIdx)
	}
	return recoveredIndexSizes
}

// recoveredSegmentIndexesForTest returns segment index records in scan order:
// several per segment and collection, every task state, deleted tasks, sizes
// to estimate, unfinished tasks well after the task-stats LRU fills, a build
// recorded twice (the later record wins) and an index of a segment built
// twice under two build IDs.
func recoveredSegmentIndexesForTest() []*model.SegmentIndex {
	states := []commonpb.IndexState{
		commonpb.IndexState_Finished, commonpb.IndexState_Finished, commonpb.IndexState_Finished,
		commonpb.IndexState_InProgress, commonpb.IndexState_Unissued, commonpb.IndexState_Failed,
		commonpb.IndexState_Retry, commonpb.IndexState_Finished, commonpb.IndexState_IndexStateNone,
	}
	var records []*model.SegmentIndex
	buildID := int64(1)
	for segment := range int64(1500) {
		for indexID := range 1 + segment%3 {
			record := &model.SegmentIndex{
				CollectionID:        1 + segment%7,
				PartitionID:         10,
				SegmentID:           1000 + segment,
				IndexID:             100 + indexID,
				BuildID:             buildID,
				IndexState:          states[int(buildID)%len(states)],
				IsDeleted:           buildID%23 == 0,
				IndexSerializedSize: uint64(buildID * 3),
				CreatedUTCTime:      uint64(buildID),
			}
			if buildID%5 != 0 {
				record.IndexMemSize = uint64(buildID * 7)
			}
			records = append(records, record)
			buildID++
		}
	}
	// A build recorded again, in another state and segment.
	again := model.CloneSegmentIndex(records[10])
	again.SegmentID, again.IndexState = 99999, commonpb.IndexState_InProgress
	records = append(records, again)
	// A build recorded three times: deleted, then alive in another state and
	// index; and one recorded again in the same state.
	deleted := model.CloneSegmentIndex(records[30])
	deleted.IsDeleted = true
	revived := model.CloneSegmentIndex(records[30])
	revived.IndexID, revived.IndexState = 102, commonpb.IndexState_Failed
	same := model.CloneSegmentIndex(records[40])
	same.SegmentID = 99998
	records = append(records, deleted, same, revived)
	// An index of a segment built again under another build ID.
	rebuilt := model.CloneSegmentIndex(records[20])
	rebuilt.BuildID, rebuilt.IndexState = buildID, commonpb.IndexState_Unissued
	records = append(records, rebuilt)
	return records
}

func cloneSegmentIndexes(records []*model.SegmentIndex) []*model.SegmentIndex {
	clones := make([]*model.SegmentIndex, len(records))
	for i, record := range records {
		clones[i] = model.CloneSegmentIndex(record)
	}
	return clones
}

type indexCacheSnapshot struct {
	segmentIndexes map[int64]map[int64]model.SegmentIndex
	builds         map[int64]model.SegmentIndex
	byIndex        map[[2]int64]indexTaskStats
	total          indexTaskStateCounts
	taskStatKeys   []int64
	taskStats      []*metricsinfo.IndexTaskStats
}

func snapshotIndexCache(m *IndexMeta) indexCacheSnapshot {
	snapshot := indexCacheSnapshot{
		segmentIndexes: make(map[int64]map[int64]model.SegmentIndex),
		builds:         make(map[int64]model.SegmentIndex),
		byIndex:        make(map[[2]int64]indexTaskStats),
	}
	m.SegmentIndexes.Range(func(segmentID int64, indexes *typeutil.ConcurrentMap[int64, *model.SegmentIndex]) bool {
		snapshot.segmentIndexes[segmentID] = make(map[int64]model.SegmentIndex)
		indexes.Range(func(indexID int64, segIdx *model.SegmentIndex) bool {
			snapshot.segmentIndexes[segmentID][indexID] = *segIdx
			return true
		})
		return true
	})
	m.SegmentBuildInfo.Range(func(segIdx *model.SegmentIndex) bool {
		snapshot.builds[segIdx.BuildID] = *segIdx
		return true
	})
	m.taskCounts.mu.RLock()
	for key, counts := range m.taskCounts.byIndex {
		snapshot.byIndex[key] = counts
	}
	snapshot.total = m.taskCounts.total
	m.taskCounts.mu.RUnlock()
	snapshot.taskStatKeys = m.SegmentBuildInfo.taskStats.Keys()
	snapshot.taskStats = m.SegmentBuildInfo.taskStats.Values()
	return snapshot
}

// The parallel rebuild leaves the segment index cache, the task counts, the
// task-stats LRU and the stored sizes as one pass over the records in scan
// order does, whether the field indexes register before or after it.
func TestRebuildSegmentIndexCacheLikeOnePassInOrder(t *testing.T) {
	records := recoveredSegmentIndexesForTest()
	registerIndexes := func(m *IndexMeta) {
		for collectionID := range int64(5) {
			m.updateCollectionIndex(&model.Index{CollectionID: 1 + collectionID, IndexID: 100})
			m.updateCollectionIndex(&model.Index{CollectionID: 1 + collectionID, IndexID: 101, IsDeleted: collectionID == 2})
		}
	}

	reference := newRecoveryTestIndexMeta()
	registerIndexes(reference)
	rebuildSegmentIndexCacheInOrder(reference, cloneSegmentIndexes(records))
	want := snapshotIndexCache(reference)
	require.Len(t, want.taskStatKeys, taskStatsLRUCapacity)
	require.Equal(t, commonpb.IndexState_InProgress, want.builds[records[10].BuildID].IndexState, "the later record of a build wins")
	require.Equal(t, records[len(records)-1].BuildID, want.segmentIndexes[records[20].SegmentID][records[20].IndexID].BuildID,
		"the later build of a segment's index wins")

	for _, registerFirst := range []bool{true, false} {
		for range 3 {
			parallel := newRecoveryTestIndexMeta()
			if registerFirst {
				registerIndexes(parallel)
			}
			parallel.rebuildSegmentIndexCache(cloneSegmentIndexes(records))
			if !registerFirst {
				registerIndexes(parallel)
			}
			got := snapshotIndexCache(parallel)
			require.Equal(t, want.segmentIndexes, got.segmentIndexes)
			require.Equal(t, want.builds, got.builds)
			require.Equal(t, want.byIndex, got.byIndex)
			require.Equal(t, want.total, got.total)
			require.Equal(t, want.taskStatKeys, got.taskStatKeys, "the LRU holds the same tasks in the same order")
			require.Equal(t, want.taskStats, got.taskStats)
		}
	}
}

func TestRecoveryKeepsNewestIndexBuildAcrossPlacements(t *testing.T) {
	m := newRecoveryTestIndexMeta()
	current := &model.SegmentIndex{CollectionID: 1, SegmentID: 2, IndexID: 3, BuildID: 20}
	older := model.CloneSegmentIndex(current)
	older.BuildID = 10
	m.rebuildSegmentIndexCache([]*model.SegmentIndex{current, older})
	manifest := model.CloneSegmentIndex(older)
	manifest.BuildID = 5
	manifest.ManifestPublished = true
	recovered := m.NewManifestIndexRecovery()
	recovered.Install(manifest)
	// A manifest may also contain a build already recovered from its catalog row.
	duplicate := model.CloneSegmentIndex(current)
	duplicate.ManifestPublished = true
	recovered.Install(duplicate)
	require.Equal(t, 1, recovered.Installed())
	snapshot := snapshotIndexCache(m)
	require.Equal(t, current.BuildID, snapshot.segmentIndexes[2][3].BuildID)
	require.False(t, snapshot.builds[20].ManifestPublished)
	require.False(t, snapshot.builds[10].ManifestPublished)
	require.True(t, snapshot.builds[5].ManifestPublished)
}

// Recovering the index meta from a catalog that reads key ranges recovers what
// one walk of each prefix recovers.
func TestRecoverIndexMetaInRangesLikeOneWalk(t *testing.T) {
	ctx := context.Background()
	paramtable.Init()
	client, err := etcd.GetEtcdClient(
		paramtable.Get().EtcdCfg.UseEmbedEtcd.GetAsBool(),
		paramtable.Get().EtcdCfg.EtcdUseSSL.GetAsBool(),
		paramtable.Get().EtcdCfg.Endpoints.GetAsStrings(),
		paramtable.Get().EtcdCfg.EtcdTLSCert.GetValue(),
		paramtable.Get().EtcdCfg.EtcdTLSKey.GetValue(),
		paramtable.Get().EtcdCfg.EtcdTLSCACert.GetValue(),
		paramtable.Get().EtcdCfg.EtcdTLSMinVersion.GetValue())
	require.NoError(t, err)
	root := fmt.Sprintf("unittest/index-meta-ranges/%d", time.Now().UnixNano())
	t.Cleanup(func() {
		_, _ = client.Delete(context.Background(), root, clientv3.WithPrefix())
		_ = client.Close()
	})
	catalog := datacoord.NewCatalog(etcdkv.NewEtcdKV(client, root), "", root)
	records := recoveredSegmentIndexesForTest()
	for start := 0; start < len(records); start += 64 {
		require.NoError(t, catalog.AlterSegmentIndexes(ctx, records[start:min(start+64, len(records))]))
	}
	var collectionIDs []int64
	for collectionID := range int64(9) {
		collectionIDs = append(collectionIDs, collectionID+1)
		require.NoError(t, catalog.CreateIndex(ctx, &model.Index{CollectionID: collectionID + 1, IndexID: 100}))
		require.NoError(t, catalog.CreateIndex(ctx, &model.Index{CollectionID: collectionID + 1, IndexID: 101}))
	}

	recoverWith := func(ranges int) (*IndexMeta, indexCacheSnapshot) {
		key := paramtable.Get().MetaStoreCfg.ParallelScanRanges.Key
		paramtable.Get().Save(key, strconv.Itoa(ranges))
		defer paramtable.Get().Reset(key)
		m, err := NewIndexMeta(ctx, catalog, collectionIDs)
		require.NoError(t, err)
		return m, snapshotIndexCache(m)
	}
	oneWalk, want := recoverWith(1)
	for _, ranges := range []int{2, 8} {
		m, got := recoverWith(ranges)
		require.Equal(t, oneWalk.Indexes, m.Indexes)
		require.Equal(t, want.segmentIndexes, got.segmentIndexes)
		require.Equal(t, want.builds, got.builds)
		require.Equal(t, want.byIndex, got.byIndex)
		require.Equal(t, want.total, got.total)
		require.Equal(t, want.taskStatKeys, got.taskStatKeys)
	}
}
