package datacoord

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

func newRecoveryTestIndexMeta() *indexMeta {
	return &indexMeta{
		ctx:              context.Background(),
		indexes:          make(map[UniqueID]map[UniqueID]*model.Index),
		keyLock:          lock.NewKeyLock[UniqueID](),
		segmentBuildInfo: newSegmentIndexBuildInfo(),
		segmentIndexes:   typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
	}
}

// rebuildSegmentIndexCacheInOrder is the recovery rebuild as one pass over the
// records in order, the reference of rebuildSegmentIndexCache.
func rebuildSegmentIndexCacheInOrder(m *indexMeta, segmentIndexes []*model.SegmentIndex) map[UniqueID]map[UniqueID]uint64 {
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
		indexes, ok := m.segmentIndexes.Get(segIdx.SegmentID)
		if ok {
			indexes.Insert(segIdx.IndexID, segIdx)
		} else {
			indexes = typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
			indexes.Insert(segIdx.IndexID, segIdx)
			m.segmentIndexes.Insert(segIdx.SegmentID, indexes)
		}
		old, _ := m.segmentBuildInfo.Get(segIdx.BuildID)
		m.taskCounts.replaceTask(old, segIdx)
		m.segmentBuildInfo.AddForRecovery(segIdx)
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
		for indexID := range int64(1 + segment%3) {
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
	byIndex        map[[2]int64]indexTaskStateCounts
	total          indexTaskStateCounts
	taskStatKeys   []int64
	taskStats      []*metricsinfo.IndexTaskStats
}

func snapshotIndexCache(m *indexMeta) indexCacheSnapshot {
	snapshot := indexCacheSnapshot{
		segmentIndexes: make(map[int64]map[int64]model.SegmentIndex),
		builds:         make(map[int64]model.SegmentIndex),
		byIndex:        make(map[[2]int64]indexTaskStateCounts),
	}
	m.segmentIndexes.Range(func(segmentID int64, indexes *typeutil.ConcurrentMap[int64, *model.SegmentIndex]) bool {
		snapshot.segmentIndexes[segmentID] = make(map[int64]model.SegmentIndex)
		indexes.Range(func(indexID int64, segIdx *model.SegmentIndex) bool {
			snapshot.segmentIndexes[segmentID][indexID] = *segIdx
			return true
		})
		return true
	})
	m.segmentBuildInfo.Range(func(segIdx *model.SegmentIndex) bool {
		snapshot.builds[segIdx.BuildID] = *segIdx
		return true
	})
	m.taskCounts.mu.RLock()
	for key, counts := range m.taskCounts.byIndex {
		snapshot.byIndex[key] = counts
	}
	snapshot.total = m.taskCounts.total
	m.taskCounts.mu.RUnlock()
	snapshot.taskStatKeys = m.segmentBuildInfo.taskStats.Keys()
	snapshot.taskStats = m.segmentBuildInfo.taskStats.Values()
	return snapshot
}

// The parallel rebuild leaves the segment index cache, the task counts, the
// task-stats LRU and the stored sizes as one pass over the records in scan
// order does, whether the field indexes register before or after it.
func TestRebuildSegmentIndexCacheLikeOnePassInOrder(t *testing.T) {
	records := recoveredSegmentIndexesForTest()
	registerIndexes := func(m *indexMeta) {
		for collectionID := range int64(5) {
			m.updateCollectionIndex(&model.Index{CollectionID: 1 + collectionID, IndexID: 100})
			m.updateCollectionIndex(&model.Index{CollectionID: 1 + collectionID, IndexID: 101, IsDeleted: collectionID == 2})
		}
	}

	reference := newRecoveryTestIndexMeta()
	registerIndexes(reference)
	wantSizes := rebuildSegmentIndexCacheInOrder(reference, cloneSegmentIndexes(records))
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
			sizes := parallel.rebuildSegmentIndexCache(cloneSegmentIndexes(records))
			if !registerFirst {
				registerIndexes(parallel)
			}
			got := snapshotIndexCache(parallel)
			require.Equal(t, wantSizes, sizes)
			require.Equal(t, want.segmentIndexes, got.segmentIndexes)
			require.Equal(t, want.builds, got.builds)
			require.Equal(t, want.byIndex, got.byIndex)
			require.Equal(t, want.total, got.total)
			require.Equal(t, want.taskStatKeys, got.taskStatKeys, "the LRU holds the same tasks in the same order")
			require.Equal(t, want.taskStats, got.taskStats)
		}
	}
}

// Recovering the index meta from a catalog that reads key ranges recovers what
// one walk of each prefix recovers.
func TestRecoverIndexMetaInRangesLikeOneWalk(t *testing.T) {
	ctx := context.Background()
	paramtable.Init()
	client, err := etcd.GetEtcdClient(
		Params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		Params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		Params.EtcdCfg.Endpoints.GetAsStrings(),
		Params.EtcdCfg.EtcdTLSCert.GetValue(),
		Params.EtcdCfg.EtcdTLSKey.GetValue(),
		Params.EtcdCfg.EtcdTLSCACert.GetValue(),
		Params.EtcdCfg.EtcdTLSMinVersion.GetValue())
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

	recoverWith := func(ranges int) (*indexMeta, indexCacheSnapshot) {
		key := paramtable.Get().MetaStoreCfg.ParallelScanRanges.Key
		paramtable.Get().Save(key, strconv.Itoa(ranges))
		defer paramtable.Get().Reset(key)
		m, err := recoverIndexMeta(ctx, catalog, collectionIDs)
		require.NoError(t, err)
		return m, snapshotIndexCache(m)
	}
	oneWalk, want := recoverWith(1)
	for _, ranges := range []int{2, 8} {
		m, got := recoverWith(ranges)
		require.Equal(t, oneWalk.indexes, m.indexes)
		require.Equal(t, want.segmentIndexes, got.segmentIndexes)
		require.Equal(t, want.builds, got.builds)
		require.Equal(t, want.byIndex, got.byIndex)
		require.Equal(t, want.total, got.total)
		require.Equal(t, want.taskStatKeys, got.taskStatKeys)
	}
}
