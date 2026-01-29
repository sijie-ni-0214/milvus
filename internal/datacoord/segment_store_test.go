package datacoord

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const testSegmentStoreRoot = "by-dev/meta"

// newCASTestMeta builds a meta whose segments live in an in-memory optimistic
// persist, so writes go through casSegmentStore.
func newCASTestMeta(t *testing.T) (*meta, *memPersist[string, *datapb.SegmentInfo]) {
	persist := NewOptimisticTxnMemoryPersist[string, *datapb.SegmentInfo](&SegmentInfoMarshaler{}).(*memPersist[string, *datapb.SegmentInfo])
	m := &meta{
		ctx:      context.Background(),
		catalog:  datacoord.NewCatalog(NewMetaMemoryKV(), "", ""),
		segments: NewSegmentsInfo(),
	}
	m.segmentStore = newCASSegmentStore(persist, m.catalog, testSegmentStoreRoot, m.segments)
	return m, persist
}

func newFlushedTestSegment(id int64) *SegmentInfo {
	return NewSegmentInfo(&datapb.SegmentInfo{
		ID:            id,
		CollectionID:  1,
		PartitionID:   2,
		InsertChannel: "ch-1",
		State:         commonpb.SegmentState_Flushed,
		NumOfRows:     10,
	})
}

func storedSegmentKey(id int64) string {
	return testSegmentStoreRoot + "/" + segmentKey(1, 2, id)
}

func TestOptimisticTxnPutIfVersion(t *testing.T) {
	ctx := context.Background()
	persist := NewOptimisticTxnMemoryPersist[string, *datapb.SegmentInfo](&SegmentInfoMarshaler{})

	txn := persist.Txn(ctx)
	txn.PutIfVersion("k", &datapb.SegmentInfo{ID: 1}, 0, false)
	results, err := txn.Commit()
	require.NoError(t, err)
	v1 := results[0].Version

	// Version 0 means the key must be absent.
	txn = persist.Txn(ctx)
	txn.PutIfVersion("k", &datapb.SegmentInfo{ID: 1, NumOfRows: 1}, 0, false)
	_, err = txn.Commit()
	require.ErrorIs(t, err, ErrVersionConflict)

	txn = persist.Txn(ctx)
	txn.PutIfVersion("k", &datapb.SegmentInfo{ID: 1, NumOfRows: 1}, v1, false)
	results, err = txn.Commit()
	require.NoError(t, err)
	require.Greater(t, results[0].Version, v1)

	// The old version is stale now.
	txn = persist.Txn(ctx)
	txn.DeleteIfVersion("k", v1)
	_, err = txn.Commit()
	require.ErrorIs(t, err, ErrVersionConflict)
}

func TestOptimisticTxnPlanBatches(t *testing.T) {
	ops := &txnOps[string, *datapb.SegmentInfo]{}
	ops.Raw(RawOp{Key: "a"})
	ops.Raw(RawOp{Key: "marker", Commit: true})
	ops.Raw(RawOp{Key: "b"})
	ops.Raw(RawOp{Key: "c"})

	batches, err := ops.planBatches(10)
	require.NoError(t, err)
	require.Equal(t, [][]int{{0, 1, 2, 3}}, batches)

	// Over the limit: non-commit ops flush first in recorded order, the
	// commit-marked op lands alone in the final batch.
	batches, err = ops.planBatches(2)
	require.NoError(t, err)
	require.Equal(t, [][]int{{0, 2}, {3}, {1}}, batches)

	ops.AtomicOnly()
	_, err = ops.planBatches(2)
	require.Error(t, err)
}

func TestCASSegmentStoreWritesFullRecordsAgainstCachedVersions(t *testing.T) {
	ctx := context.Background()
	m, persist := newCASTestMeta(t)

	segment := newFlushedTestSegment(100)
	segment.Binlogs = []*datapb.FieldBinlog{{FieldID: 1, Binlogs: []*datapb.Binlog{{LogID: 7, EntriesNum: 10}}}}
	require.NoError(t, m.AddSegment(ctx, segment))
	version, ok := m.segments.Version(100)
	require.True(t, ok)
	stored := persist.data[storedSegmentKey(100)]
	require.NotNil(t, stored)
	require.Equal(t, version, stored.version)
	// Binlogs are kept inline in the record.
	require.Len(t, stored.value.GetBinlogs(), 1)

	require.NoError(t, m.UpdateSegmentsInfo(ctx, UpdateStatusOperator(100, commonpb.SegmentState_Dropped)))
	newVersion, ok := m.segments.Version(100)
	require.True(t, ok)
	require.Greater(t, newVersion, version)
	require.Equal(t, commonpb.SegmentState_Dropped, persist.data[storedSegmentKey(100)].value.GetState())
	require.Equal(t, commonpb.SegmentState_Dropped, m.GetSegment(ctx, 100).GetState())

	require.NoError(t, m.DropSegment(ctx, 100))
	require.Nil(t, m.GetSegment(ctx, 100))
	require.NotContains(t, persist.data, storedSegmentKey(100))
}

func TestCASSegmentStoreConflictRefreshesCache(t *testing.T) {
	ctx := context.Background()
	m, persist := newCASTestMeta(t)
	require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(100)))

	// Another writer moves the record under the cache.
	external := newFlushedTestSegment(100)
	external.NumOfRows = 42
	cached, _ := m.segments.Version(100)
	txn := persist.Txn(ctx)
	txn.PutIfVersion(storedSegmentKey(100), external.SegmentInfo, cached, false)
	results, err := txn.Commit()
	require.NoError(t, err)

	err = m.UpdateSegmentsInfo(ctx, UpdateStatusOperator(100, commonpb.SegmentState_Dropped))
	require.ErrorIs(t, err, ErrVersionConflict)

	// The cache now holds the persisted value, so a rebuilt write succeeds.
	refreshed, _ := m.segments.Version(100)
	require.Equal(t, results[0].Version, refreshed)
	require.EqualValues(t, 42, m.GetSegment(ctx, 100).GetNumOfRows())
	require.NoError(t, m.UpdateSegmentsInfo(ctx, UpdateStatusOperator(100, commonpb.SegmentState_Dropped)))
	require.EqualValues(t, 42, persist.data[storedSegmentKey(100)].value.GetNumOfRows())
}

func TestCASSegmentStoreComposesCatalogActions(t *testing.T) {
	ctx := context.Background()
	m, persist := newCASTestMeta(t)
	require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(100)))

	dataView := &viewpb.DataViewOfCollection{
		CollectionId: 1,
		DataVersion:  &viewpb.DataVersion{StreamingVersion: 3},
	}
	committed, err := m.UpdateSegmentsInfoAndDataView(ctx, dataView, SetSegmentIsInvisible(100, true))
	require.NoError(t, err)
	require.True(t, committed)
	require.True(t, persist.data[storedSegmentKey(100)].value.GetIsInvisible())

	// The DataView is written by the catalog encoding, under the root path,
	// in the same transaction as the segment record.
	b, err := m.catalog.(catalogUpdateEncoder).EncodeUpdate(ctx, metastore.SaveDataView(dataView))
	require.NoError(t, err)
	require.Len(t, b.Ops(), 1)
	value, ok := persist.raw[testSegmentStoreRoot+"/"+b.Ops()[0].Key]
	require.True(t, ok)
	require.Equal(t, b.Ops()[0].Value, value)
}

func TestCASSegmentStoreKeepsRuntimeFields(t *testing.T) {
	ctx := context.Background()
	m, _ := newCASTestMeta(t)
	require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(100)))

	exist, canDo := m.CheckAndSetSegmentsCompacting(ctx, []int64{100})
	require.True(t, exist)
	require.True(t, canDo)
	require.NoError(t, m.UpdateSegmentsInfo(ctx, UpdateSegmentLevelOperator(100, datapb.SegmentLevel_L2)))
	require.True(t, m.GetSegment(ctx, 100).isCompacting, "a persisted update must keep the compacting flag")
	require.Equal(t, datapb.SegmentLevel_L2, m.GetSegment(ctx, 100).GetLevel())
}

func TestCASSegmentStoreParallelWritersOfDifferentSegments(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()
	m, persist := newCASTestMeta(t)
	const n = 32
	for i := int64(0); i < n; i++ {
		require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(100+i)))
	}
	// memPersist is not safe for concurrent use; serialize its commits
	// while the meta-level writers run in parallel.
	m.segmentStore = &serializedSegmentStore{inner: m.segmentStore}

	var wg sync.WaitGroup
	for i := int64(0); i < n; i++ {
		wg.Add(1)
		go func(id int64) {
			defer wg.Done()
			for round := 0; round < 5; round++ {
				assert.NoError(t, m.UpdateSegmentsInfo(ctx, MutationsOperator(map[int64][]MutateFunc{
					id: {func(seg *datapb.SegmentInfo) bool {
						seg.NumOfRows++
						return true
					}},
				})))
			}
		}(100 + i)
	}
	wg.Wait()
	for i := int64(0); i < n; i++ {
		require.EqualValues(t, 15, m.GetSegment(ctx, 100+i).GetNumOfRows())
		require.EqualValues(t, 15, persist.data[storedSegmentKey(100+i)].value.GetNumOfRows())
	}
}

func TestMutationsOperatorSkipsRejectedSegments(t *testing.T) {
	ctx := context.Background()
	m, persist := newCASTestMeta(t)
	require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(100)))
	require.NoError(t, m.AddSegment(ctx, newFlushedTestSegment(101)))
	before := persist.data[storedSegmentKey(101)].version

	require.NoError(t, m.UpdateSegmentsInfo(ctx, MutationsOperator(map[int64][]MutateFunc{
		100: {func(seg *datapb.SegmentInfo) bool { seg.NumOfRows = 99; return true }},
		101: {func(seg *datapb.SegmentInfo) bool { seg.NumOfRows = 99; return false }},
	})))
	require.EqualValues(t, 99, m.GetSegment(ctx, 100).GetNumOfRows())
	require.EqualValues(t, 10, m.GetSegment(ctx, 101).GetNumOfRows())
	require.Equal(t, before, persist.data[storedSegmentKey(101)].version, "a rejected segment is not rewritten")
}

func TestMergeFieldBinlogsDoesNotAliasInput(t *testing.T) {
	incoming := []*datapb.FieldBinlog{
		{FieldID: 1, Binlogs: []*datapb.Binlog{{LogID: 1}}},
		{FieldID: 1, Binlogs: []*datapb.Binlog{{LogID: 2}}},
	}
	first := mergeFieldBinlogs(nil, incoming)
	second := mergeFieldBinlogs(nil, incoming)
	require.Len(t, first, 1)
	require.Len(t, first[0].GetBinlogs(), 2)
	require.Len(t, second[0].GetBinlogs(), 2, "a rerun from the same input must produce the same result")
	require.Len(t, incoming[0].GetBinlogs(), 1, "the input is never modified")
}

// serializedSegmentStore serializes a segmentStore for test persists that are
// not safe for concurrent use.
type serializedSegmentStore struct {
	mu    sync.Mutex
	inner segmentStore
}

func (s *serializedSegmentStore) ListSegments(ctx context.Context, target partitionScanTarget) ([]*datapb.SegmentInfo, []int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.ListSegments(ctx, target)
}

func (s *serializedSegmentStore) AddSegment(ctx context.Context, segment *datapb.SegmentInfo) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.AddSegment(ctx, segment)
}

func (s *serializedSegmentStore) DropSegment(ctx context.Context, segment *datapb.SegmentInfo) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.DropSegment(ctx, segment)
}

func (s *serializedSegmentStore) AlterSegments(ctx context.Context, segments []*datapb.SegmentInfo, binlogs ...metastore.BinlogsIncrement) (map[int64]int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.AlterSegments(ctx, segments, binlogs...)
}

func (s *serializedSegmentStore) SaveDroppedSegmentsInBatch(ctx context.Context, segments []*datapb.SegmentInfo) (map[int64]int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.SaveDroppedSegmentsInBatch(ctx, segments)
}

func (s *serializedSegmentStore) Update(ctx context.Context, actions ...metastore.UpdateAction) (map[int64]int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inner.Update(ctx, actions...)
}

func newTestEtcdPersist(t *testing.T) (OptimisticTxnPersist[string, *datapb.SegmentInfo], *clientv3.Client, string) {
	paramtable.Init()
	cli, err := etcd.GetEtcdClient(
		Params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		Params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		Params.EtcdCfg.Endpoints.GetAsStrings(),
		Params.EtcdCfg.EtcdTLSCert.GetValue(),
		Params.EtcdCfg.EtcdTLSKey.GetValue(),
		Params.EtcdCfg.EtcdTLSCACert.GetValue(),
		Params.EtcdCfg.EtcdTLSMinVersion.GetValue())
	require.NoError(t, err)
	root := fmt.Sprintf("/test/optimistic-txn/%s/%d", t.Name(), time.Now().UnixNano())
	t.Cleanup(func() {
		_, _ = cli.Delete(context.Background(), root, clientv3.WithPrefix())
		_ = cli.Close()
	})
	return NewOptimisticTxnEtcdPersist[string, *datapb.SegmentInfo](cli, &SegmentInfoMarshaler{}), cli, root
}

func TestOptimisticTxnEtcdVersionedWrites(t *testing.T) {
	ctx := context.Background()
	persist, cli, root := newTestEtcdPersist(t)
	key := root + "/s/1"

	txn := persist.Txn(ctx)
	txn.PutIfVersion(key, &datapb.SegmentInfo{ID: 1}, 0, false)
	txn.Raw(RawOp{Key: root + "/marker", Value: "v1", Commit: true})
	results, err := txn.Commit()
	require.NoError(t, err)
	v1 := results[0].Version
	resp, err := cli.Get(ctx, root+"/marker")
	require.NoError(t, err)
	require.Equal(t, "v1", string(resp.Kvs[0].Value), "raw ops commit in the same transaction")

	// A stale version is a conflict, not a retry loop.
	txn = persist.Txn(ctx)
	txn.PutIfVersion(key, &datapb.SegmentInfo{ID: 1, NumOfRows: 1}, 0, false)
	_, err = txn.Commit()
	require.ErrorIs(t, err, ErrVersionConflict)

	txn = persist.Txn(ctx)
	txn.PutIfVersion(key, &datapb.SegmentInfo{ID: 1, NumOfRows: 2}, v1, false)
	results, err = txn.Commit()
	require.NoError(t, err)
	v2 := results[0].Version
	require.Greater(t, v2, v1)

	current, err := persist.Get(ctx, []string{key})
	require.NoError(t, err)
	require.Equal(t, v2, current[key].Version)
	require.EqualValues(t, 2, current[key].Value.GetNumOfRows())
}

// A versioned write whose response was lost is retried with the same expected
// version; finding its own value in the store, the retry succeeds instead of
// reporting a conflict.
func TestOptimisticTxnEtcdRetryAfterLostResponse(t *testing.T) {
	ctx := context.Background()
	persist, cli, root := newTestEtcdPersist(t)
	key := root + "/s/1"

	txn := persist.Txn(ctx)
	txn.PutIfVersion(key, &datapb.SegmentInfo{ID: 1}, 0, false)
	results, err := txn.Commit()
	require.NoError(t, err)
	v1 := results[0].Version

	value := &datapb.SegmentInfo{ID: 1, NumOfRows: 5}
	retried := persist.Txn(ctx)
	retried.PutIfVersion(key, value, v1, false)
	retried.Raw(RawOp{Key: root + "/marker", Value: "m", Commit: true})
	first, err := retried.Commit()
	require.NoError(t, err)
	_, err = cli.Delete(ctx, root+"/marker")
	require.NoError(t, err)

	again, err := retried.Commit()
	require.NoError(t, err)
	require.Equal(t, first[0].Version, again[0].Version)
	resp, err := cli.Get(ctx, root+"/marker")
	require.NoError(t, err)
	require.Len(t, resp.Kvs, 1, "the raw ops of an applied batch are re-applied")

	// A different value at a newer version is still a conflict.
	conflicting := persist.Txn(ctx)
	conflicting.PutIfVersion(key, &datapb.SegmentInfo{ID: 1, NumOfRows: 6}, v1, false)
	_, err = conflicting.Commit()
	require.ErrorIs(t, err, ErrVersionConflict)
}

func TestOptimisticTxnEtcdOrderedBatches(t *testing.T) {
	ctx := context.Background()
	persist, cli, root := newTestEtcdPersist(t)
	key := paramtable.Get().MetaStoreCfg.MaxEtcdTxnNum.Key
	paramtable.Get().Save(key, "2")
	defer paramtable.Get().Reset(key)

	txn := persist.Txn(ctx)
	for i := 0; i < 5; i++ {
		txn.PutIfVersion(fmt.Sprintf("%s/s/%d", root, i), &datapb.SegmentInfo{ID: int64(i)}, 0, false)
	}
	txn.Raw(RawOp{Key: root + "/marker", Value: "done", Commit: true})
	results, err := txn.Commit()
	require.NoError(t, err)
	resp, err := cli.Get(ctx, root+"/marker")
	require.NoError(t, err)
	markerRevision := resp.Kvs[0].ModRevision
	for i := 0; i < 5; i++ {
		require.Less(t, results[i].Version, markerRevision, "the commit marker lands after every other op")
	}

	atomic := persist.Txn(ctx)
	for i := 10; i < 13; i++ {
		atomic.PutIfVersion(fmt.Sprintf("%s/s/%d", root, i), &datapb.SegmentInfo{ID: int64(i)}, 0, false)
	}
	atomic.AtomicOnly()
	_, err = atomic.Commit()
	require.Error(t, err)
}
