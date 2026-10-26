package datacoord

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// writeRangeTestSegments writes segments in the catalog encoding for the
// returned collections (one holding most of them) and for orphan collections
// before, between and after them in key order, records that still carry their
// binlogs, and a record keyed exactly at the first key of a collection. It
// returns the collections and the number of records.
func writeRangeTestSegments(t *testing.T, persist OptimisticTxnPersist[string, *datapb.SegmentInfo], catalog *datacoord.Catalog, root string) ([]int64, int) {
	t.Helper()
	ctx := context.Background()
	listed := []int64{20, 200, 201, 2000, 3, 30, 449412345678901234, 7, 77}
	for i := range 40 {
		listed = append(listed, 100000+int64(i)*7919)
	}
	nextSegment := int64(1)
	addSegments := func(collectionID int64, n int) {
		for range n {
			segment := compatTestSegment(nextSegment)
			segment.CollectionID = collectionID
			segment.PartitionID = nextSegment % 3
			require.NoError(t, catalog.AddSegment(ctx, segment))
			nextSegment++
		}
	}
	for i, collectionID := range listed {
		n := 1 + i%3
		if collectionID == 200 {
			n = 60
		}
		addSegments(collectionID, n)
	}
	for _, orphan := range []int64{1, 2, 25, 99999999} {
		addSegments(orphan, 2)
	}

	txn := persist.Txn(ctx)
	for _, collectionID := range []int64{7, 1} {
		inline := compatTestSegment(nextSegment)
		inline.CollectionID = collectionID
		txn.PutIfVersion(root+"/"+segmentRecordKey(collectionID, 2, nextSegment), inline, 0, false)
		nextSegment++
	}
	atBound := compatTestSegment(nextSegment)
	atBound.CollectionID, atBound.Binlogs, atBound.Deltalogs, atBound.Statslogs, atBound.Bm25Statslogs = 30, nil, nil, nil, nil
	txn.PutIfVersion(root+"/"+datacoord.SegmentPrefix+"/30/", atBound, 0, false)
	_, err := txn.Commit()
	require.NoError(t, err)
	return listed, int(nextSegment)
}

func listSegmentsInRanges(t *testing.T, store *casSegmentStore, collectionIDs []int64, ranges int) ([]*datapb.SegmentInfo, []int64, error) {
	t.Helper()
	key := paramtable.Get().MetaStoreCfg.ParallelScanRanges.Key
	paramtable.Get().Save(key, strconv.Itoa(ranges))
	defer paramtable.Get().Reset(key)
	return store.ListSegments(context.Background(), collectionIDs)
}

// Segment records scanned as parallel key ranges recover exactly what one
// scan recovers, in the same order: orphan collections, inline records and a
// record keyed at a range bound included.
func TestCASSegmentStoreListsRangesLikeOneScan(t *testing.T) {
	persist, cli, root := newTestEtcdPersist(t)
	_, ok := persist.(rangeScanner[*datapb.SegmentInfo])
	require.True(t, ok, "the etcd persist reads ranges")
	catalog := datacoord.NewCatalog(etcdkv.NewEtcdKV(cli, root), "files", root)
	listed, numRecords := writeRangeTestSegments(t, persist, catalog, root)
	// An etcd without RangeStream serves every range in several pages.
	paginationKey := paramtable.Get().MetaStoreCfg.PaginationSize.Key
	paramtable.Get().Save(paginationKey, "7")
	defer paramtable.Get().Reset(paginationKey)

	oneScan := newCASSegmentStore(persist, catalog, root)
	want, wantVersions, err := listSegmentsInRanges(t, oneScan, listed, 1)
	require.NoError(t, err)
	require.Len(t, want, numRecords)

	for _, ranges := range []int{2, 3, 8, 64} {
		store := newCASSegmentStore(persist, catalog, root)
		got, versions, err := listSegmentsInRanges(t, store, listed, ranges)
		require.NoError(t, err)
		require.Equal(t, wantVersions, versions, "ranges %d", ranges)
		require.Len(t, got, len(want))
		for i := range want {
			require.True(t, proto.Equal(want[i], got[i]), "ranges %d: record %d differs", ranges, i)
		}
		require.ElementsMatch(t, oneScan.inline.Collect(), store.inline.Collect())
		require.Len(t, store.inline.Collect(), 2)
	}
}

// A record that does not decode fails the ranged scan with the error one scan
// meets first.
func TestCASSegmentStoreRangesFailOnTheFirstCorruptRecord(t *testing.T) {
	ctx := context.Background()
	persist, cli, root := newTestEtcdPersist(t)
	catalog := datacoord.NewCatalog(etcdkv.NewEtcdKV(cli, root), "", root)
	listed, _ := writeRangeTestSegments(t, persist, catalog, root)
	for _, collectionID := range []int64{2000, 77, 3} {
		_, err := cli.Put(ctx, fmt.Sprintf("%s/%s/%d/1/1", root, datacoord.SegmentPrefix, collectionID), "corrupt")
		require.NoError(t, err)
	}

	_, _, want := listSegmentsInRanges(t, newCASSegmentStore(persist, catalog, root), listed, 1)
	require.ErrorIs(t, want, merr.ErrDataIntegrity)
	require.Contains(t, want.Error(), "/2000/1/1", "one scan meets the first corrupt key first")
	for _, ranges := range []int{2, 8, 64} {
		_, _, err := listSegmentsInRanges(t, newCASSegmentStore(persist, catalog, root), listed, ranges)
		require.Equal(t, want.Error(), err.Error(), "ranges %d", ranges)
		require.ErrorIs(t, err, merr.ErrDataIntegrity)
	}
}
