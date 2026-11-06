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
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// writeRangeTestSegments writes segments in the catalog encoding for the
// returned collections (one holding most of them) and for orphan collections
// before, between and after them in key order, records that still carry their
// binlogs, and a record keyed exactly at the first key of a collection. It
// returns the collections and the number of records.
func writeRangeTestSegments(t *testing.T, catalog *Catalog) ([]int64, int) {
	t.Helper()
	ctx := context.Background()
	listed := []int64{20, 200, 201, 2000, 3, 30, 449412345678901234, 7, 77}
	for i := range 40 {
		listed = append(listed, 100000+int64(i)*7919)
	}
	nextSegment := int64(1)
	addSegments := func(collectionID int64, n int) {
		for range n {
			segment := rangeTestSegment(nextSegment)
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

	save := func(key string, record *datapb.SegmentInfo) {
		value, err := proto.Marshal(record)
		require.NoError(t, err)
		require.NoError(t, catalog.MetaKv.Save(ctx, key, string(value)))
	}
	for _, collectionID := range []int64{7, 1} {
		inline := rangeTestSegment(nextSegment)
		inline.CollectionID = collectionID
		save(buildSegmentPath(collectionID, 2, nextSegment), inline)
		nextSegment++
	}
	atBound := rangeTestSegment(nextSegment)
	atBound.CollectionID, atBound.Binlogs, atBound.Deltalogs, atBound.Statslogs, atBound.Bm25Statslogs = 30, nil, nil, nil, nil
	save(SegmentPrefix+"/30/", atBound)
	return listed, int(nextSegment)
}

func listSegmentsInRanges(t *testing.T, catalog *Catalog, collectionIDs []int64, ranges int) ([]*datapb.SegmentInfo, error) {
	t.Helper()
	key := paramtable.Get().MetaStoreCfg.ParallelScanRanges.Key
	paramtable.Get().Save(key, strconv.Itoa(ranges))
	defer paramtable.Get().Reset(key)
	return catalog.ListAllSegmentsForRecovery(context.Background(), collectionIDs)
}

// Segment records scanned as parallel key ranges recover exactly what one
// scan recovers, in the same order: orphan collections, inline records and a
// record keyed at a range bound included.
func TestCatalogSegmentListsRangesLikeOneScan(t *testing.T) {
	catalog, _, _, _ := newIndexRangeTestCatalog(t)
	listed, numRecords := writeRangeTestSegments(t, catalog)
	// An etcd without RangeStream serves every range in several pages.
	paginationKey := paramtable.Get().MetaStoreCfg.PaginationSize.Key
	paramtable.Get().Save(paginationKey, "7")
	defer paramtable.Get().Reset(paginationKey)

	want, err := listSegmentsInRanges(t, catalog, listed, 1)
	require.NoError(t, err)
	require.Len(t, want, numRecords)

	for _, ranges := range []int{2, 3, 8, 64} {
		got, err := listSegmentsInRanges(t, catalog, listed, ranges)
		require.NoError(t, err)
		require.Len(t, got, len(want))
		for i := range want {
			require.True(t, proto.Equal(want[i], got[i]), "ranges %d: record %d differs", ranges, i)
		}
	}
}

// A record that does not decode fails the ranged scan with the error one scan
// meets first.
func TestCatalogSegmentRangesFailOnTheFirstCorruptRecord(t *testing.T) {
	ctx := context.Background()
	catalog, cli, root, _ := newIndexRangeTestCatalog(t)
	listed, _ := writeRangeTestSegments(t, catalog)
	for _, collectionID := range []int64{2000, 77, 3} {
		_, err := cli.Put(ctx, fmt.Sprintf("%s/%s/%d/1/1", root, SegmentPrefix, collectionID), "corrupt")
		require.NoError(t, err)
	}

	_, want := listSegmentsInRanges(t, catalog, listed, 1)
	require.ErrorIs(t, want, merr.ErrDataIntegrity)
	require.Contains(t, want.Error(), "/2000/1/1", "one scan meets the first corrupt key first")
	for _, ranges := range []int{2, 8, 64} {
		_, err := listSegmentsInRanges(t, catalog, listed, ranges)
		require.Equal(t, want.Error(), err.Error(), "ranges %d", ranges)
		require.ErrorIs(t, err, merr.ErrDataIntegrity)
	}
}

func rangeTestSegment(id int64) *datapb.SegmentInfo {
	return &datapb.SegmentInfo{
		ID: id, CollectionID: 1, PartitionID: 2, InsertChannel: "ch-1", State: commonpb.SegmentState_Flushed, NumOfRows: 10,
		Binlogs:       []*datapb.FieldBinlog{{FieldID: 101, Binlogs: []*datapb.Binlog{{LogID: id*10 + 1, EntriesNum: 10, LogSize: 8}}}},
		Deltalogs:     []*datapb.FieldBinlog{{FieldID: 0, Binlogs: []*datapb.Binlog{{LogID: id*10 + 2, EntriesNum: 1}}}},
		Statslogs:     []*datapb.FieldBinlog{{FieldID: 100, Binlogs: []*datapb.Binlog{{LogID: id*10 + 3}}}},
		Bm25Statslogs: []*datapb.FieldBinlog{{FieldID: 102, Binlogs: []*datapb.Binlog{{LogID: id*10 + 4}}}},
	}
}
