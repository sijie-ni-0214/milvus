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

package datacoord

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/model"
	pkgutil "github.com/milvus-io/milvus/pkg/v3/util"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
)

// newIndexRangeTestCatalog returns a catalog over the configured etcd that
// holds field and segment indexes of the returned collections, one of them
// holding most segment indexes, and of orphan collections before, between and
// after them, plus an index record keyed exactly at the first key of a
// collection.
func newIndexRangeTestCatalog(t *testing.T) (*Catalog, *clientv3.Client, string, []int64) {
	t.Helper()
	ctx := context.Background()
	client, err := etcd.GetEtcdClient(
		Params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		Params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		Params.EtcdCfg.Endpoints.GetAsStrings(),
		Params.EtcdCfg.EtcdTLSCert.GetValue(),
		Params.EtcdCfg.EtcdTLSKey.GetValue(),
		Params.EtcdCfg.EtcdTLSCACert.GetValue(),
		Params.EtcdCfg.EtcdTLSMinVersion.GetValue())
	require.NoError(t, err)
	root := fmt.Sprintf("unittest/index-ranges/%s/%d", t.Name(), time.Now().UnixNano())
	t.Cleanup(func() {
		_, _ = client.Delete(context.Background(), root, clientv3.WithPrefix())
		_ = client.Close()
	})
	catalog := NewCatalog(etcdkv.NewEtcdKV(client, root), "", root)

	listed := []int64{3, 30, 300, 31, 4, 449412345678901234, 7, 77, 8000}
	for i := range 30 {
		listed = append(listed, 100000+int64(i)*7919)
	}
	buildID := int64(1)
	addIndexes := func(collectionID int64, segments int) {
		for indexID := range int64(2) {
			require.NoError(t, catalog.CreateIndex(ctx, &model.Index{CollectionID: collectionID, IndexID: 10 + indexID, IndexName: "idx"}))
		}
		indexes := make([]*model.SegmentIndex, 0, 2*segments)
		for segment := range int64(segments) {
			for indexID := range int64(2) {
				indexes = append(indexes, &model.SegmentIndex{
					CollectionID: collectionID, PartitionID: segment % 2, SegmentID: collectionID*1000 + segment,
					IndexID: 10 + indexID, BuildID: buildID, IndexState: commonpb.IndexState_Finished,
					IndexSerializedSize: uint64(buildID),
				})
				buildID++
			}
		}
		for start := 0; start < len(indexes); start += 32 {
			require.NoError(t, catalog.AlterSegmentIndexes(ctx, indexes[start:min(start+32, len(indexes))]))
		}
	}
	for i, collectionID := range listed {
		segments := 1 + i%3
		if collectionID == 30 {
			segments = 150
		}
		addIndexes(collectionID, segments)
	}
	for _, orphan := range []int64{1, 2, 35, 99999999} {
		addIndexes(orphan, 2)
	}
	atBound, err := proto.Marshal(model.MarshalSegmentIndexModel(&model.SegmentIndex{CollectionID: 31, SegmentID: 31, BuildID: buildID}))
	require.NoError(t, err)
	require.NoError(t, catalog.MetaKv.Save(ctx, pkgutil.SegmentIndexPrefix+"/31/", string(atBound)))
	return catalog, client, root, listed
}

func withParallelScanRanges(t *testing.T, ranges int, fn func()) {
	t.Helper()
	key := Params.MetaStoreCfg.ParallelScanRanges.Key
	Params.Save(key, strconv.Itoa(ranges))
	defer Params.Reset(key)
	fn()
}

// The index prefixes read as parallel key ranges list what one walk of each
// lists, in the same order.
func TestCatalogListIndexesForRecoveryInRanges(t *testing.T) {
	ctx := context.Background()
	catalog, _, _, listed := newIndexRangeTestCatalog(t)
	// An etcd without RangeStream serves every range in several pages.
	catalog.paginationSize = 7
	wantIndexes, err := catalog.ListIndexes(ctx)
	require.NoError(t, err)
	wantSegmentIndexes, err := catalog.ListAllSegmentIndexes(ctx)
	require.NoError(t, err)
	require.Len(t, wantIndexes, 2*(len(listed)+4))
	require.Greater(t, len(wantSegmentIndexes), 300)

	for _, ranges := range []int{1, 2, 3, 8, 64} {
		withParallelScanRanges(t, ranges, func() {
			indexes, err := catalog.ListIndexesForRecovery(ctx, listed)
			require.NoError(t, err)
			require.Equal(t, wantIndexes, indexes, "ranges %d", ranges)
			segmentIndexes, err := catalog.ListAllSegmentIndexesForRecovery(ctx, listed)
			require.NoError(t, err)
			require.Equal(t, wantSegmentIndexes, segmentIndexes, "ranges %d", ranges)
		})
	}
}

// An index record that does not decode fails the ranged listing with the
// error one walk meets first.
func TestCatalogListIndexesForRecoveryFailsOnTheFirstCorruptRecord(t *testing.T) {
	ctx := context.Background()
	catalog, client, root, listed := newIndexRangeTestCatalog(t)
	for _, key := range []string{
		BuildIndexKey(8000, 99), BuildIndexKey(4, 99), BuildIndexKey(30, 99),
		BuildSegmentIndexKey(8000, 1, 1, 1), BuildSegmentIndexKey(4, 1, 1, 1), BuildSegmentIndexKey(30, 1, 1, 1),
	} {
		_, err := client.Put(ctx, root+"/"+key, "corrupt-"+key)
		require.NoError(t, err)
	}
	_, wantIndexErr := catalog.ListIndexes(ctx)
	require.Error(t, wantIndexErr)
	_, wantSegmentIndexErr := catalog.ListAllSegmentIndexes(ctx)
	require.Error(t, wantSegmentIndexErr)
	for _, ranges := range []int{2, 8, 64} {
		withParallelScanRanges(t, ranges, func() {
			_, err := catalog.ListIndexesForRecovery(ctx, listed)
			require.Equal(t, wantIndexErr.Error(), err.Error(), "ranges %d", ranges)
			_, err = catalog.ListAllSegmentIndexesForRecovery(ctx, listed)
			require.Equal(t, wantSegmentIndexErr.Error(), err.Error(), "ranges %d", ranges)
		})
	}
}
