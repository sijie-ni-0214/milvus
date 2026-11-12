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
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/datacoord/broker"
	"github.com/milvus-io/milvus/internal/datacoord/meta"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestMetaPrefetchReplacesRecoveryScans(t *testing.T) {
	ctx := context.Background()
	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().ListIndexes(mock.Anything).Return(nil, nil).Once()
	catalog.EXPECT().ListSegmentIndexes(mock.Anything, int64(1)).Return([]*model.SegmentIndex{{
		CollectionID: 1, PartitionID: 100, SegmentID: 10, IndexID: 1000, BuildID: 10000,
	}}, nil).Once()
	catalog.EXPECT().ListSegmentIndexes(mock.Anything, int64(2)).Return(nil, nil).Once()
	catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return([]*datapb.SegmentInfo{{ID: 10, CollectionID: 1, PartitionID: 100, State: commonpb.SegmentState_Flushed}}, nil).Once()
	catalog.EXPECT().ListSegments(mock.Anything, int64(2)).Return([]*datapb.SegmentInfo{{ID: 20, CollectionID: 2, PartitionID: 200, State: commonpb.SegmentState_Flushed}}, nil).Once()
	catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(map[string]*msgpb.MsgPosition{"ch-1": {Timestamp: 1000}}, nil).Once()
	catalog.EXPECT().ListSegmentChangeGroups(mock.Anything).Return(nil, nil).Once()
	catalog.EXPECT().ListAnalyzeTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListCompactionTask(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListCompactionTargets(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().ListPartitionStatsInfos(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListStatsTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListSnapshots(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListExternalCollectionRefreshJobs(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListExternalCollectionRefreshTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().AlterSegmentIndexes(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, _ []*model.SegmentIndex) error {
		return ctx.Err()
	}).Once()
	prefetched := startMetaPrefetch(ctx, catalog, []int64{2, 1})
	defer prefetched.close()
	indexMeta, err := prefetched.indexMeta.Await()
	require.NoError(t, err)
	dm := &domainMetas{}
	// RootCoord already supplies this shared collection view; no broker RPCs.
	store := meta.NewMetaStoreWithCollections(catalog, 1, 2)
	recovered, err := meta.NewMetaWithoutManifestIndexes(ctx, catalog, nil, store, broker.NewMockBroker(t), loadDomainMetas(catalog, nil, dm, prefetched.indexMeta), prefetched.recovery)
	require.NoError(t, err)
	defer recovered.CloseManifestCommitExecutor()
	require.Same(t, indexMeta, recovered.IndexCatalog)
	require.NotNil(t, recovered.GetSegment(ctx, 10))
	require.NotNil(t, recovered.GetSegment(ctx, 20))
	require.EqualValues(t, 1000, recovered.GetChannelCheckpoint("ch-1").GetTimestamp())
	// initMeta adopts this object before its deferred prefetch cleanup runs.
	prefetched.close()
	require.NoError(t, dm.indexMeta.UpdateVersion(10000, 7), "adopted metadata must remain writable after cleanup")
}
