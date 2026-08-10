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

package metacache

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/etcdpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type globalRecoveryCatalog struct {
	metastore.DataCoordCatalog
	segments      []*datapb.SegmentInfo
	scanErr       error
	checkpointErr error
	scans         int
}

func (c *globalRecoveryCatalog) ListAllSegments(ctx context.Context) ([]*datapb.SegmentInfo, error) {
	c.scans++
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return c.segments, c.scanErr
}

func (c *globalRecoveryCatalog) ListChannelCheckpoint(context.Context) (map[string]*msgpb.MsgPosition, error) {
	return map[string]*msgpb.MsgPosition{"channel": {Timestamp: 1}}, c.checkpointErr
}

func TestLoadFromCatalogGlobalScopeAndRetry(t *testing.T) {
	paramtable.Init()
	catalog := &globalRecoveryCatalog{segments: []*datapb.SegmentInfo{
		{ID: 10, CollectionID: 1},
		{ID: 20, CollectionID: 2},
		{ID: 30, CollectionID: 3},
	}}
	store := NewMetaStore(catalog)
	store.PutCollection(&CollectionInfo{ID: 1, State: etcdpb.CollectionState_CollectionCreated})
	store.PutCollection(&CollectionInfo{ID: 2, State: etcdpb.CollectionState_CollectionDropping})
	normalize := func(segment *datapb.SegmentInfo) (*datapb.SegmentInfo, error) {
		require.NotEqual(t, int64(3), segment.GetCollectionID(), "orphan metadata must not be normalized")
		copy := proto.Clone(segment).(*datapb.SegmentInfo)
		copy.NumOfRows = 42
		return copy, nil
	}
	require.NoError(t, store.LoadFromCatalog(context.Background(), store.GetCollectionIDs(), normalize))
	require.Equal(t, 1, catalog.scans)
	require.Len(t, store.GetAllSegments(), 2)
	for _, id := range []int64{10, 20} {
		segment, ok := store.GetSegment(id)
		require.True(t, ok)
		require.EqualValues(t, 42, segment.GetNumOfRows())
	}
	require.Zero(t, catalog.segments[0].GetNumOfRows(), "normalization must not mutate catalog records")
	require.Equal(t, "channel", store.GetChannelCheckpoint("channel").GetChannelName())

	// Recovery reuses the store. A later snapshot must not retain stale data.
	store.RemoveCollection(1)
	require.NoError(t, store.LoadFromCatalog(context.Background(), store.GetCollectionIDs(), normalize))
	require.Len(t, store.GetAllSegments(), 1)
	_, found := store.GetSegment(10)
	require.False(t, found)
	store.RemoveCollection(2)
	require.NoError(t, store.LoadFromCatalog(context.Background(), store.GetCollectionIDs(), normalize))
	require.Empty(t, store.GetAllSegments())
	require.Equal(t, 2, catalog.scans, "empty collection snapshots must not trigger a global scan")
}

func TestLoadFromCatalogGlobalFailureDoesNotPublish(t *testing.T) {
	paramtable.Init()
	for _, failure := range []string{"scan", "normalize", "checkpoint", "cancel"} {
		t.Run(failure, func(t *testing.T) {
			catalog := &globalRecoveryCatalog{segments: []*datapb.SegmentInfo{{ID: 10, CollectionID: 1}}}
			store := NewMetaStore(catalog)
			store.LoadSegments([]*datapb.SegmentInfo{{ID: 99, CollectionID: 9}})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var normalize SegmentNormalizer
			expected := merr.ErrServiceUnavailable
			switch failure {
			case "scan":
				catalog.scanErr = expected
			case "normalize":
				expected = merr.ErrDataIntegrity
				normalize = func(*datapb.SegmentInfo) (*datapb.SegmentInfo, error) { return nil, expected }
			case "checkpoint":
				catalog.checkpointErr = expected
			case "cancel":
				cancel()
			}
			err := store.LoadFromCatalog(ctx, []int64{1}, normalize)
			if failure == "cancel" {
				require.ErrorIs(t, err, context.Canceled)
			} else {
				require.ErrorIs(t, err, expected)
			}
			require.Empty(t, store.GetAllSegments())
			require.Empty(t, store.GetChannelCheckpoints())
		})
	}
}
