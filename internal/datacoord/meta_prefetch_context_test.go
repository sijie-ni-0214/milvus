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
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/datacoord/index"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

// Exercise the production recovery lifecycle and the real IndexMeta write path.
// The catalog boundary honors caller cancellation, as TiKV does.
func TestMetaPrefetchIndexWriteContext(t *testing.T) {
	newCatalog := func(t *testing.T) *catalogmocks.DataCoordCatalog {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		catalog.EXPECT().ListIndexes(mock.Anything).Return(nil, nil).Once()
		catalog.EXPECT().ListSegmentIndexes(mock.Anything, int64(1)).Return([]*model.SegmentIndex{{
			CollectionID: 1, PartitionID: 2, SegmentID: 10, IndexID: 100, BuildID: 1000,
		}}, nil).Once()
		catalog.EXPECT().AlterSegmentIndexes(mock.Anything, mock.Anything).
			RunAndReturn(func(ctx context.Context, _ []*model.SegmentIndex) error {
				t.Logf("catalog write: context error = %v", ctx.Err())
				return ctx.Err()
			})
		return catalog
	}

	t.Run("prefetch_cleanup_preserves_runtime_writes", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		catalog := newCatalog(t)
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, nil).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, nil).Once()
		prefetched := startMetaPrefetch(ctx, catalog, []int64{1})
		defer prefetched.close()
		indexMeta, err := prefetched.indexMeta.Await()
		require.NoError(t, err)
		require.NoError(t, indexMeta.UpdateVersion(1000, 7), "write before cleanup")

		prefetched.close()
		require.NoError(t, ctx.Err(), "server context must remain alive")
		// initMeta publishes this same object, then runs the deferred close.
		// Repeated runtime writes must still succeed after recovery completes.
		for attempt := 1; attempt <= 3; attempt++ {
			require.NoError(t, indexMeta.UpdateVersion(1000, 8), "runtime write attempt %d after cleanup", attempt)
		}
		cancel()
		require.ErrorIs(t, indexMeta.UpdateVersion(1000, 8), context.Canceled,
			"server shutdown must still cancel writes after prefetch cleanup")
	})

	t.Run("direct_recovery_control", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		indexMeta, err := index.NewIndexMeta(ctx, newCatalog(t), []int64{1})
		require.NoError(t, err)
		_, cancelRecovery := context.WithCancel(ctx)
		cancelRecovery()
		require.NoError(t, indexMeta.UpdateVersion(1000, 7))
		cancel()
		require.ErrorIs(t, indexMeta.UpdateVersion(1000, 8), context.Canceled,
			"real server shutdown must still cancel writes")
	})
}

func TestMetaPrefetchCloseCancelsIndexRecovery(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().ListIndexes(mock.Anything).RunAndReturn(func(ctx context.Context) ([]*model.Index, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}).Once()
	catalog.EXPECT().ListSegmentIndexes(mock.Anything, int64(1)).Return(nil, nil).Once()
	catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, nil).Once()
	catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, nil).Once()
	prefetched := startMetaPrefetch(ctx, catalog, []int64{1})
	t.Cleanup(func() {
		cancel()
		prefetched.close()
	})
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("index recovery did not start")
	}
	closed := conc.Go(func() (struct{}, error) {
		prefetched.close()
		return struct{}{}, nil
	})
	select {
	case <-closed.Inner():
	case <-time.After(5 * time.Second):
		t.Fatal("prefetch cleanup did not cancel and join index recovery")
	}
	require.NoError(t, ctx.Err(), "canceling recovery must not stop the server")
	indexMeta, err := prefetched.indexMeta.Await()
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, indexMeta, "failed recovery must not publish partial metadata")
}
