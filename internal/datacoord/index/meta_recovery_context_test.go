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

package index

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/metastore"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type globalRecoveryContextCatalog struct {
	*catalogmocks.DataCoordCatalog
}

func (c *globalRecoveryContextCatalog) ListAllSegmentIndexes(ctx context.Context) ([]*model.SegmentIndex, error) {
	return c.ListSegmentIndexes(ctx, 1)
}

type rangedRecoveryContextCatalog struct {
	*globalRecoveryContextCatalog
}

func (c *rangedRecoveryContextCatalog) ListIndexesForRecovery(ctx context.Context, _ []int64) ([]*model.Index, error) {
	return c.ListIndexes(ctx)
}

func (c *rangedRecoveryContextCatalog) ListAllSegmentIndexesForRecovery(ctx context.Context, _ []int64) ([]*model.SegmentIndex, error) {
	return c.ListAllSegmentIndexes(ctx)
}

// Exercise every catalog scan implementation with reads blocked until their
// context is canceled. A lifetime context accidentally used for any scan would
// keep recovery blocked while the server remains alive.
func TestIndexMetaRecoveryContextCancellation(t *testing.T) {
	paramtable.Init()
	for _, mode := range []string{"per_collection", "global", "ranged"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			recoveryCtx, cancelRecovery := context.WithCancel(ctx)
			defer cancelRecovery()
			started := make(chan struct{}, 2)
			base := catalogmocks.NewDataCoordCatalog(t)
			base.EXPECT().ListIndexes(mock.Anything).RunAndReturn(func(ctx context.Context) ([]*model.Index, error) {
				started <- struct{}{}
				<-ctx.Done()
				return nil, ctx.Err()
			}).Once()
			base.EXPECT().ListSegmentIndexes(mock.Anything, int64(1)).RunAndReturn(func(ctx context.Context, _ int64) ([]*model.SegmentIndex, error) {
				started <- struct{}{}
				<-ctx.Done()
				return nil, ctx.Err()
			}).Once()
			var catalog metastore.DataCoordCatalog = base
			switch mode {
			case "global":
				catalog = &globalRecoveryContextCatalog{base}
			case "ranged":
				catalog = &rangedRecoveryContextCatalog{&globalRecoveryContextCatalog{base}}
			}
			recovered := conc.Go(func() (*IndexMeta, error) {
				return NewIndexMetaWithRecoveryContext(ctx, recoveryCtx, catalog, []int64{1})
			})
			// Ensure both independent scans are in flight before canceling.
			for range 2 {
				select {
				case <-started:
				case <-time.After(5 * time.Second):
					t.Fatal("catalog scan did not start")
				}
			}
			cancelRecovery()
			select {
			case <-recovered.Inner():
			case <-time.After(5 * time.Second):
				t.Fatal("catalog scans ignored recovery cancellation")
			}
			require.NoError(t, ctx.Err())
			indexMeta, err := recovered.Await()
			require.ErrorIs(t, err, context.Canceled)
			require.Nil(t, indexMeta)
		})
	}
}
