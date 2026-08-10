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

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type globalIndexRecoveryCatalog struct {
	metastore.DataCoordCatalog
	indexes []*model.SegmentIndex
	err     error
	scans   int
}

func (c *globalIndexRecoveryCatalog) ListAllSegmentIndexes(context.Context) ([]*model.SegmentIndex, error) {
	c.scans++
	return c.indexes, c.err
}

func (c *globalIndexRecoveryCatalog) ListIndexes(context.Context) ([]*model.Index, error) {
	return nil, nil
}

func TestIndexMetaGlobalRecoveryScope(t *testing.T) {
	paramtable.Init()
	catalog := &globalIndexRecoveryCatalog{indexes: []*model.SegmentIndex{
		{CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 1000},
		{CollectionID: 2, SegmentID: 20, IndexID: 200, BuildID: 2000},
		{CollectionID: 3, SegmentID: 30, IndexID: 300, BuildID: 3000},
	}}
	meta, err := NewIndexMeta(context.Background(), catalog, []int64{1, 2})
	require.NoError(t, err)
	require.Equal(t, 1, catalog.scans)
	for _, buildID := range []int64{1000, 2000} {
		_, ok := meta.SegmentBuildInfo.Get(buildID)
		require.True(t, ok)
	}
	_, found := meta.SegmentBuildInfo.Get(3000)
	require.False(t, found, "orphan collection indexes must not enter recovery state")
	_, err = NewIndexMeta(context.Background(), catalog, nil)
	require.NoError(t, err)
	require.Equal(t, 1, catalog.scans)
}

func TestIndexMetaGlobalRecoveryFailure(t *testing.T) {
	paramtable.Init()
	catalog := &globalIndexRecoveryCatalog{err: merr.ErrServiceUnavailable}
	meta, err := NewIndexMeta(context.Background(), catalog, []int64{1})
	require.ErrorIs(t, err, merr.ErrServiceUnavailable)
	require.Nil(t, meta)
}
