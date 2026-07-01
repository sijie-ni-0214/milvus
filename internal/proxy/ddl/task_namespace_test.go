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

package ddl

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func newNamespaceTestCache(t *testing.T, schema *schemapb.CollectionSchema) *MockCache {
	cache := NewMockCache(t)
	cache.EXPECT().GetCollectionSchema(mock.Anything, mock.Anything, mock.Anything).
		Return(mustNewSchemaInfo(schema), nil).Maybe()
	return cache
}

func partitionModeNamespaceSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Name:            "coll",
		EnableNamespace: true,
		Properties: []*commonpb.KeyValuePair{
			{Key: common.NamespaceModeKey, Value: common.NamespaceModePartition},
		},
	}
}

func TestCreateNamespaceTaskPreExecuteModeValidation(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()

	tests := []struct {
		name    string
		schema  *schemapb.CollectionSchema
		wantErr bool
	}{
		{
			name: "namespace disabled",
			schema: &schemapb.CollectionSchema{
				Name:            "coll",
				EnableNamespace: false,
			},
			wantErr: true,
		},
		{
			name: "partition key mode",
			schema: &schemapb.CollectionSchema{
				Name:            "coll",
				EnableNamespace: true,
				Properties: []*commonpb.KeyValuePair{
					{Key: common.NamespaceModeKey, Value: common.NamespaceModePartitionKey},
				},
			},
			wantErr: true,
		},
		{
			name:   "partition mode",
			schema: partitionModeNamespaceSchema(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task := &CreateNamespaceTask{
				CreateNamespaceRequest: &milvuspb.CreateNamespaceRequest{
					CollectionName: "coll",
					NamespaceName:  "tenant_1",
				},
			}
			task.MetaCache = newNamespaceTestCache(t, tt.schema)
			err := task.PreExecute(ctx)
			if tt.wantErr {
				assert.ErrorIs(t, err, merr.ErrParameterInvalid)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestCreateNamespaceTaskPreExecuteRejectsInvalidName(t *testing.T) {
	paramtable.Init()
	task := &CreateNamespaceTask{
		CreateNamespaceRequest: &milvuspb.CreateNamespaceRequest{
			CollectionName: "coll",
			NamespaceName:  "bad name",
		},
	}
	task.MetaCache = newNamespaceTestCache(t, partitionModeNamespaceSchema())
	require.Error(t, task.PreExecute(context.Background()))
}

func TestDropNamespaceTaskPreExecuteRejectsLoadedNamespace(t *testing.T) {
	paramtable.Init()
	mockey.PatchConvey("drop namespace loaded rejection", t, func() {
		mockey.Mock(isCollectionLoaded).Return(true, nil).Build()
		mockey.Mock(isPartitionLoaded).Return(true, nil).Build()

		cache := newNamespaceTestCache(t, partitionModeNamespaceSchema())
		cache.EXPECT().GetCollectionID(mock.Anything, mock.Anything, mock.Anything).Return(int64(100), nil).Maybe()
		cache.EXPECT().GetPartitionID(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(int64(200), nil).Maybe()

		task := &DropNamespaceTask{
			DropNamespaceRequest: &milvuspb.DropNamespaceRequest{
				CollectionName: "coll",
				NamespaceName:  "tenant_1",
			},
		}
		task.MetaCache = cache
		require.ErrorIs(t, task.PreExecute(context.Background()), merr.ErrParameterInvalid)
	})
}

func TestGetNamespaceStatsTaskPreExecuteRejectsExact(t *testing.T) {
	task := &GetNamespaceStatsTask{
		GetNamespaceStatsRequest: &milvuspb.GetNamespaceStatsRequest{
			Exact: true,
		},
	}
	require.ErrorIs(t, task.PreExecute(context.Background()), merr.ErrParameterInvalid)
}

func TestGetNamespaceStatsTaskExecuteRejectsDefaultPartition(t *testing.T) {
	paramtable.Init()
	cache := NewMockCache(t)
	cache.EXPECT().GetCollectionID(mock.Anything, mock.Anything, mock.Anything).Return(int64(100), nil).Maybe()
	cache.EXPECT().GetPartitionID(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(int64(200), nil).Maybe()

	task := &GetNamespaceStatsTask{
		GetNamespaceStatsRequest: &milvuspb.GetNamespaceStatsRequest{
			CollectionName: "coll",
			NamespaceName:  paramtable.Get().CommonCfg.DefaultPartitionName.GetValue(),
		},
	}
	task.MetaCache = cache
	require.ErrorIs(t, task.Execute(context.Background()), merr.ErrNamespaceNotFound)
}

func TestNamespaceStatsFromKVs(t *testing.T) {
	stats := namespaceStatsFromKVs(context.Background(), []*commonpb.KeyValuePair{
		{Key: "row_count", Value: "42"},
		{Key: "other", Value: "ignored"},
	})
	require.Equal(t, int64(42), stats.GetApproxEntityCount())
	require.Equal(t, "approximate", stats.GetEntityCountType())
	require.Len(t, stats.GetStats(), 2)
}
