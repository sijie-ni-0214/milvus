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
package meta

import (
	"context"
	"github.com/bytedance/mockey"
	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/stretchr/testify/require"
	"path"
	"testing"
)

// The prefetch reads the manifest of every recovery candidate among the
// scanned records, delivers one final result for each, and leaves the records
// as scanned. It reads nothing on local storage.
func TestStartManifestReadPrefetch(t *testing.T) {
	ctx := context.Background()
	record := func(id int64, level datapb.SegmentLevel, hasIndex bool) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID: id, CollectionID: 1, PartitionID: int64(7), State: commonpb.SegmentState_Flushed, Level: level,
			StorageVersion: storage.StorageV3, ManifestHasIndex: hasIndex,
			ManifestPath: packed.MarshalManifestPath(path.Join("root", "insert_log", metautil.JoinIDPath(1, int64(7), id)), 1),
		}
	}
	records := []*datapb.SegmentInfo{
		record(1, datapb.SegmentLevel_L1, true),
		record(2, datapb.SegmentLevel_L0, true),
		record(3, datapb.SegmentLevel_L1, false),
		record(4, datapb.SegmentLevel_L2, true),
	}
	serve := func(_ context.Context, manifest string, complete func([]packed.ManifestIndexInfo, error)) error {
		complete([]packed.ManifestIndexInfo{{IndexID: 1, BuildID: 1}}, nil)
		return nil
	}
	withProperties := mockey.Mock(packed.SubmitManifestIndexInfosWithProperties).To(func(ctx context.Context, _ *packed.ManifestIOContext, manifest string, _ *packed.ManifestReadProperties, complete func([]packed.ManifestIndexInfo, error)) error {
		return serve(ctx, manifest, complete)
	}).Build()
	defer withProperties.UnPatch()
	withConfig := mockey.Mock(packed.SubmitManifestIndexInfos).To(func(ctx context.Context, _ *packed.ManifestIOContext, manifest string, _ *indexpb.StorageConfig, complete func([]packed.ManifestIndexInfo, error)) error {
		return serve(ctx, manifest, complete)
	}).Build()
	defer withConfig.UnPatch()

	t.Run("reads the candidates", func(t *testing.T) {
		old := Params.CommonCfg.StorageType.SwapTempValue("minio")
		defer Params.CommonCfg.StorageType.SwapTempValue(old)
		prefetch := StartManifestReadPrefetch(ctx, records)
		require.NotNil(t, prefetch)
		require.Equal(t, CreateStorageConfig().GetRootPath(), prefetch.rootPath)
		var ids []int64
		for result := range prefetch.results {
			require.NoError(t, result.err)
			ids = append(ids, result.request.segment.GetID())
		}
		require.ElementsMatch(t, []int64{1, 4}, ids)
		for _, r := range records {
			require.Nil(t, r.GetStats())
		}
	})

	t.Run("reads nothing on local storage", func(t *testing.T) {
		old := Params.CommonCfg.StorageType.SwapTempValue("local")
		defer Params.CommonCfg.StorageType.SwapTempValue(old)
		require.Nil(t, StartManifestReadPrefetch(ctx, records))
	})
}
