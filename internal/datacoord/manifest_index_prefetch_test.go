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
	"cmp"
	"context"
	"path"
	"slices"
	"sync"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

// Recovery installs from prefetched reads what it installs from its own reads.
// It reads again only the segments the prefetch did not read successfully at
// their installed manifest path, and ignores the prefetch when it read under
// another storage root.
func TestManifestIndexRecoveryUsesPrefetchedReads(t *testing.T) {
	old := Params.DataCoordCfg.SegmentIndexManifestLoadConcurrency.SwapTempValue("1")
	defer Params.DataCoordCfg.SegmentIndexManifestLoadConcurrency.SwapTempValue(old)
	ctx := context.Background()
	fixture := newManifestRecoveryFixture(t, 3, false, false)

	var mu sync.Mutex
	var reads []string
	mock := mockey.Mock(packed.SubmitManifestIndexInfosWithProperties).To(func(_ context.Context, _ *packed.ManifestIOContext, manifest string, _ *packed.ManifestReadProperties, complete func([]packed.ManifestIndexInfo, error)) error {
		read, ok := fixture.reads[manifest]
		if !ok {
			return merr.WrapErrServiceInternalMsg("unexpected manifest read %s", manifest)
		}
		mu.Lock()
		reads = append(reads, manifest)
		mu.Unlock()
		complete(read.entries, read.err)
		return nil
	}).Build()
	defer mock.UnPatch()
	takeReads := func() []string {
		mu.Lock()
		defer mu.Unlock()
		taken := reads
		reads = nil
		return taken
	}
	// The task-stats LRU keeps the latest installs, which only the install
	// order decides.
	normalized := func(state recoveredIndexState) recoveredIndexState {
		slices.Sort(state.LRUKeys)
		slices.SortFunc(state.LRUValues, func(a, b metricsinfo.IndexTaskStats) int { return cmp.Compare(a.BuildID, b.BuildID) })
		return state
	}

	want := fixture.boot(t)
	require.NoError(t, want.reloadSegmentIndexesFromManifests(ctx))
	wantState := normalized(snapshotRecoveredIndexState(t, want))
	manifests := takeReads()
	require.Len(t, manifests, len(fixture.reads))

	prefetch := func(m *meta, rootPath string, results []manifestIndexReadResult) {
		ch := make(chan manifestIndexReadResult, len(results))
		for _, result := range results {
			ch <- result
		}
		close(ch)
		m.manifestReadPrefetch = conc.Go(func() (*manifestReadPrefetch, error) {
			return &manifestReadPrefetch{rootPath: rootPath, results: ch}, nil
		})
	}
	read := func(m *meta, manifest string) manifestIndexReadResult {
		var record *datapb.SegmentInfo
		for _, segment := range fixture.segments {
			if segment.GetManifestPath() == manifest {
				record = proto.Clone(segment).(*datapb.SegmentInfo)
			}
		}
		require.NotNil(t, record)
		segment := &SegmentInfo{SegmentInfo: record}
		r := fixture.reads[manifest]
		result := manifestIndexReadResult{request: manifestIndexReadRequest{segment: segment}, entries: r.entries, err: r.err}
		if r.err == nil {
			result.indexes, result.invalid = projectManifestIndexes(m.chunkManager.RootPath(), segment, r.entries)
		}
		return result
	}

	t.Run("prefetched reads are installed, the rest read again", func(t *testing.T) {
		got := fixture.boot(t)
		half := len(manifests) / 2
		var results []manifestIndexReadResult
		var reread []string
		for _, manifest := range manifests[:half] {
			result := read(got, manifest)
			if result.err != nil {
				reread = append(reread, manifest)
			}
			results = append(results, result)
		}
		failed := read(got, manifests[half])
		failed.entries, failed.indexes, failed.err = nil, nil, merr.WrapErrServiceUnavailable("prefetch failed")
		stale := read(got, manifests[half+1])
		stale.request.segment = &SegmentInfo{SegmentInfo: proto.Clone(stale.request.segment.SegmentInfo).(*datapb.SegmentInfo)}
		stale.request.segment.ManifestPath = packed.MarshalManifestPath(path.Join(fixture.root, "insert_log",
			metautil.JoinIDPath(stale.request.segment.GetCollectionID(), recoveryPartitionID, stale.request.segment.GetID())), 2)
		results = append(results, failed, stale)
		reread = append(reread, manifests[half:]...)
		prefetch(got, got.chunkManager.RootPath(), results)

		require.NoError(t, got.reloadSegmentIndexesFromManifests(ctx))
		require.Equal(t, wantState, normalized(snapshotRecoveredIndexState(t, got)))
		require.ElementsMatch(t, reread, takeReads())
		require.Nil(t, got.manifestReadPrefetch)
	})

	t.Run("a prefetch under another root is ignored", func(t *testing.T) {
		got := fixture.boot(t)
		results := []manifestIndexReadResult{read(got, manifests[0])}
		prefetch(got, "another-root", results)

		require.NoError(t, got.reloadSegmentIndexesFromManifests(ctx))
		require.Equal(t, wantState, normalized(snapshotRecoveredIndexState(t, got)))
		require.ElementsMatch(t, manifests, takeReads())
	})
}

// The prefetch reads the manifest of every recovery candidate among the
// scanned records, delivers one final result for each, and leaves the records
// as scanned. It reads nothing on local storage.
func TestStartManifestReadPrefetch(t *testing.T) {
	ctx := context.Background()
	record := func(id int64, level datapb.SegmentLevel, hasIndex bool) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID: id, CollectionID: 1, PartitionID: recoveryPartitionID, State: commonpb.SegmentState_Flushed, Level: level,
			StorageVersion: storage.StorageV3, ManifestHasIndex: hasIndex,
			ManifestPath: packed.MarshalManifestPath(path.Join("root", "insert_log", metautil.JoinIDPath(1, recoveryPartitionID, id)), 1),
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
		prefetch := startManifestReadPrefetch(ctx, records)
		require.NotNil(t, prefetch)
		require.Equal(t, createStorageConfig().GetRootPath(), prefetch.rootPath)
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
		require.Nil(t, startManifestReadPrefetch(ctx, records))
	})
}

// Parallel workers installing every prefetched read leave the state the
// regular reads leave, but for which records the task-stats LRU keeps, which
// the install order decides.
func TestManifestIndexRecoveryAppliesPrefetchedReadsInParallel(t *testing.T) {
	ctx := context.Background()
	fixture := newManifestRecoveryFixture(t, 260, false, false)
	reads := mockManifestRecoveryReads(t, fixture.reads, true)

	want := fixture.boot(t)
	require.NoError(t, want.reloadSegmentIndexesFromManifests(ctx))
	wantState := snapshotRecoveredIndexState(t, want)
	require.Len(t, reads(), len(fixture.reads))

	got := fixture.boot(t)
	rootPath := got.chunkManager.RootPath()
	results := make(chan manifestIndexReadResult, len(fixture.segments))
	for _, record := range fixture.segments {
		segment := &SegmentInfo{SegmentInfo: proto.Clone(record).(*datapb.SegmentInfo)}
		read := fixture.reads[record.GetManifestPath()]
		result := manifestIndexReadResult{request: manifestIndexReadRequest{segment: segment}, err: read.err}
		if read.err == nil {
			result.indexes, result.invalid = projectManifestIndexes(rootPath, segment, read.entries)
		}
		results <- result
	}
	close(results)
	got.manifestReadPrefetch = conc.Go(func() (*manifestReadPrefetch, error) {
		return &manifestReadPrefetch{rootPath: rootPath, results: results}, nil
	})
	require.NoError(t, got.reloadSegmentIndexesFromManifests(ctx))
	gotState := snapshotRecoveredIndexState(t, got)

	// Only the dropped segment whose manifest is gone failed to read.
	require.Len(t, reads(), len(fixture.reads)+1)
	require.Len(t, gotState.LRUKeys, taskStatsLRUCapacity)
	wantState.LRUKeys, wantState.LRUValues, gotState.LRUKeys, gotState.LRUValues = nil, nil, nil, nil
	require.Equal(t, wantState, gotState)
}
