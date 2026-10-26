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
	"cmp"
	"context"
	"path"
	"slices"
	"sync"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/datacoord/meta"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
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
	require.NoError(t, want.ReloadSegmentIndexesFromManifests(ctx))
	wantState := normalized(snapshotRecoveredIndexState(t, want))
	manifests := takeReads()
	require.Len(t, manifests, len(fixture.reads))

	prefetch := func(m *meta.Meta, rootPath string, results []meta.ManifestReadResultForTest) {
		m.SetManifestReadPrefetchForTest(rootPath, results)
	}
	read := func(m *meta.Meta, manifest string) meta.ManifestReadResultForTest {
		var record *datapb.SegmentInfo
		for _, segment := range fixture.segments {
			if segment.GetManifestPath() == manifest {
				record = proto.Clone(segment).(*datapb.SegmentInfo)
			}
		}
		require.NotNil(t, record)
		segment := &meta.SegmentInfo{SegmentInfo: record}
		r := fixture.reads[manifest]
		result := meta.ManifestReadResultForTest{Segment: segment, Entries: r.entries, Err: r.err}
		if r.err == nil {
			result.Indexes, result.Invalid = meta.ProjectManifestIndexesForTest(m.ChunkManager.RootPath(), segment, r.entries)
		}
		return result
	}

	t.Run("prefetched reads are installed, the rest read again", func(t *testing.T) {
		got := fixture.boot(t)
		half := len(manifests) / 2
		var results []meta.ManifestReadResultForTest
		var reread []string
		for _, manifest := range manifests[:half] {
			result := read(got, manifest)
			if result.Err != nil {
				reread = append(reread, manifest)
			}
			results = append(results, result)
		}
		failed := read(got, manifests[half])
		failed.Entries, failed.Indexes, failed.Err = nil, nil, merr.WrapErrServiceUnavailable("prefetch failed")
		stale := read(got, manifests[half+1])
		stale.Segment = &meta.SegmentInfo{SegmentInfo: proto.Clone(stale.Segment.SegmentInfo).(*datapb.SegmentInfo)}
		stale.Segment.ManifestPath = packed.MarshalManifestPath(path.Join(fixture.root, "insert_log",
			metautil.JoinIDPath(stale.Segment.GetCollectionID(), recoveryPartitionID, stale.Segment.GetID())), 2)
		results = append(results, failed, stale)
		reread = append(reread, manifests[half:]...)
		prefetch(got, got.ChunkManager.RootPath(), results)

		require.NoError(t, got.ReloadSegmentIndexesFromManifests(ctx))
		require.Equal(t, wantState, normalized(snapshotRecoveredIndexState(t, got)))
		require.ElementsMatch(t, reread, takeReads())
		require.False(t, got.HasManifestReadPrefetchForTest())
	})

	t.Run("a prefetch under another root is ignored", func(t *testing.T) {
		got := fixture.boot(t)
		results := []meta.ManifestReadResultForTest{read(got, manifests[0])}
		prefetch(got, "another-root", results)

		require.NoError(t, got.ReloadSegmentIndexesFromManifests(ctx))
		require.Equal(t, wantState, normalized(snapshotRecoveredIndexState(t, got)))
		require.ElementsMatch(t, manifests, takeReads())
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
	require.NoError(t, want.ReloadSegmentIndexesFromManifests(ctx))
	wantState := snapshotRecoveredIndexState(t, want)
	require.Len(t, reads(), len(fixture.reads))

	got := fixture.boot(t)
	rootPath := got.ChunkManager.RootPath()
	results := make([]meta.ManifestReadResultForTest, 0, len(fixture.segments))
	for _, record := range fixture.segments {
		segment := &meta.SegmentInfo{SegmentInfo: proto.Clone(record).(*datapb.SegmentInfo)}
		read := fixture.reads[record.GetManifestPath()]
		result := meta.ManifestReadResultForTest{Segment: segment, Err: read.err}
		if read.err == nil {
			result.Indexes, result.Invalid = meta.ProjectManifestIndexesForTest(rootPath, segment, read.entries)
		}
		results = append(results, result)
	}
	got.SetManifestReadPrefetchForTest(rootPath, results)
	require.NoError(t, got.ReloadSegmentIndexesFromManifests(ctx))
	gotState := snapshotRecoveredIndexState(t, got)

	// Only the dropped segment whose manifest is gone failed to read.
	require.Len(t, reads(), len(fixture.reads)+1)
	require.Len(t, gotState.LRUKeys, taskStatsLRUCapacity)
	wantState.LRUKeys, wantState.LRUValues, gotState.LRUKeys, gotState.LRUValues = nil, nil, nil, nil
	require.Equal(t, wantState, gotState)
}
