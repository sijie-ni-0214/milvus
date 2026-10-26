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
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/metastore"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// recoveredSegmentsForTest includes shared compaction sources, growing rows,
// missing stats and duplicate IDs across collections. Catalog scan order now
// governs duplicates; the old CAS revision-order cases remain as input shapes.
func recoveredSegmentsForTest() []*datapb.SegmentInfo {
	var segments []*datapb.SegmentInfo
	add := func(segment *datapb.SegmentInfo, _ int64) {
		segments = append(segments, segment)
	}
	newSegment := func(id, collectionID int64, compactionFrom ...int64) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID:             id,
			CollectionID:   collectionID,
			PartitionID:    collectionID * 10,
			InsertChannel:  fmt.Sprintf("ch-%d", id%5),
			State:          commonpb.SegmentState_Flushed,
			Level:          datapb.SegmentLevel_L1,
			NumOfRows:      100,
			CompactionFrom: compactionFrom,
			Binlogs:        []*datapb.FieldBinlog{{FieldID: 101, Binlogs: []*datapb.Binlog{{LogID: id*10 + 1, EntriesNum: 100, LogSize: 64, MemorySize: 64}}}},
			Statslogs:      []*datapb.FieldBinlog{{FieldID: 100, Binlogs: []*datapb.Binlog{{LogID: id*10 + 2}}}},
			Deltalogs:      []*datapb.FieldBinlog{{FieldID: 0, Binlogs: []*datapb.Binlog{{LogID: id*10 + 3, EntriesNum: 1}}}},
		}
	}
	for group := range int64(200) {
		// A clustering compaction: five outputs of two sources.
		for output := range int64(5) {
			add(newSegment(10000+group*5+output, 1+group%7, group, group+1000), 100+group)
		}
		// A sort compaction: one output of one source.
		add(newSegment(20000+group, 1+group%3, group+2000), 100+group)
		growing := newSegment(30000+group, 1+group%5)
		growing.State = commonpb.SegmentState_Growing
		growing.Stats = nil
		add(growing, 50)
		stale := newSegment(40000+group, 2)
		stale.NumOfRows = 1
		add(stale, 50)
	}
	// Records of segment IDs already recorded.
	replacing := newSegment(10000, 9, 99999)
	add(replacing, 1000) // later catalog row replaces the segment
	stale := newSegment(10001, 9, 99998)
	add(stale, 1) // later catalog row replaces the segment
	tie := newSegment(20003, 3, 99997)
	add(tie, 103) // later catalog row replaces the segment
	again := newSegment(10002, 1)
	add(again, 5000) // replacement without compaction sources
	// A record of an ID whose first record comes later in the scan.
	add(newSegment(50000, 4, 3000), 10)
	add(newSegment(50000, 5, 3001), 20)
	return segments
}

func cloneSegmentProtos(segments []*datapb.SegmentInfo) []*datapb.SegmentInfo {
	clones := make([]*datapb.SegmentInfo, len(segments))
	for i, segment := range segments {
		clones[i] = proto.Clone(segment).(*datapb.SegmentInfo)
	}
	return clones
}

// installInOrder is the recovery install as one pass over the records in
// order, the reference of installRecoveredSegments.
func installInOrder(m *Meta, segments []*datapb.SegmentInfo) error {
	normalize := m.segmentLoadNormalizer()
	for i, segment := range segments {
		normalized, err := normalize(segment)
		if err != nil {
			return err
		}
		segments[i] = normalized
	}
	m.MetaStore.LoadSegments(segments)
	var compacted []*datapb.SegmentInfo
	for id, record := range m.MetaStore.GetAllSegments() {
		m.Segments.dcState.Insert(id, newDCSegmentState(record))
		compacted = append(compacted, record)
	}
	m.Segments.updateCompactionTo(nil, compacted)
	return nil
}

// dcStateShape is the DataCoord state of a segment without the times it was
// taken at.
type dcStateShape struct {
	allocations     int
	lastFlushTime   bool
	isCompacting    bool
	lastWrittenTime bool
}

type segmentCacheSnapshot struct {
	segments     map[int64]*datapb.SegmentInfo
	dcState      map[int64]dcStateShape
	compactionTo map[int64][]int64
	byCollection map[int64][]int64
	byChannel    map[string][]int64
}

func snapshotSegmentCache(s *SegmentsInfo) segmentCacheSnapshot {
	snapshot := segmentCacheSnapshot{
		segments:     make(map[int64]*datapb.SegmentInfo),
		dcState:      make(map[int64]dcStateShape),
		compactionTo: make(map[int64][]int64),
		byCollection: make(map[int64][]int64),
		byChannel:    make(map[string][]int64),
	}
	channels := make(map[string]struct{})
	for segmentID, segment := range s.store.GetAllSegments() {
		snapshot.segments[segmentID] = segment
		channels[segment.GetInsertChannel()] = struct{}{}
		snapshot.byCollection[segment.GetCollectionID()] = append(snapshot.byCollection[segment.GetCollectionID()], segmentID)
	}
	for _, ids := range snapshot.byCollection {
		slices.Sort(ids)
	}
	s.dcState.Range(func(segmentID int64, state *dcSegmentState) bool {
		snapshot.dcState[segmentID] = dcStateShape{
			allocations:     len(state.allocations),
			lastFlushTime:   !state.lastFlushTime.IsZero(),
			isCompacting:    state.isCompacting,
			lastWrittenTime: !state.lastWrittenTime.IsZero(),
		}
		return true
	})
	s.compactionTo.Range(func(from int64, to []int64) bool {
		snapshot.compactionTo[from] = slices.Clone(to)
		slices.Sort(snapshot.compactionTo[from])
		return true
	})
	for channel := range channels {
		for segmentID := range s.store.GetSegmentsByChannel(channel) {
			snapshot.byChannel[channel] = append(snapshot.byChannel[channel], segmentID)
		}
		slices.Sort(snapshot.byChannel[channel])
	}
	return snapshot
}

func requireSameSegmentCache(t *testing.T, want, got segmentCacheSnapshot) {
	t.Helper()
	require.Len(t, got.segments, len(want.segments))
	for id, segment := range want.segments {
		require.True(t, proto.Equal(segment, got.segments[id]), "segment %d", id)
	}
	require.Equal(t, want.dcState, got.dcState)
	require.Equal(t, want.compactionTo, got.compactionTo, "the same compaction edges after rebuilding the final master snapshot")
	require.Equal(t, want.byCollection, got.byCollection)
	require.Equal(t, want.byChannel, got.byChannel)
}

func newRecoveryTestMeta(t *testing.T, records []*datapb.SegmentInfo) *Meta {
	base := catalogmocks.NewDataCoordCatalog(t)
	base.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, nil).Maybe()
	catalog := &recoveryCatalog{DataCoordCatalog: base, records: records}
	store := metacache.NewMetaStore(catalog)
	return &Meta{Ctx: context.Background(), Catalog: catalog, MetaStore: store, Segments: NewSegmentsInfo(store)}
}

type recoveryCatalog struct {
	metastore.DataCoordCatalog
	records []*datapb.SegmentInfo
}

func (c *recoveryCatalog) ListAllSegments(context.Context) ([]*datapb.SegmentInfo, error) {
	return c.records, nil
}

func installRecovered(m *Meta) error {
	if err := m.MetaStore.LoadFromCatalog(context.Background(), []int64{1, 2, 3, 4, 5, 6, 7, 8, 9}, m.segmentLoadNormalizer()); err != nil {
		return err
	}
	m.Segments.RebuildFromStore()
	return nil
}

// Parallel recovery leaves the segment table and DataCoord state identical to
// the master serial loader and a serial rebuild of its final snapshot.
func TestInstallRecoveredSegmentsLikeOnePassInOrder(t *testing.T) {
	segments := recoveredSegmentsForTest()

	reference := newRecoveryTestMeta(t, nil)
	err := installInOrder(reference, cloneSegmentProtos(segments))
	require.NoError(t, err)
	want := snapshotSegmentCache(reference.Segments)
	require.Equal(t, []int64{10010, 10011, 10012, 10013, 10014}, want.compactionTo[2], "five outputs of one source")
	require.NotContains(t, want.compactionTo[0], int64(10000), "the replacement moves its compaction relation")
	require.NotContains(t, want.compactionTo[1000], int64(10000))
	require.Equal(t, []int64{10000}, want.compactionTo[99999])
	require.Equal(t, []int64{10001}, want.compactionTo[99998], "master catalog scan order is authoritative")
	require.Equal(t, []int64{20003}, want.compactionTo[99997])
	require.Equal(t, []int64{50000}, want.compactionTo[3001])

	for range 5 {
		parallel := newRecoveryTestMeta(t, cloneSegmentProtos(segments))
		err := installRecovered(parallel)
		require.NoError(t, err)
		requireSameSegmentCache(t, want, snapshotSegmentCache(parallel.Segments))
	}
}

// A record that cannot be normalized fails the install with the error a pass
// in order stops at.
func TestInstallRecoveredSegmentsFailsOnTheFirstInvalidRecord(t *testing.T) {
	setLocalManifestLegacyPrefix(t, "files")
	segments := recoveredSegmentsForTest()
	for _, i := range []int{900, 37, 38} {
		segments[i].StorageVersion = storage.StorageV3
		segments[i].ManifestPath = packed.MarshalManifestPath(fmt.Sprintf("files/insert_log/1/2/%d", i), 1)
	}
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))

	reference := newRecoveryTestMeta(t, nil)
	reference.ChunkManager = cm
	want := installInOrder(reference, cloneSegmentProtos(segments))
	require.ErrorIs(t, want, merr.ErrDataIntegrity)
	for range 5 {
		parallel := newRecoveryTestMeta(t, cloneSegmentProtos(segments))
		parallel.ChunkManager = cm
		err := installRecovered(parallel)
		require.Equal(t, want.Error(), err.Error())
	}
}

// The channel checkpoints are scanned while the segments are, and a segment
// error takes precedence over a checkpoint error.
func TestReloadFromKVScansCheckpointsAlongsideSegments(t *testing.T) {
	newMeta := func(catalog *catalogmocks.DataCoordCatalog) *Meta {
		return NewLocalManifestTestMeta(t, catalog, storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir())))
	}

	t.Run("concurrently", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		checkpointScanned := make(chan struct{})
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).RunAndReturn(func(context.Context, int64) ([]*datapb.SegmentInfo, error) {
			select {
			case <-checkpointScanned:
				return []*datapb.SegmentInfo{{ID: 3, CollectionID: 1, PartitionID: 2, State: commonpb.SegmentState_Flushed}}, nil
			case <-time.After(10 * time.Second):
				return nil, errors.New("the checkpoints were not scanned while the segments were")
			}
		}).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).RunAndReturn(func(context.Context) (map[string]*msgpb.MsgPosition, error) {
			close(checkpointScanned)
			return map[string]*msgpb.MsgPosition{"ch-1": {ChannelName: "ch-1", Timestamp: 7}}, nil
		}).Once()
		mt := newMeta(catalog)
		require.NoError(t, LoadAndNormalize(context.Background(), mt, []int64{1}))
		require.NotNil(t, mt.Segments.GetSegment(3))
		require.EqualValues(t, 7, mt.MetaStore.GetChannelCheckpoint("ch-1").GetTimestamp())
	})

	t.Run("segment error first", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		segmentErr := errors.New("segment scan failed")
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, segmentErr).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, errors.New("checkpoint scan failed")).Maybe()
		require.ErrorIs(t, LoadAndNormalize(context.Background(), newMeta(catalog), []int64{1}), segmentErr)
	})

	t.Run("checkpoint error", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		checkpointErr := errors.New("checkpoint scan failed")
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, nil).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, checkpointErr).Once()
		require.ErrorIs(t, LoadAndNormalize(context.Background(), newMeta(catalog), []int64{1}), checkpointErr)
	})
}
