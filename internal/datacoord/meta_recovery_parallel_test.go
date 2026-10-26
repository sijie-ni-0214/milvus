package datacoord

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
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// recoveredSegmentsForTest returns records in scan order with their versions:
// segments of several collections and channels, growing ones, ones without
// Stats or with a stale row count, outputs that share the segments they were
// compacted from, and segment IDs recorded twice at a higher, a lower and an
// equal version, with other compaction sources and in another collection.
func recoveredSegmentsForTest() ([]*datapb.SegmentInfo, []int64) {
	var segments []*datapb.SegmentInfo
	var versions []int64
	add := func(segment *datapb.SegmentInfo, version int64) {
		segments = append(segments, segment)
		versions = append(versions, version)
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
	add(replacing, 1000) // newer: replaces, and moves the relations of 10000
	stale := newSegment(10001, 9, 99998)
	add(stale, 1) // older: rejected
	tie := newSegment(20003, 3, 99997)
	add(tie, 103) // as new as the recorded value: rejected
	again := newSegment(10002, 1)
	add(again, 5000) // newer, without sources
	// A record of an ID whose first record comes later in the scan.
	add(newSegment(50000, 4, 3000), 10)
	add(newSegment(50000, 5, 3001), 20)
	return segments, versions
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
func installInOrder(m *meta, segments []*datapb.SegmentInfo, versions []int64) (*segmentRecoveryMetricAggregates, error) {
	aggregates := newSegmentRecoveryMetricAggregates()
	normalize := m.segmentLoadNormalizer()
	for i, segment := range segments {
		normalized, err := normalize(segment)
		if err != nil {
			return nil, err
		}
		info := NewSegmentInfo(normalized)
		m.segments.LoadSegment(normalized.GetID(), info, versions[i])
		aggregates.Add(info)
	}
	return aggregates, nil
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
	versions     map[int64]int64
	dcState      map[int64]dcStateShape
	compactionTo map[int64][]int64
	byCollection map[int64][]int64
	byChannel    map[string][]int64
}

func snapshotSegmentCache(s *SegmentsInfo) segmentCacheSnapshot {
	snapshot := segmentCacheSnapshot{
		segments:     make(map[int64]*datapb.SegmentInfo),
		versions:     make(map[int64]int64),
		dcState:      make(map[int64]dcStateShape),
		compactionTo: make(map[int64][]int64),
		byCollection: make(map[int64][]int64),
		byChannel:    make(map[string][]int64),
	}
	channels := make(map[string]struct{})
	s.store.RangeSegments(func(segmentID int64, segment *datapb.SegmentInfo) bool {
		snapshot.segments[segmentID] = segment
		snapshot.versions[segmentID], _ = s.store.SegmentVersion(segmentID)
		channels[segment.GetInsertChannel()] = struct{}{}
		return true
	})
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
		snapshot.compactionTo[from] = to
		return true
	})
	s.store.RangeSegmentCollections(func(collectionID int64) bool {
		s.store.RangeCollectionSegmentIDs(collectionID, func(segmentID int64) bool {
			snapshot.byCollection[collectionID] = append(snapshot.byCollection[collectionID], segmentID)
			return true
		})
		slices.Sort(snapshot.byCollection[collectionID])
		return true
	})
	for channel := range channels {
		s.store.RangeChannelSegmentIDs(channel, func(segmentID int64) bool {
			snapshot.byChannel[channel] = append(snapshot.byChannel[channel], segmentID)
			return true
		})
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
	require.Equal(t, want.versions, got.versions)
	require.Equal(t, want.dcState, got.dcState)
	require.Equal(t, want.compactionTo, got.compactionTo, "the relations, in the order of their segments")
	require.Equal(t, want.byCollection, got.byCollection)
	require.Equal(t, want.byChannel, got.byChannel)
}

func newRecoveryTestMeta() *meta {
	return &meta{ctx: context.Background(), segments: NewSegmentsInfo()}
}

// The parallel install leaves the segment cache and the metric aggregates as
// installing the records one after another in scan order does.
func TestInstallRecoveredSegmentsLikeOnePassInOrder(t *testing.T) {
	segments, versions := recoveredSegmentsForTest()

	reference := newRecoveryTestMeta()
	wantAggregates, err := installInOrder(reference, cloneSegmentProtos(segments), versions)
	require.NoError(t, err)
	want := snapshotSegmentCache(reference.segments)
	require.Equal(t, []int64{10010, 10011, 10012, 10013, 10014}, want.compactionTo[2], "five outputs of one source")
	require.NotContains(t, want.compactionTo, int64(0), "the newer record of 10000 removed the relations of its old value")
	require.NotContains(t, want.compactionTo, int64(1000))
	require.Equal(t, []int64{10000}, want.compactionTo[99999])
	require.NotContains(t, want.compactionTo, int64(99998), "an older record does not apply")
	require.NotContains(t, want.compactionTo, int64(99997), "nor does one as new as the recorded value")
	require.Equal(t, []int64{50000}, want.compactionTo[3001])

	for range 5 {
		parallel := newRecoveryTestMeta()
		aggregates, err := parallel.installRecoveredSegments(context.Background(), cloneSegmentProtos(segments), versions)
		require.NoError(t, err)
		requireSameSegmentCache(t, want, snapshotSegmentCache(parallel.segments))
		require.Equal(t, wantAggregates, aggregates)
	}
}

// A record that cannot be normalized fails the install with the error a pass
// in order stops at.
func TestInstallRecoveredSegmentsFailsOnTheFirstInvalidRecord(t *testing.T) {
	setLocalManifestLegacyPrefix(t, "files")
	segments, versions := recoveredSegmentsForTest()
	for _, i := range []int{900, 37, 38} {
		segments[i].StorageVersion = storage.StorageV3
		segments[i].ManifestPath = packed.MarshalManifestPath(fmt.Sprintf("files/insert_log/1/2/%d", i), 1)
	}
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))

	reference := newRecoveryTestMeta()
	reference.chunkManager = cm
	_, want := installInOrder(reference, cloneSegmentProtos(segments), versions)
	require.ErrorIs(t, want, merr.ErrDataIntegrity)
	for range 5 {
		parallel := newRecoveryTestMeta()
		parallel.chunkManager = cm
		_, err := parallel.installRecoveredSegments(context.Background(), cloneSegmentProtos(segments), versions)
		require.Equal(t, want.Error(), err.Error())
	}
}

// The channel checkpoints are scanned while the segments are, and a segment
// error takes precedence over a checkpoint error.
func TestReloadFromKVScansCheckpointsAlongsideSegments(t *testing.T) {
	newMeta := func(catalog *catalogmocks.DataCoordCatalog) *meta {
		return newLocalManifestTestMeta(t, catalog, storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir())))
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
		require.NoError(t, mt.reloadFromKV(context.Background(), []int64{1}, nil))
		require.NotNil(t, mt.segments.GetSegment(3))
		require.EqualValues(t, 7, mt.metaStore.GetChannelCheckpoint("ch-1").GetTimestamp())
	})

	t.Run("segment error first", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		segmentErr := errors.New("segment scan failed")
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, segmentErr).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, errors.New("checkpoint scan failed")).Maybe()
		require.ErrorIs(t, newMeta(catalog).reloadFromKV(context.Background(), []int64{1}, nil), segmentErr)
	})

	t.Run("checkpoint error", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		checkpointErr := errors.New("checkpoint scan failed")
		catalog.EXPECT().ListSegments(mock.Anything, int64(1)).Return(nil, nil).Once()
		catalog.EXPECT().ListChannelCheckpoint(mock.Anything).Return(nil, checkpointErr).Once()
		require.ErrorIs(t, newMeta(catalog).reloadFromKV(context.Background(), []int64{1}, nil), checkpointErr)
	})
}
