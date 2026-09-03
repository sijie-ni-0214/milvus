package datacoord

import (
	"context"
	"path"
	"strings"
	"sync/atomic"

	"github.com/cockroachdb/errors"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/kv/txn"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// segmentStore persists the segment records of meta. Its methods mirror the
// segment part of metastore.DataCoordCatalog and additionally return the store
// version of every written segment, which meta keeps next to the cached value.
//
// Two implementations exist:
//   - casSegmentStore: segments live in an OptimisticTxnPersist as full
//     SegmentInfo records and every write is a compare-and-swap against the
//     cached version, so writers of different segments never serialize on a
//     global lock. The non-segment actions of a composite write are encoded by
//     the catalog and committed in the same transaction. Used by the server.
//   - catalogSegmentStore: forwards to the catalog unchanged. Used when meta is
//     built without a persist (unit tests).
type segmentStore interface {
	// ListSegments lists every stored segment with its version. collectionIDs
	// bounds the scan only for stores that cannot list globally.
	ListSegments(ctx context.Context, collectionIDs []int64) ([]*datapb.SegmentInfo, []int64, error)
	// AddSegment persists a segment that must not exist yet.
	AddSegment(ctx context.Context, segment *datapb.SegmentInfo) (int64, error)
	// DropSegment removes a segment record.
	DropSegment(ctx context.Context, segment *datapb.SegmentInfo) error
	// DropSegments removes segment records in one transaction.
	DropSegments(ctx context.Context, segments []*datapb.SegmentInfo) error
	// AlterSegments persists new values of segments.
	AlterSegments(ctx context.Context, segments []*datapb.SegmentInfo, binlogs ...metastore.BinlogsIncrement) (map[int64]int64, error)
	// SaveDroppedSegmentsInBatch persists dropped segments.
	SaveDroppedSegmentsInBatch(ctx context.Context, segments []*datapb.SegmentInfo) (map[int64]int64, error)
	// Update applies a composite write with the catalog's atomicity contract.
	Update(ctx context.Context, actions ...metastore.UpdateAction) (map[int64]int64, error)
}

// ============================================================
// catalogSegmentStore
// ============================================================

type catalogSegmentStore struct {
	// catalog returns the catalog at call time, so tests that swap
	// meta.catalog after construction keep observing segment writes.
	catalog func() metastore.DataCoordCatalog
	version atomic.Int64
}

func newCatalogSegmentStore(catalog func() metastore.DataCoordCatalog) *catalogSegmentStore {
	return &catalogSegmentStore{catalog: catalog}
}

func (s *catalogSegmentStore) nextVersions(segments []*datapb.SegmentInfo) map[int64]int64 {
	versions := make(map[int64]int64, len(segments))
	for _, segment := range segments {
		versions[segment.GetID()] = s.version.Add(1)
	}
	return versions
}

func (s *catalogSegmentStore) ListSegments(ctx context.Context, collectionIDs []int64) ([]*datapb.SegmentInfo, []int64, error) {
	var segments []*datapb.SegmentInfo
	for _, collectionID := range collectionIDs {
		collectionSegments, err := s.catalog().ListSegments(ctx, collectionID)
		if err != nil {
			return nil, nil, err
		}
		segments = append(segments, collectionSegments...)
	}
	versions := make([]int64, len(segments))
	for i := range segments {
		versions[i] = s.version.Add(1)
	}
	return segments, versions, nil
}

func (s *catalogSegmentStore) AddSegment(ctx context.Context, segment *datapb.SegmentInfo) (int64, error) {
	if err := s.catalog().AddSegment(ctx, segment); err != nil {
		return 0, err
	}
	return s.version.Add(1), nil
}

func (s *catalogSegmentStore) DropSegment(ctx context.Context, segment *datapb.SegmentInfo) error {
	return s.catalog().DropSegment(ctx, segment)
}

// DropSegments drops the segments one by one: the catalog has no batch drop.
func (s *catalogSegmentStore) DropSegments(ctx context.Context, segments []*datapb.SegmentInfo) error {
	for _, segment := range segments {
		if err := s.catalog().DropSegment(ctx, segment); err != nil {
			return err
		}
	}
	return nil
}

func (s *catalogSegmentStore) AlterSegments(ctx context.Context, segments []*datapb.SegmentInfo, binlogs ...metastore.BinlogsIncrement) (map[int64]int64, error) {
	if err := s.catalog().AlterSegments(ctx, segments, binlogs...); err != nil {
		return nil, err
	}
	return s.nextVersions(segments), nil
}

func (s *catalogSegmentStore) SaveDroppedSegmentsInBatch(ctx context.Context, segments []*datapb.SegmentInfo) (map[int64]int64, error) {
	if err := s.catalog().SaveDroppedSegmentsInBatch(ctx, segments); err != nil {
		return nil, err
	}
	return s.nextVersions(segments), nil
}

func (s *catalogSegmentStore) Update(ctx context.Context, actions ...metastore.UpdateAction) (map[int64]int64, error) {
	if err := s.catalog().Update(ctx, actions...); err != nil {
		return nil, err
	}
	return s.nextVersions(segmentsOfActions(actions)), nil
}

func segmentsOfActions(actions []metastore.UpdateAction) []*datapb.SegmentInfo {
	var segments []*datapb.SegmentInfo
	for _, action := range actions {
		if entry, ok := action.Entry.(metastore.SegmentEntry); ok && entry.Segment != nil {
			segments = append(segments, entry.Segment)
		}
	}
	return segments
}

// ============================================================
// casSegmentStore
// ============================================================

// catalogUpdateEncoder encodes composite actions into ops without committing
// them (implemented by the kv DataCoord catalog).
type catalogUpdateEncoder interface {
	EncodeUpdate(ctx context.Context, actions ...metastore.UpdateAction) (*txn.Builder, error)
}

type casSegmentStore struct {
	persist OptimisticTxnPersist[string, *datapb.SegmentInfo]
	encoder catalogUpdateEncoder
	// rootPath is the MetaKv root of the catalog keys; segment keys and the
	// raw ops encoded by the catalog are joined under it.
	rootPath string
	// segments supplies the versions the writes compare against and receives
	// the refreshed values after a version conflict.
	segments *CachedSegmentsInfo
}

func newCASSegmentStore(persist OptimisticTxnPersist[string, *datapb.SegmentInfo], catalog metastore.DataCoordCatalog, rootPath string, segments *CachedSegmentsInfo) *casSegmentStore {
	encoder, _ := catalog.(catalogUpdateEncoder)
	return &casSegmentStore{
		persist:  persist,
		encoder:  encoder,
		rootPath: rootPath,
		segments: segments,
	}
}

func (s *casSegmentStore) join(key string) string {
	if s.rootPath == "" {
		return key
	}
	return path.Join(s.rootPath, key)
}

func (s *casSegmentStore) key(segment *datapb.SegmentInfo) string {
	return s.join(segmentKey(segment.GetCollectionID(), segment.GetPartitionID(), segment.GetID()))
}

// scanPrefix is the prefix of every segment record. join drops a trailing
// separator, which must stay: without it the scan would also cover sibling
// catalog prefixes such as statslog and stats-task.
func (s *casSegmentStore) scanPrefix() string {
	return strings.TrimSuffix(s.join(segmentMetaPrefix), "/") + "/"
}

// version returns the version the write of segment is built from: the cached
// version, or 0 (must not exist) for a segment the cache does not hold.
func (s *casSegmentStore) version(segmentID int64) int64 {
	version, _ := s.segments.Version(segmentID)
	return version
}

// ListSegments scans all segment records in one walk; the store is keyed by
// segment, so collectionIDs is not needed to bound the scan.
func (s *casSegmentStore) ListSegments(ctx context.Context, _ []int64) ([]*datapb.SegmentInfo, []int64, error) {
	_, values, versions, err := s.persist.Scan(ctx, s.scanPrefix())
	if err != nil {
		return nil, nil, err
	}
	return values, versions, nil
}

func (s *casSegmentStore) AddSegment(ctx context.Context, segment *datapb.SegmentInfo) (int64, error) {
	txn := s.persist.Txn(ctx)
	txn.PutIfVersion(s.key(segment), record(segment), s.version(segment.GetID()), false)
	versions, err := s.commit(ctx, txn, []*datapb.SegmentInfo{segment}, []int{0})
	if err != nil {
		return 0, err
	}
	return versions[segment.GetID()], nil
}

func (s *casSegmentStore) DropSegment(ctx context.Context, segment *datapb.SegmentInfo) error {
	return s.DropSegments(ctx, []*datapb.SegmentInfo{segment})
}

// DropSegments deletes every record against its cached version. A version
// conflict fails the whole transaction and refreshes the cached copies.
func (s *casSegmentStore) DropSegments(ctx context.Context, segments []*datapb.SegmentInfo) error {
	if len(segments) == 0 {
		return nil
	}
	txn := s.persist.Txn(ctx)
	positions := make([]int, 0, len(segments))
	for i, segment := range segments {
		txn.DeleteIfVersion(s.key(segment), s.version(segment.GetID()))
		positions = append(positions, i)
	}
	_, err := s.commit(ctx, txn, segments, positions)
	return err
}

// AlterSegments writes the segments as full records: the binlogs live inside
// the record, so the binlog increments of the catalog encoding are not needed.
func (s *casSegmentStore) AlterSegments(ctx context.Context, segments []*datapb.SegmentInfo, _ ...metastore.BinlogsIncrement) (map[int64]int64, error) {
	if len(segments) == 0 {
		return map[int64]int64{}, nil
	}
	txn := s.persist.Txn(ctx)
	positions := make([]int, 0, len(segments))
	for i, segment := range segments {
		txn.PutIfVersion(s.key(segment), record(segment), s.version(segment.GetID()), false)
		positions = append(positions, i)
	}
	return s.commit(ctx, txn, segments, positions)
}

func (s *casSegmentStore) SaveDroppedSegmentsInBatch(ctx context.Context, segments []*datapb.SegmentInfo) (map[int64]int64, error) {
	return s.AlterSegments(ctx, segments)
}

// Update commits a composite write in one versioned transaction. Segment
// entries become versioned puts of the full record; every other action is
// encoded by the catalog and composed as raw ops, in action order, keeping the
// catalog's commit markers so an over-limit write is flushed in the same
// order as catalog.Update's chunked fallback.
func (s *casSegmentStore) Update(ctx context.Context, actions ...metastore.UpdateAction) (map[int64]int64, error) {
	txn := s.persist.Txn(ctx)
	var segments []*datapb.SegmentInfo
	var positions []int
	numOps := 0
	for _, action := range actions {
		entry, ok := action.Entry.(metastore.SegmentEntry)
		if !ok {
			n, err := s.composeRaw(ctx, txn, action)
			if err != nil {
				return nil, err
			}
			numOps += n
			continue
		}
		if entry.Segment == nil {
			return nil, merr.WrapErrServiceInternalMsg("segment store: nil segment in UpdateAction")
		}
		if action.Type != metastore.ActionAdd && action.Type != metastore.ActionUpdate {
			return nil, merr.WrapErrServiceInternalMsg("segment store cannot apply action type %v to a segment", action.Type)
		}
		// A first Flush publication must never become visible before its
		// DataView: on the ordered fallback the record lands in the final
		// batch together with the commit-marked DataView.
		commit := publishesSegmentDataVersion(entry.Segment, actions)
		txn.PutIfVersion(s.key(entry.Segment), record(entry.Segment), s.version(entry.Segment.GetID()), commit)
		segments = append(segments, entry.Segment)
		positions = append(positions, numOps)
		numOps++
	}
	if containsSegmentIndexAction(actions) {
		// A segment index removal and the manifest pointer that publishes or
		// retracts its artifact must land together.
		txn.AtomicOnly()
	}
	return s.commit(ctx, txn, segments, positions)
}

// composeRaw appends the catalog encoding of a non-segment action to txn and
// returns the number of ops it added.
func (s *casSegmentStore) composeRaw(ctx context.Context, txn Txn[string, *datapb.SegmentInfo], action metastore.UpdateAction) (int, error) {
	if s.encoder == nil {
		return 0, merr.WrapErrServiceInternalMsg("segment store: catalog cannot encode %T", action.Entry)
	}
	b, err := s.encoder.EncodeUpdate(ctx, action)
	if err != nil {
		return 0, err
	}
	if b.Guarded() {
		return 0, merr.WrapErrServiceInternalMsg("segment store: conditional commit of %T is not supported", action.Entry)
	}
	ops := b.Ops()
	for _, op := range ops {
		if op.DeletePrefix {
			return 0, merr.WrapErrServiceInternalMsg("segment store: prefix removal of %T is not supported", action.Entry)
		}
		txn.Raw(RawOp{Key: s.join(op.Key), Value: op.Value, Delete: op.Delete, Commit: op.Commit})
	}
	return len(ops), nil
}

// commit runs txn and maps the results at positions (the op index of each
// segment write) to segment versions. On a version conflict the cached values
// of the written segments are refreshed from the store, so a rebuilt write
// starts from the persisted state.
func (s *casSegmentStore) commit(ctx context.Context, txn Txn[string, *datapb.SegmentInfo], segments []*datapb.SegmentInfo, positions []int) (map[int64]int64, error) {
	results, err := txn.Commit()
	if err != nil {
		if errors.Is(err, ErrVersionConflict) {
			s.refresh(ctx, segments)
		}
		return nil, err
	}
	versions := make(map[int64]int64, len(segments))
	for i, segment := range segments {
		versions[segment.GetID()] = results[positions[i]].Version
	}
	return versions, nil
}

// refresh reloads the persisted values of segments into the cache.
func (s *casSegmentStore) refresh(ctx context.Context, segments []*datapb.SegmentInfo) {
	keys := make([]string, 0, len(segments))
	byKey := make(map[string]*datapb.SegmentInfo, len(segments))
	for _, segment := range segments {
		key := s.key(segment)
		keys = append(keys, key)
		byKey[key] = segment
	}
	current, err := s.persist.Get(ctx, keys)
	if err != nil {
		mlog.Warn(ctx, "failed to refresh segments after a version conflict", mlog.Err(err))
		return
	}
	for _, key := range keys {
		segmentID := byKey[key].GetID()
		value, ok := current[key]
		if !ok {
			if cached := s.segments.GetSegment(segmentID); cached != nil {
				mlog.Warn(ctx, "segment is gone from the store, dropping the cached copy", mlog.FieldSegmentID(segmentID))
				s.segments.DropSegment(segmentID)
			}
			continue
		}
		s.segments.PublishSegment(segmentID, NewSegmentInfo(proto.Clone(value.Value).(*datapb.SegmentInfo)), value.Version)
	}
}

// record copies the segment for persistence: marshaling caches sizes inside
// the message, and the in-memory value (which may share sub-messages with
// requests and readers) must not be touched by the write, as with the catalog.
func record(segment *datapb.SegmentInfo) *datapb.SegmentInfo {
	return proto.Clone(segment).(*datapb.SegmentInfo)
}

// publishesSegmentDataVersion identifies a first Flush publication: the
// segment carries the DataVersion of a DataView written by the same action set.
func publishesSegmentDataVersion(segment *datapb.SegmentInfo, actions []metastore.UpdateAction) bool {
	if segment.GetSealedAtDataVersion() == nil {
		return false
	}
	for _, action := range actions {
		if entry, ok := action.Entry.(metastore.DataViewEntry); ok &&
			entry.DataView.GetCollectionId() == segment.GetCollectionID() &&
			proto.Equal(entry.DataView.GetDataVersion(), segment.GetSealedAtDataVersion()) {
			return true
		}
	}
	return false
}

func containsSegmentIndexAction(actions []metastore.UpdateAction) bool {
	for _, action := range actions {
		if _, ok := action.Entry.(metastore.SegmentIndexEntry); ok {
			return true
		}
	}
	return false
}
