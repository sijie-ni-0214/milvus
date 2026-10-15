package datacoord

import (
	"context"
	"sync/atomic"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/metastore"
	datacoordkv "github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/internal/metastore/kv/txn"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// segmentStore persists the segment records of meta. Its methods mirror the
// segment part of metastore.DataCoordCatalog and additionally return the store
// version of every written segment, which meta keeps next to the cached value.
//
// Two implementations exist:
//   - casSegmentStore: segments live in an OptimisticTxnPersist in the
//     catalog's encoding (a record without binlogs plus per-field binlog KVs)
//     and every record write is a compare-and-swap against the cached
//     version, so writers of different segments never serialize on a global
//     lock. The non-segment actions of a composite write are encoded by the
//     catalog and committed in the same transaction. Used by the server.
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
	// codec keeps the persisted segments in the catalog encoding, so the
	// catalog (and an older DataCoord) reads what this store writes and the
	// other way round.
	codec datacoordkv.SegmentCodec
	// rootPath is the MetaKv root of the catalog keys; segment keys and the
	// raw ops encoded by the catalog are joined under it.
	rootPath string
	// segments supplies the versions the writes compare against and receives
	// the refreshed values after a version conflict.
	segments *CachedSegmentsInfo
	// inline holds the segments whose persisted record still carries its
	// binlogs: records written before this store kept the catalog encoding.
	// Their next write moves every binlog family out of the record.
	inline *typeutil.ConcurrentSet[int64]
}

func newCASSegmentStore(persist OptimisticTxnPersist[string, *datapb.SegmentInfo], catalog metastore.DataCoordCatalog, rootPath string, segments *CachedSegmentsInfo) *casSegmentStore {
	encoder, _ := catalog.(catalogUpdateEncoder)
	chunkManagerRootPath := ""
	if kc, ok := catalog.(*datacoordkv.Catalog); ok {
		chunkManagerRootPath = kc.ChunkManagerRootPath
	}
	return &casSegmentStore{
		persist: persist,
		encoder: encoder,
		// Every segment this store writes keeps its binlogs in binlog KVs (an
		// inline record moves them out on its first write), so the GC
		// compatibility write of a dropped segment never applies.
		codec:    datacoordkv.NewSegmentCodec(chunkManagerRootPath, nil),
		rootPath: rootPath,
		segments: segments,
		inline:   typeutil.NewConcurrentSet[int64](),
	}
}

// join joins a key under the root the way the catalog's MetaKv does, keeping
// the trailing separator of a prefix (segment 3 must not cover segment 30).
func (s *casSegmentStore) join(key string) string {
	return util.GetPath(s.rootPath, key)
}

// version returns the version the write of segment is built from: the cached
// version, or 0 (must not exist) for a segment the cache does not hold.
func (s *casSegmentStore) version(segmentID int64) int64 {
	version, _ := s.segments.Version(segmentID)
	return version
}

// ListSegments scans every segment record and binlog KV in one walk per
// prefix; the store is keyed by segment, so collectionIDs is not needed to
// bound the scans.
func (s *casSegmentStore) ListSegments(ctx context.Context, _ []int64) ([]*datapb.SegmentInfo, []int64, error) {
	var records []*datapb.SegmentInfo
	var versions []int64
	var binlogs []map[int64][]*datapb.FieldBinlog
	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		var err error
		_, records, versions, err = s.persist.Scan(gctx, s.join(datacoordkv.SegmentPrefix+"/"))
		return err
	})
	g.Go(func() error {
		var err error
		binlogs, err = s.loadBinlogs(gctx, datacoordkv.SegmentBinlogPrefixes())
		return err
	})
	if err := g.Wait(); err != nil {
		return nil, nil, err
	}

	inline := 0
	for _, record := range records {
		if datacoordkv.HasInlineBinlogs(record) {
			s.inline.Insert(record.GetID())
			inline++
		}
		s.codec.DecodeRecord(record)
	}
	if err := datacoordkv.FillSegmentBinlogs(records, binlogs); err != nil {
		return nil, nil, err
	}
	if inline > 0 {
		mlog.Info(ctx, "segment records still carry their binlogs; their next write moves them out", mlog.Int("numSegments", inline))
	}
	return records, versions, nil
}

// loadBinlogs reads the binlog KVs under prefixes (one prefix per family, in
// the order of datacoordkv.SegmentBinlogPrefixes) into one map per family,
// keyed by segment.
func (s *casSegmentStore) loadBinlogs(ctx context.Context, prefixes []string) ([]map[int64][]*datapb.FieldBinlog, error) {
	binlogs := make([]map[int64][]*datapb.FieldBinlog, len(prefixes))
	g, gctx := errgroup.WithContext(ctx)
	for i, prefix := range prefixes {
		family := make(map[int64][]*datapb.FieldBinlog)
		binlogs[i] = family
		g.Go(func() error {
			return s.persist.WalkRaw(gctx, s.join(prefix), func(key string, value []byte) error {
				segmentID, fieldBinlog, err := datacoordkv.DecodeFieldBinlog(key, value)
				if err != nil {
					return err
				}
				family[segmentID] = append(family[segmentID], fieldBinlog)
				return nil
			})
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return binlogs, nil
}

func (s *casSegmentStore) AddSegment(ctx context.Context, segment *datapb.SegmentInfo) (int64, error) {
	b := s.newTxn(ctx)
	if err := b.write(ctx, metastore.ActionAdd, metastore.SegmentEntry{Segment: segment}, false); err != nil {
		return 0, err
	}
	versions, err := s.commit(ctx, b)
	if err != nil {
		return 0, err
	}
	return versions[segment.GetID()], nil
}

func (s *casSegmentStore) DropSegment(ctx context.Context, segment *datapb.SegmentInfo) error {
	return s.DropSegments(ctx, []*datapb.SegmentInfo{segment})
}

// DropSegments deletes every record against its cached version, with its
// binlog KVs. A transaction holds whole segments, as many as the op limit
// takes, so no segment is dropped in part. A version conflict fails its
// transaction and the ones after it.
func (s *casSegmentStore) DropSegments(ctx context.Context, segments []*datapb.SegmentInfo) error {
	perTxn := len(segments)
	if limit := paramtable.Get().MetaStoreCfg.MaxEtcdTxnNum.GetAsInt(); limit > 0 {
		perTxn = max(1, limit/(1+len(datacoordkv.SegmentBinlogPrefixes())))
	}
	for _, chunk := range lo.Chunk(segments, max(1, perTxn)) {
		b := s.newTxn(ctx)
		for _, segment := range chunk {
			b.drop(segment)
		}
		if _, err := s.commit(ctx, b); err != nil {
			return err
		}
	}
	return nil
}

// AlterSegments writes the segments in the catalog's AlterSegments encoding:
// each record plus the binlog increments of its segment.
func (s *casSegmentStore) AlterSegments(ctx context.Context, segments []*datapb.SegmentInfo, binlogs ...metastore.BinlogsIncrement) (map[int64]int64, error) {
	if len(segments) == 0 && len(binlogs) == 0 {
		return map[int64]int64{}, nil
	}
	increments := lo.GroupBy(binlogs, func(b metastore.BinlogsIncrement) int64 { return b.Segment.GetID() })
	b := s.newTxn(ctx)
	for _, segment := range segments {
		entry := metastore.SegmentEntry{Segment: segment, Binlogs: increments[segment.GetID()], AlterEncoding: true}
		delete(increments, segment.GetID())
		if err := b.write(ctx, metastore.ActionUpdate, entry, false); err != nil {
			return nil, err
		}
	}
	// The catalog persists the increments of segments it does not rewrite as
	// bare binlog KVs.
	for _, segmentIncrements := range increments {
		ops, err := datacoordkv.EncodeBinlogIncrements(segmentIncrements)
		if err != nil {
			return nil, err
		}
		for _, op := range ops {
			b.raw(op)
		}
	}
	return s.commit(ctx, b)
}

// SaveDroppedSegmentsInBatch rewrites the records alone, like the catalog.
func (s *casSegmentStore) SaveDroppedSegmentsInBatch(ctx context.Context, segments []*datapb.SegmentInfo) (map[int64]int64, error) {
	if len(segments) == 0 {
		return map[int64]int64{}, nil
	}
	b := s.newTxn(ctx)
	for _, segment := range segments {
		if err := b.write(ctx, metastore.ActionUpdate, metastore.SegmentEntry{Segment: segment}, false); err != nil {
			return nil, err
		}
	}
	return s.commit(ctx, b)
}

// Update commits a composite write in one versioned transaction. Segment
// entries become versioned puts of their records with the binlog KVs of the
// catalog encoding; every other action is encoded by the catalog and composed
// as raw ops, in action order, keeping the catalog's commit markers so an
// over-limit write is flushed in the same order as catalog.Update's chunked
// fallback.
func (s *casSegmentStore) Update(ctx context.Context, actions ...metastore.UpdateAction) (map[int64]int64, error) {
	b := s.newTxn(ctx)
	for _, action := range actions {
		entry, ok := action.Entry.(metastore.SegmentEntry)
		if !ok {
			if err := s.composeRaw(ctx, b, action); err != nil {
				return nil, err
			}
			continue
		}
		// A first Flush publication must never become visible before its
		// DataView: on the ordered fallback the record lands in the final
		// batch together with the commit-marked DataView.
		if err := b.write(ctx, action.Type, entry, publishesSegmentDataVersion(entry.Segment, actions)); err != nil {
			return nil, err
		}
	}
	if containsSegmentIndexAction(actions) {
		// A segment index removal and the manifest pointer that publishes or
		// retracts its artifact must land together.
		b.txn.AtomicOnly()
	}
	return s.commit(ctx, b)
}

// composeRaw appends the catalog encoding of a non-segment action to b.
func (s *casSegmentStore) composeRaw(ctx context.Context, b *segmentTxn, action metastore.UpdateAction) error {
	if s.encoder == nil {
		return merr.WrapErrServiceInternalMsg("segment store: catalog cannot encode %T", action.Entry)
	}
	builder, err := s.encoder.EncodeUpdate(ctx, action)
	if err != nil {
		return err
	}
	if builder.Guarded() {
		return merr.WrapErrServiceInternalMsg("segment store: conditional commit of %T is not supported", action.Entry)
	}
	for _, op := range builder.Ops() {
		b.raw(op)
	}
	return nil
}

// segmentTxn stages segment writes into a store transaction and remembers the
// op index of every segment record.
type segmentTxn struct {
	store     *casSegmentStore
	txn       Txn[string, *datapb.SegmentInfo]
	numOps    int
	segments  []*datapb.SegmentInfo
	positions []int
}

func (s *casSegmentStore) newTxn(ctx context.Context) *segmentTxn {
	return &segmentTxn{store: s, txn: s.persist.Txn(ctx)}
}

func (b *segmentTxn) raw(op txn.Op) {
	b.txn.Raw(RawOp{Key: b.store.join(op.Key), Value: op.Value, Delete: op.Delete, DeletePrefix: op.DeletePrefix, Commit: op.Commit})
	b.numOps++
}

func (b *segmentTxn) track(segment *datapb.SegmentInfo) {
	b.segments = append(b.segments, segment)
	b.positions = append(b.positions, b.numOps)
	b.numOps++
}

// write stages a segment action in the catalog encoding. The binlog KVs go
// before the versioned record, so on the ordered fallback a record never
// lands ahead of its binlogs.
//
// A segment whose persisted record still carries its binlogs gets every
// binlog family written out (keeping the caller's field removals): the record
// this write stores no longer holds them.
func (b *segmentTxn) write(ctx context.Context, t metastore.ActionType, e metastore.SegmentEntry, commit bool) error {
	if e.Segment != nil && b.store.inline.Contain(e.Segment.GetID()) {
		all := metastore.BinlogsIncrement{Segment: e.Segment}
		for _, increment := range e.Binlogs {
			all.DroppedBinlogFieldIDs = append(all.DroppedBinlogFieldIDs, increment.DroppedBinlogFieldIDs...)
		}
		e.Binlogs = []metastore.BinlogsIncrement{all}
		e.AlterEncoding = true
	}
	w, err := b.store.codec.EncodeAction(ctx, t, e)
	if err != nil {
		return err
	}
	for _, op := range w.Ops {
		b.raw(op)
	}
	b.txn.PutIfVersion(b.store.join(w.RecordKey), w.Record, b.store.version(e.Segment.GetID()), commit)
	b.track(e.Segment)
	return nil
}

// drop stages the removal of a segment: its versioned record first, then its
// binlog KVs, so on the ordered fallback the record never outlives them.
func (b *segmentTxn) drop(segment *datapb.SegmentInfo) {
	b.txn.DeleteIfVersion(b.store.join(datacoordkv.SegmentRecordKey(segment)), b.store.version(segment.GetID()))
	b.track(segment)
	for _, prefix := range datacoordkv.SegmentBinlogPrefixesOf(segment.GetCollectionID(), segment.GetPartitionID(), segment.GetID()) {
		b.raw(txn.Op{Key: prefix, DeletePrefix: true})
	}
}

// commit runs the transaction of b and maps the result of each segment record
// to the segment's version. On a version conflict the cached values of the
// written segments are refreshed from the store, so a rebuilt write starts
// from the persisted state. Any other failure leaves the cache alone: a resend
// compares against the versions it was built from and so recognizes a write
// that landed.
func (s *casSegmentStore) commit(ctx context.Context, b *segmentTxn) (map[int64]int64, error) {
	results, err := b.txn.Commit()
	if err != nil {
		if errors.Is(err, ErrVersionConflict) {
			s.refresh(ctx, b.segments)
		}
		return nil, err
	}
	versions := make(map[int64]int64, len(b.segments))
	for i, segment := range b.segments {
		versions[segment.GetID()] = results[b.positions[i]].Version
		s.inline.Remove(segment.GetID())
	}
	return versions, nil
}

// refresh reloads the persisted values of segments into the cache.
func (s *casSegmentStore) refresh(ctx context.Context, segments []*datapb.SegmentInfo) {
	keys := make([]string, 0, len(segments))
	byKey := make(map[string]*datapb.SegmentInfo, len(segments))
	for _, segment := range segments {
		key := s.join(datacoordkv.SegmentRecordKey(segment))
		keys = append(keys, key)
		byKey[key] = segment
	}
	current, err := s.persist.Get(ctx, keys)
	if err != nil {
		mlog.Warn(ctx, "failed to refresh segments after a version conflict", mlog.Err(err))
		return
	}
	for _, key := range keys {
		segment := byKey[key]
		value, ok := current[key]
		if !ok {
			if cached := s.segments.GetSegment(segment.GetID()); cached != nil {
				mlog.Warn(ctx, "segment is gone from the store, dropping the cached copy", mlog.FieldSegmentID(segment.GetID()))
				s.segments.DropSegment(segment.GetID())
			}
			s.inline.Remove(segment.GetID())
			continue
		}
		record := value.Value
		if datacoordkv.HasInlineBinlogs(record) {
			s.inline.Insert(segment.GetID())
		} else {
			s.inline.Remove(segment.GetID())
		}
		binlogs, err := s.loadBinlogs(ctx, datacoordkv.SegmentBinlogPrefixesOf(segment.GetCollectionID(), segment.GetPartitionID(), segment.GetID()))
		if err == nil {
			s.codec.DecodeRecord(record)
			err = datacoordkv.FillSegmentBinlogs([]*datapb.SegmentInfo{record}, binlogs)
		}
		if err != nil {
			mlog.Warn(ctx, "failed to refresh a segment after a version conflict", mlog.FieldSegmentID(segment.GetID()), mlog.Err(err))
			continue
		}
		s.segments.PublishSegment(segment.GetID(), NewSegmentInfo(record), value.Version)
	}
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
