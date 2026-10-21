package datacoord

import (
	"context"
	"slices"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// SegmentsInfo is the segment metadata store of meta.
type SegmentsInfo = CachedSegmentsInfo

// NewSegmentsInfo creates an empty segment metadata store. With a store, the
// cache serves the store's segment reads (see AttachStore).
func NewSegmentsInfo(store ...metacache.MetaStore) *SegmentsInfo {
	s := NewCachedSegmentsInfo()
	if len(store) > 0 && store[0] != nil {
		s.AttachStore(store[0])
	}
	return s
}

// CachedSegmentsInfo is a concurrent, version-tracked segment metadata store
// backed by Cache. Reads are lock-free. Every live entry carries the store
// version of the persisted value it holds, which versioned writes compare
// against. It provides secondary indexes for fast lookups by collection,
// channel, and compaction relationships. Collection, channel and
// compaction_from of a segment never change, so the indexes only grow on
// insert and shrink on drop, and a lookup never misses a live segment.
//
// Every accepted change of a quota input (state, rows, binlogs/stats, level,
// importing flag) invalidates the per-collection quota statistics of the
// segment (quota_cache.go). Runtime-only setters do not touch quota inputs.
type CachedSegmentsInfo struct {
	segments *Cache[UniqueID, *SegmentInfo]

	// secondary indexes: collection/channel -> set of segment IDs
	coll2Segments    *typeutil.ConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, struct{}]]
	channel2Segments *typeutil.ConcurrentMap[string, *typeutil.ConcurrentMap[UniqueID, struct{}]]
	// compactionTo maps a compaction input to its outputs. Writers serialize on
	// compactionToMu and replace the slice; readers are lock-free.
	compactionToMu sync.Mutex
	compactionTo   *typeutil.ConcurrentMap[UniqueID, []UniqueID]

	// pubMu makes installing a multi-segment mutation atomic (PublishSegments)
	// for readers that need a consistent cut of a collection (ReadConsistent,
	// loadableProjection, quota refresh). It is held only around in-memory
	// installs, never across store I/O. It and the quota state stay after the
	// fields every lock-free reader loads.
	pubMu sync.RWMutex
	quota collectionQuotaCache
}

func NewCachedSegmentsInfo() *CachedSegmentsInfo {
	return &CachedSegmentsInfo{
		segments: NewCache[UniqueID, *SegmentInfo](func(s *SegmentInfo) *SegmentInfo {
			return s.Clone()
		}),
		coll2Segments:    typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, struct{}]](),
		channel2Segments: typeutil.NewConcurrentMap[string, *typeutil.ConcurrentMap[UniqueID, struct{}]](),
		compactionTo:     typeutil.NewConcurrentMap[UniqueID, []UniqueID](),
	}
}

// AttachStore makes the cache the segment storage of store: every segment read
// of the store, including QueryCoord's through the shared metacache.MetaView,
// is served by the cache from then on, so the segments exist only once in
// memory. The cache counts DataCoordNumSegments itself (segmentMetricTransition).
func (s *CachedSegmentsInfo) AttachStore(store metacache.MetaStore) {
	store.SetSegmentSource(segmentSource{s})
}

// segmentSource serves a MetaStore's segment reads from the cache.
type segmentSource struct{ segments *CachedSegmentsInfo }

func (src segmentSource) GetSegment(segmentID int64) (*datapb.SegmentInfo, bool) {
	segment := src.segments.GetSegment(segmentID)
	if segment == nil {
		return nil, false
	}
	return segment.SegmentInfo, true
}

func (src segmentSource) GetSegmentsByIDs(segmentIDs []int64) map[int64]*datapb.SegmentInfo {
	result := make(map[int64]*datapb.SegmentInfo, len(segmentIDs))
	for _, id := range segmentIDs {
		if segment := src.segments.GetSegment(id); segment != nil {
			result[id] = segment.SegmentInfo
		}
	}
	return result
}

func (src segmentSource) GetAllSegments() map[int64]*datapb.SegmentInfo {
	result := make(map[int64]*datapb.SegmentInfo, src.segments.Len())
	src.segments.Range(func(id UniqueID, segment *SegmentInfo) bool {
		result[id] = segment.SegmentInfo
		return true
	})
	return result
}

func (src segmentSource) GetSegments(collectionID int64) map[int64]*datapb.SegmentInfo {
	return segmentProtos(src.segments.GetSegmentsBySelector(WithCollection(collectionID)))
}

func (src segmentSource) GetSegmentsByChannel(channel string) map[int64]*datapb.SegmentInfo {
	return segmentProtos(src.segments.GetSegmentsByChannel(channel))
}

func segmentProtos(segments []*SegmentInfo) map[int64]*datapb.SegmentInfo {
	result := make(map[int64]*datapb.SegmentInfo, len(segments))
	for _, segment := range segments {
		result[segment.GetID()] = segment.SegmentInfo
	}
	return result
}

// segmentMetricTransition counts an accepted change of a segment in
// DataCoordNumSegments; a nil old is a new segment, a nil updated a removed one.
func segmentMetricTransition(old, updated *SegmentInfo) {
	var before, after *datapb.SegmentInfo
	if old != nil {
		before = old.SegmentInfo
	}
	if updated != nil {
		after = updated.SegmentInfo
	}
	metacache.SegmentMetricTransition(before, after)
}

// GetSegment returns SegmentInfo
func (s *CachedSegmentsInfo) GetSegment(segmentID UniqueID) *SegmentInfo {
	v, ok := s.segments.Lookup(segmentID)
	if !ok {
		return nil
	}
	return v
}

// GetSegmentWithVersion returns the segment and the store version of that
// value, read together: a write built from the segment compares against the
// version. Returns (nil, 0) if not found.
func (s *CachedSegmentsInfo) GetSegmentWithVersion(segmentID UniqueID) (*SegmentInfo, int64) {
	v, version, ok := s.segments.LookupWithVersion(segmentID)
	if !ok {
		return nil, 0
	}
	return v, version
}

// Version returns the store version of the cached value of segmentID.
func (s *CachedSegmentsInfo) Version(segmentID UniqueID) (int64, bool) {
	return s.segments.Version(segmentID)
}

// GetSegments returns all live segments. No deep copy applied.
func (s *CachedSegmentsInfo) GetSegments() []*SegmentInfo {
	result := make([]*SegmentInfo, 0, s.segments.Len())
	s.segments.Range(func(_ UniqueID, v *SegmentInfo) bool {
		result = append(result, v)
		return true
	})
	return result
}

// Range iterates over all live segments. Return false to stop.
func (s *CachedSegmentsInfo) Range(fn func(segmentID UniqueID, segment *SegmentInfo) bool) {
	s.segments.Range(fn)
}

func (s *CachedSegmentsInfo) GetSegmentsBySelector(filters ...SegmentFilter) []*SegmentInfo {
	criterion := &segmentCriterion{}
	for _, filter := range filters {
		filter.AddFilter(criterion)
	}

	candidates := s.getCandidates(criterion)
	result := make([]*SegmentInfo, 0, len(candidates))
	for _, seg := range candidates {
		if criterion.Match(seg) {
			result = append(result, seg)
		}
	}
	return result
}

func (s *CachedSegmentsInfo) getCandidates(criterion *segmentCriterion) []*SegmentInfo {
	if criterion.collectionID > 0 {
		idSet, ok := s.coll2Segments.Get(criterion.collectionID)
		if !ok {
			return nil
		}
		var result []*SegmentInfo
		idSet.Range(func(id UniqueID, _ struct{}) bool {
			if seg := s.GetSegment(id); seg != nil && seg.GetCollectionID() == criterion.collectionID {
				if criterion.channel == "" || seg.InsertChannel == criterion.channel {
					result = append(result, seg)
				}
			}
			return true
		})
		return result
	}

	if criterion.channel != "" {
		return s.GetSegmentsByChannel(criterion.channel)
	}

	return s.GetSegments()
}

func (s *CachedSegmentsInfo) GetRealSegmentsForChannel(channel string) []*SegmentInfo {
	idSet, ok := s.channel2Segments.Get(channel)
	if !ok {
		return nil
	}
	var result []*SegmentInfo
	idSet.Range(func(id UniqueID, _ struct{}) bool {
		if seg := s.GetSegment(id); seg != nil && seg.GetInsertChannel() == channel && !seg.GetIsFake() {
			result = append(result, seg)
		}
		return true
	})
	return result
}

// GetCompactionTo returns the segment that the provided segment is compacted to.
// Return (nil, false) if given segmentID can not found in the meta and compact to is nil.
// Return (nil, true) if given segmentID can be found with no compaction to.
// Return (notnil, true) if given segmentID can be found and has compaction to.
func (s *CachedSegmentsInfo) GetCompactionTo(fromSegmentID int64) ([]*SegmentInfo, bool) {
	exist := s.GetSegment(fromSegmentID) != nil
	compactTos, ok := s.compactionTo.Get(fromSegmentID)
	if !ok {
		return nil, exist
	}
	result := make([]*SegmentInfo, 0, len(compactTos))
	for _, toID := range compactTos {
		to := s.GetSegment(toID)
		if to == nil {
			mlog.Warn(context.TODO(), "compactionTo relation is broken", zap.Int64("from", fromSegmentID), zap.Int64("to", toID))
			return nil, exist
		}
		result = append(result, to)
	}
	return result, exist
}

// SetSegment inserts or overwrites a segment. A version > 0 is the store
// version of the value and is stale-protected (rejected when the cache already
// holds an equal or newer version); without one the write is forced and keeps
// the cached version. Returns the previous value and whether it existed.
func (s *CachedSegmentsInfo) SetSegment(segmentID UniqueID, segment *SegmentInfo, version ...int64) (old *SegmentInfo, existed bool) {
	old, existed, applied := s.install(segmentID, segment, versionArg(version))
	if applied {
		segmentMetricTransition(liveOrNil(old, existed), segment)
	}
	return old, existed
}

// LoadSegment installs a segment read back by recovery at its store version
// like SetSegment, but does not count it in DataCoordNumSegments: recovery
// publishes the counts of all loaded segments at once.
func (s *CachedSegmentsInfo) LoadSegment(segmentID UniqueID, segment *SegmentInfo, version int64) {
	s.install(segmentID, segment, version)
}

// install swaps segment in at version and reindexes it if the swap applied.
func (s *CachedSegmentsInfo) install(segmentID UniqueID, segment *SegmentInfo, version int64) (old *SegmentInfo, existed bool, applied bool) {
	normalizeSegmentRowCount(segment.SegmentInfo)
	old, existed, applied = s.segments.Swap(segmentID, version, func(*SegmentInfo, bool) *SegmentInfo {
		return segment
	})
	if applied {
		s.reindex(old, existed, segment)
		s.invalidateQuota(old, segment)
	}
	return old, existed, applied
}

// PublishSegment installs a newly persisted value of a segment at version.
// The runtime-only fields (allocations, flush/written time, compacting flag)
// of the current entry are kept: they may have changed after the writer
// cloned the segment, and no persisted write owns them.
func (s *CachedSegmentsInfo) PublishSegment(segmentID UniqueID, segment *SegmentInfo, version int64) (old *SegmentInfo, existed bool) {
	normalizeSegmentRowCount(segment.SegmentInfo)
	old, existed, applied := s.segments.Swap(segmentID, version, func(current *SegmentInfo, live bool) *SegmentInfo {
		if !live {
			return segment
		}
		// LastExpireTime only advances in memory (allocation, seal); a writer
		// that cloned the segment earlier must not move it back.
		if current.GetLastExpireTime() > segment.GetLastExpireTime() {
			segment.LastExpireTime = current.GetLastExpireTime()
		}
		return &SegmentInfo{
			SegmentInfo:     segment.SegmentInfo,
			allocations:     current.allocations,
			lastFlushTime:   current.lastFlushTime,
			isCompacting:    current.isCompacting,
			lastWrittenTime: current.lastWrittenTime,
		}
	})
	if !applied {
		// The cache already holds this or a newer version.
		currentVersion, _ := s.segments.Version(segmentID)
		mlog.Debug(context.TODO(), "stale segment publication ignored",
			mlog.FieldSegmentID(segmentID), mlog.Int64("version", version), mlog.Int64("currentVersion", currentVersion))
		return old, existed
	}
	s.reindex(old, existed, segment)
	segmentMetricTransition(liveOrNil(old, existed), segment)
	s.invalidateQuota(old, segment)
	return old, existed
}

// liveOrNil returns the replaced value of a cache swap, nil when there was none.
func liveOrNil(old *SegmentInfo, existed bool) *SegmentInfo {
	if !existed {
		return nil
	}
	return old
}

// PublishSegments installs the persisted values of one mutation through
// PublishSegment and returns the live values they replaced. A multi-segment
// mutation is installed under pubMu, so a consistent reader observes all of
// it or none of it.
func (s *CachedSegmentsInfo) PublishSegments(segments []*SegmentInfo, versions map[int64]int64) map[int64]*SegmentInfo {
	if len(segments) > 1 {
		s.pubMu.Lock()
		defer s.pubMu.Unlock()
	}
	previous := make(map[int64]*SegmentInfo, len(segments))
	for _, segment := range segments {
		if old, existed := s.PublishSegment(segment.GetID(), segment, versions[segment.GetID()]); existed {
			previous[segment.GetID()] = old
		}
	}
	return previous
}

// ReadConsistent runs read without observing a half-installed PublishSegments.
func (s *CachedSegmentsInfo) ReadConsistent(read func()) {
	s.pubMu.RLock()
	defer s.pubMu.RUnlock()
	read()
}

// reindex updates the secondary indexes after old was replaced by segment.
// Entries are only added for an unchanged identity, so a lock-free reader
// never misses a live segment; collection, channel and compaction_from only
// change in tests that rewrite a segment wholesale.
func (s *CachedSegmentsInfo) reindex(old *SegmentInfo, existed bool, segment *SegmentInfo) {
	if existed && (old.GetCollectionID() != segment.GetCollectionID() || old.GetInsertChannel() != segment.GetInsertChannel()) {
		s.removeSecondaryIndex(old)
	}
	if existed && !slices.Equal(old.GetCompactionFrom(), segment.GetCompactionFrom()) {
		s.deleteCompactTo(old)
	}
	s.addSecondaryIndex(segment)
	s.addCompactTo(segment)
}

// DropSegment removes the segment from the cache. A version > 0 is retained
// as a tombstone until PruneSegment so that stale writes are rejected.
func (s *CachedSegmentsInfo) DropSegment(segmentID UniqueID, version ...int64) {
	v := versionArg(version)
	// Erase first: a stale drop is rejected and must leave the indexes intact;
	// a drop over a tombstone still advances the tombstone version. Only the
	// drop that erases the live value counts it, so concurrent drops of one
	// segment decrement once.
	if old := s.segments.Erase(segmentID, v); old != nil {
		s.removeSecondaryIndex(old)
		s.deleteCompactTo(old)
		segmentMetricTransition(old, nil)
		s.invalidateQuota(old, nil)
	}
	if v <= 0 {
		s.segments.Prune(segmentID)
	}
}

// PruneSegment removes a tombstone entry, freeing memory.
func (s *CachedSegmentsInfo) PruneSegment(segmentID UniqueID) {
	s.segments.Prune(segmentID)
}

func versionArg(version []int64) int64 {
	if len(version) == 0 {
		return 0
	}
	return version[0]
}

// --- Local-only updates (no version bump, not persisted) ---

// SetRowCount sets rowCount info for SegmentInfo with provided segmentID
// if SegmentInfo not found, do nothing
func (s *CachedSegmentsInfo) SetRowCount(segmentID UniqueID, rowCount int64) {
	if collectionID, ok := s.cloneSegment(segmentID, SetRowCount(rowCount)); ok {
		s.invalidateQuotaCollection(collectionID)
	}
}

// SetDmlPosition sets DmlPosition info (checkpoint for recovery) for SegmentInfo with provided segmentID
// if SegmentInfo not found, do nothing
func (s *CachedSegmentsInfo) SetDmlPosition(segmentID UniqueID, pos *msgpb.MsgPosition) {
	s.cloneSegment(segmentID, SetDmlPosition(pos))
}

// SetStartPosition sets StartPosition info (recovery info when no checkout point found) for SegmentInfo with provided segmentID
// if SegmentInfo not found, do nothing
func (s *CachedSegmentsInfo) SetStartPosition(segmentID UniqueID, pos *msgpb.MsgPosition) {
	s.cloneSegment(segmentID, SetStartPosition(pos))
}

// SetAllocations sets allocations for segment with specified id
// if the segment id is not found, do nothing
// uses `ShadowClone` since internal SegmentInfo is not changed
func (s *CachedSegmentsInfo) SetAllocations(segmentID UniqueID, allocations []*Allocation) {
	s.shadowCloneSegment(segmentID, SetAllocations(allocations))
}

// AddAllocation adds a new allocation to specified segment
// if the segment is not found, do nothing
// uses `Clone` since internal SegmentInfo's LastExpireTime is changed
func (s *CachedSegmentsInfo) AddAllocation(segmentID UniqueID, allocation *Allocation) {
	s.cloneSegment(segmentID, AddAllocation(allocation))
}

// SetLastWrittenTime updates segment last writtent time to now.
// if the segment is not found, do nothing
// uses `ShadowClone` since internal SegmentInfo is not changed
func (s *CachedSegmentsInfo) SetLastWrittenTime(segmentID UniqueID) {
	s.shadowCloneSegment(segmentID, SetLastWrittenTime())
}

// SetFlushTime sets flush time for segment
// if the segment is not found, do nothing
// uses `ShadowClone` since internal SegmentInfo is not changed
func (s *CachedSegmentsInfo) SetFlushTime(segmentID UniqueID, t time.Time) {
	s.shadowCloneSegment(segmentID, SetFlushTime(t))
}

// SetIsCompacting sets compaction status for segment.
func (s *CachedSegmentsInfo) SetIsCompacting(segmentID UniqueID, isCompacting bool) {
	s.shadowCloneSegment(segmentID, SetIsCompacting(isCompacting))
}

// SetLastExpire sets the last expire time of the segment allocations.
func (s *CachedSegmentsInfo) SetLastExpire(segmentID UniqueID, lastExpire uint64) {
	s.cloneSegment(segmentID, SetExpireTime(lastExpire))
}

func (s *CachedSegmentsInfo) GetSegmentsByChannel(channel string) []*SegmentInfo {
	idSet, ok := s.channel2Segments.Get(channel)
	if !ok {
		return nil
	}
	var result []*SegmentInfo
	idSet.Range(func(id UniqueID, _ struct{}) bool {
		if seg := s.GetSegment(id); seg != nil && seg.GetInsertChannel() == channel {
			result = append(result, seg)
		}
		return true
	})
	return result
}

func (s *CachedSegmentsInfo) Len() int {
	return s.segments.Len()
}

// cloneSegment applies a local-only update to a deep clone (the option changes
// the persisted proto, which readers of the previous value may still hold).
// Returns the collection of the segment and whether it was updated.
func (s *CachedSegmentsInfo) cloneSegment(segmentID UniqueID, opt SegmentInfoOption) (collectionID int64, updated bool) {
	// Transform may rerun fn after a lost race; the last run is the applied one.
	var before, after *SegmentInfo
	updated = s.segments.Transform(segmentID, func(seg *SegmentInfo) (*SegmentInfo, bool) {
		collectionID = seg.GetCollectionID()
		before, after = seg, seg.Clone(opt)
		return after, true
	})
	if updated {
		segmentMetricTransition(before, after)
	}
	return collectionID, updated
}

// shadowCloneSegment applies a local-only update of runtime fields, sharing
// the persisted proto.
func (s *CachedSegmentsInfo) shadowCloneSegment(segmentID UniqueID, opt SegmentInfoOption) {
	s.segments.Transform(segmentID, func(seg *SegmentInfo) (*SegmentInfo, bool) {
		return seg.ShadowClone(opt), true
	})
}

// --- secondary index maintenance ---

func (s *CachedSegmentsInfo) addSecondaryIndex(segment *SegmentInfo) {
	collID := segment.GetCollectionID()
	channel := segment.GetInsertChannel()

	collSet, _ := s.coll2Segments.GetOrInsert(collID, typeutil.NewConcurrentMap[UniqueID, struct{}]())
	collSet.Insert(segment.GetID(), struct{}{})

	chSet, _ := s.channel2Segments.GetOrInsert(channel, typeutil.NewConcurrentMap[UniqueID, struct{}]())
	chSet.Insert(segment.GetID(), struct{}{})
}

func (s *CachedSegmentsInfo) removeSecondaryIndex(segment *SegmentInfo) {
	collID := segment.GetCollectionID()
	channel := segment.GetInsertChannel()

	if collSet, ok := s.coll2Segments.Get(collID); ok {
		collSet.Remove(segment.GetID())
	}
	if chSet, ok := s.channel2Segments.Get(channel); ok {
		chSet.Remove(segment.GetID())
	}
}

// addCompactTo adds the compact relation to the segment
func (s *CachedSegmentsInfo) addCompactTo(segment *SegmentInfo) {
	if len(segment.GetCompactionFrom()) == 0 {
		return
	}
	s.compactionToMu.Lock()
	defer s.compactionToMu.Unlock()
	for _, from := range segment.GetCompactionFrom() {
		existing, _ := s.compactionTo.Get(from)
		found := false
		for _, id := range existing {
			if id == segment.GetID() {
				found = true
				break
			}
		}
		if found {
			continue
		}
		updated := make([]UniqueID, 0, len(existing)+1)
		updated = append(updated, existing...)
		s.compactionTo.Insert(from, append(updated, segment.GetID()))
	}
}

// deleteCompactTo deletes the compact relation to the segment
func (s *CachedSegmentsInfo) deleteCompactTo(segment *SegmentInfo) {
	if len(segment.GetCompactionFrom()) == 0 {
		return
	}
	s.compactionToMu.Lock()
	defer s.compactionToMu.Unlock()
	for _, from := range segment.GetCompactionFrom() {
		s.compactionTo.Remove(from)
	}
}
