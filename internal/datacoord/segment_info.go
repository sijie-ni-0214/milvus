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
	"fmt"
	"slices"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/segmentutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// SegmentsInfo is the segment metadata of meta. The persisted segment protos
// live in the shared MetaStore, which serves them lock-free to DataCoord and,
// through MetaView, to QueryCoord (see metacache.VersionedSegments): writes
// compare-and-swap the store version of a segment, so they never serialize on a
// lock. The DataCoord-only state of a segment and the compaction relations,
// which never go through MetaStore, live here.
//
// Every accepted change of a quota input (state, rows, binlogs/stats, level,
// importing flag) invalidates the per-collection quota statistics of the
// segment (quota_cache.go). Runtime-only setters do not touch quota inputs.
type SegmentsInfo struct {
	store metacache.MetaStore

	// dcState holds the DataCoord-only state of the segments whose state is not
	// the zero one. A stored state is never modified: a change stores a new one
	// (updateDCState).
	dcState *typeutil.ConcurrentMap[UniqueID, *dcSegmentState]

	// compactionTo maps a compaction input to its outputs. Writers serialize on
	// compactionToMu and replace the slice; readers are lock-free.
	compactionToMu sync.Mutex
	compactionTo   *typeutil.ConcurrentMap[UniqueID, []UniqueID]

	quota collectionQuotaCache
}

// dcSegmentState holds the DataCoord-only fields of a segment, which are not
// part of the shared datapb.SegmentInfo proto.
type dcSegmentState struct {
	allocations     []*Allocation
	lastFlushTime   time.Time
	isCompacting    bool
	lastWrittenTime time.Time
}

// newDCSegmentState returns the state of a segment that enters memory without
// one: a growing segment from recovery can be considered idle.
func newDCSegmentState(segProto *datapb.SegmentInfo) dcSegmentState {
	var dc dcSegmentState
	if segProto.GetState() == commonpb.SegmentState_Growing {
		dc.allocations = make([]*Allocation, 0, 16)
		dc.lastFlushTime = time.Now().Add(-1 * paramtable.Get().DataCoordCfg.SegmentFlushInterval.GetAsDuration(time.Second))
		dc.lastWrittenTime = getZeroTime()
	}
	return dc
}

func dcStateOf(segment *SegmentInfo) dcSegmentState {
	return dcSegmentState{
		allocations:     segment.allocations,
		lastFlushTime:   segment.lastFlushTime,
		isCompacting:    segment.isCompacting,
		lastWrittenTime: segment.lastWrittenTime,
	}
}

func (dc *dcSegmentState) isZero() bool {
	return len(dc.allocations) == 0 && dc.lastFlushTime.IsZero() && !dc.isCompacting && dc.lastWrittenTime.IsZero()
}

// NewSegmentsInfo creates the segment metadata of meta over store, the shared
// MetaStore; without one (unit tests) it uses a private store.
func NewSegmentsInfo(store ...metacache.MetaStore) *SegmentsInfo {
	s := &SegmentsInfo{
		dcState:      typeutil.NewConcurrentMap[UniqueID, *dcSegmentState](),
		compactionTo: typeutil.NewConcurrentMap[UniqueID, []UniqueID](),
	}
	if len(store) > 0 && store[0] != nil {
		s.store = store[0]
	} else {
		s.store = metacache.NewMetaStore(nil)
	}
	return s
}

// fill sets si to a segment proto of the store combined with the DataCoord
// state of the segment. Callers MUST NOT mutate the proto.
func (s *SegmentsInfo) fill(si *SegmentInfo, segmentID UniqueID, segProto *datapb.SegmentInfo) {
	*si = SegmentInfo{SegmentInfo: segProto}
	if dc, ok := s.dcState.Get(segmentID); ok {
		si.allocations = dc.allocations
		si.lastFlushTime = dc.lastFlushTime
		si.isCompacting = dc.isCompacting
		si.lastWrittenTime = dc.lastWrittenTime
	}
}

func (s *SegmentsInfo) assemble(segmentID UniqueID, segProto *datapb.SegmentInfo) *SegmentInfo {
	si := &SegmentInfo{}
	s.fill(si, segmentID, segProto)
	return si
}

// GetSegment returns SegmentInfo
// Callers MUST NOT mutate the returned SegmentInfo.SegmentInfo proto; use the
// SegmentsInfo setter methods which clone before writing.
func (s *SegmentsInfo) GetSegment(segmentID UniqueID) *SegmentInfo {
	segProto, ok := s.store.GetSegment(segmentID)
	if !ok {
		return nil
	}
	return s.assemble(segmentID, segProto)
}

// GetSegmentWithVersion returns the segment and the store version of that
// value, read together: a write built from the segment compares against the
// version. Returns (nil, 0) if not found.
func (s *SegmentsInfo) GetSegmentWithVersion(segmentID UniqueID) (*SegmentInfo, int64) {
	segProto, version, ok := s.store.GetSegmentWithVersion(segmentID)
	if !ok {
		return nil, 0
	}
	return s.assemble(segmentID, segProto), version
}

// Version returns the store version of the value of segmentID.
func (s *SegmentsInfo) Version(segmentID UniqueID) (int64, bool) {
	return s.store.SegmentVersion(segmentID)
}

// GetSegments returns all live segments. No deep copy applied.
func (s *SegmentsInfo) GetSegments() []*SegmentInfo {
	result := make([]*SegmentInfo, 0, s.store.NumSegments())
	s.store.RangeSegments(func(segmentID int64, segProto *datapb.SegmentInfo) bool {
		result = append(result, s.assemble(segmentID, segProto))
		return true
	})
	return result
}

// Range iterates over all live segments. Return false to stop.
func (s *SegmentsInfo) Range(fn func(segmentID UniqueID, segment *SegmentInfo) bool) {
	s.store.RangeSegments(func(segmentID int64, segProto *datapb.SegmentInfo) bool {
		return fn(segmentID, s.assemble(segmentID, segProto))
	})
}

func (s *SegmentsInfo) GetSegmentsBySelector(filters ...SegmentFilter) []*SegmentInfo {
	result := make([]*SegmentInfo, 0)
	s.RangeReused(func(segment *SegmentInfo) bool {
		result = append(result, segment.ShadowClone())
		return true
	}, filters...)
	return result
}

// RangeReused visits the live segments the filters select with one reused
// value, valid only during fn; fn returns false to stop. A visitor that keeps
// a segment keeps segment.ShadowClone(). Unlike the reads returning segments,
// it allocates nothing per segment, which matters to scans of every segment.
func (s *SegmentsInfo) RangeReused(fn func(segment *SegmentInfo) bool, filters ...SegmentFilter) {
	criterion := &segmentCriterion{}
	for _, filter := range filters {
		filter.AddFilter(criterion)
	}
	segment := &SegmentInfo{}
	s.rangeCandidates(criterion, func(segmentID int64, segProto *datapb.SegmentInfo) bool {
		s.fill(segment, segmentID, segProto)
		if !criterion.Match(segment) {
			return true
		}
		return fn(segment)
	})
}

// rangeCandidates visits the live segments of the criterion's collection and
// channel, every live segment when it has neither.
func (s *SegmentsInfo) rangeCandidates(criterion *segmentCriterion, fn func(segmentID int64, segProto *datapb.SegmentInfo) bool) {
	switch {
	case criterion.collectionID > 0:
		s.store.RangeCollectionSegmentIDs(criterion.collectionID, func(segmentID int64) bool {
			segProto, ok := s.store.GetSegment(segmentID)
			if !ok || segProto.GetCollectionID() != criterion.collectionID ||
				(criterion.channel != "" && segProto.GetInsertChannel() != criterion.channel) {
				return true
			}
			return fn(segmentID, segProto)
		})
	case criterion.channel != "":
		s.store.RangeChannelSegmentIDs(criterion.channel, func(segmentID int64) bool {
			segProto, ok := s.store.GetSegment(segmentID)
			if !ok || segProto.GetInsertChannel() != criterion.channel {
				return true
			}
			return fn(segmentID, segProto)
		})
	default:
		s.store.RangeSegments(fn)
	}
}

func (s *SegmentsInfo) getCandidates(criterion *segmentCriterion) []*SegmentInfo {
	var result []*SegmentInfo
	s.rangeCandidates(criterion, func(segmentID int64, segProto *datapb.SegmentInfo) bool {
		result = append(result, s.assemble(segmentID, segProto))
		return true
	})
	return result
}

func (s *SegmentsInfo) GetRealSegmentsForChannel(channel string) []*SegmentInfo {
	var result []*SegmentInfo
	s.rangeCandidates(&segmentCriterion{channel: channel}, func(segmentID int64, segProto *datapb.SegmentInfo) bool {
		if !segProto.GetIsFake() {
			result = append(result, s.assemble(segmentID, segProto))
		}
		return true
	})
	return result
}

func (s *SegmentsInfo) GetSegmentsByChannel(channel string) []*SegmentInfo {
	return s.getCandidates(&segmentCriterion{channel: channel})
}

func (s *SegmentsInfo) Len() int {
	return s.store.NumSegments()
}

// GetCompactionTo returns the segment that the provided segment is compacted to.
// Return (nil, false) if given segmentID can not found in the meta and compact to is nil.
// Return (nil, true) if given segmentID can be found with no compaction to.
// Return (notnil, true) if given segmentID can be found and has compaction to.
func (s *SegmentsInfo) GetCompactionTo(fromSegmentID int64) ([]*SegmentInfo, bool) {
	exist := s.GetSegment(fromSegmentID) != nil
	compactTos, ok := s.compactionTo.Get(fromSegmentID)
	if !ok {
		return nil, exist
	}
	result := make([]*SegmentInfo, 0, len(compactTos))
	for _, toID := range compactTos {
		to := s.GetSegment(toID)
		if to == nil {
			mlog.Warn(context.TODO(), "compactionTo relation is broken", mlog.Int64("from", fromSegmentID), mlog.Int64("to", toID))
			return nil, exist
		}
		result = append(result, to)
	}
	return result, exist
}

// SetSegment inserts or overwrites a segment, its DataCoord state included. A
// version > 0 is the store version of the value and is stale-protected
// (rejected when the store already holds an equal or newer version); without
// one the write is forced and keeps the stored version.
func (s *SegmentsInfo) SetSegment(segmentID UniqueID, segment *SegmentInfo, version ...int64) {
	s.install(segmentID, segment, versionArg(version), s.store.InstallSegment)
}

// LoadSegment installs a segment read back by recovery at its store version
// like SetSegment, but does not count it in DataCoordNumSegments: recovery
// publishes the counts of all loaded segments at once.
func (s *SegmentsInfo) LoadSegment(segmentID UniqueID, segment *SegmentInfo, version int64) {
	s.install(segmentID, segment, version, s.store.LoadSegment)
}

// loadRecovered installs a segment read back by recovery like LoadSegment,
// except for its compaction relations, which recovery replays once every
// segment is installed (see replayCompactTo). It returns the value the
// segment replaced and whether the write applied. Installs of different
// segments may run concurrently.
func (s *SegmentsInfo) loadRecovered(segmentID UniqueID, segment *SegmentInfo, version int64) (*datapb.SegmentInfo, bool) {
	normalizeSegmentRowCount(segment.SegmentInfo)
	old, applied := s.store.LoadSegment(segmentID, segment.SegmentInfo, version)
	if !applied {
		return nil, false
	}
	s.setDCState(segmentID, dcStateOf(segment))
	s.invalidateQuota(old, segment.SegmentInfo)
	return old, true
}

// compactToChange is a change of the compaction relations that an install
// makes (see reindexCompactTo): the relation of segment from is removed, or
// segment is added to it. pos orders the changes of the installs.
type compactToChange struct {
	from    int64
	segment int64
	pos     int32
	remove  bool
}

// appendCompactToChanges appends to shards, by the shard of the segment
// compacted from, the changes reindexCompactTo makes for the install at pos
// of segment over old.
func appendCompactToChanges(shards [][]compactToChange, pos int, old, segment *datapb.SegmentInfo) {
	if old != nil && !slices.Equal(old.GetCompactionFrom(), segment.GetCompactionFrom()) {
		for _, from := range old.GetCompactionFrom() {
			shard := shardOf(from, len(shards))
			shards[shard] = append(shards[shard], compactToChange{from: from, pos: int32(pos), remove: true})
		}
	}
	for _, from := range segment.GetCompactionFrom() {
		shard := shardOf(from, len(shards))
		shards[shard] = append(shards[shard], compactToChange{from: from, segment: segment.GetID(), pos: int32(pos)})
	}
}

// replayCompactTo applies the compaction relation changes of the installs of
// recovery. changes[w][shard] holds, in install order, the changes worker w
// made to the relations of shard. Every shard is replayed on its own
// goroutine, the changes of a relation in install order, so each relation
// ends as installing the segments one after another leaves it. Recovery is
// the only writer of the relations while it runs.
func (s *SegmentsInfo) replayCompactTo(changes [][][]compactToChange) {
	if len(changes) == 0 {
		return
	}
	var wg sync.WaitGroup
	for shard := range changes[0] {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var mine []compactToChange
			for _, workerChanges := range changes {
				mine = append(mine, workerChanges[shard]...)
			}
			slices.SortFunc(mine, func(a, b compactToChange) int {
				if c := cmp.Compare(a.from, b.from); c != 0 {
					return c
				}
				if c := cmp.Compare(a.pos, b.pos); c != 0 {
					return c
				}
				// An install removes the relations of the old value first.
				switch {
				case a.remove == b.remove:
					return 0
				case a.remove:
					return -1
				default:
					return 1
				}
			})
			for start := 0; start < len(mine); {
				from := mine[start].from
				end := start
				for end < len(mine) && mine[end].from == from {
					end++
				}
				s.applyCompactToChanges(from, mine[start:end])
				start = end
			}
		}()
	}
	wg.Wait()
}

// applyCompactToChanges applies the changes of the relation of from in order,
// like deleteCompactTo and addCompactTo do.
func (s *SegmentsInfo) applyCompactToChanges(from int64, changes []compactToChange) {
	existing, existed := s.compactionTo.Get(from)
	children := slices.Clone(existing)
	present := existed
	for _, change := range changes {
		switch {
		case change.remove:
			children, present = nil, false
		case !slices.Contains(children, change.segment):
			children, present = append(children, change.segment), true
		}
	}
	switch {
	case present:
		s.compactionTo.Insert(from, children)
	case existed:
		s.compactionTo.Remove(from)
	}
}

// install writes segment through write and, when the write applied, sets the
// DataCoord state and the compaction relations of the segment.
func (s *SegmentsInfo) install(segmentID UniqueID, segment *SegmentInfo, version int64,
	write func(int64, *datapb.SegmentInfo, int64) (*datapb.SegmentInfo, bool),
) {
	normalizeSegmentRowCount(segment.SegmentInfo)
	old, applied := write(segmentID, segment.SegmentInfo, version)
	if !applied {
		return
	}
	s.setDCState(segmentID, dcStateOf(segment))
	s.reindexCompactTo(old, segment.SegmentInfo)
	s.invalidateQuota(old, segment.SegmentInfo)
}

// PublishSegment installs a newly persisted value of a segment at version.
// The DataCoord state of a live segment is kept: it may have changed after the
// writer cloned the segment, and no persisted write owns it.
func (s *SegmentsInfo) PublishSegment(segmentID UniqueID, segment *SegmentInfo, version int64) (old *SegmentInfo, existed bool) {
	old = s.PublishSegments([]*SegmentInfo{segment}, map[int64]int64{segmentID: version})[segmentID]
	return old, old != nil
}

// PublishSegments installs the persisted values of one mutation and returns
// the live values they replaced. A multi-segment mutation is installed
// atomically (metacache.VersionedSegments.PublishSegments), so a consistent
// reader observes all of it or none of it.
func (s *SegmentsInfo) PublishSegments(segments []*SegmentInfo, versions map[int64]int64) map[int64]*SegmentInfo {
	protos := make([]*datapb.SegmentInfo, len(segments))
	for i, segment := range segments {
		normalizeSegmentRowCount(segment.SegmentInfo)
		protos[i] = segment.SegmentInfo
	}
	results := s.store.PublishSegments(protos, versions, keepLastExpireTime)
	previous := make(map[int64]*SegmentInfo, len(segments))
	for i, result := range results {
		segmentID := protos[i].GetID()
		if result.Applied {
			if result.Old == nil {
				// A new segment brings its DataCoord state, unless a setter
				// already gave it one.
				s.initDCState(segmentID, dcStateOf(segments[i]))
			}
			s.reindexCompactTo(result.Old, protos[i])
			s.invalidateQuota(result.Old, protos[i])
		} else {
			mlog.Debug(context.TODO(), "stale segment publication ignored", mlog.FieldSegmentID(segmentID))
		}
		if result.Old != nil {
			previous[segmentID] = s.assemble(segmentID, result.Old)
		}
	}
	return previous
}

// keepLastExpireTime is the publication merge of DataCoord: LastExpireTime
// only advances in memory (allocation, seal), so a writer that cloned the
// segment earlier must not move it back.
func keepLastExpireTime(current, next *datapb.SegmentInfo) {
	if current.GetLastExpireTime() > next.GetLastExpireTime() {
		next.LastExpireTime = current.GetLastExpireTime()
	}
}

// ReadConsistent runs read without observing a half-installed PublishSegments.
func (s *SegmentsInfo) ReadConsistent(read func()) {
	s.store.ReadConsistent(read)
}

// DropSegment removes the segment. A version > 0 is retained as a tombstone
// until PruneSegment so that stale writes are rejected.
func (s *SegmentsInfo) DropSegment(segmentID UniqueID, version ...int64) {
	if old := s.store.DropSegment(segmentID, versionArg(version)); old != nil {
		s.deleteCompactTo(old)
		s.invalidateQuota(old, nil)
		s.dcState.Remove(segmentID)
	}
}

// PruneSegment removes a tombstone entry, freeing memory.
func (s *SegmentsInfo) PruneSegment(segmentID UniqueID) {
	s.store.PruneSegment(segmentID)
}

func versionArg(version []int64) int64 {
	if len(version) == 0 {
		return 0
	}
	return version[0]
}

// --- Local-only updates (no version bump, not persisted) ---

// updateProto applies a local-only change to a clone of the persisted proto:
// readers of the previous value may still hold it. Returns the updated proto,
// nil when the segment is not live.
func (s *SegmentsInfo) updateProto(segmentID UniqueID, change func(*datapb.SegmentInfo)) *datapb.SegmentInfo {
	_, updated := s.store.UpdateSegment(segmentID, func(current *datapb.SegmentInfo) *datapb.SegmentInfo {
		next := proto.Clone(current).(*datapb.SegmentInfo)
		change(next)
		return next
	})
	return updated
}

// updateDCState stores the DataCoord state of a live segment changed by
// change, starting from the default state when the segment has none.
func (s *SegmentsInfo) updateDCState(segmentID UniqueID, change func(*dcSegmentState)) {
	for {
		current, existed := s.dcState.Get(segmentID)
		var next dcSegmentState
		if existed {
			next = *current
		} else {
			segProto, live := s.store.GetSegment(segmentID)
			if !live {
				return
			}
			next = newDCSegmentState(segProto)
		}
		change(&next)
		if s.replaceDCState(segmentID, current, existed, &next) {
			break
		}
	}
	// A segment dropped meanwhile does not keep the state this change gave it.
	if _, live := s.store.GetSegment(segmentID); !live {
		s.dcState.Remove(segmentID)
	}
}

// replaceDCState replaces current, the stored state of the segment (absent
// when !existed), with next, and reports whether current was still stored.
func (s *SegmentsInfo) replaceDCState(segmentID UniqueID, current *dcSegmentState, existed bool, next *dcSegmentState) bool {
	switch {
	case next.isZero() && !existed:
		return true
	case next.isZero():
		return s.dcState.CompareAndDelete(segmentID, current)
	case !existed:
		_, loaded := s.dcState.GetOrInsert(segmentID, next)
		return !loaded
	default:
		return s.dcState.CompareAndSwap(segmentID, current, next)
	}
}

// setDCState stores the DataCoord state of an installed segment.
func (s *SegmentsInfo) setDCState(segmentID UniqueID, state dcSegmentState) {
	if state.isZero() {
		s.dcState.Remove(segmentID)
		return
	}
	s.dcState.Insert(segmentID, &state)
}

// initDCState stores the DataCoord state a new segment brings, unless a setter
// already gave the segment one.
func (s *SegmentsInfo) initDCState(segmentID UniqueID, state dcSegmentState) {
	if !state.isZero() {
		s.dcState.GetOrInsert(segmentID, &state)
	}
}

// SetRowCount sets rowCount info for SegmentInfo with provided segmentID
// if SegmentInfo not found, do nothing
func (s *SegmentsInfo) SetRowCount(segmentID UniqueID, rowCount int64) {
	if updated := s.updateProto(segmentID, func(segment *datapb.SegmentInfo) {
		segment.NumOfRows = rowCount
	}); updated != nil {
		s.invalidateQuotaCollection(updated.GetCollectionID())
	}
}

// SetAllocations sets allocations for segment with specified id
// if the segment id is not found, do nothing
func (s *SegmentsInfo) SetAllocations(segmentID UniqueID, allocations []*Allocation) {
	s.updateDCState(segmentID, func(dc *dcSegmentState) {
		dc.allocations = allocations
	})
}

// AddAllocation adds a new allocation to specified segment
// if the segment is not found, do nothing
func (s *SegmentsInfo) AddAllocation(segmentID UniqueID, allocation *Allocation) {
	s.updateProto(segmentID, func(segment *datapb.SegmentInfo) {
		segment.LastExpireTime = allocation.ExpireTime
	})
	s.updateDCState(segmentID, func(dc *dcSegmentState) {
		// Clip: a reader of the previous state may hold the same array.
		dc.allocations = append(slices.Clip(dc.allocations), allocation)
	})
}

// SetLastWrittenTime updates segment last writtent time to now.
// if the segment is not found, do nothing
func (s *SegmentsInfo) SetLastWrittenTime(segmentID UniqueID) {
	s.updateDCState(segmentID, func(dc *dcSegmentState) {
		dc.lastWrittenTime = time.Now()
	})
}

// SetFlushTime sets flush time for segment
// if the segment is not found, do nothing
func (s *SegmentsInfo) SetFlushTime(segmentID UniqueID, t time.Time) {
	s.updateDCState(segmentID, func(dc *dcSegmentState) {
		dc.lastFlushTime = t
	})
}

// SetIsCompacting sets compaction status for segment.
func (s *SegmentsInfo) SetIsCompacting(segmentID UniqueID, isCompacting bool) {
	s.updateDCState(segmentID, func(dc *dcSegmentState) {
		dc.isCompacting = isCompacting
	})
}

// SetLastExpire sets the last expire time of the segment allocations.
func (s *SegmentsInfo) SetLastExpire(segmentID UniqueID, lastExpire uint64) {
	s.updateProto(segmentID, func(segment *datapb.SegmentInfo) {
		segment.LastExpireTime = lastExpire
	})
}

// --- compaction relations ---

// reindexCompactTo updates the compaction relations after old (nil when the
// segment was not live) was replaced by segment.
func (s *SegmentsInfo) reindexCompactTo(old, segment *datapb.SegmentInfo) {
	if old != nil && !slices.Equal(old.GetCompactionFrom(), segment.GetCompactionFrom()) {
		s.deleteCompactTo(old)
	}
	s.addCompactTo(segment)
}

// addCompactTo adds the compact relation to the segment
func (s *SegmentsInfo) addCompactTo(segment *datapb.SegmentInfo) {
	if len(segment.GetCompactionFrom()) == 0 {
		return
	}
	s.compactionToMu.Lock()
	defer s.compactionToMu.Unlock()
	for _, from := range segment.GetCompactionFrom() {
		existing, _ := s.compactionTo.Get(from)
		if slices.Contains(existing, segment.GetID()) {
			continue
		}
		updated := make([]UniqueID, 0, len(existing)+1)
		updated = append(updated, existing...)
		s.compactionTo.Insert(from, append(updated, segment.GetID()))
	}
}

// deleteCompactTo deletes the compact relation to the segment
func (s *SegmentsInfo) deleteCompactTo(segment *datapb.SegmentInfo) {
	if len(segment.GetCompactionFrom()) == 0 {
		return
	}
	s.compactionToMu.Lock()
	defer s.compactionToMu.Unlock()
	for _, from := range segment.GetCompactionFrom() {
		s.compactionTo.Remove(from)
	}
}

// SegmentInfo wraps datapb.SegmentInfo and patches some extra info on it
type SegmentInfo struct {
	*datapb.SegmentInfo
	allocations     []*Allocation
	lastFlushTime   time.Time
	isCompacting    bool
	lastWrittenTime time.Time
}

// EnsureStats returns a non-nil Statistics view for read-only aggregate
// queries. It does NOT mutate s — concurrent lock-free readers would race
// otherwise. The persisted s.Stats is populated eagerly by
// NewSegmentInfo on construction and by the array-mutating operators
// (AddBinlogsOperator, UpdateBinlogsFromSaveBinlogPathsOperator,
// UpdateSegmentStats), both of which run on the update pack's private clone. When a
// caller hands us a SegmentInfo built via the struct literal
// `&SegmentInfo{SegmentInfo: ...}` with a nil Stats (the only remaining
// path is now legacy tests), we fall back to a transient recompute so
// readers see the right number; we just don't write it back.
func (s *SegmentInfo) EnsureStats() *datapb.Statistics {
	if s.SegmentInfo == nil {
		return nil
	}
	if stats := s.GetStats(); stats != nil {
		return stats
	}
	return storage.BuildStatsFromFieldBinlogs(s.GetBinlogs(), s.GetStatslogs(), s.GetBm25Statslogs(), s.GetDeltalogs())
}

func (s *SegmentInfo) GetResidualSegmentSize() int64 {
	if s.GetNumOfRows() == 0 {
		return 0
	}
	deltaRatio := float64(s.EnsureStats().GetDeleteNumRows()) / float64(s.GetNumOfRows())
	if deltaRatio >= 1.0 {
		// segments with too many deleted rows should be considered as prioritized segments and be compacted definitely
		return s.getSegmentSize()
	}
	residualRatio := 1.0 - deltaRatio
	return int64(residualRatio * float64(s.getSegmentSize()))
}

func (s *SegmentInfo) GetEarliestTs() uint64 {
	// For import segments, row timestamps predate the actual commit time.
	// Use commit_timestamp as the effective data age so compaction priority
	// and TTL decisions are not distorted by stale row timestamps.
	if commitTs := s.GetCommitTimestamp(); commitTs != 0 {
		return commitTs
	}
	// Stats.TimestampFrom is the exact min(TimestampFrom) across all insert
	// binlogs (populated by StatisticsCollector on the writer side, or by
	// BuildStatsFromFieldBinlogs on V2 fallback / migration).
	return s.EnsureStats().GetTimestampFrom()
}

// NewSegmentInfo create `SegmentInfo` wrapper from `datapb.SegmentInfo`
// assign current rows to last checkpoint and pre-allocate `allocations` slice
// Note that the allocation information is not preserved,
// the worst case scenario is to have a segment with twice size we expects
//
// Stats is populated from the FieldBinlog arrays when nil so legacy
// segments (persisted before Statistics existed) and live aggregate
// reads agree without callers needing a fallback. EnsureStats covers the
// struct-literal construction path that bypasses this constructor.
func NewSegmentInfo(info *datapb.SegmentInfo) *SegmentInfo {
	if info.Stats == nil {
		info.Stats = storage.BuildStatsFromFieldBinlogs(info.GetBinlogs(), info.GetStatslogs(), info.GetBm25Statslogs(), info.GetDeltalogs())
	}
	dc := newDCSegmentState(info)
	return &SegmentInfo{
		SegmentInfo:     info,
		allocations:     dc.allocations,
		lastFlushTime:   dc.lastFlushTime,
		isCompacting:    dc.isCompacting,
		lastWrittenTime: dc.lastWrittenTime,
	}
}

// normalizeSegmentRowCount reconciles a V1/V2 segment's NumOfRows with its
// insert binlogs before the segment enters memory. The catalog applies the same
// correction, but only to the copy it persists (kv_catalog.go), which would
// leave the in-memory segment — the one QueryCoord reads through the shared
// store — carrying the stale count until the next reload.
//
// V3 segments are exempt: their binlog arrays are a pass-through cache that can
// be empty after recovery or delta-only during a growing-source flush, so an
// array-derived count would be wrong.
func normalizeSegmentRowCount(segProto *datapb.SegmentInfo) {
	if segProto.GetStorageVersion() == storage.StorageV3 || segProto.GetManifestPath() != "" {
		return
	}
	segmentutil.ReCalcRowCount(segProto, segProto)
}

func (s *SegmentInfo) IsDeltaLogExists(logID int64) bool {
	for _, deltaLogs := range s.GetDeltalogs() {
		for _, l := range deltaLogs.GetBinlogs() {
			if l.GetLogID() == logID {
				return true
			}
		}
	}
	return false
}

func (s *SegmentInfo) IsStatsLogExists(logID int64) bool {
	for _, statsLogs := range s.GetStatslogs() {
		for _, l := range statsLogs.GetBinlogs() {
			if l.GetLogID() == logID {
				return true
			}
		}
	}
	return false
}

// Clone deep clone the segment info and return a new instance. Stats lives
// on the proto and is copied by proto.Clone, so the cloned segment's
// aggregate reads stay consistent with its (cloned) binlog arrays. Opts
// that replace binlogs should also refresh Stats eagerly (recompute via
// storage.BuildStatsFromFieldBinlogs); EnsureStats no longer writes back lazily —
// concurrent RLock readers would race.
func (s *SegmentInfo) Clone(opts ...SegmentInfoOption) *SegmentInfo {
	info := proto.Clone(s.SegmentInfo).(*datapb.SegmentInfo)
	cloned := &SegmentInfo{
		SegmentInfo:     info,
		allocations:     s.allocations,
		lastFlushTime:   s.lastFlushTime,
		isCompacting:    s.isCompacting,
		lastWrittenTime: s.lastWrittenTime,
	}
	for _, opt := range opts {
		opt(cloned)
	}
	return cloned
}

// ShadowClone shadow clone the segment and return a new instance
func (s *SegmentInfo) ShadowClone(opts ...SegmentInfoOption) *SegmentInfo {
	cloned := &SegmentInfo{
		SegmentInfo:     s.SegmentInfo,
		allocations:     s.allocations,
		lastFlushTime:   s.lastFlushTime,
		isCompacting:    s.isCompacting,
		lastWrittenTime: s.lastWrittenTime,
	}
	for _, opt := range opts {
		opt(cloned)
	}
	return cloned
}

// SegmentInfoOption is the option to set fields in segment info
type SegmentInfoOption func(segment *SegmentInfo)

// SetRowCount is the option to set row count for segment info
func SetRowCount(rowCount int64) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.NumOfRows = rowCount
	}
}

// SetExpireTime is the option to set expire time for segment info
func SetExpireTime(expireTs Timestamp) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.LastExpireTime = expireTs
	}
}

// SetState is the option to set state for segment info
func SetState(state commonpb.SegmentState) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.State = state
	}
}

// SetDmlPosition is the option to set dml position for segment info
func SetDmlPosition(pos *msgpb.MsgPosition) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.DmlPosition = pos
	}
}

// SetStartPosition is the option to set start position for segment info
func SetStartPosition(pos *msgpb.MsgPosition) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.StartPosition = pos
	}
}

// SetAllocations is the option to set allocations for segment info
func SetAllocations(allocations []*Allocation) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.allocations = allocations
	}
}

// AddAllocation is the option to add allocation info for segment info
func AddAllocation(allocation *Allocation) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.allocations = append(segment.allocations, allocation)
		segment.LastExpireTime = allocation.ExpireTime
	}
}

// SetLastWrittenTime is the option to set last writtent time for segment info
func SetLastWrittenTime() SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.lastWrittenTime = time.Now()
	}
}

// SetFlushTime is the option to set flush time for segment info
func SetFlushTime(t time.Time) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.lastFlushTime = t
	}
}

// SetIsCompacting is the option to set compaction state for segment info
func SetIsCompacting(isCompacting bool) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.isCompacting = isCompacting
	}
}

func (s *SegmentInfo) getSegmentSize() int64 {
	stats := s.EnsureStats()
	return stats.GetInsertBinlogSize() + stats.GetStatsBinlogSize() + stats.GetDeltaBinlogSize()
}

func (s *SegmentInfo) getFieldBinlogSize(fieldID int64) int64 {
	var size int64
	for _, binlogs := range s.GetBinlogs() {
		if binlogs.GetFieldID() == fieldID {
			for _, l := range binlogs.GetBinlogs() {
				size += l.GetMemorySize()
			}
		} else {
			for _, childFieldID := range binlogs.GetChildFields() {
				if childFieldID == fieldID {
					for _, l := range binlogs.GetBinlogs() {
						size += l.GetMemorySize()
					}
				}
			}
		}
	}
	// StorageV3 manifest segments persist no FieldBinlog arrays (the manifest
	// is the source of truth), so the loop above sums nothing for them and
	// the whole-segment fallback below would overcount the field by every
	// non-indexed field's bytes (e.g. index task slots sized by segment size
	// instead of vector size). Their per-column-group sizes ship inside
	// Stats.LoadResource instead; mirror the array matching rules (GroupId or
	// member FieldIds) so both storage versions size the same field the same.
	if size <= 0 {
		for _, group := range s.EnsureStats().GetLoadResource().GetColumnGroups() {
			if group.GetGroupId() == fieldID || slices.Contains(group.GetFieldIds(), fieldID) {
				size += group.GetMemorySize()
			}
		}
	}
	if size <= 0 {
		return s.getSegmentSize()
	}
	return size
}

func (s *SegmentInfo) getDeltaCount() int64 {
	return s.EnsureStats().GetDeleteNumRows()
}

// SegmentInfoSelector is the function type to select SegmentInfo from meta
type SegmentInfoSelector func(*SegmentInfo) bool

// ValidateManifestSegment checks that segments with manifest_path have empty
// legacy stats fields. Returns a descriptive message if validation fails,
// or empty string if the segment is valid.
func ValidateManifestSegment(info *SegmentInfo) string {
	if info.GetManifestPath() == "" {
		return ""
	}

	var nonEmpty []string
	if len(info.GetStatslogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("statslogs(%d)", len(info.GetStatslogs())))
	}
	if len(info.GetBm25Statslogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("bm25statslogs(%d)", len(info.GetBm25Statslogs())))
	}
	if len(info.GetTextStatsLogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("textStatsLogs(%d)", len(info.GetTextStatsLogs())))
	}
	if len(info.GetJsonKeyStats()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("jsonKeyStats(%d)", len(info.GetJsonKeyStats())))
	}

	if len(nonEmpty) > 0 {
		return fmt.Sprintf("segment %d has manifest_path but non-empty legacy stats fields: %v",
			info.GetID(), nonEmpty)
	}
	return ""
}

// segmentEffectiveTs returns the start-position timestamp that governs temporal
// decisions for a segment. For import segments with a non-zero commit_timestamp,
// commit_timestamp overrides start_position.Timestamp because the data was not
// "officially present" until the import was committed.
func segmentEffectiveTs(seg *datapb.SegmentInfo) uint64 {
	if ts := seg.GetCommitTimestamp(); ts != 0 {
		return ts
	}
	return seg.GetStartPosition().GetTimestamp()
}

// segmentEffectiveDmlTs returns the DML-position timestamp for temporal decisions.
// Same override logic as segmentEffectiveTs but for dml_position consumers
// (GC eligibility, TruncateChannelByTime).
func segmentEffectiveDmlTs(seg *datapb.SegmentInfo) uint64 {
	if ts := seg.GetCommitTimestamp(); ts != 0 {
		return ts
	}
	return seg.GetDmlPosition().GetTimestamp()
}
