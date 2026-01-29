package datacoord

import (
	"context"
	"slices"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/lock"
)

// Segment writes on the optimistic segment store.
//
// Readers use the lock-free segment cache directly. A writer
//  1. takes the writer locks of every segment it changes (lockSegments), so
//     writers of one segment serialize while writers of different segments
//     run in parallel;
//  2. builds the new values from the cached ones;
//  3. persists them through the segment store, which compares each written
//     segment against its cached version (casSegmentStore). A conflict means
//     the store moved under the cache (another process wrote it): the cache is
//     refreshed and the write fails, to be rebuilt by the caller;
//  4. publishes the persisted values with their new versions
//     (publishSegments) before releasing the locks.
//
// Lock order: segment manifest lock -> index keyLock(buildID) -> segment
// writer locks -> groupMu -> pubMu. While holding writer locks, never block on
// another writer lock outside one lockSegments call.

// store returns the segment store, defaulting to the catalog for a meta built
// without one (unit tests construct meta directly).
func (m *meta) store() segmentStore {
	m.segmentStoreOnce.Do(func() {
		if m.segmentStore == nil {
			m.segmentStore = newCatalogSegmentStore(func() metastore.DataCoordCatalog { return m.catalog })
		}
	})
	return m.segmentStore
}

func (m *meta) getSegmentWriteLocks() *lock.KeyLock[int64] {
	m.segmentWriteLocksOnce.Do(func() {
		if m.segmentWriteLocks == nil {
			m.segmentWriteLocks = lock.NewKeyLock[int64]()
		}
	})
	return m.segmentWriteLocks
}

// lockSegments takes the writer locks of segmentIDs in ascending order and
// returns their release.
func (m *meta) lockSegments(segmentIDs ...int64) func() {
	ids := slices.Clone(segmentIDs)
	slices.Sort(ids)
	ids = slices.Compact(ids)
	locks := m.getSegmentWriteLocks()
	lock.LockManyOrdered(locks, ids)
	return func() { locks.UnlockMany(ids) }
}

// publishSegment installs the persisted value of one segment.
func (m *meta) publishSegment(segment *SegmentInfo, version int64) *SegmentInfo {
	previous, _ := m.segments.PublishSegment(segment.GetID(), segment, version)
	return previous
}

// publishSegments installs the persisted values of a mutation and returns the
// values they replaced. A multi-segment mutation is installed under pubMu so a
// consistent reader observes all of it or none of it.
func (m *meta) publishSegments(segments []*SegmentInfo, versions map[int64]int64) map[int64]*SegmentInfo {
	previous := make(map[int64]*SegmentInfo, len(segments))
	if len(segments) > 1 {
		m.pubMu.Lock()
		defer m.pubMu.Unlock()
	}
	for _, segment := range segments {
		if old, existed := m.segments.PublishSegment(segment.GetID(), segment, versions[segment.GetID()]); existed {
			previous[segment.GetID()] = old
		}
	}
	return previous
}

// selectSegmentsConsistent is SelectSegments for readers that must not observe
// a half-installed multi-segment mutation.
func (m *meta) selectSegmentsConsistent(ctx context.Context, filters ...SegmentFilter) []*SegmentInfo {
	m.pubMu.RLock()
	defer m.pubMu.RUnlock()
	return m.SelectSegments(ctx, filters...)
}

// lockOperatorSegments builds the update pack of operators under the writer
// locks of every segment the operators touch. The touched set is discovered by
// a run against the unlocked cache; when the locked run touches a segment the
// discovery run did not (the cache changed in between), it releases and
// retries. Operators only mutate the pack, so the extra runs have no effect.
func (m *meta) lockOperatorSegments(operators []UpdateOperator) (*updateSegmentPack, func(), error) {
	for {
		discovery := m.newUpdateSegmentPack()
		if err := discovery.run(operators); err != nil {
			return nil, nil, err
		}
		locked := make(map[int64]struct{}, len(discovery.segments))
		ids := make([]int64, 0, len(discovery.segments))
		for id := range discovery.segments {
			locked[id] = struct{}{}
			ids = append(ids, id)
		}
		unlock := m.lockSegments(ids...)
		pack := m.newUpdateSegmentPack()
		if err := pack.run(operators); err != nil {
			unlock()
			return nil, nil, err
		}
		covered := true
		for id := range pack.segments {
			if _, ok := locked[id]; !ok {
				covered = false
				break
			}
		}
		if covered {
			return pack, unlock, nil
		}
		unlock()
	}
}

// MutateFunc modifies a *datapb.SegmentInfo in place. Returns true to proceed
// with the write, false to skip this segment's update.
type MutateFunc func(seg *datapb.SegmentInfo) bool

// MutationsOperator adapts per-segment MutateFuncs to an UpdateOperator. The
// funcs of a segment run in order against its latest cached value; when any of
// them returns false the segment is left untouched by this operator.
func MutationsOperator(mutations map[int64][]MutateFunc) UpdateOperator {
	return func(modPack *updateSegmentPack) bool {
		segmentIDs := make([]int64, 0, len(mutations))
		for segmentID := range mutations {
			segmentIDs = append(segmentIDs, segmentID)
		}
		slices.Sort(segmentIDs)
		updated := false
		for _, segmentID := range segmentIDs {
			_, inPack := modPack.segments[segmentID]
			segment := modPack.Get(segmentID)
			if segment == nil {
				continue
			}
			mutated := proto.Clone(segment.SegmentInfo).(*datapb.SegmentInfo)
			apply := true
			for _, fn := range mutations[segmentID] {
				if !fn(mutated) {
					apply = false
					break
				}
			}
			if !apply {
				if !inPack {
					delete(modPack.segments, segmentID)
				}
				continue
			}
			segment.SegmentInfo = mutated
			updated = true
		}
		return updated
	}
}

// AddSegmentsOperator inserts new segments; a segment that already exists in
// meta is left as it is.
func AddSegmentsOperator(segments ...*datapb.SegmentInfo) UpdateOperator {
	return func(modPack *updateSegmentPack) bool {
		added := false
		for _, segment := range segments {
			if modPack.meta.segments.GetSegment(segment.GetID()) != nil {
				continue
			}
			info := NewSegmentInfo(segment)
			modPack.segments[segment.GetID()] = info
			added = true
		}
		return added
	}
}
