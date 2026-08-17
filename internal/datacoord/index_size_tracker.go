// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datacoord

import (
	"context"
	"strconv"
	"sync"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
)

// storedIndexSizeTracker maintains the total serialized size of active
// segment indexes by collection and field index. It is a secondary aggregate
// over indexMeta.segmentIndexes, so dropping an index does not need to scan all
// segment indexes to update DataCoordStoredIndexFilesSize.
//
// The outer mutex only protects the collection map. Updates for different
// collections are serialized independently by collectionStoredIndexSize.mu.
type storedIndexSizeTracker struct {
	mu          sync.RWMutex
	collections map[UniqueID]*collectionStoredIndexSize
}

type collectionStoredIndexSize struct {
	mu         sync.Mutex
	indexSizes map[UniqueID]uint64
	total      uint64
}

func (t *storedIndexSizeTracker) getOrCreateCollection(collectionID UniqueID) *collectionStoredIndexSize {
	t.mu.RLock()
	state, ok := t.collections[collectionID]
	t.mu.RUnlock()
	if ok {
		return state
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if t.collections == nil {
		t.collections = make(map[UniqueID]*collectionStoredIndexSize)
	}
	state, ok = t.collections[collectionID]
	if !ok {
		state = &collectionStoredIndexSize{
			indexSizes: make(map[UniqueID]uint64),
		}
		t.collections[collectionID] = state
	}
	return state
}

func (t *storedIndexSizeTracker) getCollection(collectionID UniqueID) (*collectionStoredIndexSize, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	state, ok := t.collections[collectionID]
	return state, ok
}

// activate makes an index eligible for subsequent size updates. The caller
// holds the collection-level index DDL write lock.
func (t *storedIndexSizeTracker) activate(collectionID, indexID UniqueID) {
	state := t.getOrCreateCollection(collectionID)
	state.mu.Lock()
	defer state.mu.Unlock()

	if _, ok := state.indexSizes[indexID]; !ok {
		state.indexSizes[indexID] = 0
	}
}

// recover rebuilds the aggregate once from the segment indexes loaded during
// startup. Only non-deleted field indexes contribute to the active-size gauge.
func (t *storedIndexSizeTracker) recover(
	indexes map[UniqueID]map[UniqueID]*model.Index,
	recoveredIndexSizes map[UniqueID]map[UniqueID]uint64,
) {
	recovered := make(map[UniqueID]*collectionStoredIndexSize)
	for collectionID, collectionIndexes := range indexes {
		for indexID, index := range collectionIndexes {
			if index.IsDeleted {
				continue
			}
			state, ok := recovered[collectionID]
			if !ok {
				state = &collectionStoredIndexSize{
					indexSizes: make(map[UniqueID]uint64),
				}
				recovered[collectionID] = state
			}
			size := recoveredIndexSizes[collectionID][indexID]
			state.indexSizes[indexID] = size
			state.total += size
		}
	}

	t.mu.Lock()
	t.collections = recovered
	t.mu.Unlock()

	for collectionID, state := range recovered {
		setStoredIndexSizeMetric(collectionID, state.total)
	}
}

// update applies one segment index's size transition. The caller first
// verifies that the field index is active while holding the collection-level
// index DDL read lock.
func (t *storedIndexSizeTracker) update(ctx context.Context, collectionID, indexID UniqueID, oldSize, newSize uint64) {
	state := t.getOrCreateCollection(collectionID)
	state.mu.Lock()
	defer state.mu.Unlock()

	currentSize, ok := state.indexSizes[indexID]
	if !ok {
		// This is expected for zero-value indexMeta instances used by tests. In
		// production, recovery and CreateIndex activate every live index first.
		state.indexSizes[indexID] = 0
		currentSize = 0
	}

	if newSize >= oldSize {
		delta := newSize - oldSize
		state.indexSizes[indexID] = currentSize + delta
		state.total += delta
	} else {
		delta := oldSize - newSize
		if delta > currentSize || delta > state.total {
			mlog.Warn(ctx, "stored index size tracker underflow",
				mlog.Int64("collectionID", collectionID),
				mlog.Int64("indexID", indexID),
				mlog.Uint64("oldSize", oldSize),
				mlog.Uint64("newSize", newSize),
				mlog.Uint64("trackedIndexSize", currentSize),
				mlog.Uint64("trackedCollectionSize", state.total))
			if currentSize < delta {
				delta = currentSize
			}
			if state.total < delta {
				delta = state.total
			}
		}
		state.indexSizes[indexID] = currentSize - delta
		state.total -= delta
	}

	setStoredIndexSizeMetric(collectionID, state.total)
}

// deactivate removes active indexes from the aggregate in O(number of index
// IDs). The caller holds the collection-level index DDL write lock so a late
// FinishTask cannot recreate the metric.
func (t *storedIndexSizeTracker) deactivate(ctx context.Context, collectionID UniqueID, indexIDs []UniqueID) {
	state, ok := t.getCollection(collectionID)
	if !ok {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	changed := false
	for _, indexID := range indexIDs {
		size, ok := state.indexSizes[indexID]
		if !ok {
			continue
		}
		if size > state.total {
			mlog.Warn(ctx, "stored index size tracker collection total underflow",
				mlog.Int64("collectionID", collectionID),
				mlog.Int64("indexID", indexID),
				mlog.Uint64("trackedIndexSize", size),
				mlog.Uint64("trackedCollectionSize", state.total))
			size = state.total
		}
		state.total -= size
		delete(state.indexSizes, indexID)
		changed = true
	}
	if changed {
		setStoredIndexSizeMetric(collectionID, state.total)
	}
}

func setStoredIndexSizeMetric(collectionID UniqueID, size uint64) {
	metrics.DataCoordStoredIndexFilesSize.WithLabelValues("", "",
		strconv.FormatInt(int64(collectionID), 10)).Set(float64(size))
}
