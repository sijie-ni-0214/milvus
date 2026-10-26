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
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
)

var indexTaskMetricStates = [...]indexpb.JobState{
	indexpb.JobState_JobStateNone,
	indexpb.JobState_JobStateInit,
	indexpb.JobState_JobStateInProgress,
	indexpb.JobState_JobStateFinished,
	indexpb.JobState_JobStateFailed,
	indexpb.JobState_JobStateRetry,
}

type indexTaskStateCounts [len(indexTaskMetricStates)]int64

// indexTaskCounts uses a short, memory-only lock independent of fieldIndexLock.
// Manifest installs can update it without waiting for index DDL
// persistence. It adds no per-build objects or pointers.
// Raw counts include orphaned/dropped field indexes, but exclude deleted
// tasks. Keeping them allows index DDL to adjust totals without visiting builds.
type indexTaskCounts struct {
	mu      sync.RWMutex
	active  map[[2]int64]bool
	byIndex map[[2]int64]indexTaskStateCounts
	total   indexTaskStateCounts
}

func indexTaskStateSlot(state commonpb.IndexState) int {
	switch state {
	case commonpb.IndexState_IndexStateNone:
		return 0
	case commonpb.IndexState_Unissued:
		return 1
	case commonpb.IndexState_InProgress:
		return 2
	case commonpb.IndexState_Finished:
		return 3
	case commonpb.IndexState_Failed:
		return 4
	case commonpb.IndexState_Retry:
		return 5
	default:
		return -1
	}
}

// taskCountSlot returns where a task is counted: its index and the slot of its
// state; ok is false for a task that is not counted.
func taskCountSlot(task *model.SegmentIndex) (key [2]int64, slot int, ok bool) {
	if task == nil || task.IsDeleted {
		return key, 0, false
	}
	slot = indexTaskStateSlot(task.IndexState)
	if slot < 0 {
		return key, 0, false
	}
	return [2]int64{task.CollectionID, task.IndexID}, slot, true
}

func (c *indexTaskCounts) adjustTask(task *model.SegmentIndex, delta int64) {
	key, slot, ok := taskCountSlot(task)
	if !ok {
		return
	}
	if c.byIndex == nil {
		c.byIndex = make(map[[2]int64]indexTaskStateCounts)
	}
	counts := c.byIndex[key]
	counts[slot] += delta
	if counts == (indexTaskStateCounts{}) {
		delete(c.byIndex, key)
	} else {
		c.byIndex[key] = counts
	}
	if c.active[key] {
		c.total[slot] += delta
	}
}

func (c *indexTaskCounts) replaceTask(old, current *model.SegmentIndex) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if old != nil && current != nil && old.CollectionID == current.CollectionID &&
		old.IndexID == current.IndexID && old.IndexState == current.IndexState && old.IsDeleted == current.IsDeleted {
		return
	}
	c.adjustTask(old, -1)
	c.adjustTask(current, 1)
}

// addTasks adds tasks counted by index and state slot (see taskCountSlot), as
// adjustTask adds them one at a time.
func (c *indexTaskCounts) addTasks(tasks map[[2]int64]indexTaskStateCounts) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.byIndex == nil {
		c.byIndex = make(map[[2]int64]indexTaskStateCounts)
	}
	for key, delta := range tasks {
		counts := c.byIndex[key]
		for slot := range counts {
			counts[slot] += delta[slot]
		}
		if counts == (indexTaskStateCounts{}) {
			delete(c.byIndex, key)
		} else {
			c.byIndex[key] = counts
		}
		if c.active[key] {
			for slot := range c.total {
				c.total[slot] += delta[slot]
			}
		}
	}
}

func (c *indexTaskCounts) replaceIndex(old, current *model.Index) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.active == nil {
		c.active = make(map[[2]int64]bool)
	}
	for _, change := range [...]struct {
		index *model.Index
		delta int64
	}{{old, -1}, {current, 1}} {
		if change.index == nil || change.index.IsDeleted {
			continue
		}
		key := [2]int64{change.index.CollectionID, change.index.IndexID}
		if change.delta < 0 {
			delete(c.active, key)
		} else {
			c.active[key] = true
		}
		counts := c.byIndex[key]
		for state, count := range counts {
			c.total[state] += count * change.delta
		}
	}
}

func (c *indexTaskCounts) snapshot() indexTaskStateCounts {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.total
}
