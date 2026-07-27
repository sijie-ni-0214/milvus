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

package qnview

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
)

const (
	segmentLoadTimingLogInterval = 5 * time.Second
	segmentLoadTimingLogBatch    = 2000
)

var sqnSegmentLoadTimingStats = segmentLoadTimingStats{
	interval:  segmentLoadTimingLogInterval,
	batchSize: segmentLoadTimingLogBatch,
}

type segmentLoadTimingSample struct {
	total           time.Duration
	updateIndexMeta time.Duration
	reserveResource time.Duration
	physicalLoad    time.Duration
	releaseResource time.Duration
	onLoaded        time.Duration
	failed          bool
}

type durationStats struct {
	total time.Duration
	max   time.Duration
}

func (s *durationStats) add(duration time.Duration) {
	s.total += duration
	s.max = max(s.max, duration)
}

func (s durationStats) average(count int64) time.Duration {
	if count == 0 {
		return 0
	}
	return s.total / time.Duration(count)
}

type segmentLoadTimingSnapshot struct {
	count          int64
	failed         int64
	windowDuration time.Duration

	total           durationStats
	updateIndexMeta durationStats
	reserveResource durationStats
	physicalLoad    durationStats
	releaseResource durationStats
	onLoaded        durationStats
}

type segmentLoadTimingStats struct {
	mu         sync.Mutex
	interval   time.Duration
	batchSize  int64
	windowFrom time.Time

	count  int64
	failed int64

	total           durationStats
	updateIndexMeta durationStats
	reserveResource durationStats
	physicalLoad    durationStats
	releaseResource durationStats
	onLoaded        durationStats
}

func recordSQNSegmentLoadTiming(ctx context.Context, sample segmentLoadTimingSample) {
	snapshot, ok := sqnSegmentLoadTimingStats.add(time.Now(), sample)
	if !ok {
		return
	}
	logSQNSegmentLoadTiming(ctx, snapshot)
}

func (s *segmentLoadTimingStats) add(now time.Time, sample segmentLoadTimingSample) (segmentLoadTimingSnapshot, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.windowFrom.IsZero() {
		s.windowFrom = now
	}
	s.count++
	if sample.failed {
		s.failed++
	}
	s.total.add(sample.total)
	s.updateIndexMeta.add(sample.updateIndexMeta)
	s.reserveResource.add(sample.reserveResource)
	s.physicalLoad.add(sample.physicalLoad)
	s.releaseResource.add(sample.releaseResource)
	s.onLoaded.add(sample.onLoaded)

	if s.count < s.batchSize && now.Sub(s.windowFrom) < s.interval {
		return segmentLoadTimingSnapshot{}, false
	}
	snapshot := segmentLoadTimingSnapshot{
		count:           s.count,
		failed:          s.failed,
		windowDuration:  max(now.Sub(s.windowFrom), time.Duration(0)),
		total:           s.total,
		updateIndexMeta: s.updateIndexMeta,
		reserveResource: s.reserveResource,
		physicalLoad:    s.physicalLoad,
		releaseResource: s.releaseResource,
		onLoaded:        s.onLoaded,
	}
	s.reset()
	return snapshot, true
}

func (s *segmentLoadTimingStats) reset() {
	s.windowFrom = time.Time{}
	s.count = 0
	s.failed = 0
	s.total = durationStats{}
	s.updateIndexMeta = durationStats{}
	s.reserveResource = durationStats{}
	s.physicalLoad = durationStats{}
	s.releaseResource = durationStats{}
	s.onLoaded = durationStats{}
}

func logSQNSegmentLoadTiming(ctx context.Context, snapshot segmentLoadTimingSnapshot) {
	mlog.Info(
		ctx, "[SN recovery] SQN segment load timing",
		mlog.String("phase", "sqn_segment_load_timing"),
		mlog.String("component", "queryNode"),
		mlog.Int64("count", snapshot.count),
		mlog.Int64("failed", snapshot.failed),
		mlog.Duration("windowDuration", snapshot.windowDuration),
		mlog.Duration("avgTotal", snapshot.total.average(snapshot.count)),
		mlog.Duration("maxTotal", snapshot.total.max),
		mlog.Duration("avgUpdateIndexMeta", snapshot.updateIndexMeta.average(snapshot.count)),
		mlog.Duration("maxUpdateIndexMeta", snapshot.updateIndexMeta.max),
		mlog.Duration("avgReserveResource", snapshot.reserveResource.average(snapshot.count)),
		mlog.Duration("maxReserveResource", snapshot.reserveResource.max),
		mlog.Duration("avgPhysicalLoad", snapshot.physicalLoad.average(snapshot.count)),
		mlog.Duration("maxPhysicalLoad", snapshot.physicalLoad.max),
		mlog.Duration("avgReleaseResource", snapshot.releaseResource.average(snapshot.count)),
		mlog.Duration("maxReleaseResource", snapshot.releaseResource.max),
		mlog.Duration("avgOnLoaded", snapshot.onLoaded.average(snapshot.count)),
		mlog.Duration("maxOnLoaded", snapshot.onLoaded.max),
	)
}
