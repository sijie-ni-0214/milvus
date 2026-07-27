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

package idf

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
)

const (
	bm25RuntimePrepareTimingLogInterval = 5 * time.Second
	bm25RuntimePrepareTimingLogBatch    = 2000
)

var (
	eagerBM25RuntimePrepareTimingStats = bm25RuntimePrepareTimingStats{
		interval:  bm25RuntimePrepareTimingLogInterval,
		batchSize: bm25RuntimePrepareTimingLogBatch,
	}
	lazyBM25RuntimePrepareTimingStats = bm25RuntimePrepareTimingStats{
		interval:  bm25RuntimePrepareTimingLogInterval,
		batchSize: bm25RuntimePrepareTimingLogBatch,
	}
)

type bm25RuntimePrepareTimingSample struct {
	startedAt time.Time
	total     time.Duration
	failed    bool
}

type bm25RuntimePrepareDurationStats struct {
	total time.Duration
	max   time.Duration
}

func (s *bm25RuntimePrepareDurationStats) add(duration time.Duration) {
	s.total += duration
	s.max = max(s.max, duration)
}

func (s bm25RuntimePrepareDurationStats) average(count int64) time.Duration {
	if count == 0 {
		return 0
	}
	return s.total / time.Duration(count)
}

type bm25RuntimePrepareTimingSnapshot struct {
	count          int64
	failed         int64
	windowDuration time.Duration
	total          bm25RuntimePrepareDurationStats
}

type bm25RuntimePrepareTimingStats struct {
	mu         sync.Mutex
	interval   time.Duration
	batchSize  int64
	windowFrom time.Time

	count  int64
	failed int64
	total  bm25RuntimePrepareDurationStats
}

func recordBM25RuntimePrepareTiming(ctx context.Context, lazy bool, sample bm25RuntimePrepareTimingSample) {
	stats := &eagerBM25RuntimePrepareTimingStats
	if lazy {
		stats = &lazyBM25RuntimePrepareTimingStats
	}
	snapshot, ok := stats.add(time.Now(), sample)
	if !ok {
		return
	}
	mlog.Info(
		ctx, "[SN recovery] BM25 runtime prepare timing",
		mlog.String("phase", "bm25_runtime_prepare_timing"),
		mlog.String("component", "streamingNode"),
		mlog.Bool("lazyLoadSealedStats", lazy),
		mlog.Int64("count", snapshot.count),
		mlog.Int64("failed", snapshot.failed),
		mlog.Duration("windowDuration", snapshot.windowDuration),
		mlog.Duration("totalDuration", snapshot.total.total),
		mlog.Duration("avgTotal", snapshot.total.average(snapshot.count)),
		mlog.Duration("maxTotal", snapshot.total.max),
	)
}

func (s *bm25RuntimePrepareTimingStats) add(completedAt time.Time, sample bm25RuntimePrepareTimingSample) (bm25RuntimePrepareTimingSnapshot, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	startedAt := sample.startedAt
	if startedAt.IsZero() || startedAt.After(completedAt) {
		startedAt = completedAt
	}
	if s.windowFrom.IsZero() || startedAt.Before(s.windowFrom) {
		s.windowFrom = startedAt
	}
	s.count++
	if sample.failed {
		s.failed++
	}
	s.total.add(sample.total)

	if s.count < s.batchSize && completedAt.Sub(s.windowFrom) < s.interval {
		return bm25RuntimePrepareTimingSnapshot{}, false
	}
	snapshot := bm25RuntimePrepareTimingSnapshot{
		count:          s.count,
		failed:         s.failed,
		windowDuration: max(completedAt.Sub(s.windowFrom), time.Duration(0)),
		total:          s.total,
	}
	s.reset()
	return snapshot, true
}

func (s *bm25RuntimePrepareTimingStats) reset() {
	s.windowFrom = time.Time{}
	s.count = 0
	s.failed = 0
	s.total = bm25RuntimePrepareDurationStats{}
}
