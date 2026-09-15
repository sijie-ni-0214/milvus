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

package proxy

import (
	"context"
	"time"

	"golang.org/x/time/rate"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/stage"
)

type (
	dqlTimingKey struct{}
	dqlTiming    struct {
		ctx             context.Context
		operation, path string
		started, ready  time.Time
		timer           stage.Timer
	}
)

var dqlRecorders = map[string]*stage.Recorder{
	"Search":       stage.New("proxy", "Search", "request"),
	"HybridSearch": stage.New("proxy", "HybridSearch", "request"),
	"Query":        stage.New("proxy", "Query", "request"),
}

func startDQL(ctx context.Context, operation string) (context.Context, *dqlTiming) {
	t := &dqlTiming{ctx: ctx, operation: operation, path: stage.PathUnknown, started: time.Now(), timer: dqlRecorders[operation].Begin()}
	return context.WithValue(ctx, dqlTimingKey{}, t), t
}

// pathRank orders readiness paths from warmest to coldest. A retried request
// keeps the coldest path it observed, so a later attempt on the loaded
// collection cannot relabel a request that waited for the load.
var pathRank = map[string]int{
	stage.PathUnknown:  0,
	stage.PathDisabled: 1,
	stage.PathReady:    2,
	stage.PathLoading:  3,
	stage.PathUnloaded: 4,
}

// Only the caller sets its observed readiness path. Shared load workers outlive
// callers and must never mutate this request-local state. Unknown values, such
// as an empty path from an older QueryCoord, are ignored to keep labels bounded.
func setDQLPath(ctx context.Context, path string) {
	if t, ok := ctx.Value(dqlTimingKey{}).(*dqlTiming); ok {
		if rank, known := pathRank[path]; known && rank > pathRank[t.path] {
			t.path = path
		}
	}
}

// markDQLReady ends readiness at the latest attempt whose readiness check
// succeeded; time spent in earlier failed attempts counts as readiness.
func markDQLReady(ctx context.Context) {
	if t, ok := ctx.Value(dqlTimingKey{}).(*dqlTiming); ok {
		t.Ready()
	}
}

func (t *dqlTiming) Ready() { t.ready = time.Now() }
func (t *dqlTiming) End(status *commonpb.Status, err error) {
	if err == nil {
		err = merr.Error(status)
	}
	t.finish(time.Now(), stage.Outcome(err))
}

func (t *dqlTiming) finish(ended time.Time, result stage.Result) {
	total := ended.Sub(t.started)
	readiness := total
	if !t.ready.IsZero() {
		readiness = t.ready.Sub(t.started)
	}
	execution := total - readiness
	cohort := "le1s"
	if total > time.Second {
		cohort = "gt1s"
	}
	for i, name := range [...]string{"total", "readiness", "execution"} {
		d := [...]time.Duration{total, readiness, execution}[i]
		metrics.QueryRequestStageDuration.WithLabelValues(t.operation, t.path, cohort, name, result.String()).Observe(d.Seconds())
	}
	if cohort == "gt1s" || result != stage.Success {
		metrics.QueryStageItems.WithLabelValues("proxy", t.operation, "slow_summary", "candidate").Inc()
		if mlog.LevelEnabled(mlog.InfoLevel) && dqlSummaryLimiter.Allow() {
			metrics.QueryStageItems.WithLabelValues("proxy", t.operation, "slow_summary", "emitted").Inc()
			mlog.Info(t.ctx, "DQL request stages", mlog.String("operation", t.operation), mlog.String("loadPath", t.path),
				mlog.String("result", result.String()), mlog.Duration("total", total), mlog.Duration("readiness", readiness), mlog.Duration("execution", execution))
		} else {
			metrics.QueryStageItems.WithLabelValues("proxy", t.operation, "slow_summary", "dropped").Inc()
		}
	}
	t.timer.EndResult(result)
}

var dqlSummaryLimiter = rate.NewLimiter(2, 10)
