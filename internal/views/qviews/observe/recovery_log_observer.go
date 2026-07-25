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

package observe

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
)

const recoveryProgressLogInterval = 5 * time.Second

type recoveryProgressLogFunc func(context.Context, recoveryProgressSnapshot)

// recoveryLogObserver aggregates QueryView recovery events into low-frequency
// progress logs. It does not participate in state transitions or scheduling.
type recoveryLogObserver struct {
	mu         sync.Mutex
	now        func() time.Time
	interval   time.Duration
	components map[string]*recoveryProgress
	log        recoveryProgressLogFunc
}

type recoveryProgress struct {
	active map[qviews.QueryViewKey]*recoveryViewProgress
	ctx    context.Context

	startedAt time.Time
	lastLogAt time.Time
	total     int
	completed int

	expectedSegments    int
	readySegments       int
	windowReadySegments int
	windowCompleted     int
}

type recoveryViewProgress struct {
	state            qviews.QueryViewState
	expectedSegments int
	readySegments    int
}

type recoveryProgressSnapshot struct {
	component string
	terminal  bool

	totalViews        int
	activeViews       int
	completedViews    int
	abortedViews      int
	preparingViews    int
	readyViews        int
	upRecoveringViews int
	otherActiveViews  int

	expectedSegments int
	readySegments    int
	pendingSegments  int
	windowReady      int
	windowCompleted  int
	readyRate        float64
	windowDuration   time.Duration
	elapsed          time.Duration
}

func newDefaultRecoveryLogObserver() *recoveryLogObserver {
	observer := newRecoveryLogObserver(time.Now, logRecoveryProgress)
	go observer.reportLoop()
	return observer
}

func newRecoveryLogObserver(now func() time.Time, log recoveryProgressLogFunc) *recoveryLogObserver {
	return &recoveryLogObserver{
		now:        now,
		interval:   recoveryProgressLogInterval,
		components: make(map[string]*recoveryProgress),
		log:        log,
	}
}

func (o *recoveryLogObserver) Observe(ctx context.Context, event Event) {
	now := o.now()
	o.mu.Lock()
	component, progress := o.recordLocked(event, now)
	if progress == nil {
		o.mu.Unlock()
		return
	}
	progress.ctx = ctx
	terminal := len(progress.active) == 0
	if !terminal {
		o.mu.Unlock()
		return
	}
	snapshot := progress.snapshot(component, now, terminal)
	delete(o.components, component)
	o.mu.Unlock()
	o.log(ctx, snapshot)
}

func (o *recoveryLogObserver) reportLoop() {
	ticker := time.NewTicker(o.interval)
	defer ticker.Stop()
	for range ticker.C {
		o.reportProgress()
	}
}

func (o *recoveryLogObserver) reportProgress() {
	type pendingLog struct {
		ctx      context.Context
		snapshot recoveryProgressSnapshot
	}
	now := o.now()
	o.mu.Lock()
	logs := make([]pendingLog, 0, len(o.components))
	for component, progress := range o.components {
		if len(progress.active) == 0 || now.Sub(progress.lastLogAt) < o.interval {
			continue
		}
		ctx := progress.ctx
		if ctx == nil {
			ctx = context.TODO()
		}
		logs = append(logs, pendingLog{ctx: ctx, snapshot: progress.snapshot(component, now, false)})
		progress.lastLogAt = now
		progress.windowReadySegments = 0
		progress.windowCompleted = 0
	}
	o.mu.Unlock()
	for _, pending := range logs {
		o.log(pending.ctx, pending.snapshot)
	}
}

func (o *recoveryLogObserver) recordLocked(event Event, now time.Time) (string, *recoveryProgress) {
	component := event.ComponentInfo()
	progress := o.components[component]

	switch e := event.(type) {
	case CoordViewCreatedEvent:
		progress, _ = o.trackView(component, e.View, e.State, 0, now)
	case CoordViewReportAppliedEvent:
		if progress == nil && e.From != qviews.QueryViewStatePreparing {
			return "", nil
		}
		progress, view := o.trackView(component, e.View, e.From, 0, now)
		view.state = e.To
		progress.setExpectedSegments(view, e.ExpectedSegmentCount)
		progress.setReadySegments(view, e.ReadySegmentCount)
		if e.To == qviews.QueryViewStateUp {
			progress.finishView(e.View)
		} else if e.To == qviews.QueryViewStateUnrecoverable ||
			e.To == qviews.QueryViewStateDropping || e.To == qviews.QueryViewStateDropped {
			progress.removeView(e.View)
		}
	case CoordViewPreemptedEvent:
		if progress != nil {
			progress.removeView(e.View)
		}
	case CoordViewQueryNodeLostAppliedEvent:
		if progress != nil && e.To == qviews.QueryViewStateUnrecoverable {
			progress.removeView(e.View)
		}
	case CoordViewReleaseRequestedEvent:
		if progress != nil {
			progress.removeView(e.View)
		}
	case QueryNodeAcquireSegmentsEvent:
		progress, _ = o.trackView(component, e.View, qviews.QueryViewStatePreparing, e.SegmentCount, now)
	case QueryNodeSegmentsReadyEvent:
		if progress == nil && e.From != qviews.QueryViewStatePreparing {
			return "", nil
		}
		progress, view := o.trackView(component, e.View, e.From, 0, now)
		view.state = e.To
		progress.setReadySegments(view, e.ReadySegmentCount)
		if e.To == qviews.QueryViewStateReady {
			progress.finishView(e.View)
		}
	case QueryNodeSegmentUnrecoverableEvent:
		if progress != nil {
			progress.removeView(e.View)
		}
	case QueryNodeReleaseDoneEvent:
		if progress != nil {
			progress.removeView(e.View)
		}
	case StreamingNodeAcquireResourceEvent:
		progress, _ = o.trackView(component, e.View, qviews.QueryViewStatePreparing, 0, now)
	case StreamingNodeRecoverAcquireResourceEvent:
		progress, _ = o.trackView(component, e.View, qviews.QueryViewStateUpRecovering, 0, now)
	case StreamingNodeResourceReadyEvent:
		if progress != nil {
			progress.finishView(e.View)
		}
	case StreamingNodeRecoveringDoneEvent:
		if progress != nil {
			progress.finishView(e.View)
		}
	case StreamingNodeReleaseDoneEvent:
		if progress != nil {
			progress.removeView(e.View)
		}
	default:
		return "", nil
	}
	return component, progress
}

func (o *recoveryLogObserver) trackView(
	component string,
	key qviews.QueryViewKey,
	state qviews.QueryViewState,
	expectedSegments int,
	now time.Time,
) (*recoveryProgress, *recoveryViewProgress) {
	progress := o.components[component]
	if progress == nil {
		progress = &recoveryProgress{
			active:    make(map[qviews.QueryViewKey]*recoveryViewProgress),
			startedAt: now,
			lastLogAt: now,
		}
		o.components[component] = progress
	}
	view := progress.active[key]
	if view == nil {
		view = &recoveryViewProgress{state: state}
		progress.active[key] = view
		progress.total++
	}
	view.state = state
	progress.setExpectedSegments(view, expectedSegments)
	return progress, view
}

func (p *recoveryProgress) setExpectedSegments(view *recoveryViewProgress, expected int) {
	if expected <= view.expectedSegments {
		return
	}
	p.expectedSegments += expected - view.expectedSegments
	view.expectedSegments = expected
}

func (p *recoveryProgress) setReadySegments(view *recoveryViewProgress, ready int) {
	if ready <= view.readySegments {
		return
	}
	if view.expectedSegments > 0 {
		ready = min(ready, view.expectedSegments)
	} else {
		p.setExpectedSegments(view, ready)
	}
	delta := ready - view.readySegments
	view.readySegments = ready
	p.readySegments += delta
	p.windowReadySegments += delta
}

func (p *recoveryProgress) finishView(key qviews.QueryViewKey) {
	if _, ok := p.active[key]; !ok {
		return
	}
	delete(p.active, key)
	p.completed++
	p.windowCompleted++
}

func (p *recoveryProgress) removeView(key qviews.QueryViewKey) {
	delete(p.active, key)
}

func (p *recoveryProgress) snapshot(component string, now time.Time, terminal bool) recoveryProgressSnapshot {
	windowDuration := max(now.Sub(p.lastLogAt), time.Duration(0))
	snapshot := recoveryProgressSnapshot{
		component:        component,
		terminal:         terminal,
		totalViews:       p.total,
		activeViews:      len(p.active),
		completedViews:   p.completed,
		abortedViews:     p.total - len(p.active) - p.completed,
		expectedSegments: p.expectedSegments,
		readySegments:    p.readySegments,
		pendingSegments:  max(0, p.expectedSegments-p.readySegments),
		windowReady:      p.windowReadySegments,
		windowCompleted:  p.windowCompleted,
		windowDuration:   windowDuration,
		elapsed:          max(now.Sub(p.startedAt), time.Duration(0)),
	}
	if windowDuration > 0 {
		snapshot.readyRate = float64(p.windowReadySegments) / windowDuration.Seconds()
	}
	for _, view := range p.active {
		switch view.state {
		case qviews.QueryViewStatePreparing:
			snapshot.preparingViews++
		case qviews.QueryViewStateReady:
			snapshot.readyViews++
		case qviews.QueryViewStateUpRecovering:
			snapshot.upRecoveringViews++
		default:
			snapshot.otherActiveViews++
		}
	}
	return snapshot
}

func logRecoveryProgress(ctx context.Context, snapshot recoveryProgressSnapshot) {
	message := "[SN recovery] query view recovery progress"
	if snapshot.terminal {
		message = "[SN recovery] query view recovery completed"
	}
	mlog.Info(
		ctx, message,
		mlog.String("phase", "query_view_recovery_progress"),
		mlog.String("component", snapshot.component),
		mlog.Bool("terminal", snapshot.terminal),
		mlog.Int("totalViews", snapshot.totalViews),
		mlog.Int("activeViews", snapshot.activeViews),
		mlog.Int("completedViews", snapshot.completedViews),
		mlog.Int("abortedViews", snapshot.abortedViews),
		mlog.Int("preparingViews", snapshot.preparingViews),
		mlog.Int("readyViews", snapshot.readyViews),
		mlog.Int("upRecoveringViews", snapshot.upRecoveringViews),
		mlog.Int("otherActiveViews", snapshot.otherActiveViews),
		mlog.Int("expectedSegments", snapshot.expectedSegments),
		mlog.Int("readySegments", snapshot.readySegments),
		mlog.Int("pendingSegments", snapshot.pendingSegments),
		mlog.Int("windowReadySegments", snapshot.windowReady),
		mlog.Int("windowCompletedViews", snapshot.windowCompleted),
		mlog.Float64("readySegmentsPerSecond", snapshot.readyRate),
		mlog.Duration("windowDuration", snapshot.windowDuration),
		mlog.Duration("elapsed", snapshot.elapsed),
	)
}
