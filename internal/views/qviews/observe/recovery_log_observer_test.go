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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/views/qviews"
)

func TestRecoveryLogObserverAggregatesQueryNodeProgress(t *testing.T) {
	now := time.Unix(100, 0)
	var snapshots []recoveryProgressSnapshot
	observer := newRecoveryLogObserver(func() time.Time { return now }, func(_ context.Context, snapshot recoveryProgressSnapshot) {
		snapshots = append(snapshots, snapshot)
	})
	view := testQueryViewKey()

	observer.Observe(context.Background(), QueryNodeAcquireSegmentsEvent{
		View:         view,
		SegmentCount: 10,
	})
	now = now.Add(6 * time.Second)
	observer.Observe(context.Background(), QueryNodeSegmentsReadyEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStatePreparing,
			To:   qviews.QueryViewStatePreparing,
		},
		ReadySegmentCount: 4,
	})
	observer.reportProgress()

	require.Len(t, snapshots, 1)
	assert.False(t, snapshots[0].terminal)
	assert.Equal(t, 10, snapshots[0].expectedSegments)
	assert.Equal(t, 4, snapshots[0].readySegments)
	assert.Equal(t, 6, snapshots[0].pendingSegments)
	assert.Equal(t, 1, snapshots[0].activeViews)
	assert.InDelta(t, 4.0/6.0, snapshots[0].readyRate, 0.001)

	now = now.Add(time.Second)
	observer.Observe(context.Background(), QueryNodeSegmentsReadyEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStatePreparing,
			To:   qviews.QueryViewStateReady,
		},
		ReadySegmentCount: 10,
	})

	require.Len(t, snapshots, 2)
	terminal := snapshots[1]
	assert.True(t, terminal.terminal)
	assert.Equal(t, 1, terminal.completedViews)
	assert.Zero(t, terminal.activeViews)
	assert.Equal(t, 10, terminal.readySegments)
	assert.Zero(t, terminal.pendingSegments)
}

func TestRecoveryLogObserverKeepsCoordViewActiveUntilUp(t *testing.T) {
	now := time.Unix(100, 0)
	var snapshots []recoveryProgressSnapshot
	observer := newRecoveryLogObserver(func() time.Time { return now }, func(_ context.Context, snapshot recoveryProgressSnapshot) {
		snapshots = append(snapshots, snapshot)
	})
	view := testQueryViewKey()

	observer.Observe(context.Background(), CoordViewCreatedEvent{
		View:  view,
		State: qviews.QueryViewStatePreparing,
	})
	now = now.Add(6 * time.Second)
	observer.Observe(context.Background(), CoordViewReportAppliedEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStatePreparing,
			To:   qviews.QueryViewStatePreparing,
		},
		Node:                 qviews.NewQueryNode(10),
		ReportedState:        qviews.QueryViewStatePreparing,
		ExpectedSegmentCount: 10,
		ReadySegmentCount:    4,
	})
	observer.reportProgress()
	now = now.Add(time.Second)
	observer.Observe(context.Background(), CoordViewReportAppliedEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStatePreparing,
			To:   qviews.QueryViewStateReady,
		},
		Node:                 qviews.NewQueryNode(10),
		ReportedState:        qviews.QueryViewStateReady,
		ExpectedSegmentCount: 10,
		ReadySegmentCount:    10,
	})

	require.Len(t, snapshots, 1)
	assert.Equal(t, 1, snapshots[0].activeViews)

	now = now.Add(time.Second)
	observer.Observe(context.Background(), CoordViewReportAppliedEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStateReady,
			To:   qviews.QueryViewStateUp,
		},
		Node:          qviews.NewStreamingNodeFromVChannel(view.ShardID.VChannel),
		ReportedState: qviews.QueryViewStateUp,
	})

	require.Len(t, snapshots, 2)
	terminal := snapshots[1]
	assert.True(t, terminal.terminal)
	assert.Equal(t, 1, terminal.completedViews)
	assert.Equal(t, 10, terminal.expectedSegments)
	assert.Equal(t, 10, terminal.readySegments)
}

func TestRecoveryLogObserverIgnoresLateTerminalReports(t *testing.T) {
	now := time.Unix(100, 0)
	var snapshots []recoveryProgressSnapshot
	observer := newRecoveryLogObserver(func() time.Time { return now }, func(_ context.Context, snapshot recoveryProgressSnapshot) {
		snapshots = append(snapshots, snapshot)
	})
	view := testQueryViewKey()

	observer.Observe(context.Background(), CoordViewReportAppliedEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStateUp,
			To:   qviews.QueryViewStateUp,
		},
		Node:          qviews.NewStreamingNodeFromVChannel(view.ShardID.VChannel),
		ReportedState: qviews.QueryViewStateUp,
	})
	observer.Observe(context.Background(), QueryNodeSegmentsReadyEvent{
		ViewStateTransition: ViewStateTransition{
			View: view,
			From: qviews.QueryViewStateReady,
			To:   qviews.QueryViewStateReady,
		},
		ReadySegmentCount: 1,
	})

	assert.Empty(t, snapshots)
	assert.Empty(t, observer.components)
}
