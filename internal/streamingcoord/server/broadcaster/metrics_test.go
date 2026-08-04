package broadcaster

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
)

func TestTaskMetricsGuardObserveBroadcastStageDurationDoesNotPanicOnLabels(t *testing.T) {
	metrics.StreamingCoordBroadcasterTaskBroadcastStageDurationSeconds.Reset()

	guard := newBroadcasterMetrics().NewBroadcastTask(
		message.MessageTypeCreateCollection,
		streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_PENDING,
		nil,
	)
	require.NotPanics(t, func() {
		guard.ObserveBroadcastStageDuration(broadcastStageAppendMessages, time.Now())
	})
	require.Equal(t, 1, testutil.CollectAndCount(metrics.StreamingCoordBroadcasterTaskBroadcastStageDurationSeconds))
}

func TestTaskMetricsGuardObserveAckWaitDoneDoesNotPanicOnLabels(t *testing.T) {
	metrics.StreamingCoordBroadcasterTaskAckWaitDurationSeconds.Reset()

	guard := newBroadcasterMetrics().NewBroadcastTask(
		message.MessageTypeDropCollection,
		streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_PENDING,
		nil,
	)
	require.NotPanics(t, func() {
		guard.ObserveBroadcastDone()
		guard.ObserveAckWaitDone()
	})
	require.Equal(t, 1, testutil.CollectAndCount(metrics.StreamingCoordBroadcasterTaskAckWaitDurationSeconds))
}
