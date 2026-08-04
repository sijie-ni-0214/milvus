package broadcaster

import (
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const (
	broadcastStageWorkerWait         = "worker_wait"
	broadcastStageInitializeRecovery = "initialize_recovery"
	broadcastStageAppendMessages     = "append_messages"
	broadcastStageAppendResults      = "append_results"
	broadcastStageFastAckLockWait    = "fast_ack_lock_wait"
)

// newBroadcasterMetrics creates a new broadcaster metrics.
func newBroadcasterMetrics() *broadcasterMetrics {
	constLabel := prometheus.Labels{
		metrics.NodeIDLabelName: paramtable.GetStringNodeID(),
	}
	return &broadcasterMetrics{
		taskTotal:           metrics.StreamingCoordBroadcasterTaskTotal.MustCurryWith(constLabel),
		executionDuration:   metrics.StreamingCoordBroadcasterTaskExecutionDurationSeconds.MustCurryWith(constLabel),
		broadcastDuration:   metrics.StreamingCoordBroadcasterTaskBroadcastDurationSeconds.MustCurryWith(constLabel),
		broadcastStage:      metrics.StreamingCoordBroadcasterTaskBroadcastStageDurationSeconds.MustCurryWith(constLabel),
		ackWaitDuration:     metrics.StreamingCoordBroadcasterTaskAckWaitDurationSeconds.MustCurryWith(constLabel),
		ackCallbackDuration: metrics.StreamingCoordBroadcasterTaskAckCallbackDurationSeconds.MustCurryWith(constLabel),
		acquireLockDuration: metrics.StreamingCoordBroadcasterTaskAcquireLockDurationSeconds.MustCurryWith(constLabel),
	}
}

// broadcasterMetrics is the metrics of the broadcaster.
type broadcasterMetrics struct {
	taskTotal           *prometheus.GaugeVec
	executionDuration   prometheus.ObserverVec
	broadcastDuration   prometheus.ObserverVec
	broadcastStage      prometheus.ObserverVec
	ackWaitDuration     prometheus.ObserverVec
	ackCallbackDuration prometheus.ObserverVec
	acquireLockDuration prometheus.ObserverVec
}

// ObserveAcquireLockDuration observes the acquire lock duration.
func (m *broadcasterMetrics) ObserveAcquireLockDuration(from time.Time, rks []message.ResourceKey) {
	m.acquireLockDuration.WithLabelValues(formatResourceKeys(rks)).Observe(time.Since(from).Seconds())
}

// fromStateToState updates the metrics when the state of the broadcast task changes.
func (m *broadcasterMetrics) fromStateToState(msgType message.MessageType, from streamingpb.BroadcastTaskState, to streamingpb.BroadcastTaskState) {
	if from != streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_UNKNOWN {
		m.taskTotal.WithLabelValues(msgType.String(), from.String()).Dec()
	}
	if to != streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_DONE {
		m.taskTotal.WithLabelValues(msgType.String(), to.String()).Inc()
	}
}

// NewBroadcastTask creates a new broadcast task.
func (m *broadcasterMetrics) NewBroadcastTask(msgType message.MessageType, state streamingpb.BroadcastTaskState, rks []message.ResourceKey) *taskMetricsGuard {
	rks = uniqueSortResourceKeys(rks)
	g := &taskMetricsGuard{
		start:              time.Now(),
		ackCallbackBegin:   time.Now(),
		state:              state,
		resourceKeys:       formatResourceKeys(rks),
		broadcasterMetrics: m,
		messageType:        msgType,
	}
	g.fromStateToState(msgType, streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_UNKNOWN, state)
	return g
}

type taskMetricsGuard struct {
	mu                    sync.Mutex
	start                 time.Time
	broadcastDone         time.Time
	broadcastDoneObserved bool
	workerStartObserved   bool
	ackWaitDone           time.Time
	ackWaitDoneObserved   bool
	ackWaitObserved       bool
	ackCallbackBegin      time.Time
	state                 streamingpb.BroadcastTaskState
	resourceKeys          string
	messageType           message.MessageType
	*broadcasterMetrics
}

// ObserveBroadcastWorkerStart observes the duration from task creation to the first worker execution.
func (g *taskMetricsGuard) ObserveBroadcastWorkerStart() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.workerStartObserved {
		return
	}
	g.broadcastStage.WithLabelValues(g.messageType.String(), broadcastStageWorkerWait).Observe(time.Since(g.start).Seconds())
	g.workerStartObserved = true
}

// ObserveBroadcastStageDuration observes the duration of a broadcast sub-stage.
func (g *taskMetricsGuard) ObserveBroadcastStageDuration(stage string, start time.Time) {
	g.broadcastStage.WithLabelValues(g.messageType.String(), stage).Observe(time.Since(start).Seconds())
}

// ObserveStateChanged updates the state of the broadcast task.
func (g *taskMetricsGuard) ObserveStateChanged(state streamingpb.BroadcastTaskState) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.fromStateToState(g.messageType, g.state, state)
	if state == streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_TOMBSTONE {
		g.executionDuration.WithLabelValues(g.messageType.String()).Observe(time.Since(g.start).Seconds())
	}
	g.state = state
}

// ObserveBroadcastDone observes the broadcast done.
func (g *taskMetricsGuard) ObserveBroadcastDone() {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := time.Now()
	g.broadcastDone = now
	g.broadcastDoneObserved = true
	g.broadcastDuration.WithLabelValues(g.messageType.String()).Observe(now.Sub(g.start).Seconds())
	g.observeAckWaitDurationLocked()
}

// ObserveAckWaitDone observes the duration between broadcast append completion and all vchannels acked.
func (g *taskMetricsGuard) ObserveAckWaitDone() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.ackWaitDone = time.Now()
	g.ackWaitDoneObserved = true
	g.observeAckWaitDurationLocked()
}

func (g *taskMetricsGuard) observeAckWaitDurationLocked() {
	if g.ackWaitObserved || !g.broadcastDoneObserved || !g.ackWaitDoneObserved {
		return
	}
	duration := g.ackWaitDone.Sub(g.broadcastDone)
	if duration < 0 {
		duration = 0
	}
	g.ackWaitDuration.WithLabelValues(g.messageType.String()).Observe(duration.Seconds())
	g.ackWaitObserved = true
}

// ObserveAckCallbackBegin observes the ack callback begin.
func (g *taskMetricsGuard) ObserveAckCallbackBegin() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.ackCallbackBegin = time.Now()
}

// ObserveAckCallbackDone observes the ack callback done.
func (g *taskMetricsGuard) ObserveAckCallbackDone() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.ackCallbackDuration.WithLabelValues(g.messageType.String()).Observe(time.Since(g.ackCallbackBegin).Seconds())
}

// formatResourceKeys formats the resource keys.
func formatResourceKeys(rks []message.ResourceKey) string {
	keys := make([]string, 0, len(rks))
	for _, rk := range rks {
		keys = append(keys, rk.ShortString())
	}
	return strings.Join(keys, "|")
}
