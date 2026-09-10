package broadcaster

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/mocks/mock_metastore"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/registry"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/resource"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// configureTombstoneGCTest keeps the tombstone scheduler from collecting while
// a test drives ACK callbacks.
func configureTombstoneGCTest(t *testing.T) {
	t.Helper()
	paramtable.Init()
	cfg := &paramtable.Get().StreamingCfg
	for _, setting := range []struct {
		param *paramtable.ParamItem
		value string
	}{
		{&cfg.WALBroadcasterTombstoneMaxCount, "0"},
		{&cfg.WALBroadcasterTombstoneMaxLifetime, "1h"},
		{&cfg.WALBroadcasterTombstoneCheckInternal, "1h"},
	} {
		previous := setting.param.SwapTempValue(setting.value)
		t.Cleanup(func() { setting.param.SwapTempValue(previous) })
	}
}

func TestAckCallbackUnlocksAfterPersistenceBeforeGCHandoff(t *testing.T) {
	configureTombstoneGCTest(t)
	registry.ResetRegistration()
	registerDropCollectionNoopCallbacks()
	s := newAckCallbackScheduler(mlog.With())
	rk := message.NewExclusiveCollectionNameResourceKey("db", "collection")
	meta := mock_metastore.NewMockStreamingCoordCataLog(t)
	meta.EXPECT().SaveBroadcastTask(mock.Anything, uint64(1), mock.Anything).
		RunAndReturn(func(ctx context.Context, id uint64, task *streamingpb.BroadcastTask) error {
			guards, err := s.rkLocker.FastLock(rk)
			if err == nil {
				guards.Unlock()
				t.Error("ACK resource lock was released before TOMBSTONE persistence")
			}
			return nil
		}).Once()
	resource.InitForTest(resource.OptStreamingCatalog(meta))
	task := newBroadcastTaskFromProto(createNewWaitAckBroadcastTaskFromMessage(
		createNewBroadcastMsg([]string{"v1"}, rk).WithBroadcastID(1),
		streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_PENDING, []byte{1}), newBroadcasterMetrics(), s)
	task.SetLogger(mlog.With())
	guards := s.rkLocker.Lock(rk)

	// Hold the enqueue lock to verify the resource lock's lifetime independently
	// of whether GC handoff normally completes quickly.
	s.tombstoneScheduler.pendingMu.Lock()
	unlockPending := sync.OnceFunc(s.tombstoneScheduler.pendingMu.Unlock)
	var worker sync.WaitGroup
	defer func() {
		s.notifier.Cancel()
		unlockPending()
		worker.Wait()
	}()
	completed := make(chan error, 1)
	worker.Add(1)
	go func() {
		defer worker.Done()
		completed <- s.doAckCallback(task, guards)
	}()
	select {
	case <-s.triggerChan:
	case <-time.After(5 * time.Second):
		t.Fatal("ACK lock release waited for GC handoff")
	}
	nextGuards, err := s.rkLocker.FastLock(rk)
	require.NoError(t, err)
	nextGuards.Unlock()
	requireNoResult(t, completed, 10*time.Millisecond, "GC handoff should still be blocked")
	unlockPending()
	select {
	case err := <-completed:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("ACK callback did not finish GC handoff")
	}
}

func TestAckCallbackCancellationDoesNotEnqueueTombstone(t *testing.T) {
	for _, phase := range []string{"waiting_for_ack", "persisting_tombstone"} {
		t.Run(phase, func(t *testing.T) {
			configureTombstoneGCTest(t)
			registry.ResetRegistration()
			registerDropCollectionNoopCallbacks()
			s := newAckCallbackScheduler(mlog.With())
			defer s.notifier.Cancel()
			meta := mock_metastore.NewMockStreamingCoordCataLog(t)
			bitmap := []byte{0}
			if phase == "waiting_for_ack" {
				s.notifier.Cancel()
			} else {
				bitmap[0] = 1
				meta.EXPECT().SaveBroadcastTask(mock.Anything, uint64(1), mock.Anything).
					RunAndReturn(func(ctx context.Context, id uint64, task *streamingpb.BroadcastTask) error {
						s.notifier.Cancel()
						return ctx.Err()
					}).Once()
			}
			resource.InitForTest(resource.OptStreamingCatalog(meta))
			rk := message.NewExclusiveCollectionNameResourceKey("db", "collection")
			task := newBroadcastTaskFromProto(createNewWaitAckBroadcastTaskFromMessage(
				createNewBroadcastMsg([]string{"v1"}, rk).WithBroadcastID(1),
				streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_PENDING, bitmap), newBroadcasterMetrics(), s)
			task.SetLogger(mlog.With())
			require.ErrorIs(t, s.doAckCallback(task, s.rkLocker.Lock(rk)), context.Canceled)
			require.Zero(t, s.tombstoneScheduler.pending.Len())
			nextGuards, err := s.rkLocker.FastLock(rk)
			require.NoError(t, err)
			nextGuards.Unlock()
		})
	}
}

func registerDropCollectionNoopCallbacks() {
	registry.RegisterDropCollectionV1AckCallback(func(ctx context.Context, msg message.BroadcastResultDropCollectionMessageV1) error {
		return nil
	})
}
