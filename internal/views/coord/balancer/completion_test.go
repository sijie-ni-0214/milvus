package balancer

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	balancercache "github.com/milvus-io/milvus/internal/views/coord/balancer/cache"
	"github.com/milvus-io/milvus/internal/views/coord/coordview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/syncutil"
)

type completionTestPolicy func(balancercache.Reader, []qviews.ShardID) *BalancePlan

func (p completionTestPolicy) Plan(reader balancercache.Reader, dirty []qviews.ShardID) *BalancePlan {
	return p(reader, dirty)
}

func pendingCompletion(t *testing.T, b *DefaultBalancer, count int) *syncutil.Future[error] {
	t.Helper()
	var result *syncutil.Future[error]
	require.Eventually(t, func() bool {
		b.queue.mu.Lock()
		defer b.queue.mu.Unlock()
		result = b.queue.completion
		return result != nil && len(b.queue.dirtyColls) == count
	}, 5*time.Second, time.Millisecond)
	return result
}

func TestTriggerWithCompletionCoalescesConcurrentReleases(t *testing.T) {
	for _, fail := range []bool{false, true} {
		name := "success"
		if fail {
			name = "planning failure"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			c := balancercache.New(nil)
			t.Cleanup(c.Close)
			c.PublishShard(cacheShard(1000, 10000), &coordview.ShardStats{})
			c.MarkReady()
			entered := make(chan struct{})
			gate := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(gate) })
			planned := make(chan []qviews.ShardID, 1)
			var calls atomic.Int32
			policy := completionTestPolicy(func(_ balancercache.Reader, dirty []qviews.ShardID) *BalancePlan {
				if calls.Add(1) == 1 {
					close(entered)
					<-gate
					return nil
				}
				select {
				case planned <- slices.Clone(dirty):
				default:
				}
				if fail {
					return &BalancePlan{Retries: dirty}
				}
				return nil
			})
			b := NewDefaultBalancer(c, &coordview.ShardViewRegistry{}, policy, nil)
			b.Start(ctx)
			t.Cleanup(b.Stop)
			t.Cleanup(unblock)
			<-entered

			const waiters = 64
			results := make(chan *syncutil.Future[error], waiters)
			var expected []qviews.ShardID
			for i := range waiters {
				id := cacheShard(int64(i+1), int64(i+1)*10)
				c.PublishShard(id, &coordview.ShardStats{})
				expected = append(expected, id)
				go func() {
					results <- b.TriggerWithCompletion(TriggerScope{DirtyCollections: []int64{int64(i + 1)}})
				}()
			}
			completion := pendingCompletion(t, b, waiters)
			for range waiters {
				require.Same(t, completion, <-results, "concurrent releases share the pending batch")
			}
			require.False(t, completion.Ready())
			unblock()
			result, err := completion.GetWithContext(ctx)
			require.NoError(t, err)
			if fail {
				require.ErrorIs(t, result, merr.ErrServiceUnavailable)
				require.Equal(t, merr.Code(merr.ErrServiceUnavailable), merr.Code(result))
			} else {
				require.NoError(t, result)
			}
			require.ElementsMatch(t, expected, <-planned)
			if !fail {
				require.Equal(t, int32(2), calls.Load(), "one initial pass and one shared pass for all 64 releases")
			}
		})
	}
}

func TestTriggerWithCompletionUsesSuccessorBatch(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	c := balancercache.New(nil)
	t.Cleanup(c.Close)
	a, late := cacheShard(1, 10), cacheShard(2, 20)
	c.PublishShard(a, &coordview.ShardStats{})
	firstStarted, secondStarted := make(chan struct{}), make(chan struct{})
	firstGate, secondGate := make(chan struct{}), make(chan struct{})
	unblockFirst := sync.OnceFunc(func() { close(firstGate) })
	unblockSecond := sync.OnceFunc(func() { close(secondGate) })
	var calls atomic.Int32
	policy := completionTestPolicy(func(_ balancercache.Reader, _ []qviews.ShardID) *BalancePlan {
		switch calls.Add(1) {
		case 1:
			close(firstStarted)
			<-firstGate
		case 2:
			close(secondStarted)
			<-secondGate
		}
		return nil
	})
	b := NewDefaultBalancer(c, &coordview.ShardViewRegistry{}, policy, nil)
	b.Start(ctx)
	t.Cleanup(b.Stop)
	t.Cleanup(unblockFirst)
	t.Cleanup(unblockSecond)
	first := b.TriggerWithCompletion(TriggerScope{DirtyCollections: []int64{1}})
	c.MarkReady()
	<-firstStarted

	c.PublishShard(late, &coordview.ShardStats{})
	second := b.TriggerWithCompletion(TriggerScope{DirtyCollections: []int64{2}})
	require.NotSame(t, first, second)
	unblockFirst()
	result, err := first.GetWithContext(ctx)
	require.NoError(t, err)
	require.NoError(t, result)
	<-secondStarted
	require.False(t, second.Ready(), "a release arriving during a pass cannot use that pass's completion")
	unblockSecond()
	result, err = second.GetWithContext(ctx)
	require.NoError(t, err)
	require.NoError(t, result)
}

func TestTriggerWithCompletionRetainsPendingBatch(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	c := balancercache.New(nil)
	t.Cleanup(c.Close)
	b := NewDefaultBalancer(c, &coordview.ShardViewRegistry{}, nil, nil)
	scope := TriggerScope{DirtyCollections: []int64{1}}
	completion := b.TriggerWithCompletion(scope)
	b.Start(ctx)
	t.Cleanup(b.Stop)
	b.Stop()
	require.False(t, completion.Ready(), "the runtime owns shutdown notification to callers")
	require.Same(t, completion, b.TriggerWithCompletion(scope))
	c.MarkReady()
	b.Start(ctx)
	result, err := completion.GetWithContext(ctx)
	require.NoError(t, err)
	require.NoError(t, result)
}

func TestTriggerWithCompletionDrainsOldPreparation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	registry := emptyRegistry(t)
	c := newTestCache(cfgFor(1, 10, []int64{1}, nil))
	t.Cleanup(c.Close)
	id := cacheShard(1, 10)
	c.PublishDataView(1, cacheData(1, id.VChannel, 100))
	c.PublishNode(1, &NodeInfo{NodeID: 1, Alive: true, ResourceGroup: "rg1"})
	t.Cleanup(registry.RegisterPublicationListener(c.PublishShard))
	c.MarkReady()
	oldStarted, releaseStarted := make(chan struct{}), make(chan struct{})
	oldGate, releaseGate := make(chan struct{}), make(chan struct{})
	unblockOld := sync.OnceFunc(func() { close(oldGate) })
	unblockRelease := sync.OnceFunc(func() { close(releaseGate) })
	var calls atomic.Int32
	policy := completionTestPolicy(func(_ balancercache.Reader, _ []qviews.ShardID) *BalancePlan {
		if calls.Add(1) == 1 {
			builder := qviews.NewQueryViewAtCoordBuilder(id.ReplicaID, &viewpb.DataViewOfCollection{
				CollectionId: 1, DataVersion: &viewpb.DataVersion{StreamingVersion: 1},
				Shards: []*viewpb.DataViewOfShard{{Vchannel: id.VChannel}},
			}, id.VChannel)
			builder.SetAssignments(map[int64]map[int64][]int64{1: {1: {1000}}})
			close(oldStarted)
			<-oldGate
			return &BalancePlan{Prepares: map[qviews.ShardID]*qviews.QueryViewAtCoordBuilder{id: builder}}
		}
		if calls.Load() == 2 {
			close(releaseStarted)
			<-releaseGate
			return &BalancePlan{Releases: []qviews.ShardID{id}}
		}
		return nil
	})
	b := NewDefaultBalancer(c, registry, policy, nil)
	b.Start(ctx)
	t.Cleanup(b.Stop)
	t.Cleanup(unblockOld)
	t.Cleanup(unblockRelease)
	<-oldStarted
	require.Empty(t, registry.CollectionShards(1))
	c.PublishLoadConfig(1, nil, 2)
	completion := b.TriggerWithCompletion(TriggerScope{DirtyCollections: []int64{1}})
	unblockOld()
	<-releaseStarted
	require.NotNil(t, registry.Get(id), "the old pass can publish a prepare after desired state was removed")
	require.False(t, completion.Ready(), "release must wait for its successor pass, not an empty-registry check")
	unblockRelease()
	result, err := completion.GetWithContext(ctx)
	require.NoError(t, err)
	require.NoError(t, result)
}
