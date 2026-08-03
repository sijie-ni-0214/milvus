package idf

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/semaphore"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/walview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/messagespb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/nodescheduler"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	testBM25InputFieldID  = int64(101)
	testBM25OutputFieldID = int64(102)
)

type blockingNodeSchedulerTask struct {
	started chan struct{}
	release chan struct{}
}

type observedDoneContext struct {
	context.Context
	doneObserved chan struct{}
	once         sync.Once
}

func (c *observedDoneContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.doneObserved) })
	return c.Context.Done()
}

func (t *blockingNodeSchedulerTask) Execute(ctx context.Context) error {
	close(t.started)
	select {
	case <-t.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func newScheduledOracleRuntime(scheduler nodescheduler.Scheduler, current qviews.DataVersion) *oracleRuntime {
	return &oracleRuntime{
		provider:       &Provider{},
		scheduler:      scheduler,
		collectionID:   1,
		vchannel:       "v1",
		currentVersion: current,
		currentStats:   make(bm25Stats),
		currentSealed:  make(map[int64]sealedContribution),
		currentGrowing: make(map[int64]growingContribution),
		growingStore:   newGrowingStatsStore(nil),
	}
}

func TestOracleRuntimeSchedulesCoalescedAdvance(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10, CompactVersion: 1}
	first := qviews.DataVersion{StreamingVersion: 11, CompactVersion: 1}
	latest := qviews.DataVersion{StreamingVersion: 12, CompactVersion: 1}
	scheduler := nodescheduler.New(1)
	defer scheduler.Close()

	blocker := &blockingNodeSchedulerTask{started: make(chan struct{}), release: make(chan struct{})}
	scheduler.Submit(blocker)
	<-blocker.started

	var callsMu sync.Mutex
	calls := make([]qviews.DataVersion, 0, 1)
	mock := mockey.Mock((*Provider).getSealedBM25Resources).To(func(
		_ *Provider,
		_ context.Context,
		_ int64,
		_ string,
		version qviews.DataVersion,
		_ []int64,
		_ uint64,
	) ([]*datapb.StreamingNodeBM25Resource, error) {
		callsMu.Lock()
		calls = append(calls, version)
		callsMu.Unlock()
		return nil, nil
	}).Build()
	defer mock.UnPatch()

	runtime := newScheduledOracleRuntime(scheduler, current)
	runtime.MaybeAdvance(first)
	runtime.MaybeAdvance(latest)

	runtime.mu.RLock()
	require.True(t, runtime.advanceScheduled)
	require.True(t, runtime.pending.EQ(latest))
	runtime.mu.RUnlock()

	close(blocker.release)
	require.Eventually(t, func() bool {
		runtime.mu.RLock()
		defer runtime.mu.RUnlock()
		return runtime.currentVersion.EQ(latest) && !runtime.advanceScheduled
	}, time.Second, 10*time.Millisecond)
	runtime.Close()

	callsMu.Lock()
	require.Equal(t, []qviews.DataVersion{latest}, calls)
	callsMu.Unlock()
}

func TestAcquireSealedContributionsUsesSharedLimit(t *testing.T) {
	stats := storage.NewBM25Stats()
	stats.Append(map[uint32]float32{1: 1})
	statsBytes, err := stats.Serialize()
	require.NoError(t, err)

	chunkManager := mocks.NewChunkManager(t)
	var active atomic.Int32
	var maxActive atomic.Int32
	var started atomic.Int32
	firstBatchStarted := make(chan struct{})
	release := make(chan struct{})
	chunkManager.EXPECT().Read(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, _ string) ([]byte, error) {
			current := active.Add(1)
			defer active.Add(-1)
			for {
				observed := maxActive.Load()
				if current <= observed || maxActive.CompareAndSwap(observed, current) {
					break
				}
			}
			if started.Add(1) == 2 {
				close(firstBatchStarted)
			}
			select {
			case <-release:
				return statsBytes, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}).Times(4)

	limiter := semaphore.NewWeighted(2)
	providers := []*Provider{
		{chunkManager: chunkManager, sealedCache: newSegmentCache(), sealedStatsLoadLimiter: limiter},
		{chunkManager: chunkManager, sealedCache: newSegmentCache(), sealedStatsLoadLimiter: limiter},
	}
	type result struct {
		contributions map[int64]sealedContribution
		err           error
	}
	results := make(chan result, len(providers))
	for i, provider := range providers {
		resources := testSealedBM25Resources(int64(i*2+1), 2)
		go func(provider *Provider, resources []*datapb.StreamingNodeBM25Resource) {
			contributions, err := provider.acquireSealedContributions(context.Background(), resources)
			results <- result{contributions: contributions, err: err}
		}(provider, resources)
	}

	select {
	case <-firstBatchStarted:
	case <-time.After(time.Second):
		t.Fatal("sealed BM25 stats loads did not start")
	}
	close(release)
	for range providers {
		result := <-results
		require.NoError(t, result.err)
		require.Len(t, result.contributions, 2)
		for _, contribution := range result.contributions {
			contribution.lease.Close()
		}
	}
	require.Equal(t, int32(2), maxActive.Load())
}

func TestAcquireSealedContributionsReleasesLeasesAfterError(t *testing.T) {
	stats := storage.NewBM25Stats()
	stats.Append(map[uint32]float32{1: 1})
	statsBytes, err := stats.Serialize()
	require.NoError(t, err)

	goodRead := make(chan struct{})
	chunkManager := mocks.NewChunkManager(t)
	chunkManager.EXPECT().Read(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, path string) ([]byte, error) {
			if path == "stats-1" {
				close(goodRead)
				return statsBytes, nil
			}
			<-goodRead
			return nil, merr.WrapErrDataIntegrityMsg("test sealed stats read failure")
		}).Times(2)

	provider := &Provider{
		chunkManager:           chunkManager,
		sealedCache:            newSegmentCache(),
		sealedStatsLoadLimiter: semaphore.NewWeighted(2),
	}
	_, err = provider.acquireSealedContributions(context.Background(), testSealedBM25Resources(1, 2))
	require.Error(t, err)
	require.Empty(t, provider.sealedCache.entries)
}

func TestOracleRuntimeCloseCancelsScheduledAdvance(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10, CompactVersion: 1}
	target := qviews.DataVersion{StreamingVersion: 11, CompactVersion: 1}
	scheduler := nodescheduler.New(1)
	defer scheduler.Close()

	started := make(chan struct{})
	canceled := make(chan struct{})
	mock := mockey.Mock((*Provider).getSealedBM25Resources).To(func(
		_ *Provider,
		ctx context.Context,
		_ int64,
		_ string,
		_ qviews.DataVersion,
		_ []int64,
		_ uint64,
	) ([]*datapb.StreamingNodeBM25Resource, error) {
		close(started)
		<-ctx.Done()
		close(canceled)
		return nil, ctx.Err()
	}).Build()
	defer mock.UnPatch()

	runtime := newScheduledOracleRuntime(scheduler, current)
	runtime.MaybeAdvance(target)
	<-started
	runtime.Close()

	select {
	case <-canceled:
	default:
		t.Fatal("scheduled IDF advance was not canceled")
	}
}

func TestOracleRuntimeRejectsStaleAdvanceDiff(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10, CompactVersion: 1}
	target := qviews.DataVersion{StreamingVersion: 11, CompactVersion: 1}
	runtime := &oracleRuntime{
		currentVersion: current,
		currentStats:   make(bm25Stats),
		currentSealed:  make(map[int64]sealedContribution),
		currentGrowing: map[int64]growingContribution{
			20: {
				segmentID:   20,
				partitionID: 10,
				stats:       make(bm25Stats),
			},
		},
		growingStore: newGrowingStatsStore(nil),
		revision:     2,
	}

	committed, retry := runtime.commitDiff(&idfDiff{
		target:      target,
		revision:    1,
		positive:    make(bm25Stats),
		negative:    make(bm25Stats),
		nextSealed:  make(map[int64]sealedContribution),
		nextGrowing: make(map[int64]growingContribution),
	})

	require.False(t, committed)
	require.True(t, retry)
	require.True(t, runtime.currentVersion.EQ(current))
	require.Contains(t, runtime.currentGrowing, int64(20))
}

func TestOracleRuntimePreparesAndServesExactDataVersion(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10}
	target := qviews.DataVersion{StreamingVersion: 11}
	fieldID := int64(102)
	schema := &schemapb.CollectionSchema{Functions: []*schemapb.FunctionSchema{{
		Type:           schemapb.FunctionType_BM25,
		OutputFieldIds: []int64{fieldID},
	}}}
	growingStore := newGrowingStatsStore(schema)
	stats := storage.NewBM25Stats()
	stats.Append(map[uint32]float32{7: 2})
	growingStore.appendStats(20, 10, bm25Stats{fieldID: stats})

	mock := mockey.Mock((*Provider).getSealedBM25Resources).Return(nil, nil).Build()
	defer mock.UnPatch()
	runtime := &oracleRuntime{
		provider:       &Provider{sealedCache: newSegmentCache()},
		collectionID:   1,
		vchannel:       "v1",
		schema:         schema,
		currentVersion: current,
		currentStats:   newBM25StatsFromSchema(schema),
		currentSealed:  make(map[int64]sealedContribution),
		currentGrowing: make(map[int64]growingContribution),
		growingStore:   growingStore,
	}

	require.NoError(t, runtime.PrepareDataVersion(context.Background(), target))
	query := &schemapb.SparseFloatArray{Contents: [][]byte{
		typeutil.CreateAndSortSparseFloatRow(map[uint32]float32{7: 1}),
	}}
	_, avgdl, err := runtime.BuildIDF(context.Background(), target, fieldID, query)
	require.NoError(t, err)
	require.Equal(t, float64(2), avgdl)

	runtime.ReleaseDataVersion(target)
	_, _, err = runtime.BuildIDF(context.Background(), target, fieldID, query)
	require.Error(t, err)
}

func TestOracleRuntimeLazyPreparesAndServesExactDataVersion(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10}
	target := qviews.DataVersion{StreamingVersion: 11}
	client := mocks.NewMockDataCoordClient(t)
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			require.True(t, qviews.FromProtoDataVersion(req.GetDataVersion()).EQ(target))
			return testBM25ResourceResponse(req), nil
		}).Once()
	oracle := newTestLazyOracleRuntime(t, client, current)
	defer oracle.Close()

	stats := storage.NewBM25Stats()
	stats.Append(map[uint32]float32{7: 2})
	oracle.growingStore.appendStats(20, 10, bm25Stats{testBM25OutputFieldID: stats})

	require.NoError(t, oracle.PrepareDataVersion(context.Background(), target))
	require.Empty(t, client.Calls)
	_, avgdl, err := oracle.BuildIDF(context.Background(), target, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.Equal(t, float64(2), avgdl)
	require.Len(t, client.Calls, 1)
	require.True(t, oracleStatsReady(oracle, target))

	oracle.ReleaseDataVersion(target)
	_, _, err = oracle.BuildIDF(context.Background(), target, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.Error(t, err)
	require.Len(t, client.Calls, 1)
}

func TestOracleRuntimeEagerPreparationCancellationIsIsolated(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10}
	target := qviews.DataVersion{StreamingVersion: 11}
	client := mocks.NewMockDataCoordClient(t)
	firstStarted := make(chan struct{})
	var calls atomic.Int32
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			if calls.Add(1) == 1 {
				close(firstStarted)
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return testBM25ResourceResponse(req), nil
		}).Twice()
	oracle := newTestEagerOracleRuntime(t, client, current)
	defer oracle.Close()

	firstCtx, cancelFirst := context.WithCancel(context.Background())
	defer cancelFirst()
	firstResult := make(chan error, 1)
	go func() {
		firstResult <- oracle.PrepareDataVersion(firstCtx, target)
	}()
	<-firstStarted

	secondResult := make(chan error, 1)
	go func() {
		secondResult <- oracle.PrepareDataVersion(context.Background(), target)
	}()
	select {
	case err := <-secondResult:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("second eager preparation waited for the canceled caller")
	}
	require.True(t, oracleStatsReady(oracle, target))

	cancelFirst()
	require.ErrorIs(t, <-firstResult, context.Canceled)
	require.Len(t, client.Calls, 2)
}

func TestOracleRuntimeSharesCanceledVersionMaterialization(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}
	client := mocks.NewMockDataCoordClient(t)
	rpcStarted := make(chan struct{})
	var calls atomic.Int32
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			if calls.Add(1) == 1 {
				close(rpcStarted)
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return testBM25ResourceResponse(req), nil
		}).Twice()
	oracle := newTestLazyOracleRuntime(t, client, version)
	defer oracle.Close()

	firstCtx, cancelFirst := context.WithCancel(context.Background())
	firstResult := make(chan error, 1)
	go func() {
		_, _, err := oracle.BuildIDF(firstCtx, version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
		firstResult <- err
	}()
	<-rpcStarted

	waiterCtx := &observedDoneContext{Context: context.Background(), doneObserved: make(chan struct{})}
	secondResult := make(chan error, 1)
	go func() {
		_, _, err := oracle.BuildIDF(waiterCtx, version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
		secondResult <- err
	}()
	<-waiterCtx.doneObserved

	cancelFirst()
	require.ErrorIs(t, <-firstResult, context.Canceled)
	require.ErrorIs(t, <-secondResult, context.Canceled)
	require.False(t, oracleStatsReady(oracle, version))
	require.Len(t, client.Calls, 1)

	_, _, err := oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.Len(t, client.Calls, 2)
	require.True(t, oracleStatsReady(oracle, version))
}

func TestOracleRuntimeCancellationBeforeCommitDoesNotPublish(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}
	client := mocks.NewMockDataCoordClient(t)
	rpcStarted := make(chan struct{})
	releaseRPC := make(chan struct{})
	rpcReturned := make(chan struct{})
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			close(rpcStarted)
			<-releaseRPC
			close(rpcReturned)
			return testBM25ResourceResponse(req), nil
		}).Once()
	oracle := newTestLazyOracleRuntime(t, client, version)
	defer oracle.Close()

	oracle.growingStore.mu.Lock()
	storeLocked := true
	defer func() {
		if storeLocked {
			oracle.growingStore.mu.Unlock()
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, _, err := oracle.BuildIDF(ctx, version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
		result <- err
	}()
	<-rpcStarted
	close(releaseRPC)
	<-rpcReturned

	require.Eventually(t, func() bool {
		if oracle.mu.TryLock() {
			oracle.mu.Unlock()
			return false
		}
		return true
	}, time.Second, time.Millisecond)
	cancel()
	oracle.growingStore.mu.Unlock()
	storeLocked = false

	require.ErrorIs(t, <-result, context.Canceled)
	require.False(t, oracleStatsReady(oracle, version))
}

func TestOracleRuntimeAllowsNextQueryAfterMaterializationFailure(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}
	client := mocks.NewMockDataCoordClient(t)
	var calls atomic.Int32
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			if calls.Add(1) == 1 {
				return nil, merr.WrapErrServiceUnavailableMsg("temporary resource lookup failure")
			}
			return testBM25ResourceResponse(req), nil
		}).Twice()
	oracle := newTestLazyOracleRuntime(t, client, version)
	defer oracle.Close()

	_, _, err := oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.Error(t, err)
	require.False(t, oracleStatsReady(oracle, version))
	require.Len(t, client.Calls, 1)

	_, _, err = oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.Len(t, client.Calls, 2)
	require.True(t, oracleStatsReady(oracle, version))
}

func TestOracleRuntimeIncludesGrowingStatsAddedDuringMaterialization(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}
	client := mocks.NewMockDataCoordClient(t)
	rpcStarted := make(chan struct{})
	releaseRPC := make(chan struct{})
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			close(rpcStarted)
			select {
			case <-releaseRPC:
				return testBM25ResourceResponse(req), nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}).Once()
	oracle := newTestLazyOracleRuntime(t, client, version)
	defer oracle.Close()

	type result struct {
		avgdl float64
		err   error
	}
	buildResult := make(chan result, 1)
	go func() {
		_, avgdl, err := oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
		buildResult <- result{avgdl: avgdl, err: err}
	}()
	<-rpcStarted

	stats := storage.NewBM25Stats()
	stats.Append(map[uint32]float32{7: 2})
	oracle.mu.Lock()
	oracle.growingStore.appendStats(20, 10, bm25Stats{testBM25OutputFieldID: stats})
	oracle.mu.Unlock()
	close(releaseRPC)

	resultValue := <-buildResult
	require.NoError(t, resultValue.err)
	require.Equal(t, float64(2), resultValue.avgdl)
}

func TestOracleRuntimeLazyAdvanceDefersTargetMaterialization(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10}
	target := qviews.DataVersion{StreamingVersion: 12}
	client := mocks.NewMockDataCoordClient(t)
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			require.True(t, qviews.FromProtoDataVersion(req.GetDataVersion()).EQ(target))
			return testBM25ResourceResponse(req), nil
		}).Once()
	oracle := newTestLazyOracleRuntime(t, client, current)
	defer oracle.Close()

	require.NoError(t, oracle.PrepareDataVersion(context.Background(), target))
	oracle.Advance(target)
	require.Empty(t, client.Calls)
	require.True(t, oracleCurrentVersion(oracle).EQ(target))
	require.False(t, oracleStatsReady(oracle, target))

	_, _, err := oracle.BuildIDF(context.Background(), target, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.Len(t, client.Calls, 1)
	require.True(t, oracleStatsReady(oracle, target))
}

func TestOracleRuntimeLazyOnlyDefersInitialMaterialization(t *testing.T) {
	current := qviews.DataVersion{StreamingVersion: 10}
	target := qviews.DataVersion{StreamingVersion: 11}
	client := mocks.NewMockDataCoordClient(t)
	var calls atomic.Int32
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			if calls.Add(1) == 1 {
				require.True(t, qviews.FromProtoDataVersion(req.GetDataVersion()).EQ(current))
			} else {
				require.True(t, qviews.FromProtoDataVersion(req.GetDataVersion()).EQ(target))
			}
			return testBM25ResourceResponse(req), nil
		}).Twice()
	oracle := newTestLazyOracleRuntime(t, client, current)
	defer oracle.Close()

	_, _, err := oracle.BuildIDF(context.Background(), current, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.True(t, oracleStatsReady(oracle, current))

	require.NoError(t, oracle.PrepareDataVersion(context.Background(), target))
	require.True(t, oracleStatsReady(oracle, target))
	require.Len(t, client.Calls, 2)

	oracle.Advance(target)
	require.True(t, oracleCurrentVersion(oracle).EQ(target))
	require.True(t, oracleStatsReady(oracle, target))
	_, _, err = oracle.BuildIDF(context.Background(), target, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.NoError(t, err)
	require.Len(t, client.Calls, 2)
}

func TestOracleRuntimeCloseCancelsVersionMaterialization(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}
	client := mocks.NewMockDataCoordClient(t)
	rpcStarted := make(chan struct{})
	client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, _ *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
			close(rpcStarted)
			<-ctx.Done()
			return nil, ctx.Err()
		}).Once()
	oracle := newTestLazyOracleRuntime(t, client, version)

	result := make(chan error, 1)
	go func() {
		_, _, err := oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
		result <- err
	}()
	<-rpcStarted
	oracle.Close()

	require.ErrorIs(t, <-result, context.Canceled)
	_, _, err := oracle.BuildIDF(context.Background(), version, testBM25OutputFieldID, &schemapb.SparseFloatArray{})
	require.ErrorIs(t, err, context.Canceled)
}

func newTestLazyOracleRuntime(t *testing.T, client *mocks.MockDataCoordClient, version qviews.DataVersion) *oracleRuntime {
	t.Helper()
	scheduler := nodescheduler.New(1)
	t.Cleanup(scheduler.Close)
	oracle, err := newOracleRuntime(
		context.Background(),
		NewProvider(client, WithNodeScheduler(scheduler)),
		testBM25WALView(version),
		nil,
		true,
	)
	require.NoError(t, err)
	return oracle
}

func newTestEagerOracleRuntime(t *testing.T, client *mocks.MockDataCoordClient, version qviews.DataVersion) *oracleRuntime {
	t.Helper()
	scheduler := nodescheduler.New(1)
	t.Cleanup(scheduler.Close)
	oracle, err := newOracleRuntime(
		context.Background(),
		NewProvider(client, WithNodeScheduler(scheduler)),
		testBM25WALView(version),
		nil,
		false,
	)
	require.NoError(t, err)
	return oracle
}

func testBM25WALView(version qviews.DataVersion) walview.VChannelWALView {
	return walview.VChannelWALView{
		CollectionID: 1,
		VChannel:     "test-vchannel",
		Schema: &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: testBM25InputFieldID, Name: "text", DataType: schemapb.DataType_VarChar},
				{FieldID: testBM25OutputFieldID, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector, IsFunctionOutput: true},
			},
			Functions: []*schemapb.FunctionSchema{{
				Name:             "bm25",
				Type:             schemapb.FunctionType_BM25,
				InputFieldIds:    []int64{testBM25InputFieldID},
				OutputFieldIds:   []int64{testBM25OutputFieldID},
				InputFieldNames:  []string{"text"},
				OutputFieldNames: []string{"sparse"},
			}},
		},
		LoadFields: []*messagespb.LoadFieldConfig{{FieldId: testBM25OutputFieldID}},
		SegmentSnapshot: walview.VisibleSegmentSnapshot{
			CollectionID: 1,
			VChannel:     "test-vchannel",
			DataVersion:  version,
		},
	}
}

func testBM25ResourceResponse(req *datapb.GetStreamingNodeQueryViewResourcesRequest) *datapb.GetStreamingNodeQueryViewResourcesResponse {
	return &datapb.GetStreamingNodeQueryViewResourcesResponse{
		Status:       merr.Success(),
		CollectionId: req.GetCollectionId(),
		Vchannel:     req.GetVchannel(),
		DataVersion:  req.GetDataVersion(),
	}
}

func oracleStatsReady(oracle *oracleRuntime, version qviews.DataVersion) bool {
	oracle.mu.RLock()
	defer oracle.mu.RUnlock()
	_, ready := oracle.statsForVersionLocked(version)
	return ready
}

func oracleCurrentVersion(oracle *oracleRuntime) qviews.DataVersion {
	oracle.mu.RLock()
	defer oracle.mu.RUnlock()
	return oracle.currentVersion
}

func testSealedBM25Resources(firstSegmentID int64, count int) []*datapb.StreamingNodeBM25Resource {
	resources := make([]*datapb.StreamingNodeBM25Resource, 0, count)
	for i := range count {
		segmentID := firstSegmentID + int64(i)
		resources = append(resources, &datapb.StreamingNodeBM25Resource{
			SegmentId:      segmentID,
			PartitionId:    segmentID,
			StorageVersion: storage.StorageV2,
			Bm25Binlogs: []*datapb.FieldBinlog{{
				FieldID: testBM25OutputFieldID,
				Binlogs: []*datapb.Binlog{{
					LogPath: fmt.Sprintf("stats-%d", segmentID),
				}},
			}},
		})
	}
	return resources
}
