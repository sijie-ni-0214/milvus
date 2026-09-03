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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/internal/views/queryclient"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestEnsureCollectionReadyRejectsUnhealthyProxy(t *testing.T) {
	client := &autoLoadViewQueryClient{}
	node := &Proxy{ctx: context.Background(), viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Abnormal)

	err := node.ensureCollectionReady(context.Background(), "db", "collection")
	require.ErrorIs(t, err, merr.ErrServiceNotReady)
	require.Equal(t, 0, client.checkCalls)
	require.Equal(t, 0, client.waitCalls)
}

func TestEnsureCollectionReadySkipsWhenAutoLoadDisabled(t *testing.T) {
	require.NoError(t, Params.Save(Params.ProxyCfg.EnableAutoLoad.Key, "false"))
	t.Cleanup(func() { require.NoError(t, Params.Reset(Params.ProxyCfg.EnableAutoLoad.Key)) })

	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	require.NoError(t, node.ensureCollectionReady(context.Background(), "db", "collection"))
	require.Equal(t, 0, client.checkCalls)
	require.Equal(t, 0, client.waitCalls)
}

func TestEnsureCollectionReadyFastPath(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	require.NoError(t, node.ensureCollectionReady(context.Background(), "db", "collection"))
	require.Equal(t, 1, client.checkCalls)
	require.Equal(t, 0, client.waitCalls)
	require.Equal(t, int64(100), client.collectionID)
	require.Equal(t, []string{"v0", "v1"}, client.expectedVChannels)
}

func TestEnsureCollectionReadyWaitsForAssignment(t *testing.T) {
	enableAutoLoad(t)
	tests := []struct {
		name          string
		loadState     commonpb.LoadState
		expectedLoads int
	}{
		{name: "not loaded", loadState: commonpb.LoadState_LoadStateNotLoad, expectedLoads: 1},
		{name: "loading", loadState: commonpb.LoadState_LoadStateLoading, expectedLoads: 0},
		{name: "loaded before assignment propagation", loadState: commonpb.LoadState_LoadStateLoaded, expectedLoads: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
			client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
			node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
			node.UpdateStateCode(commonpb.StateCode_Healthy)
			loadCalls := 0
			getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).To(
				func(_ *Proxy, _ context.Context, request *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
					require.Equal(t, "db", request.GetDbName())
					require.Equal(t, "collection", request.GetCollectionName())
					return &milvuspb.GetLoadStateResponse{Status: merr.Success(), State: test.loadState}, nil
				}).Build()
			loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
				func(_ *Proxy, _ context.Context, request *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
					loadCalls++
					require.Equal(t, "db", request.GetDbName())
					require.Equal(t, "collection", request.GetCollectionName())
					return merr.Success(), nil
				}).Build()
			t.Cleanup(func() { getLoadStateMock.UnPatch() })
			t.Cleanup(func() { loadCollectionMock.UnPatch() })

			require.NoError(t, node.ensureCollectionReady(context.Background(), "db", "collection"))
			require.Equal(t, test.expectedLoads, loadCalls)
			require.Equal(t, 1, client.waitCalls)
			require.Equal(t, int64(100), client.collectionID)
			require.Equal(t, []string{"v0", "v1"}, client.expectedVChannels)
		})
	}
}

func TestEnsureCollectionReadyRequiresLoadPrivilege(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).Return(&milvuspb.GetLoadStateResponse{
		Status: merr.Success(),
		State:  commonpb.LoadState_LoadStateNotLoad,
	}, nil).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, request interface{}) (context.Context, error) {
			loadRequest, ok := request.(*milvuspb.LoadCollectionRequest)
			require.True(t, ok)
			require.Equal(t, "db", loadRequest.GetDbName())
			require.Equal(t, "collection", loadRequest.GetCollectionName())
			return ctx, status.Error(codes.PermissionDenied, "PrivilegeLoad: permission denied")
		}).Build()
	loadCalls := 0
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			loadCalls++
			return merr.Success(), nil
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()

	err := node.ensureCollectionReady(context.Background(), "db", "collection")
	require.ErrorIs(t, err, merr.ErrPrivilegeNotPermitted)
	resultStatus := merr.Status(err)
	require.Equal(t, merr.Code(merr.ErrPrivilegeNotPermitted), resultStatus.GetCode())
	require.Equal(t, "true", resultStatus.GetExtraInfo()[merr.InputErrorFlagKey])
	require.Equal(t, 0, loadCalls)
	require.Equal(t, 0, client.waitCalls)
}

func TestEnsureCollectionReadyCoalescesConcurrentLoad(t *testing.T) {
	enableAutoLoad(t)
	const concurrency = 16

	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	var loading atomic.Bool
	var stateChecks atomic.Int32
	var loadCalls atomic.Int32
	allStateChecks := make(chan struct{})
	var closeStateChecks sync.Once
	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
			if stateChecks.Add(1) >= concurrency {
				closeStateChecks.Do(func() { close(allStateChecks) })
			}
			state := commonpb.LoadState_LoadStateNotLoad
			if loading.Load() {
				state = commonpb.LoadState_LoadStateLoading
			}
			return &milvuspb.GetLoadStateResponse{Status: merr.Success(), State: state}, nil
		}).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, _ interface{}) (context.Context, error) {
			return ctx, nil
		}).Build()
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, ctx context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			loadCalls.Add(1)
			select {
			case <-allStateChecks:
				loading.Store(true)
				return merr.Success(), nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := make(chan struct{})
	errs := make(chan error, concurrency)
	var wg sync.WaitGroup
	for range concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs <- node.ensureCollectionReady(ctx, "db", "collection")
		}()
	}
	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		require.NoError(t, err)
	}
	require.Equal(t, int32(1), loadCalls.Load())
}

func TestEnsureCollectionReadyCallerCancellationDoesNotCancelLoad(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).Return(&milvuspb.GetLoadStateResponse{
		Status: merr.Success(),
		State:  commonpb.LoadState_LoadStateNotLoad,
	}, nil).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, _ interface{}) (context.Context, error) {
			return ctx, nil
		}).Build()
	loadStarted := make(chan struct{})
	loadFinished := make(chan struct{})
	loadContextCanceled := make(chan struct{})
	releaseLoad := make(chan struct{})
	var releaseOnce sync.Once
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, ctx context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			close(loadStarted)
			defer close(loadFinished)
			select {
			case <-releaseLoad:
				return merr.Success(), nil
			case <-ctx.Done():
				close(loadContextCanceled)
				return nil, context.Cause(ctx)
			}
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()
	release := func() {
		releaseOnce.Do(func() { close(releaseLoad) })
	}
	t.Cleanup(release)

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- node.ensureCollectionReady(ctx, "db", "collection")
	}()

	select {
	case <-loadStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for auto-load to start")
	}
	cancel()

	select {
	case err := <-errCh:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("caller did not stop after its context was canceled")
	}
	select {
	case <-loadContextCanceled:
		t.Fatal("caller cancellation propagated to the shared load")
	default:
	}
	select {
	case <-loadFinished:
		t.Fatal("shared load stopped before it was released")
	default:
	}

	release()
	select {
	case <-loadFinished:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for shared load to finish")
	}
}

func TestEnsureCollectionReadyWaiterCanCancelIndependently(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	var stateChecks atomic.Int32
	var loading atomic.Bool
	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
			stateChecks.Add(1)
			state := commonpb.LoadState_LoadStateNotLoad
			if loading.Load() {
				state = commonpb.LoadState_LoadStateLoading
			}
			return &milvuspb.GetLoadStateResponse{Status: merr.Success(), State: state}, nil
		}).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, _ interface{}) (context.Context, error) {
			return ctx, nil
		}).Build()
	loadStarted := make(chan struct{})
	loadFinished := make(chan struct{})
	releaseLoad := make(chan struct{})
	var releaseOnce sync.Once
	var loadStartOnce sync.Once
	var loadFinishOnce sync.Once
	var loadCalls atomic.Int32
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, ctx context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			loadCalls.Add(1)
			loadStartOnce.Do(func() { close(loadStarted) })
			defer loadFinishOnce.Do(func() { close(loadFinished) })
			select {
			case <-releaseLoad:
				loading.Store(true)
				return merr.Success(), nil
			case <-ctx.Done():
				return nil, context.Cause(ctx)
			}
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()
	release := func() {
		releaseOnce.Do(func() { close(releaseLoad) })
	}
	t.Cleanup(release)

	leaderErrCh := make(chan error, 1)
	go func() {
		leaderErrCh <- node.ensureCollectionReady(context.Background(), "db", "collection")
	}()
	select {
	case <-loadStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for auto-load to start")
	}

	waiterCtx, cancelWaiter := context.WithCancel(context.Background())
	waiterErrCh := make(chan error, 1)
	go func() {
		waiterErrCh <- node.ensureCollectionReady(waiterCtx, "db", "collection")
	}()
	require.Eventually(t, func() bool {
		return stateChecks.Load() >= 3
	}, time.Second, time.Millisecond)
	cancelWaiter()

	select {
	case err := <-waiterErrCh:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("waiter did not stop after its context was canceled")
	}
	select {
	case <-loadFinished:
		t.Fatal("waiter cancellation stopped the shared load")
	default:
	}

	release()
	select {
	case err := <-leaderErrCh:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("leader did not finish after the shared load completed")
	}
	require.Equal(t, int32(1), loadCalls.Load())
}

func TestEnsureCollectionReadyProxyCancellationStopsLoad(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	nodeCtx, cancelNode := context.WithCancel(context.Background())
	node := &Proxy{ctx: nodeCtx, metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).Return(&milvuspb.GetLoadStateResponse{
		Status: merr.Success(),
		State:  commonpb.LoadState_LoadStateNotLoad,
	}, nil).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, _ interface{}) (context.Context, error) {
			return ctx, nil
		}).Build()
	loadStarted := make(chan struct{})
	loadContextCanceled := make(chan struct{})
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, ctx context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			close(loadStarted)
			<-ctx.Done()
			close(loadContextCanceled)
			return nil, context.Cause(ctx)
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()

	errCh := make(chan error, 1)
	go func() {
		errCh <- node.ensureCollectionReady(context.Background(), "db", "collection")
	}()
	select {
	case <-loadStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for auto-load to start")
	}
	cancelNode()

	select {
	case <-loadContextCanceled:
	case <-time.After(time.Second):
		t.Fatal("Proxy cancellation did not stop the shared load")
	}
	select {
	case err := <-errCh:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("caller did not receive the Proxy cancellation")
	}
}

func TestEnsureCollectionReadyRechecksLoadState(t *testing.T) {
	enableAutoLoad(t)
	metaCache := mockSearchCollectionMeta(t, 100, []string{"v0", "v1"})
	client := &autoLoadViewQueryClient{checkErr: merr.WrapErrCollectionNotLoaded(100)}
	node := &Proxy{ctx: context.Background(), metaCache: metaCache, viewQueryClient: client}
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	stateChecks := 0
	getLoadStateMock := mockey.Mock((*Proxy).GetLoadState).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
			stateChecks++
			state := commonpb.LoadState_LoadStateNotLoad
			if stateChecks > 1 {
				state = commonpb.LoadState_LoadStateLoading
			}
			return &milvuspb.GetLoadStateResponse{Status: merr.Success(), State: state}, nil
		}).Build()
	privilegeMock := mockey.Mock(PrivilegeInterceptor).To(
		func(ctx context.Context, _ interface{}) (context.Context, error) {
			return ctx, nil
		}).Build()
	loadCalls := 0
	loadCollectionMock := mockey.Mock((*Proxy).LoadCollection).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.LoadCollectionRequest) (*commonpb.Status, error) {
			loadCalls++
			return merr.Success(), nil
		}).Build()
	defer getLoadStateMock.UnPatch()
	defer privilegeMock.UnPatch()
	defer loadCollectionMock.UnPatch()

	require.NoError(t, node.ensureCollectionReady(context.Background(), "db", "collection"))
	require.Equal(t, 2, stateChecks)
	require.Equal(t, 0, loadCalls)
	require.Equal(t, 1, client.waitCalls)
}

func TestSearchStopsBeforeExecutionWhenCollectionIsNotReady(t *testing.T) {
	node := &Proxy{}
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	readinessCalls := 0
	searchCalls := 0
	readinessMock := mockey.Mock((*Proxy).ensureCollectionReady).To(
		func(_ *Proxy, _ context.Context, dbName, collectionName string) error {
			readinessCalls++
			require.Equal(t, "db", dbName)
			require.Equal(t, "collection", collectionName)
			return merr.WrapErrCollectionNotLoaded(100)
		}).Build()
	searchMock := mockey.Mock((*Proxy).search).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.SearchRequest, _, _ bool) (*milvuspb.SearchResults, bool, bool, bool, error) {
			searchCalls++
			return &milvuspb.SearchResults{Status: merr.Success()}, false, false, false, nil
		}).Build()
	defer readinessMock.UnPatch()
	defer searchMock.UnPatch()

	response, err := node.Search(context.Background(), &milvuspb.SearchRequest{
		DbName:         "db",
		CollectionName: "collection",
	})
	require.NoError(t, err)
	require.ErrorIs(t, merr.Error(response.GetStatus()), merr.ErrCollectionNotLoaded)
	require.Equal(t, 1, readinessCalls)
	require.Equal(t, 0, searchCalls)
}

func TestHybridSearchStopsBeforeExecutionWhenCollectionIsNotReady(t *testing.T) {
	node := &Proxy{}
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	readinessCalls := 0
	hybridSearchCalls := 0
	readinessMock := mockey.Mock((*Proxy).ensureCollectionReady).To(
		func(_ *Proxy, _ context.Context, dbName, collectionName string) error {
			readinessCalls++
			require.Equal(t, "db", dbName)
			require.Equal(t, "collection", collectionName)
			return merr.WrapErrCollectionNotLoaded(100)
		}).Build()
	hybridSearchMock := mockey.Mock((*Proxy).hybridSearch).To(
		func(_ *Proxy, _ context.Context, _ *milvuspb.HybridSearchRequest, _ bool) (*milvuspb.SearchResults, bool, bool, error) {
			hybridSearchCalls++
			return &milvuspb.SearchResults{Status: merr.Success()}, false, false, nil
		}).Build()
	defer readinessMock.UnPatch()
	defer hybridSearchMock.UnPatch()

	response, err := node.HybridSearch(context.Background(), &milvuspb.HybridSearchRequest{
		DbName:         "db",
		CollectionName: "collection",
	})
	require.NoError(t, err)
	require.ErrorIs(t, merr.Error(response.GetStatus()), merr.ErrCollectionNotLoaded)
	require.Equal(t, 1, readinessCalls)
	require.Equal(t, 0, hybridSearchCalls)
}

func TestQueryStopsBeforeExecutionWhenCollectionIsNotReady(t *testing.T) {
	node := &Proxy{}
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	readinessCalls := 0
	queryCalls := 0
	readinessMock := mockey.Mock((*Proxy).ensureCollectionReady).To(
		func(_ *Proxy, _ context.Context, dbName, collectionName string) error {
			readinessCalls++
			require.Equal(t, "db", dbName)
			require.Equal(t, "collection", collectionName)
			return merr.WrapErrCollectionNotLoaded(100)
		}).Build()
	queryMock := mockey.Mock((*Proxy).query).To(
		func(_ *Proxy, _ context.Context, _ *queryTask, _ trace.Span) (*milvuspb.QueryResults, segcore.StorageCost, error) {
			queryCalls++
			return &milvuspb.QueryResults{Status: merr.Success()}, segcore.StorageCost{}, nil
		}).Build()
	defer readinessMock.UnPatch()
	defer queryMock.UnPatch()

	response, err := node.Query(context.Background(), &milvuspb.QueryRequest{
		DbName:         "db",
		CollectionName: "collection",
	})
	require.NoError(t, err)
	require.ErrorIs(t, merr.Error(response.GetStatus()), merr.ErrCollectionNotLoaded)
	require.Equal(t, 1, readinessCalls)
	require.Equal(t, 0, queryCalls)
}

func TestQueryExecutesWhenCollectionIsReady(t *testing.T) {
	node := &Proxy{}
	previousRateCol := rateCol
	t.Cleanup(func() { rateCol = previousRateCol })
	require.NoError(t, node.initRateCollector())
	node.UpdateStateCode(commonpb.StateCode_Healthy)
	readinessCalls := 0
	queryCalls := 0
	readinessMock := mockey.Mock((*Proxy).ensureCollectionReady).To(
		func(_ *Proxy, _ context.Context, dbName, collectionName string) error {
			readinessCalls++
			require.Equal(t, "db", dbName)
			require.Equal(t, "collection", collectionName)
			return nil
		}).Build()
	queryMock := mockey.Mock((*Proxy).query).To(
		func(_ *Proxy, _ context.Context, _ *queryTask, _ trace.Span) (*milvuspb.QueryResults, segcore.StorageCost, error) {
			queryCalls++
			return &milvuspb.QueryResults{Status: merr.Success()}, segcore.StorageCost{}, nil
		}).Build()
	defer readinessMock.UnPatch()
	defer queryMock.UnPatch()

	response, err := node.Query(context.Background(), &milvuspb.QueryRequest{
		DbName:         "db",
		CollectionName: "collection",
	})
	require.NoError(t, err)
	require.NoError(t, merr.Error(response.GetStatus()))
	require.Equal(t, 1, readinessCalls)
	require.Equal(t, 1, queryCalls)
}

type autoLoadViewQueryClient struct {
	mu                sync.Mutex
	checkErr          error
	waitErr           error
	checkCalls        int
	waitCalls         int
	collectionID      int64
	expectedVChannels []string
}

func (c *autoLoadViewQueryClient) Legacy() queryclient.LegacyClient {
	return nil
}

func (c *autoLoadViewQueryClient) CheckCollectionReady(_ context.Context, collectionID int64, expectedVChannels []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checkCalls++
	c.collectionID = collectionID
	c.expectedVChannels = append([]string(nil), expectedVChannels...)
	return c.checkErr
}

func (c *autoLoadViewQueryClient) WaitForCollectionReady(_ context.Context, collectionID int64, expectedVChannels []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.waitCalls++
	c.collectionID = collectionID
	c.expectedVChannels = append([]string(nil), expectedVChannels...)
	return c.waitErr
}

func mockSearchCollectionMeta(t *testing.T, collectionID int64, vchannels []string) *MetaCache {
	metaCache := &MetaCache{}
	getCollectionIDMock := mockey.Mock((*MetaCache).GetCollectionID).Return(collectionID, nil).Build()
	getCollectionInfoMock := mockey.Mock((*MetaCache).GetCollectionInfo).Return(&collectionInfo{
		CollID:    collectionID,
		VChannels: vchannels,
	}, nil).Build()
	t.Cleanup(func() {
		getCollectionInfoMock.UnPatch()
		getCollectionIDMock.UnPatch()
	})
	return metaCache
}

func enableAutoLoad(t *testing.T) {
	t.Helper()
	require.NoError(t, Params.Save(Params.ProxyCfg.EnableAutoLoad.Key, "true"))
	t.Cleanup(func() { require.NoError(t, Params.Reset(Params.ProxyCfg.EnableAutoLoad.Key)) })
}
