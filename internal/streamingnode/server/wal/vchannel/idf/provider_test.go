package idf

import (
	"context"
	"os"
	"strconv"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/walview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/nodescheduler"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestMain(m *testing.M) {
	paramtable.InitWithBaseTable(paramtable.NewBaseTable(paramtable.SkipRemote(true)))
	os.Exit(m.Run())
}

func TestProviderCreatesNoopRuntimeWhenBM25IsNotLoaded(t *testing.T) {
	provider := NewProvider(nil)
	runtime, err := provider.NewRuntime()
	require.NoError(t, err)
	require.NotNil(t, runtime)
	require.NoError(t, runtime.Prepare(context.Background(), walview.VChannelWALView{
		CollectionID: 1,
		VChannel:     "ch",
		SegmentSnapshot: walview.VisibleSegmentSnapshot{
			DataVersion: qviews.DataVersion{StreamingVersion: 10},
		},
	}))
	versioned := runtime.(interface {
		PrepareDataVersion(context.Context, qviews.DataVersion) error
	})
	require.NoError(t, versioned.PrepareDataVersion(context.Background(), qviews.DataVersion{StreamingVersion: 11}))
	runtime.Close()
}

func TestRuntimePrepareRespectsLazyLoadSealedStats(t *testing.T) {
	version := qviews.DataVersion{StreamingVersion: 10}

	t.Run("eager", func(t *testing.T) {
		setLazyLoadSealedStats(t, false)
		client := mocks.NewMockDataCoordClient(t)
		client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
			RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
				return testBM25ResourceResponse(req), nil
			}).Once()

		runtime := newTestRuntime(t, client)
		defer runtime.Close()
		require.NoError(t, runtime.Prepare(context.Background(), testBM25WALView(version)))
		require.Len(t, client.Calls, 1)
		require.True(t, oracleStatsReady(runtime.currentOracle(), version))

		_, _, err := runtime.BuildIDF(context.Background(), version, testBM25OutputFieldID, nil)
		require.NoError(t, err)
		require.Len(t, client.Calls, 1)
	})

	t.Run("lazy", func(t *testing.T) {
		setLazyLoadSealedStats(t, true)
		client := mocks.NewMockDataCoordClient(t)
		client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
			RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
				return testBM25ResourceResponse(req), nil
			}).Once()

		runtime := newTestRuntime(t, client)
		defer runtime.Close()
		require.NoError(t, runtime.Prepare(context.Background(), testBM25WALView(version)))
		require.Empty(t, client.Calls)
		require.False(t, oracleStatsReady(runtime.currentOracle(), version))

		_, _, err := runtime.BuildIDF(context.Background(), version, testBM25OutputFieldID, nil)
		require.NoError(t, err)
		require.Len(t, client.Calls, 1)
		require.True(t, oracleStatsReady(runtime.currentOracle(), version))
	})

	t.Run("manifest metadata lazy", func(t *testing.T) {
		setLazyLoadSealedStats(t, false)
		setLazyManifestMetadataRead(t, true)
		client := mocks.NewMockDataCoordClient(t)
		client.EXPECT().GetStreamingNodeQueryViewResources(mock.Anything, mock.Anything).
			RunAndReturn(func(_ context.Context, req *datapb.GetStreamingNodeQueryViewResourcesRequest, _ ...grpc.CallOption) (*datapb.GetStreamingNodeQueryViewResourcesResponse, error) {
				return testBM25ResourceResponse(req), nil
			}).Once()

		runtime := newTestRuntime(t, client)
		defer runtime.Close()
		require.NoError(t, runtime.Prepare(context.Background(), testBM25WALView(version)))
		require.Empty(t, client.Calls)
		require.False(t, oracleStatsReady(runtime.currentOracle(), version))

		_, _, err := runtime.BuildIDF(context.Background(), version, testBM25OutputFieldID, nil)
		require.NoError(t, err)
		require.Len(t, client.Calls, 1)
		require.True(t, oracleStatsReady(runtime.currentOracle(), version))
	})
}

func newTestRuntime(t *testing.T, client *mocks.MockDataCoordClient) *Runtime {
	t.Helper()
	scheduler := nodescheduler.New(1)
	t.Cleanup(scheduler.Close)
	module, err := NewProvider(client, WithNodeScheduler(scheduler)).NewRuntime()
	require.NoError(t, err)
	runtime, ok := module.(*Runtime)
	require.True(t, ok)
	return runtime
}

func setLazyLoadSealedStats(t *testing.T, enabled bool) {
	t.Helper()
	params := paramtable.Get()
	key := params.QueryNodeCfg.IDFLazyLoadSealedStats.Key
	require.NoError(t, params.Save(key, strconv.FormatBool(enabled)))
	t.Cleanup(func() {
		require.NoError(t, params.Reset(key))
	})
}

func setLazyManifestMetadataRead(t *testing.T, enabled bool) {
	t.Helper()
	params := paramtable.Get()
	key := params.QueryNodeCfg.TieredLazyManifestMetadataReadEnabled.Key
	require.NoError(t, params.Save(key, strconv.FormatBool(enabled)))
	t.Cleanup(func() {
		require.NoError(t, params.Reset(key))
	})
}

func TestSealedStatsLoadConcurrency(t *testing.T) {
	require.Equal(t, int64(64), sealedStatsLoadConcurrency(16, 4))
	require.Equal(t, int64(8), sealedStatsLoadConcurrency(16, 0.5))
	require.Equal(t, int64(1), sealedStatsLoadConcurrency(1, 0.5))
	require.Equal(t, int64(1), sealedStatsLoadConcurrency(0, 4))
	require.Equal(t, int64(1), sealedStatsLoadConcurrency(16, 0))
}

func TestProvidersShareSealedStatsLoadLimiter(t *testing.T) {
	provider := NewProvider(nil)
	futureProvider := NewFutureProvider(nil)
	require.Same(t, provider.sealedStatsLoadLimiter, futureProvider.sealedStatsLoadLimiter)
}
