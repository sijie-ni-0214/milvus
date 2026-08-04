package rootcoord

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
)

func TestObserveCreateCollectionCallbackStageDoesNotPanicOnLabels(t *testing.T) {
	metrics.RootCoordDDLCallbackDuration.Reset()

	require.NotPanics(t, func() {
		observeCreateCollectionStage(createCollectionStageWatchChannelRPC, time.Now())
	})
	require.Equal(t, 1, testutil.CollectAndCount(metrics.RootCoordDDLCallbackDuration))
}

func TestObserveDropCollectionCallbackStageDoesNotPanicOnLabels(t *testing.T) {
	metrics.RootCoordDDLCallbackDuration.Reset()

	require.NotPanics(t, func() {
		observeDropCollectionCallbackStage("drop_virtual_channel", time.Now())
	})
	require.Equal(t, 1, testutil.CollectAndCount(metrics.RootCoordDDLCallbackDuration))
}
