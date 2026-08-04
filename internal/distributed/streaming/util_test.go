package streaming

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
)

func TestObserveAppendMessagesStageDoesNotPanicOnLabels(t *testing.T) {
	metrics.StreamingServiceClientAppendMessagesStageDurationSeconds.Reset()

	require.NotPanics(t, func() {
		observeAppendMessagesStage(message.MessageTypeCreateCollection.String(), appendMessagesStageLifetimeGuard, time.Now())
	})
	require.Equal(t, 1, testutil.CollectAndCount(metrics.StreamingServiceClientAppendMessagesStageDurationSeconds))
}
