//go:build test
// +build test

package broadcaster

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/mocks/mock_metastore"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/registry"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/resource"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/messagespb"
	"github.com/milvus-io/milvus/pkg/v3/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/streaming/walimpls/impls/walimplstest"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestLoadConfigControlChannelFastAck(t *testing.T) {
	paramtable.Init()
	registry.ResetRegistration()

	const controlChannel = "by-dev-rootcoord-dml_0_vcchan"
	vchannels := []string{controlChannel}

	tests := []struct {
		name   string
		build  func([]string) message.BroadcastMutableMessage
		msgTyp message.MessageType
	}{
		{
			name: "alter_load_config",
			build: func(channels []string) message.BroadcastMutableMessage {
				return message.NewAlterLoadConfigMessageBuilderV2().
					WithHeader(&messagespb.AlterLoadConfigMessageHeader{CollectionId: 100}).
					WithBody(&messagespb.AlterLoadConfigMessageBody{}).
					WithBroadcast(channels).
					MustBuildBroadcast()
			},
			msgTyp: message.MessageTypeAlterLoadConfig,
		},
		{
			name: "drop_load_config",
			build: func(channels []string) message.BroadcastMutableMessage {
				return message.NewDropLoadConfigMessageBuilderV2().
					WithHeader(&messagespb.DropLoadConfigMessageHeader{CollectionId: 100}).
					WithBody(&messagespb.DropLoadConfigMessageBody{}).
					WithBroadcast(channels).
					MustBuildBroadcast()
			},
			msgTyp: message.MessageTypeDropLoadConfig,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			msg := test.build(vchannels).WithBroadcastID(1)
			header := msg.BroadcastHeader()
			require.Equal(t, test.msgTyp, msg.MessageType())
			require.False(t, header.AckSyncUp)

			catalog := mock_metastore.NewMockStreamingCoordCataLog(t)
			catalog.EXPECT().
				SaveBroadcastTask(mock.Anything, uint64(1), mock.Anything).
				Return(nil).
				Once()
			resource.InitForTest(resource.OptStreamingCatalog(catalog))

			ackScheduler := newAckCallbackScheduler(mlog.With())
			taskProto := createNewWaitAckBroadcastTaskFromMessage(
				msg,
				streamingpb.BroadcastTaskState_BROADCAST_TASK_STATE_PENDING,
				make([]byte, len(header.VChannels)),
			)
			task := newBroadcastTaskFromProto(taskProto, newBroadcasterMetrics(), ackScheduler)
			task.SetLogger(mlog.With())

			appendResults := testFastAckAppendResults(header.VChannels)
			require.NoError(t, task.FastAck(context.Background(), appendResults))
			require.NoError(t, task.BlockUntilAllAck(context.Background()))
			assert.Len(t, ackScheduler.pending, 1)

			_, result := task.BroadcastResult()
			require.Len(t, result, len(header.VChannels))
			for vchannel, appendResult := range appendResults {
				assert.Equal(t, appendResult.TimeTick, result[vchannel].TimeTick)
				assert.True(t, appendResult.MessageID.EQ(result[vchannel].MessageID))
				assert.True(t, appendResult.LastConfirmedMessageID.EQ(result[vchannel].LastConfirmedMessageID))
			}
		})
	}
}

func testFastAckAppendResults(vchannels []string) map[string]*types.AppendResult {
	results := make(map[string]*types.AppendResult, len(vchannels))
	for idx, vchannel := range vchannels {
		results[vchannel] = &types.AppendResult{
			MessageID:              walimplstest.NewTestMessageID(int64(idx + 1)),
			LastConfirmedMessageID: walimplstest.NewTestMessageID(int64(idx + 101)),
			TimeTick:               uint64(idx + 1001),
		}
	}
	return results
}
