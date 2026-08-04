package streaming

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/internal/distributed/streaming/internal/producer"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type (
	AppendResponses = types.AppendResponses
	AppendResponse  = types.AppendResponse
)

const (
	appendMessagesStageLifetimeGuard = "lifetime_guard"
	appendMessagesStageDispatch      = "dispatch"
	appendMessagesStageBeginProduce  = "begin_produce"
	appendMessagesStageBatchCommit   = "batch_commit"
	appendMessagesStageFillResponses = "fill_responses"
)

// AppendMessagesToWAL appends messages to the wal.
// It it a helper utility function to append messages to the wal.
// If the messages is belong to one vchannel, it will be sent as a transaction.
// Otherwise, it will be sent as individual messages.
// !!! This function do not promise the atomicity and deliver order of the messages appending.
func (w *walAccesserImpl) AppendMessages(ctx context.Context, msgs ...message.MutableMessage) AppendResponses {
	return w.AppendMessagesWithOptions(ctx, msgs)
}

func (w *walAccesserImpl) AppendMessagesWithOptions(ctx context.Context, msgs []message.MutableMessage, opts ...AppendOption) AppendResponses {
	assertValidMessage(msgs...)
	msgs = applyOpt(msgs, opts...)
	messageType := appendMessagesTypeLabel(msgs...)

	stageStart := time.Now()
	if !w.lifetime.Add(typeutil.LifetimeStateWorking) {
		observeAppendMessagesStage(messageType, appendMessagesStageLifetimeGuard, stageStart)
		err := types.NewAppendResponseN(len(msgs))
		err.FillAllError(ErrWALAccesserClosed)
		return err
	}
	observeAppendMessagesStage(messageType, appendMessagesStageLifetimeGuard, stageStart)
	defer w.lifetime.Done()

	// dispatch the messages into different vchannel.
	stageStart = time.Now()
	dispatchedMessages, indexes := w.dispatchMessages(msgs...)
	observeAppendMessagesStage(messageType, appendMessagesStageDispatch, stageStart)

	// Use a slice to maintain the order of vchannels and their corresponding indexes.
	type vchannelTask struct {
		vchannel string
		indexes  []int
	}
	tasks := make([]vchannelTask, 0, len(dispatchedMessages))
	guards := make([]*producer.ProduceGuard, 0, len(dispatchedMessages))
	resp := types.NewAppendResponseN(len(msgs))
	produceOpts := toProduceOptions(opts...)
	stageStart = time.Now()
	for vchannel, vchannelMsgs := range dispatchedMessages {
		g, err := w.getProducer(vchannel).BeginProduceWithOptions(ctx, vchannelMsgs, produceOpts...)
		if err != nil {
			observeAppendMessagesStage(messageType, appendMessagesStageBeginProduce, stageStart)
			for _, guard := range guards {
				guard.Cancel()
			}
			resp.FillAllError(err)
			return resp
		}
		guards = append(guards, g)
		tasks = append(tasks, vchannelTask{
			vchannel: vchannel,
			indexes:  indexes[vchannel],
		})
	}
	observeAppendMessagesStage(messageType, appendMessagesStageBeginProduce, stageStart)

	// Batch commit and get responses per vchannel.
	stageStart = time.Now()
	guardResps := producer.BatchCommitProduce(ctx, guards...)
	observeAppendMessagesStage(messageType, appendMessagesStageBatchCommit, stageStart)

	// Map the responses back to the original order using indexes.
	stageStart = time.Now()
	for i, task := range tasks {
		guardResp := guardResps.Responses[i]
		for _, origIdx := range task.indexes {
			resp.FillResponseAtIdx(guardResp, origIdx)
		}
	}
	observeAppendMessagesStage(messageType, appendMessagesStageFillResponses, stageStart)

	return resp
}

func (w *walAccesserImpl) appendReplicateMessageToWAL(ctx context.Context, msg message.MutableMessage) (*types.AppendResult, error) {
	guard, err := w.getProducer(msg.VChannel()).BeginProduce(ctx, msg)
	if err != nil {
		return nil, err
	}
	resp := producer.BatchCommitProduce(ctx, guard)
	return resp.Responses[0].AppendResult, resp.Responses[0].Error
}

func appendMessagesTypeLabel(msgs ...message.MutableMessage) string {
	if len(msgs) == 0 {
		return message.MessageTypeUnknown.String()
	}
	msgType := msgs[0].MessageType()
	for _, msg := range msgs[1:] {
		if msg.MessageType() != msgType {
			return "Mixed"
		}
	}
	return msgType.String()
}

func observeAppendMessagesStage(messageType string, stage string, start time.Time) {
	metrics.StreamingServiceClientAppendMessagesStageDurationSeconds.WithLabelValues(paramtable.GetStringNodeID(), messageType, stage).Observe(time.Since(start).Seconds())
}

// dispatchMessages dispatches the messages into different vchannel.
func (w *walAccesserImpl) dispatchMessages(msgs ...message.MutableMessage) (map[string][]message.MutableMessage, map[string][]int) {
	dispatchedMessages := make(map[string][]message.MutableMessage, 0)
	indexes := make(map[string][]int, 0)
	for idx, msg := range msgs {
		vchannel := msg.VChannel()
		if _, ok := dispatchedMessages[vchannel]; !ok {
			dispatchedMessages[vchannel] = make([]message.MutableMessage, 0)
			indexes[vchannel] = make([]int, 0)
		}
		dispatchedMessages[vchannel] = append(dispatchedMessages[vchannel], msg)
		indexes[vchannel] = append(indexes[vchannel], idx)
	}
	return dispatchedMessages, indexes
}

// applyOpt applies the append options to the message.
func applyOpt(msgs []message.MutableMessage, opts ...AppendOption) []message.MutableMessage {
	if len(opts) == 0 {
		return msgs
	}
	if opts[0].BarrierTimeTick > 0 {
		for idx, msg := range msgs {
			msgs[idx] = msg.WithBarrierTimeTick(opts[0].BarrierTimeTick)
		}
	}
	return msgs
}

func toProduceOptions(opts ...AppendOption) []producer.ProduceOption {
	if len(opts) == 0 {
		return nil
	}
	produceOpts := make([]producer.ProduceOption, 0, len(opts))
	for _, opt := range opts {
		if opt.IdempotencyKey == "" {
			continue
		}
		produceOpts = append(produceOpts, producer.ProduceOption{
			IdempotencyKey: opt.IdempotencyKey,
		})
	}
	return produceOpts
}
