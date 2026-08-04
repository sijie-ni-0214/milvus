package adaptor

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/interceptors"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/recovery"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
)

type buildParamRetainingInterceptorBuilder struct {
	param               *interceptors.InterceptorBuildParam
	snapshotDuringBuild *recovery.RecoverySnapshot
}

func (b *buildParamRetainingInterceptorBuilder) Build(param *interceptors.InterceptorBuildParam) interceptors.Interceptor {
	b.param = param
	b.snapshotDuringBuild = param.InitialRecoverSnapshot
	return passThroughInterceptor{}
}

type passThroughInterceptor struct{}

func (passThroughInterceptor) DoAppend(
	ctx context.Context,
	msg message.MutableMessage,
	append interceptors.Append,
) (message.MessageID, error) {
	return append(ctx, msg)
}

func (passThroughInterceptor) Close() {}

func TestBuildInterceptorReleasesInitialRecoverSnapshot(t *testing.T) {
	snapshot := &recovery.RecoverySnapshot{}
	param := &interceptors.InterceptorBuildParam{InitialRecoverSnapshot: snapshot}
	builder := &buildParamRetainingInterceptorBuilder{}

	result := buildInterceptor([]interceptors.InterceptorBuilder{builder}, param)
	t.Cleanup(result.Close)

	assert.Same(t, snapshot, builder.snapshotDuringBuild)
	assert.Same(t, param, builder.param)
	assert.Nil(t, param.InitialRecoverSnapshot)
}
