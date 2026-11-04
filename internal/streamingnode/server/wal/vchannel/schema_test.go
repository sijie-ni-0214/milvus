package vchannel

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/fieldmaskpb"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/streamingutil/status"
	"github.com/milvus-io/milvus/pkg/v3/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/streaming/walimpls/impls/walimplstest"
)

func TestWritePathRecoveryStateSharesSchemaAcrossHistoryChanges(t *testing.T) {
	view := NewVChannelViewFromMeta(restoreSchemaTestMeta())
	_, initial := view.GetSchema(0)
	expected := proto.Clone(initial).(*schemapb.CollectionSchema)
	state, ok := view.WritePathRecoveryState()
	require.True(t, ok)
	require.Same(t, initial, state.Schema)

	updated := proto.Clone(initial).(*schemapb.CollectionSchema)
	updated.Version++
	updated.Description = "updated"
	msg := message.NewAlterCollectionMessageBuilderV2().WithVChannel("v1").
		WithHeader(&message.AlterCollectionMessageHeader{CollectionId: 1, UpdateMask: &fieldmaskpb.FieldMask{Paths: []string{message.FieldMaskCollectionSchema}}}).
		WithBody(&message.AlterCollectionMessageBody{Updates: &message.AlterCollectionMessageUpdates{Schema: updated}}).
		MustBuildMutable().WithTimeTick(40).WithLastConfirmedUseMessageID().IntoImmutableMessage(walimplstest.NewTestMessageID(40))
	require.True(t, view.ObserveAlterCollectionMessageV2(message.MustAsImmutableAlterCollectionMessageV2(msg)))
	current, ok := view.WritePathRecoveryState()
	require.True(t, ok)
	_, latest := view.GetSchema(0)
	require.Same(t, latest, current.Schema)
	require.NotSame(t, state.Schema, current.Schema)
	require.True(t, proto.Equal(expected, state.Schema))

	view.SchemaCleanupPlan(40, nil)
	snapshot, _ := view.ConsumeDirtyAndGetSnapshot()
	require.NotNil(t, snapshot)
	view.MarkSnapshotPersisted(snapshot)
	deletion, _ := view.SchemaCleanupPlan(40, nil)
	require.NotNil(t, deletion)
	view.MarkSchemaCleanupPersisted(deletion)
	require.Len(t, view.AssignmentMeta().GetCollectionInfo().GetSchemas(), 1)
	require.True(t, proto.Equal(expected, state.Schema), "retiring history must preserve retained schema contents")
	_, latest = view.GetSchema(0)
	require.Same(t, latest, current.Schema)
}

func TestExactSchemaVersionAfterRecovery(t *testing.T) {
	meta := restoreSchemaTestMeta()
	encoded, err := proto.Marshal(meta)
	require.NoError(t, err)
	recovered := new(streamingpb.VChannelMeta)
	require.NoError(t, proto.Unmarshal(encoded, recovered))
	module, err := NewModule(ModuleConfig{PChannel: "p1", VChannel: "v1", VChannelMeta: recovered})
	require.NoError(t, err)
	for _, version := range []int32{0, 1} {
		snapshot, err := module.GetSchemaVersion(1, version)
		require.NoError(t, err)
		require.Equal(t, version, snapshot.Schema.Version)
		require.Equal(t, uint64(version+1)*10, snapshot.CheckpointTimeTick)
		snapshot.Schema.Fields[0].FieldID = 999
		again, err := module.GetSchemaVersion(1, version)
		require.NoError(t, err)
		require.EqualValues(t, 100, again.Schema.Fields[0].FieldID, "callers must own their snapshot")
	}
	state, ok := module.vchannelView.WritePathRecoveryState()
	require.True(t, ok)
	require.EqualValues(t, 1, state.Schema.Version)
	require.EqualValues(t, 20, state.SchemaTimeTick, "write admission recovery must preserve the schema's WAL boundary")
}

func TestExactSchemaVersionAvailability(t *testing.T) {
	for _, tc := range []struct {
		name         string
		collectionID int64
		version      int32
		mutate       func(*streamingpb.VChannelMeta)
		retriable    bool
	}{
		{name: "future schema", collectionID: 1, version: 2, retriable: true},
		{name: "different collection", collectionID: 2},
		{name: "retired schema", collectionID: 1, mutate: func(m *streamingpb.VChannelMeta) {
			m.CollectionInfo.Schemas[0].State = streamingpb.VChannelSchemaState_VCHANNEL_SCHEMA_STATE_DROPPED
		}},
		{name: "missing body is not version zero", collectionID: 1, mutate: func(m *streamingpb.VChannelMeta) { m.CollectionInfo.Schemas[0].Schema = nil }},
		{name: "dropped collection", collectionID: 1, mutate: func(m *streamingpb.VChannelMeta) { m.State = streamingpb.VChannelState_VCHANNEL_STATE_DROPPED }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			meta := restoreSchemaTestMeta()
			if tc.mutate != nil {
				tc.mutate(meta)
			}
			view := NewVChannelViewFromMeta(meta)
			got, err := view.GetSchemaVersion(tc.collectionID, tc.version)
			require.Nil(t, got)
			require.Error(t, err)
			require.Equal(t, !tc.retriable, status.AsStreamingError(err).IsUnrecoverable())
		})
	}
}

func TestExactSchemaVersionBeforeRecovery(t *testing.T) {
	module := &VChannelRecoveryModule{vchannel: "v1"}
	_, err := module.GetSchemaVersion(1, 0)
	require.Error(t, err)
	require.False(t, status.AsStreamingError(err).IsUnrecoverable())
}

func TestLegacyRepeatedSchemaVersionHasOneReadIdentity(t *testing.T) {
	meta := restoreSchemaTestMeta()
	first := meta.CollectionInfo.Schemas[0]
	second := proto.Clone(first).(*streamingpb.CollectionSchemaOfVChannel)
	second.CheckpointTimeTick = 15
	second.Schema.Description = "legacy semantic update without a version bump"
	second.Schema.Fields = append(second.Schema.Fields, &schemapb.FieldSchema{FieldID: 101, Name: "legacy", DataType: schemapb.DataType_Int64})
	meta.CollectionInfo.Schemas = []*streamingpb.CollectionSchemaOfVChannel{first, second}
	data, err := proto.Marshal(meta)
	require.NoError(t, err)
	recovered := &streamingpb.VChannelMeta{}
	require.NoError(t, proto.Unmarshal(data, recovered))
	view := NewVChannelViewFromMeta(recovered)
	read, err := view.GetSchemaVersion(1, 0)
	require.NoError(t, err)
	require.Equal(t, second.Schema.Description, read.Schema.Description)
	require.True(t, proto.Equal(second.Schema, read.Schema), "exact lookup must return the canonical schema payload")
}
