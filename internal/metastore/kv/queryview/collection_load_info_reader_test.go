package queryview

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestCollectionLoadInfoExactRead(t *testing.T) {
	_, storage := newTestCatalog(t)
	ctx := t.Context()
	info := &viewpb.CollectionLoadConfig{
		CollectionID: 1, Version: 10,
		PartitionIds: []int64{100}, RequiredFields: []*viewpb.LoadFieldConfig{{FieldId: 101, IndexIds: []int64{201, 202}}},
	}
	encoded, err := proto.Marshal(info)
	require.NoError(t, err)
	key := func(v uint64) string { return fmt.Sprintf("%s/1/%d", VersionedLoadInfoPrefix, v) }
	require.NoError(t, storage.Save(ctx, key(10), string(encoded)))
	got, err := GetCollectionLoadInfo(ctx, storage, 1, 10)
	require.NoError(t, err)
	require.True(t, proto.Equal(info, got))
	missing, err := GetCollectionLoadInfo(ctx, storage, 1, 9)
	require.NoError(t, err)
	require.Nil(t, missing, "never substitute the latest record")
	require.NoError(t, storage.Save(ctx, key(11), string(encoded)))
	_, err = GetCollectionLoadInfo(ctx, storage, 1, 11)
	require.ErrorIs(t, err, merr.ErrDataIntegrity)
	require.NoError(t, storage.Save(ctx, key(12), "invalid protobuf"))
	_, err = GetCollectionLoadInfo(ctx, storage, 1, 12)
	require.Error(t, err)
	require.NoError(t, storage.Remove(ctx, key(10)))
	missing, err = GetCollectionLoadInfo(ctx, storage, 1, 10)
	require.NoError(t, err)
	require.Nil(t, missing, "released records must not survive in a reader cache")
}

func TestCollectionLoadInfoAutoLoadWireCompatibility(t *testing.T) {
	_, storage := newTestCatalog(t)
	catalog := NewCollectionLoadInfoCatalog(storage)
	ctx := t.Context()
	const epoch uint64 = 465125678901234567
	field9 := func(values ...uint64) []byte {
		var raw []byte
		for _, value := range values {
			raw = protowire.AppendTag(raw, 9, protowire.VarintType)
			raw = protowire.AppendVarint(raw, value)
		}
		return raw
	}
	nested := protowire.AppendBytes(protowire.AppendTag(nil, 100, protowire.BytesType), field9(epoch))
	for _, tc := range []struct {
		name string
		raw  []byte
		want bool
	}{
		{"absent", nil, false},
		{"false", field9(0), false},
		{"true", field9(1), true},
		{"historical epoch", field9(epoch), false},
		{"epoch then true", field9(epoch, 1), true},
		{"true then epoch", field9(1, epoch), false},
		{"epoch then false", field9(epoch, 0), false},
		{"nested tag", append(field9(1), nested...), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			info := &viewpb.CollectionLoadConfig{CollectionID: 25, Version: 11, PartitionIds: []int64{51, 52}}
			wire, err := proto.Marshal(info)
			require.NoError(t, err)
			wire = append(wire, tc.raw...)
			key := collectionLoadInfoVersionKey(info.CollectionID, info.Version)
			require.NoError(t, storage.Save(ctx, key, string(wire)))
			recovered, err := catalog.List(ctx)
			require.NoError(t, err)
			require.Len(t, recovered, 1)
			point, err := GetCollectionLoadInfo(ctx, storage, info.CollectionID, info.Version)
			require.NoError(t, err)
			require.Equal(t, tc.want, recovered[0].GetAutoLoad())
			require.True(t, proto.Equal(recovered[0], point))
			require.Equal(t, info.PartitionIds, point.PartitionIds)
			stored, err := storage.Load(ctx, key)
			require.NoError(t, err)
			require.Equal(t, string(wire), stored)
		})
	}
	for _, autoLoad := range []bool{true, false} {
		info := &viewpb.CollectionLoadConfig{CollectionID: 25, Version: 11, AutoLoad: autoLoad}
		require.NoError(t, catalog.Save(ctx, info))
		recovered, err := catalog.List(ctx)
		require.NoError(t, err)
		require.Len(t, recovered, 1)
		require.True(t, proto.Equal(info, recovered[0]))
		point, err := GetCollectionLoadInfo(ctx, storage, info.CollectionID, info.Version)
		require.NoError(t, err)
		require.True(t, proto.Equal(info, point))
	}
	key := collectionLoadInfoVersionKey(25, 11)
	require.NoError(t, storage.Save(ctx, key, string(append(field9(1), 0x48, 0x80))))
	_, err := catalog.List(ctx)
	require.Error(t, err)
	_, err = GetCollectionLoadInfo(ctx, storage, 25, 11)
	require.Error(t, err)
	require.NoError(t, storage.Remove(ctx, key))
}
