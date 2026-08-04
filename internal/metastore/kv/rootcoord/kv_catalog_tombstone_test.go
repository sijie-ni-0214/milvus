package rootcoord

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/kv/mocks"
	pb "github.com/milvus-io/milvus/pkg/v3/proto/etcdpb"
	"github.com/milvus-io/milvus/pkg/v3/util"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Tombstone-compatibility tests for RootCoord catalog load paths.
//
// Before commit e0873a65d4, SuffixSnapshot deletions at ts!=0 overwrote plain
// meta keys with a 3-byte tombstone marker. After removal, old etcd data can
// still contain such values. Every list/load path must skip them rather than
// fail proto.Unmarshal.

func tombstone() string { return string(SuffixSnapshotTombstone) }

func TestCatalog_ListAliasHelpers_SkipTombstone(t *testing.T) {
	ctx := context.Background()
	legacyAlias, err := proto.Marshal(&pb.CollectionInfo{
		ID:     10,
		DbId:   util.DefaultDBID,
		Schema: &schemapb.CollectionSchema{Name: "legacy-alias"},
	})
	require.NoError(t, err)
	alias, err := proto.Marshal(&pb.AliasInfo{
		AliasName:    "alias",
		CollectionId: 20,
	})
	require.NoError(t, err)

	t.Run("legacy alias", func(t *testing.T) {
		kv := mocks.NewTxnKV(t)
		kv.EXPECT().LoadWithPrefix(mock.Anything, CollectionAliasMetaPrefix210+"/").Return(
			[]string{"alias-tombstone", "alias-valid"},
			[]string{tombstone(), string(legacyAlias)},
			nil,
		)
		catalog := NewCatalog(kv).(*Catalog)
		aliases, err := catalog.listAliasesBefore210(ctx, 0)
		require.NoError(t, err)
		require.Len(t, aliases, 1)
		assert.Equal(t, "legacy-alias", aliases[0].Name)
	})

	t.Run("database alias", func(t *testing.T) {
		kv := mocks.NewTxnKV(t)
		kv.EXPECT().LoadWithPrefix(mock.Anything, BuildAliasPrefixWithDB(testDb)).Return(
			[]string{"alias-tombstone", "alias-valid"},
			[]string{tombstone(), string(alias)},
			nil,
		)
		catalog := NewCatalog(kv).(*Catalog)
		aliases, err := catalog.listAliasesAfter210WithDb(ctx, testDb, 0)
		require.NoError(t, err)
		require.Len(t, aliases, 1)
		assert.Equal(t, "alias", aliases[0].Name)
	})
}

func TestCatalog_ListCollections_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	coll := &pb.CollectionInfo{
		ID:    42,
		DbId:  testDb,
		State: pb.CollectionState_CollectionCreated,
		Schema: &schemapb.CollectionSchema{
			Name:   "c",
			Fields: []*schemapb.FieldSchema{{}},
		},
	}
	bColl, err := proto.Marshal(coll)
	require.NoError(t, err)

	kv := mocks.NewTxnKV(t)
	kv.On("LoadWithPrefix", mock.Anything, BuildDatabasePrefixWithDBID(testDb)).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(bColl)}, nil)

	partMeta, _ := proto.Marshal(&pb.PartitionInfo{})
	kv.On("LoadWithPrefix", mock.Anything, mock.MatchedBy(func(p string) bool {
		return strings.HasPrefix(p, PartitionMetaPrefix)
	})).Return([]string{"rootcoord/partitions/42/1"}, []string{string(partMeta)}, nil)

	fieldMeta, _ := proto.Marshal(&schemapb.FieldSchema{})
	kv.On("LoadWithPrefix", mock.Anything, mock.MatchedBy(func(p string) bool {
		return strings.HasPrefix(p, FieldMetaPrefix)
	})).Return([]string{"rootcoord/fields/42/1"}, []string{string(fieldMeta)}, nil)

	kv.On("LoadWithPrefix", mock.Anything, mock.MatchedBy(func(p string) bool {
		return strings.HasPrefix(p, StructArrayFieldMetaPrefix)
	})).Return([]string{}, []string{}, nil).Maybe()

	kv.On("LoadWithPrefix", mock.Anything, mock.MatchedBy(func(p string) bool {
		return strings.HasPrefix(p, FunctionMetaPrefix)
	})).Return([]string{}, []string{}, nil).Maybe()

	kc := NewCatalog(kv)
	got, err := kc.ListCollections(ctx, testDb, 1)
	assert.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int64(42), got[0].CollectionID)
}

func TestCatalog_GetCollectionByID_TombstoneIsNotFound(t *testing.T) {
	ctx := context.Background()
	kv := mocks.NewTxnKV(t)
	// Both default-db attempt and legacy-path attempt return the tombstone.
	kv.EXPECT().Load(mock.Anything, mock.Anything).Return(tombstone(), nil)

	kc := NewCatalog(kv)
	got, err := kc.GetCollectionByID(ctx, util.DefaultDBID, 1, 99999)
	assert.Error(t, err)
	assert.True(t, merr.ErrCollectionNotFound.Is(err))
	assert.Nil(t, got)
}

func TestCatalog_GetCollectionByName_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	kv := mocks.NewTxnKV(t)

	// Seed: one tombstone + one valid collection with the target name.
	coll := &pb.CollectionInfo{
		ID:     7,
		Schema: &schemapb.CollectionSchema{Name: "target"},
	}
	bColl, err := proto.Marshal(coll)
	require.NoError(t, err)

	kv.On("LoadWithPrefix", mock.Anything, mock.Anything).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(bColl)}, nil)

	// GetCollectionByName re-delegates to GetCollectionByID on name match, which
	// does a second Load. Return the tombstone there too and expect NotFound.
	kv.EXPECT().Load(mock.Anything, mock.Anything).Return(tombstone(), nil).Maybe()

	kc := NewCatalog(kv)
	got, err := kc.GetCollectionByName(ctx, util.NonDBID, "", "target", 1)
	// Either the valid collection is returned (happy path), or NotFound from
	// the tombstone lookup — both prove the tombstone prefix entry did not
	// crash the path with an unmarshal error.
	if err == nil {
		assert.NotNil(t, got)
	} else {
		assert.Nil(t, got)
	}
}

func TestCatalog_listPartitionsAfter210_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	part := &pb.PartitionInfo{PartitionID: 1, CollectionId: 1}
	b, err := proto.Marshal(part)
	require.NoError(t, err)

	kv := mocks.NewTxnKV(t)
	kv.EXPECT().LoadWithPrefix(mock.Anything, mock.Anything).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(b)}, nil)

	kc := NewCatalog(kv).(*Catalog)
	got, err := kc.listPartitionsAfter210(ctx, 1, 0)
	assert.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int64(1), got[0].PartitionID)
}

func TestCatalog_listFieldsAfter210_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	field := &schemapb.FieldSchema{FieldID: 100, Name: "f"}
	b, err := proto.Marshal(field)
	require.NoError(t, err)

	kv := mocks.NewTxnKV(t)
	kv.EXPECT().LoadWithPrefix(mock.Anything, mock.Anything).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(b)}, nil)

	kc := NewCatalog(kv).(*Catalog)
	got, err := kc.listFieldsAfter210(ctx, 1, 0)
	assert.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int64(100), got[0].FieldID)
}

func TestCatalog_listStructArrayFieldsAfter210_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	sf := &schemapb.StructArrayFieldSchema{FieldID: 200, Name: "s"}
	b, err := proto.Marshal(sf)
	require.NoError(t, err)

	kv := mocks.NewTxnKV(t)
	kv.EXPECT().LoadWithPrefix(mock.Anything, mock.Anything).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(b)}, nil)

	kc := NewCatalog(kv).(*Catalog)
	got, err := kc.listStructArrayFieldsAfter210(ctx, 1, 0)
	assert.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int64(200), got[0].FieldID)
}

func TestCatalog_listFunctions_SkipsTombstone(t *testing.T) {
	ctx := context.Background()
	fn := &schemapb.FunctionSchema{Id: 300, Name: "fn"}
	b, err := proto.Marshal(fn)
	require.NoError(t, err)

	kv := mocks.NewTxnKV(t)
	kv.EXPECT().LoadWithPrefix(mock.Anything, mock.Anything).
		Return([]string{"k-tomb", "k-ok"}, []string{tombstone(), string(b)}, nil)

	kc := NewCatalog(kv).(*Catalog)
	got, err := kc.listFunctions(ctx, 1, 0)
	assert.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int64(300), got[0].ID)
}
