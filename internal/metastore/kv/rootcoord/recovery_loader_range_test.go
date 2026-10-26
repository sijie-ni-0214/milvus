// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rootcoord

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/kv"
	pb "github.com/milvus-io/milvus/pkg/v3/proto/etcdpb"
	"github.com/milvus-io/milvus/pkg/v3/util"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func newRecoveryRangeTestClient(t *testing.T) *clientv3.Client {
	t.Helper()
	paramtable.Init()
	params := paramtable.Get()
	client, err := etcd.GetEtcdClient(
		params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		params.EtcdCfg.Endpoints.GetAsStrings(),
		params.EtcdCfg.EtcdTLSCert.GetValue(),
		params.EtcdCfg.EtcdTLSKey.GetValue(),
		params.EtcdCfg.EtcdTLSCACert.GetValue(),
		params.EtcdCfg.EtcdTLSMinVersion.GetValue())
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// writeRecoveryRangeTestMetadata writes collections whose children live in the
// child prefixes, with child keys of every shape a walk may meet: one
// collection holding most fields, children in the key order of their IDs as
// decimal strings, tombstones, orphan and malformed keys, a key with a
// non-canonical collection ID, and a legacy collection that recovery migrates,
// so it recovers twice. It returns the database IDs.
func writeRecoveryRangeTestMetadata(t *testing.T, metaKV kv.MetaKv) []int64 {
	t.Helper()
	ctx := context.Background()
	r := rand.New(rand.NewSource(11))
	collectionIDs := []int64{1, 2, 10, 11, 99, 100, 101, 999, 1000, 449412345678901234, 449412345678901235}
	for range 120 {
		collectionIDs = append(collectionIDs, 1+r.Int63n(1<<40))
	}
	kvs := make(map[string]string)
	marshal := func(message proto.Message) string {
		value, err := proto.Marshal(message)
		require.NoError(t, err)
		return string(value)
	}
	for i, collectionID := range collectionIDs {
		dbID := int64(1 + i%2)
		if collectionID == 100 {
			dbID = util.DefaultDBID
		}
		kvs[BuildCollectionKey(dbID, collectionID)] = marshal(&pb.CollectionInfo{
			ID:     collectionID,
			DbId:   dbID,
			Schema: &schemapb.CollectionSchema{Name: fmt.Sprintf("c%d", collectionID)},
			State:  pb.CollectionState_CollectionCreated,
		})
		fields := 1 + i%6
		if collectionID == 100 {
			fields = 400
		}
		for j := range fields {
			// Field IDs whose decimal order is not their numeric order.
			fieldID := int64([]int{100, 101, 1000, 99, 5, 10000}[j%6]*1000 + j)
			kvs[BuildFieldKey(collectionID, fieldID)] = marshal(&schemapb.FieldSchema{FieldID: fieldID, Name: fmt.Sprintf("f%d", fieldID)})
		}
		for j := range 1 + i%3 {
			partitionID := collectionID*10 + int64(j)
			kvs[BuildPartitionKey(collectionID, partitionID)] = marshal(&pb.PartitionInfo{
				CollectionId: collectionID, PartitionID: partitionID, PartitionName: strconv.FormatInt(partitionID, 10),
			})
		}
		if i%4 == 0 {
			kvs[BuildStructArrayFieldKey(collectionID, 7)] = marshal(&schemapb.StructArrayFieldSchema{FieldID: 7, Name: "s"})
			kvs[BuildFunctionKey(collectionID, 8)] = marshal(&schemapb.FunctionSchema{Id: 8, Name: "fn"})
			kvs[BuildFunctionKey(collectionID, 9)] = tombstone()
		}
	}
	// A legacy collection: recovery moves it to the default database, where it
	// recovers again with clones of its children.
	kvs[BuildCollectionKey(util.NonDBID, 555)] = marshal(&pb.CollectionInfo{
		ID: 555, DbId: util.NonDBID, Schema: &schemapb.CollectionSchema{Name: "legacy"}, State: pb.CollectionState_CollectionCreated,
	})
	kvs[BuildFieldKey(555, 1)] = marshal(&schemapb.FieldSchema{FieldID: 1, Name: "legacy-field"})
	kvs[BuildPartitionKey(555, 2)] = marshal(&pb.PartitionInfo{CollectionId: 555, PartitionID: 2, PartitionName: "legacy-partition"})
	for key, value := range map[string]string{
		FieldMetaPrefix + "/0/1":                 marshal(&schemapb.FieldSchema{FieldID: 1}), // before every collection
		FieldMetaPrefix + "/99999999999999/1":    marshal(&schemapb.FieldSchema{FieldID: 1}), // a collection that is not recovered
		FieldMetaPrefix + "/zzz/1":               "malformed-key-corrupt-value",
		FieldMetaPrefix + "/100":                 "no-child-corrupt-value",
		FieldMetaPrefix + "/0100/7":              marshal(&schemapb.FieldSchema{FieldID: 7, Name: "non-canonical"}),
		FieldMetaPrefix + "/101/tombstoned":      tombstone(),
		PartitionMetaPrefix + "/7777777777777/1": marshal(&pb.PartitionInfo{PartitionID: 1}),
	} {
		kvs[key] = value
	}
	keys := make([]string, 0, len(kvs))
	for key := range kvs {
		keys = append(keys, key)
	}
	for start := 0; start < len(keys); start += 64 {
		batch := make(map[string]string)
		for _, key := range keys[start:min(start+64, len(keys))] {
			batch[key] = kvs[key]
		}
		require.NoError(t, metaKV.MultiSave(ctx, batch))
	}
	return []int64{util.NonDBID, util.DefaultDBID, 2}
}

func recoverWithRanges(t *testing.T, client *clientv3.Client, ranges int, corrupt map[string]string) (map[int64][]*model.Collection, error) {
	t.Helper()
	key := paramtable.Get().MetaStoreCfg.ParallelScanRanges.Key
	paramtable.Get().Save(key, strconv.Itoa(ranges))
	defer paramtable.Get().Reset(key)
	root := fmt.Sprintf("unittest/rootcoord-recovery-ranges/%s/%d", t.Name(), time.Now().UnixNano())
	t.Cleanup(func() { _, _ = client.Delete(context.Background(), root, clientv3.WithPrefix()) })
	metaKV := etcdkv.NewEtcdKV(client, root)
	dbIDs := writeRecoveryRangeTestMetadata(t, metaKV)
	if len(corrupt) > 0 {
		require.NoError(t, metaKV.MultiSave(context.Background(), corrupt))
	}
	catalog := NewCatalog(metaKV).(*Catalog)
	return catalog.ListCollectionsForRecovery(context.Background(), dbIDs, 0)
}

// The child prefixes read as parallel key ranges recover the collections one
// walk of each recovers, every child list in the same order.
func TestListCollectionsForRecoveryInRangesLikeOneWalk(t *testing.T) {
	client := newRecoveryRangeTestClient(t)
	want, err := recoverWithRanges(t, client, 1, nil)
	require.NoError(t, err)
	c100 := findRecoveryCollection(t, want[util.DefaultDBID], 100)
	require.Len(t, c100.Fields, 400, "a key with a non-canonical collection ID is skipped")
	require.Len(t, want[util.NonDBID], 1)
	require.Len(t, findRecoveryCollection(t, want[util.DefaultDBID], 555).Fields, 1)

	for _, ranges := range []int{2, 3, 8, 64} {
		got, err := recoverWithRanges(t, client, ranges, nil)
		require.NoError(t, err)
		require.Equal(t, want, got, "ranges %d", ranges)
	}
}

// A child that does not decode fails the ranged recovery with the error one
// walk meets first.
func TestListCollectionsForRecoveryInRangesFailsOnTheFirstCorruptChild(t *testing.T) {
	client := newRecoveryRangeTestClient(t)
	corrupt := map[string]string{
		BuildFieldKey(449412345678901234, 3): "corrupt-late",
		BuildFieldKey(10, 3):                 "corrupt-early",
	}
	_, want := recoverWithRanges(t, client, 1, corrupt)
	require.ErrorIs(t, want, merr.ErrDataIntegrity)
	for _, ranges := range []int{2, 8, 64} {
		_, err := recoverWithRanges(t, client, ranges, corrupt)
		require.Equal(t, want.Error(), err.Error(), "ranges %d", ranges)
		require.ErrorIs(t, err, merr.ErrDataIntegrity)
	}
}
