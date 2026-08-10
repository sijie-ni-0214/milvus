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

package datacoord

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestCatalogListAllSegmentsPreservesBinlogsAndPrefixBoundaries(t *testing.T) {
	kv := newSnapshotExportJobMetaKV()
	put := func(key string, value proto.Message) {
		encoded, err := proto.Marshal(value)
		require.NoError(t, err)
		require.NoError(t, kv.Save(context.Background(), key, string(encoded)))
	}
	put(SegmentPrefix+"/1/2/10", &datapb.SegmentInfo{ID: 10, CollectionID: 1, PartitionID: 2})
	put(SegmentPrefix+"/10/2/20", &datapb.SegmentInfo{ID: 20, CollectionID: 10, PartitionID: 2})
	put(SegmentBinlogPathPrefix+"/1/2/10/100", &datapb.FieldBinlog{
		FieldID: 100, Binlogs: []*datapb.Binlog{{LogID: 123, LogSize: 40, EntriesNum: 5}},
	})
	// A sibling prefix must not be decoded as segment metadata.
	require.NoError(t, kv.Save(context.Background(), SegmentPrefix+"-sibling/invalid", "not-a-protobuf"))
	catalog := &Catalog{MetaKv: kv, paginationSize: 1}
	segments, err := catalog.ListAllSegments(context.Background())
	require.NoError(t, err)
	require.Len(t, segments, 2)
	require.EqualValues(t, 10, segments[0].GetID())
	require.Len(t, segments[0].GetBinlogs(), 1)
	require.EqualValues(t, 123, segments[0].GetBinlogs()[0].GetBinlogs()[0].GetLogID())
	require.EqualValues(t, 40, segments[0].GetBinlogs()[0].GetBinlogs()[0].GetMemorySize())

	scoped, err := catalog.ListSegments(context.Background(), 1)
	require.NoError(t, err)
	require.Len(t, scoped, 1, "collection 1 must not scan collection 10")
	require.EqualValues(t, 10, scoped[0].GetID())
}
