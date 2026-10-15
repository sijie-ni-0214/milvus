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
	"maps"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/proxyutil"
	"github.com/milvus-io/milvus/internal/util/quotasync"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type localQuotaTestSource struct {
	types.MixCoord
	values, changes map[int64]*metricsinfo.CollectionQuotaSnapshot
	forceFull       bool
}

func (s *localQuotaTestSource) ReadQuotaSnapshot(_ context.Context, full bool) (*metricsinfo.QuotaSnapshot, error) {
	full = full || s.forceFull
	values := s.changes
	if full {
		values = s.values
	}
	total := int64(0)
	for _, v := range s.values {
		total += v.Bytes
	}
	return &metricsinfo.QuotaSnapshot{Full: full, TotalBytes: total, Collections: maps.Clone(values)}, nil
}

func (s *localQuotaTestSource) AcknowledgeQuotaSnapshot(_ *metricsinfo.QuotaSnapshot) {
	clear(s.changes)
	s.forceFull = false
}

func (s *localQuotaTestSource) GetConnectedDataNodeMetrics(context.Context, *milvuspb.GetMetricsRequest) ([]metricsinfo.DataNodeInfos, error) {
	return nil, nil
}

func (s *localQuotaTestSource) put(id, bytes int64) {
	value := &metricsinfo.CollectionQuotaSnapshot{Present: true, HasQuota: true, HasRows: true, Bytes: bytes, Rows: 12, PartitionBytes: map[int64]int64{id * 10: bytes}}
	s.values[id] = value
	s.changes[id] = value
}

func TestQuotaLocalCalculationMatchesReference(t *testing.T) {
	setQuotaTestParam(t, &Params.QuotaConfig.CollectionStatsCacheEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.LocalQuotaCalculationEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.VersionedRatesEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.DiskProtectionEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.DiskQuotaPerCollection, "1")
	setQuotaTestParam(t, &Params.QuotaConfig.DMLLimitEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.DMLMaxInsertRatePerCollection, "10")
	setQuotaTestParam(t, &Params.QuotaConfig.DMLMaxInsertRatePerPartition, "5")
	setQuotaTestParam(t, &Params.QuotaConfig.TtProtectionEnabled, "false")
	meta := quotaMetaForTest(3)
	source := &localQuotaTestSource{values: make(map[int64]*metricsinfo.CollectionQuotaSnapshot), changes: make(map[int64]*metricsinfo.CollectionQuotaSnapshot)}
	for id := range meta.collID2Meta {
		source.put(id, 100)
	}
	manager := proxyutil.NewProxyClientManager(nil)
	manager.GetProxyClients().Insert(1, nil)
	allocator := newMockTsoAllocator()
	allocator.GenerateTSOF = func(uint32) (uint64, error) { return 100, nil }
	local := NewQuotaCenter(manager, source, allocator, meta)
	reference := NewQuotaCenter(manager, nil, newMockTsoAllocator(), meta)
	t.Cleanup(local.cancel)
	t.Cleanup(reference.cancel)
	round := func() {
		t.Helper()
		local.prepareLocalQuota()
		require.NotNil(t, local.localQuota)
		require.NoError(t, local.collectLocalQuotaData(context.Background(), typeutil.NewSet[int64]()))
		reference.clearMetrics()
		reference.dataCoordMetrics = local.dataCoordMetrics
		reference.queryNodeMetrics = local.queryNodeMetrics
		reference.dataNodeMetrics = local.dataNodeMetrics
		reference.readableCollections = local.readableCollections
		for id := range source.values {
			if coll := meta.collID2Meta[id]; coll != nil && coll.Available() {
				reference.collectionIDToDBID.Insert(id, coll.DBID)
			}
		}
		require.NoError(t, reference.calculateRates())
		require.NoError(t, local.calculateRates())
		require.Equal(t, quotaTreeSnapshot(reference.rateLimiter.GetRootLimiters()), quotaTreeSnapshot(local.rateLimiter.GetRootLimiters()))
		require.NoError(t, local.publishQuotaRules())
		expected, err := quotasync.Flatten(reference.toRatesRequest().RootLimiter)
		require.NoError(t, err)
		require.Len(t, local.publishedRules.Policies, len(expected))
		for key, p := range expected {
			require.True(t, proto.Equal(quotasync.Canonical(p), local.publishedRules.Policies[key]), "scope %v", key)
		}
	}
	round()
	version := local.publishedRules.Version
	round()
	require.Equal(t, version, local.publishedRules.Version)
	// One changed collection must leave other computation nodes untouched.
	untouched := local.rateLimiter.GetCollectionLimiters(10, 101)
	source.put(100, 2*1024*1024)
	round()
	require.Same(t, untouched, local.rateLimiter.GetCollectionLimiters(10, 101))
	require.NotNil(t, local.publishedRules.Delta)
	require.Len(t, local.publishedRules.Delta.Upserts, 1)
	source.put(100, 100)
	round()
	// Memory pressure and release without segment changes.
	local.queryNodeMetrics[1] = &metricsinfo.QueryNodeQuotaMetrics{Effect: metricsinfo.NodeEffect{CollectionIDs: []int64{100}}, Hms: metricsinfo.HardwareMetrics{Memory: 100, MemoryUsage: 99}}
	round()
	delete(local.queryNodeMetrics, 1)
	round()
	// Membership discovered outside NodeEffect still enters publication and
	// is reset after it disappears, even with no configuration/data change.
	require.NoError(t, Params.Save(Params.QuotaConfig.ForceDenyReading.Key, "true"))
	t.Cleanup(func() { _ = Params.Reset(Params.QuotaConfig.ForceDenyReading.Key) })
	round()
	local.readableCollections = map[int64]map[int64][]int64{10: {100: nil}}
	round()
	local.readableCollections = nil
	round()
	require.NoError(t, Params.Save(Params.QuotaConfig.ForceDenyReading.Key, "false"))
	// DB deny and release, including without new collection data.
	meta.ddLock.Lock()
	meta.dbName2Meta["quota"].Properties = []*commonpb.KeyValuePair{{Key: common.DatabaseForceDenyWritingKey, Value: "true"}}
	meta.quotaGeneration++
	meta.ddLock.Unlock()
	round()
	meta.ddLock.Lock()
	meta.dbName2Meta["quota"].Properties = nil
	meta.quotaGeneration++
	meta.ddLock.Unlock()
	round()
	// Changed collection properties and restoration of a default.
	meta.ddLock.Lock()
	meta.collID2Meta[100].Properties = []*commonpb.KeyValuePair{{Key: common.CollectionInsertRateMaxKey, Value: "3"}}
	meta.invalidateQuotaMetadataLocked(100)
	meta.ddLock.Unlock()
	round()
	meta.ddLock.Lock()
	meta.collID2Meta[100].Properties = nil
	meta.invalidateQuotaMetadataLocked(100)
	meta.ddLock.Unlock()
	round()
	// Proxy count changes must revisit every finite per-Proxy rate.
	manager.GetProxyClients().Insert(2, nil)
	round()
	local.prepareLocalQuota()
	require.NoError(t, local.calculateRates())
	beforeMembershipChange := proto.Clone(local.publishedRules.Full())
	manager.GetProxyClients().Insert(3, nil)
	require.ErrorIs(t, local.publishQuotaRules(), merr.ErrServiceNotReady)
	require.True(t, proto.Equal(beforeMembershipChange, local.publishedRules.Full()))
	round()
	// Explicit collection/partition deletion survives incremental publication.
	meta.ddLock.Lock()
	delete(meta.collID2Meta, 100)
	meta.invalidateQuotaMetadataLocked(100)
	meta.ddLock.Unlock()
	round()
	require.NotContains(t, local.publishedRules.Policies, quotasync.Key{Scope: internalpb.RateScope_Collection, DB: 10, Collection: 100})
	// Healthy segments for deleted collection still count in global quota.
	require.EqualValues(t, 300, local.dataCoordMetrics.TotalBinlogSize)
	// Roll back calculation alone, then reconstruct the local baseline.
	require.NoError(t, Params.Save(Params.QuotaConfig.LocalQuotaCalculationEnabled.Key, "false"))
	local.prepareLocalQuota()
	require.Nil(t, local.localQuota)
	require.NoError(t, local.calculateRates())
	require.NoError(t, local.publishQuotaRules())
	require.NoError(t, Params.Save(Params.QuotaConfig.LocalQuotaCalculationEnabled.Key, "true"))
	round()
	// A disable/re-enable entirely between two cycles must also rebuild any
	// dropped metadata and reset DataCoord inputs to the replacement baseline.
	require.NoError(t, Params.Save(Params.QuotaConfig.LocalQuotaCalculationEnabled.Key, "false"))
	meta.ddLock.Lock()
	delete(meta.collID2Meta, 101)
	meta.invalidateQuotaMetadataLocked(101)
	meta.ddLock.Unlock()
	require.NoError(t, Params.Save(Params.QuotaConfig.LocalQuotaCalculationEnabled.Key, "true"))
	delete(source.values, 100)
	source.forceFull = true
	round()
	require.NotContains(t, local.localQuota.metadata, int64(101))
	require.NotContains(t, local.dataCoordMetrics.CollectionBinlogSize, int64(100))
	require.EqualValues(t, 200, local.dataCoordMetrics.TotalBinlogSize)
	// Removing a DB must also erase every descendant in the published FULL.
	meta.ddLock.Lock()
	for id := range meta.collID2Meta {
		delete(meta.collID2Meta, id)
		meta.invalidateQuotaMetadataLocked(id)
	}
	delete(meta.dbName2Meta, "quota")
	meta.quotaGeneration++
	meta.ddLock.Unlock()
	round()
	for key := range local.publishedRules.Policies {
		require.NotEqualValues(t, 10, key.DB)
	}
}

func TestQuotaLocalMetadataAcknowledgesGeneration(t *testing.T) {
	setQuotaTestParam(t, &Params.QuotaConfig.CollectionStatsCacheEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.LocalQuotaCalculationEnabled, "true")
	mt := quotaMetaForTest(1)
	mt.ddLock.Lock()
	mt.invalidateQuotaMetadataLocked(100)
	mt.ddLock.Unlock()
	first := mt.readQuotaMetadata(true)
	mt.ddLock.Lock()
	mt.invalidateQuotaMetadataLocked(100)
	mt.ddLock.Unlock()
	mt.acknowledgeQuotaMetadata(first)
	require.Contains(t, mt.readQuotaMetadata(false).Collections, int64(100))
	mt.acknowledgeQuotaMetadata(mt.readQuotaMetadata(false))
	require.Empty(t, mt.readQuotaMetadata(false).Collections)
}

func TestQuotaPublisherKeepsLastCompleteVersionOnCalculationFailure(t *testing.T) {
	setQuotaTestParam(t, &Params.QuotaConfig.DiskProtectionEnabled, "true")
	setQuotaTestParam(t, &Params.QuotaConfig.DiskQuotaPerCollection, "1")
	manager := proxyutil.NewProxyClientManager(nil)
	manager.GetProxyClients().Insert(1, nil)
	allocator := newMockTsoAllocator()
	allocator.GenerateTSOF = func(uint32) (uint64, error) { return 100, nil }
	q := NewQuotaCenter(manager, nil, allocator, quotaMetaForTest(1))
	t.Cleanup(q.cancel)
	q.collectionIDToDBID.Insert(100, 10)
	require.NoError(t, q.calculateRates())
	require.NoError(t, q.publishQuotaRules())
	original := proto.Clone(q.publishedRules.Full())
	q.dataCoordMetrics = &metricsinfo.DataCoordQuotaMetrics{TotalBinlogSize: 2 * 1024 * 1024, CollectionBinlogSize: map[int64]int64{100: 2 * 1024 * 1024}}
	allocator.GenerateTSOF = func(uint32) (uint64, error) { return 0, merr.ErrServiceNotReady }
	require.ErrorIs(t, q.calculateRates(), merr.ErrServiceNotReady)
	bucket, _ := q.rateLimiter.GetCollectionLimiters(10, 100).GetLimiters().Get(internalpb.RateType_DMLInsert)
	require.Zero(t, bucket.Limit()) // The failed candidate did change; it is not published.
	require.True(t, proto.Equal(original, q.publishedRules.Full()))
}

func BenchmarkQuotaLocalIdle(b *testing.B) {
	setQuotaTestParam(b, &Params.QuotaConfig.CollectionStatsCacheEnabled, "true")
	setQuotaTestParam(b, &Params.QuotaConfig.LocalQuotaCalculationEnabled, "true")
	for _, count := range []int{1000, 100000} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			manager := proxyutil.NewProxyClientManager(nil)
			manager.GetProxyClients().Insert(1, nil)
			allocator := newMockTsoAllocator()
			allocator.GenerateTSOF = func(uint32) (uint64, error) { return 100, nil }
			q := NewQuotaCenter(manager, &localQuotaTestSource{}, allocator, quotaMetaForTest(count))
			b.Cleanup(q.cancel)
			q.prepareLocalQuota()
			require.NoError(b, q.calculateRates())
			require.NoError(b, q.publishQuotaRules())
			version := q.publishedRules.Version
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q.prepareLocalQuota()
				if err := q.calculateRates(); err != nil {
					b.Fatal(err)
				}
				if err := q.publishQuotaRules(); err != nil {
					b.Fatal(err)
				}
				if q.publishedRules.Version != version {
					b.Fatal("unchanged calculation advanced policy version")
				}
			}
		})
	}
}
