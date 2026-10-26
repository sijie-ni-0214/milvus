// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package etcdkv

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"

	"github.com/milvus-io/milvus/pkg/v3/kv"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// rangeTestEtcds returns the etcds the range walk is checked against: the
// configured one, and an embedded etcd of the client's version, which serves
// RangeStream (an older configured etcd only serves the paginated fallback).
func rangeTestEtcds(t *testing.T) map[string]*clientv3.Client {
	t.Helper()
	external, err := etcd.GetEtcdClient(
		Params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		Params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		Params.EtcdCfg.Endpoints.GetAsStrings(),
		Params.EtcdCfg.EtcdTLSCert.GetValue(),
		Params.EtcdCfg.EtcdTLSKey.GetValue(),
		Params.EtcdCfg.EtcdTLSCACert.GetValue(),
		Params.EtcdCfg.EtcdTLSMinVersion.GetValue())
	require.NoError(t, err)
	t.Cleanup(func() { _ = external.Close() })
	return map[string]*clientv3.Client{
		"configured etcd": external,
		"embedded etcd":   startRangeTestEtcd(t),
	}
}

func freeLocalAddr(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	return listener.Addr().String()
}

// startRangeTestEtcd starts a private embedded etcd and returns a client that
// reaches it over gRPC.
func startRangeTestEtcd(t *testing.T) *clientv3.Client {
	t.Helper()
	cfg := embed.NewConfig()
	cfg.Dir = t.TempDir()
	cfg.LogLevel = "fatal"
	cfg.UnsafeNoFsync = true
	clientURL := url.URL{Scheme: "http", Host: freeLocalAddr(t)}
	peerURL := url.URL{Scheme: "http", Host: freeLocalAddr(t)}
	cfg.ListenClientUrls, cfg.AdvertiseClientUrls = []url.URL{clientURL}, []url.URL{clientURL}
	cfg.ListenPeerUrls, cfg.AdvertisePeerUrls = []url.URL{peerURL}, []url.URL{peerURL}
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	server, err := embed.StartEtcd(cfg)
	require.NoError(t, err)
	t.Cleanup(server.Close)
	select {
	case <-server.Server.ReadyNotify():
	case <-time.After(30 * time.Second):
		require.FailNow(t, "embedded etcd did not become ready")
	}
	client, err := clientv3.New(clientv3.Config{Endpoints: []string{clientURL.String()}, DialTimeout: 5 * time.Second})
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })
	stream, err := client.GetStream(context.Background(), "unittest/range-walk", clientv3.WithPrefix())
	require.NoError(t, err)
	_, err = clientv3.GetStreamToGetResponse(stream)
	require.NoError(t, err, "the embedded etcd serves RangeStream")
	return client
}

func putRangeTestKeys(t *testing.T, client *clientv3.Client, kvs map[string]string) {
	t.Helper()
	keys := make([]string, 0, len(kvs))
	for key := range kvs {
		keys = append(keys, key)
	}
	for batch := range slices.Chunk(keys, 100) {
		ops := make([]clientv3.Op, 0, len(batch))
		for _, key := range batch {
			ops = append(ops, clientv3.OpPut(key, kvs[key]))
		}
		_, err := client.Txn(context.Background()).Then(ops...).Commit()
		require.NoError(t, err)
	}
}

// writeCollectionKeys writes keys shaped <prefix><id>/<partition>/<segment>
// for ids in their decimal order and any length, one id holding most keys,
// plus keys of other shapes before, between and after them. It returns the
// ids.
func writeCollectionKeys(t *testing.T, client *clientv3.Client, prefix string) []int64 {
	t.Helper()
	r := rand.New(rand.NewSource(7))
	ids := []int64{1, 2, 5, 9, 10, 11, 50, 99, 100, 101, 500, 7777777, 449412345678901234, 449412345678901235}
	for range 150 {
		ids = append(ids, r.Int63n(1<<40))
	}
	kvs := make(map[string]string)
	for i, id := range ids {
		segments := 1 + i%5
		if id == 100 {
			segments = 400
		}
		for segment := range segments {
			key := fmt.Sprintf("%s%d/%d/%d", prefix, id, 1+segment%3, int64(segment)*1000+id%997)
			kvs[key] = strings.Repeat("v", 1+r.Intn(300))
		}
	}
	for _, key := range []string{
		prefix,              // the prefix itself
		prefix + "-5/1/1",   // sorts before every id
		prefix + "0/1/1",    // an id below the listed ones
		prefix + "100/",     // exactly the bound of id 100
		prefix + "100",      // right before that bound
		prefix + "100./1/1", // '.' sorts before '/'
		prefix + "1000/1/1", // an id that is not listed
		prefix + "abc/1/1",  // not an id
		prefix + "zzz",      // after every id
	} {
		kvs[key] = "orphan"
	}
	putRangeTestKeys(t, client, kvs)
	return ids
}

type rangeTestKeyValue struct {
	key, value                           string
	createRevision, modRevision, version int64
}

func rangeTestKeyValueOf(keyValue *mvccpb.KeyValue) rangeTestKeyValue {
	return rangeTestKeyValue{
		key:            string(keyValue.Key),
		value:          string(keyValue.Value),
		createRevision: keyValue.CreateRevision,
		modRevision:    keyValue.ModRevision,
		version:        keyValue.Version,
	}
}

func walkOnce(t *testing.T, client clientv3.KV, prefix string) []rangeTestKeyValue {
	t.Helper()
	var all []rangeTestKeyValue
	require.NoError(t, WalkPrefix(context.Background(), client, prefix, 7, time.Second, func(keyValue *mvccpb.KeyValue) error {
		all = append(all, rangeTestKeyValueOf(keyValue))
		return nil
	}))
	return all
}

// checkRanges checks that parts, the keys walked per range, hold the keys of
// their range only and, in range order, what one walk returned.
func checkRanges(t *testing.T, prefix string, bounds []string, parts [][]rangeTestKeyValue, oneWalk []rangeTestKeyValue) {
	t.Helper()
	require.Equal(t, oneWalk, slices.Concat(parts...), "bounds %q", bounds)
	starts := append([]string{prefix}, bounds...)
	for i, part := range parts {
		for _, keyValue := range part {
			require.GreaterOrEqual(t, keyValue.key, starts[i])
			if i+1 < len(starts) {
				require.Less(t, keyValue.key, starts[i+1])
			}
		}
	}
}

func TestWalkPrefixInRangesMatchesOneWalk(t *testing.T) {
	for name, client := range rangeTestEtcds(t) {
		t.Run(name, func(t *testing.T) {
			prefix := fmt.Sprintf("unittest/range-walk/%s/s/", funcutil.RandomString(8))
			t.Cleanup(func() { _, _ = client.Delete(context.Background(), prefix, clientv3.WithPrefix()) })
			ids := writeCollectionKeys(t, client, prefix)
			oneWalk := walkOnce(t, client, prefix)

			boundSets := [][]string{
				{prefix + "\x00"},                          // a first range of the prefix key alone
				{prefix + "100/"},                          // a key exactly at a bound
				{prefix + "zz", prefix + "zzz"},            // an empty range between two others
				{prefix + "1", prefix + "5", prefix + "9"}, // bounds of another shape
			}
			for _, ranges := range []int{2, 3, 8, 64, 1000} {
				boundSets = append(boundSets, kv.RangeBounds(prefix, ids, ranges))
			}
			for _, bounds := range boundSets {
				parts := make([][]rangeTestKeyValue, len(bounds)+1)
				err := WalkPrefixInRanges(context.Background(), client, prefix, bounds, 3, time.Second,
					func(rangeIndex int, keyValue *mvccpb.KeyValue) error {
						parts[rangeIndex] = append(parts[rangeIndex], rangeTestKeyValueOf(keyValue))
						return nil
					})
				require.NoError(t, err)
				checkRanges(t, prefix, bounds, parts, oneWalk)
			}
		})
	}
}

// Writes made after the revision of a walk is taken are seen by no range.
func TestWalkPrefixInRangesReadsOneRevision(t *testing.T) {
	for name, client := range rangeTestEtcds(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			prefix := fmt.Sprintf("unittest/range-walk/%s/s/", funcutil.RandomString(8))
			t.Cleanup(func() { _, _ = client.Delete(ctx, prefix, clientv3.WithPrefix()) })
			ids := writeCollectionKeys(t, client, prefix)
			snapshot := walkOnce(t, client, prefix)
			revision, err := currentRevision(ctx, client, prefix, time.Second)
			require.NoError(t, err)

			bounds := kv.RangeBounds(prefix, ids, 8)
			require.Len(t, bounds, 7)
			for i, bound := range bounds {
				_, err := client.Delete(ctx, bound, clientv3.WithPrefix())
				require.NoError(t, err)
				_, err = client.Put(ctx, bound+fmt.Sprintf("new/%d", i), "new")
				require.NoError(t, err)
			}
			_, err = client.Put(ctx, prefix+"zzz", "changed")
			require.NoError(t, err)
			require.NotEqual(t, snapshot, walkOnce(t, client, prefix))

			starts := append([]string{prefix}, bounds...)
			parts := make([][]rangeTestKeyValue, len(starts))
			require.NoError(t, walkRangesAt(ctx, client, starts, clientv3.GetPrefixRangeEnd(prefix), revision, 2, time.Second,
				func(rangeIndex int, keyValue *mvccpb.KeyValue) error {
					parts[rangeIndex] = append(parts[rangeIndex], rangeTestKeyValueOf(keyValue))
					return nil
				}))
			checkRanges(t, prefix, bounds, parts, snapshot)
		})
	}
}

// A revision compacted away fails the walk like a compaction during one
// stream: with a retriable IO error.
func TestWalkPrefixInRangesFailsOnACompactedRevision(t *testing.T) {
	ctx := context.Background()
	client := startRangeTestEtcd(t)
	prefix := "unittest/range-walk/compacted/s/"
	ids := writeCollectionKeys(t, client, prefix)
	revision, err := currentRevision(ctx, client, prefix, time.Second)
	require.NoError(t, err)
	resp, err := client.Put(ctx, prefix+"zzz", "changed")
	require.NoError(t, err)
	_, err = client.Compact(ctx, resp.Header.Revision, clientv3.WithCompactPhysical())
	require.NoError(t, err)

	bounds := kv.RangeBounds(prefix, ids, 4)
	starts := append([]string{prefix}, bounds...)
	err = walkRangesAt(ctx, client, starts, clientv3.GetPrefixRangeEnd(prefix), revision, 2, time.Second,
		func(int, *mvccpb.KeyValue) error { return nil })
	require.ErrorIs(t, err, merr.ErrIoFailed)
	require.Contains(t, err.Error(), "compacted")
	require.Equal(t, merr.SystemError, merr.GetErrorType(err))
}

// The kv implementations walk their root like WalkWithPrefix does.
func TestKVWalkWithPrefixInRanges(t *testing.T) {
	client := startRangeTestEtcd(t)
	root := "unittest/range-walk/kv"
	metaKV := NewEtcdKV(client, root)
	for i := range 50 {
		require.NoError(t, metaKV.Save(context.Background(), fmt.Sprintf("s/%d/1/%d", i, i), "v"))
	}
	require.NoError(t, metaKV.Save(context.Background(), "t/1", "outside"))

	var oneWalk []string
	require.NoError(t, metaKV.WalkWithPrefix(context.Background(), "s/", 10, func(key, _ []byte) error {
		oneWalk = append(oneWalk, string(key))
		return nil
	}))
	bounds := kv.RangeBounds("s/", []int64{10, 20, 30, 40}, 3)
	require.Equal(t, []string{"s/20/", "s/30/"}, bounds)
	parts := make([][]string, len(bounds)+1)
	require.NoError(t, kv.WalkWithPrefixInRanges(context.Background(), metaKV, "s/", bounds, 10, func(rangeIndex int, key, _ []byte) error {
		parts[rangeIndex] = append(parts[rangeIndex], string(key))
		return nil
	}))
	require.Len(t, oneWalk, 50)
	require.Equal(t, oneWalk, slices.Concat(parts...))
	for _, part := range parts {
		require.NotEmpty(t, part)
	}
	require.True(t, strings.HasPrefix(parts[1][0], root+"/s/20/"))
}
