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

package main

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	tikverr "github.com/tikv/client-go/v2/error"
	"github.com/tikv/client-go/v2/oracle"
	"github.com/tikv/client-go/v2/txnkv"
	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"
	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	tikvutil "github.com/milvus-io/milvus/pkg/v3/util/tikv"
)

func TestMain(m *testing.M) {
	paramtable.InitWithBaseTable(paramtable.NewBaseTable(paramtable.SkipRemote(true), paramtable.SkipEnv(true)))
	m.Run()
}

func freeLocalAddr(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	return listener.Addr().String()
}

// startTestEtcd starts a private embedded etcd, which serves RangeStream, and
// returns a client that reaches it over gRPC.
func startTestEtcd(t *testing.T) *clientv3.Client {
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
	return client
}

func put(t *testing.T, cli *clientv3.Client, kvs map[string]string) {
	t.Helper()
	ops := make([]clientv3.Op, 0, 100)
	flush := func() {
		if len(ops) > 0 {
			_, err := cli.Txn(context.Background()).Then(ops...).Commit()
			require.NoError(t, err)
			ops = ops[:0]
		}
	}
	for key, value := range kvs {
		ops = append(ops, clientv3.OpPut(key, value))
		if len(ops) == cap(ops) {
			flush()
		}
	}
	flush()
}

// tikvGet reads key from TiKV, returning false when it does not exist.
func tikvGet(t *testing.T, store *txnkv.Client, key string) (string, bool) {
	t.Helper()
	ts, err := store.CurrentTimestamp(oracle.GlobalTxnScope)
	require.NoError(t, err)
	entry, err := store.GetSnapshot(ts).Get(context.Background(), []byte(key))
	if tikverr.IsErrNotFound(err) {
		return "", false
	}
	require.NoError(t, err)
	return string(entry.Value), true
}

func tikvSet(t *testing.T, store *txnkv.Client, key, value string) {
	t.Helper()
	txn, err := store.Begin()
	require.NoError(t, err)
	if value == "" {
		require.NoError(t, txn.Delete([]byte(key)))
	} else {
		require.NoError(t, txn.Set([]byte(key), []byte(value)))
	}
	require.NoError(t, txn.Commit(context.Background()))
}

// The copy moves the kv root and the meta root without the sessions, keeps an
// empty value as the placeholder of the TiKV meta store, and the verify pass
// then finds no difference; it finds each kind of difference after TiKV is
// changed.
func TestCopyAndVerify(t *testing.T) {
	ctx := context.Background()
	cli := startTestEtcd(t)
	store := tikvutil.SetupLocalTxn()
	defer store.Close()

	copied := map[string]string{
		"c1/kv/gid/idTimestamp":                                "id",
		"c1/kv/gid/timestamp":                                  "ts",
		"c1/meta/datacoord-meta/s/1/2/3":                       "segment",
		"c1/meta/empty":                                        "",
		"c1/meta/streamingnode-meta/wal/p1/consume-checkpoint": "checkpoint",
	}
	for i := 0; i < 10000; i++ {
		copied[fmt.Sprintf("c1/meta/root-coord/fields/%05d", i)] = fmt.Sprintf("field-%d", i)
	}
	put(t, cli, copied)
	kept := map[string]string{
		"c1/meta/session/proxy-1": "session",
		"c1/config/a":             "config",
		"c1/woodpecker/b":         "woodpecker",
		"c1/metadata":             "outside",
		"c2/meta/x":               "other cluster",
	}
	put(t, cli, kept)

	cfg := &config{etcdRoot: "c1", tikvRoot: "t1", ranges: 4, writers: 4, batch: 100}
	ranges, err := split(ctx, cli, cfg, cfg.ranges)
	require.NoError(t, err)
	require.Greater(t, len(ranges), len(cfg.sourceRanges()), "the big family is split")
	require.NoError(t, checkCover(cfg, ranges))

	require.NoError(t, prepareTarget(ctx, store, cfg, false))
	result := &copyResult{}
	require.NoError(t, copyKeys(ctx, cli, store, cfg, ranges, result))
	assert.Equal(t, int64(len(copied)), result.keys.Load())

	value, ok := tikvGet(t, store, "t1/meta/empty")
	require.True(t, ok)
	assert.Equal(t, tikvutil.EmptyValueString, value)
	value, ok = tikvGet(t, store, "t1/kv/gid/idTimestamp")
	require.True(t, ok)
	assert.Equal(t, "id", value)
	for key := range kept {
		_, ok := tikvGet(t, store, "t1"+key[len("c1"):])
		assert.False(t, ok, key)
	}

	verified := &verifyResult{}
	require.NoError(t, verifyKeys(ctx, cli, store, cfg, ranges, verified))
	assert.False(t, verified.failed(), "%v", verified.examples)
	assert.Equal(t, int64(len(copied)), verified.compared.Load())
	assert.Equal(t, int64(10000), verified.families["meta/root-coord"])

	tikvSet(t, store, "t1/meta/datacoord-meta/s/1/2/3", "changed")
	tikvSet(t, store, "t1/meta/root-coord/fields/00042", "")
	tikvSet(t, store, "t1/meta/root-coord/fields/00042x", "extra")
	verified = &verifyResult{}
	require.NoError(t, verifyKeys(ctx, cli, store, cfg, ranges, verified))
	assert.True(t, verified.failed())
	assert.Equal(t, int64(1), verified.different.Load())
	assert.Equal(t, int64(1), verified.missing.Load())
	assert.Equal(t, int64(1), verified.extra.Load())
	assert.ElementsMatch(t, []string{
		"different t1/meta/datacoord-meta/s/1/2/3",
		"missing t1/meta/root-coord/fields/00042",
		"extra t1/meta/root-coord/fields/00042x",
	}, verified.examples)

	// a second copy needs -overwrite, which replaces the target keys
	assert.Error(t, prepareTarget(ctx, store, cfg, false))
	require.NoError(t, prepareTarget(ctx, store, cfg, true))
	_, ok = tikvGet(t, store, "t1/meta/root-coord/fields/00042x")
	assert.False(t, ok)
	require.NoError(t, copyKeys(ctx, cli, store, cfg, ranges, &copyResult{}))
	verified = &verifyResult{}
	require.NoError(t, verifyKeys(ctx, cli, store, cfg, ranges, verified))
	assert.False(t, verified.failed(), "%v", verified.examples)
}

func TestCopyAtRevision(t *testing.T) {
	ctx := t.Context()
	cli := startTestEtcd(t)
	store := tikvutil.SetupLocalTxn()
	defer store.Close()

	put(t, cli, map[string]string{
		"c1/kv/gid/idTimestamp": "12345678",
		"c1/kv/gid/timestamp":   "87654321",
		"c1/meta/updated":       "original",
		"c1/meta/deleted":       "still present",
	})
	read, err := cli.Get(ctx, "c1/kv/gid/idTimestamp")
	require.NoError(t, err)
	cfg := &config{revision: read.Header.Revision, etcdRoot: "c1", tikvRoot: "c1", ranges: 3, writers: 3, batch: 2}
	changes, err := cli.Txn(ctx).Then(
		clientv3.OpPut("c1/meta/updated", "new value"),
		clientv3.OpPut("c1/meta/created", "new key"),
		clientv3.OpDelete("c1/meta/deleted"),
		clientv3.OpDelete("c1/kv/gid/timestamp"),
	).Commit()
	require.NoError(t, err)

	require.NoError(t, checkSource(ctx, cli, cfg, true))
	ranges, err := split(ctx, cli, cfg, cfg.ranges)
	require.NoError(t, err)
	result := &copyResult{}
	require.NoError(t, copyKeys(ctx, cli, store, cfg, ranges, result))
	require.EqualValues(t, 4, result.keys.Load())
	verified := &verifyResult{}
	require.NoError(t, verifyKeys(ctx, cli, store, cfg, ranges, verified))
	require.False(t, verified.failed(), "%v", verified.examples)

	latest := *cfg
	latest.revision = 0
	diffs := &verifyResult{}
	require.NoError(t, verifyKeys(ctx, cli, store, &latest, ranges, diffs))
	require.ElementsMatch(t, []string{
		"missing c1/meta/created", "different c1/meta/updated",
		"extra c1/meta/deleted", "extra c1/kv/gid/timestamp",
	}, diffs.examples)

	_, err = cli.Compact(ctx, changes.Header.Revision)
	require.NoError(t, err)
	require.ErrorIs(t, checkSource(ctx, cli, cfg, true), rpctypes.ErrCompacted)
	require.ErrorIs(t, verifyKeys(ctx, cli, store, cfg, ranges, &verifyResult{}), rpctypes.ErrCompacted)
	cfg.revision = changes.Header.Revision + 10
	require.ErrorIs(t, checkSource(ctx, cli, cfg, true), rpctypes.ErrFutureRev)
}

// A value equal to the placeholder would read back as empty, so the copy
// refuses it.
func TestCopyRefusesPlaceholderValue(t *testing.T) {
	ctx := context.Background()
	cli := startTestEtcd(t)
	store := tikvutil.SetupLocalTxn()
	defer store.Close()
	put(t, cli, map[string]string{"c1/meta/x": tikvutil.EmptyValueString})

	cfg := &config{etcdRoot: "c1", tikvRoot: "c1", ranges: 1, writers: 1, batch: 10}
	err := copyKeys(ctx, cli, store, cfg, cfg.sourceRanges(), &copyResult{})
	assert.ErrorContains(t, err, "reserved")
}

// A ranges file is reused only when it covers the source keys exactly.
func TestRangesFile(t *testing.T) {
	ctx := context.Background()
	cli := startTestEtcd(t)
	keys := make(map[string]string)
	for i := 0; i < 10000; i++ {
		keys[fmt.Sprintf("c1/meta/root-coord/fields/%05d", i)] = "v"
	}
	put(t, cli, keys)
	cfg := &config{etcdRoot: "c1", tikvRoot: "c1", ranges: 4}
	file := filepath.Join(t.TempDir(), "ranges.json")

	saved, err := loadOrSplit(ctx, cli, cfg, file)
	require.NoError(t, err)
	loaded, err := loadOrSplit(ctx, cli, cfg, file)
	require.NoError(t, err)
	assert.Equal(t, saved, loaded)

	other := &config{etcdRoot: "c2", tikvRoot: "c2", ranges: 4}
	_, err = loadOrSplit(ctx, cli, other, file)
	assert.Error(t, err)

	gap := append([]keyRange(nil), saved...)
	gap[1].Start = append(gap[1].Start, 'x')
	assert.Error(t, checkCover(cfg, gap))
	assert.Error(t, checkCover(cfg, saved[1:]))
}

func TestSourcePreflight(t *testing.T) {
	ctx := context.Background()
	cli := startTestEtcd(t)
	cfg := &config{etcdRoot: "c1"}
	put(t, cli, map[string]string{"c1/meta/session/proxy": "ignored"})
	require.ErrorContains(t, checkSource(ctx, cli, cfg, false), "c1/kv/")
	put(t, cli, map[string]string{"c1/kv/gid/idTimestamp": "12345678"})
	require.ErrorContains(t, checkSource(ctx, cli, cfg, false), "no source keys")
	// A catalog can have all its metadata on the far side of session/.
	put(t, cli, map[string]string{"c1/meta/z": "v"})
	require.NoError(t, checkSource(ctx, cli, cfg, false))
	require.ErrorContains(t, checkSource(ctx, cli, cfg, true), "gid/timestamp")
	put(t, cli, map[string]string{"c1/kv/gid/timestamp": "short"})
	require.ErrorContains(t, checkSource(ctx, cli, cfg, true), "8-byte")
	put(t, cli, map[string]string{"c1/kv/gid/timestamp": "12345678"})
	require.NoError(t, checkSource(ctx, cli, cfg, true))
}

type testStreamServer struct {
	pb.UnimplementedKVServer
	send func(*pb.RangeRequest, pb.KV_RangeStreamServer) error
}

func (s testStreamServer) RangeStream(req *pb.RangeRequest, stream pb.KV_RangeStreamServer) error {
	return s.send(req, stream)
}

func streamClient(t *testing.T, server testStreamServer) *clientv3.Client {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	g := grpc.NewServer()
	pb.RegisterKVServer(g, server)
	go func() { _ = g.Serve(listener) }()
	t.Cleanup(g.Stop)
	cli, err := clientv3.New(clientv3.Config{Endpoints: []string{listener.Addr().String()}, DialTimeout: time.Second})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cli.Close() })
	return cli
}

// The byte after "a" is a valid key and must survive resume after a partial
// stream, including when the transport closes without etcd's final header.
func TestStreamResume(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
	}{
		{"unavailable", status.Error(codes.Unavailable, "interrupted")},
		{"truncated", nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config{revision: 42}
			var calls atomic.Int32
			cli := streamClient(t, testStreamServer{send: func(req *pb.RangeRequest, stream pb.KV_RangeStreamServer) error {
				assert.Equal(t, cfg.revision, req.Revision)
				chunk := &pb.RangeResponse{Kvs: []*mvccpb.KeyValue{{Key: req.Key}}}
				if calls.Add(1) == 1 {
					if err := stream.Send(&pb.RangeStreamResponse{RangeResponse: chunk}); err != nil {
						return err
					}
					return test.err
				}
				chunk.Header = &pb.ResponseHeader{Revision: 43}
				return stream.Send(&pb.RangeStreamResponse{RangeResponse: chunk})
			}})
			var keys []string
			err := cfg.walk(t.Context(), cli, keyRange{Start: []byte("a"), End: []byte("z")}, false, nil, func(kvs []*mvccpb.KeyValue) error {
				for _, kv := range kvs {
					keys = append(keys, string(kv.Key))
				}
				return nil
			})
			require.NoError(t, err)
			require.Equal(t, []string{"a", "a\x00"}, keys)
			require.EqualValues(t, 2, calls.Load())
		})
	}
}

func TestStreamErrors(t *testing.T) {
	for _, test := range []struct {
		code    codes.Code
		message string
	}{
		{codes.Unimplemented, "etcd 3.7"},
		{codes.PermissionDenied, "denied"},
	} {
		t.Run(test.code.String(), func(t *testing.T) {
			var calls atomic.Int32
			cli := streamClient(t, testStreamServer{send: func(*pb.RangeRequest, pb.KV_RangeStreamServer) error {
				calls.Add(1)
				return status.Error(test.code, "denied")
			}})
			err := (&config{}).walk(t.Context(), cli, keyRange{Start: []byte("a"), End: []byte("z")}, true, nil, func([]*mvccpb.KeyValue) error { return nil })
			require.ErrorContains(t, err, test.message)
			require.EqualValues(t, 1, calls.Load())
		})
	}
}

func TestStreamChargesWholeChunk(t *testing.T) {
	cli := startTestEtcd(t)
	put(t, cli, map[string]string{"a": "v", "b": "v", "c": "v"})
	ctx, cancel := context.WithTimeout(t.Context(), 300*time.Millisecond)
	defer cancel()
	var seen int
	err := (&config{}).walk(ctx, cli, keyRange{Start: []byte("a"), End: []byte("z")}, false, rate.NewLimiter(1, 1), func(kvs []*mvccpb.KeyValue) error {
		seen += len(kvs)
		return nil
	})
	require.Error(t, err)
	require.Zero(t, seen)
}
