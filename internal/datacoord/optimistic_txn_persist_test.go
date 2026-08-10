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
	"bytes"
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type stringMarshaler struct{}

func (stringMarshaler) Marshal(value string) ([]byte, error) {
	return []byte(value), nil
}

func (stringMarshaler) Unmarshal(value []byte) (string, error) {
	return string(value), nil
}

type unmarshalErrorMarshaler struct {
	err error
}

func (unmarshalErrorMarshaler) Marshal(value string) ([]byte, error) {
	return []byte(value), nil
}

func (m unmarshalErrorMarshaler) Unmarshal([]byte) (string, error) {
	return "", m.err
}

type scanTestKVServer struct {
	etcdserverpb.UnimplementedKVServer

	kvs                  []*mvccpb.KeyValue
	streamChunkSize      int
	pageSize             int
	streamErr            error
	streamErrAfterChunks int
	streamCalls          atomic.Int32
	rangeCalls           atomic.Int32
}

func (s *scanTestKVServer) Range(_ context.Context, req *etcdserverpb.RangeRequest) (*etcdserverpb.RangeResponse, error) {
	s.rangeCalls.Add(1)
	kvs := s.filter(req)
	pageSize := s.pageSize
	if pageSize <= 0 || pageSize > len(kvs) {
		pageSize = len(kvs)
	}
	if req.Limit > 0 && int64(pageSize) > req.Limit {
		pageSize = int(req.Limit)
	}
	return &etcdserverpb.RangeResponse{
		Header: &etcdserverpb.ResponseHeader{Revision: 100},
		Kvs:    kvs[:pageSize],
		More:   pageSize < len(kvs),
		Count:  int64(len(kvs)),
	}, nil
}

func (s *scanTestKVServer) RangeStream(req *etcdserverpb.RangeRequest, stream etcdserverpb.KV_RangeStreamServer) error {
	s.streamCalls.Add(1)
	if s.streamErr != nil && s.streamErrAfterChunks == 0 {
		return s.streamErr
	}

	kvs := s.filter(req)
	chunkSize := s.streamChunkSize
	if chunkSize <= 0 || chunkSize > len(kvs) {
		chunkSize = len(kvs)
	}
	for start, chunks := 0, 0; start < len(kvs); chunks++ {
		end := min(start+chunkSize, len(kvs))
		resp := &etcdserverpb.RangeResponse{Kvs: kvs[start:end]}
		if end == len(kvs) {
			resp.Header = &etcdserverpb.ResponseHeader{Revision: 100}
			resp.Count = int64(len(kvs))
		}
		if err := stream.Send(&etcdserverpb.RangeStreamResponse{RangeResponse: resp}); err != nil {
			return err
		}
		if s.streamErr != nil && chunks+1 == s.streamErrAfterChunks {
			return s.streamErr
		}
		start = end
	}
	return nil
}

func (s *scanTestKVServer) filter(req *etcdserverpb.RangeRequest) []*mvccpb.KeyValue {
	result := make([]*mvccpb.KeyValue, 0, len(s.kvs))
	for _, kv := range s.kvs {
		if bytes.Compare(kv.Key, req.Key) < 0 {
			continue
		}
		if len(req.RangeEnd) > 0 && bytes.Compare(kv.Key, req.RangeEnd) >= 0 {
			continue
		}
		result = append(result, kv)
	}
	return result
}

func newScanTestClient(t *testing.T, kvServer etcdserverpb.KVServer) *clientv3.Client {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := grpc.NewServer()
	etcdserverpb.RegisterKVServer(server, kvServer)
	go func() {
		_ = server.Serve(listener)
	}()

	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"http://" + listener.Addr().String()},
		DialTimeout: 5 * time.Second,
	})
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = client.Close()
		server.Stop()
		_ = listener.Close()
	})
	return client
}

func scanTestKVs() []*mvccpb.KeyValue {
	return []*mvccpb.KeyValue{
		{Key: []byte("other/1"), Value: []byte("ignored"), ModRevision: 1},
		{Key: []byte("segments/1"), Value: []byte("value-1"), ModRevision: 11},
		{Key: []byte("segments/2"), Value: []byte("value-2"), ModRevision: 12},
		{Key: []byte("segments/3"), Value: []byte("value-3"), ModRevision: 13},
	}
}

func TestEtcdPersistScanUsesRangeStream(t *testing.T) {
	server := &scanTestKVServer{kvs: scanTestKVs(), streamChunkSize: 2}
	client := newScanTestClient(t, server)
	persist := NewOptimisticTxnEtcdPersist[string, string](client, stringMarshaler{})

	keys, values, versions, err := persist.Scan(context.Background(), "segments/")

	require.NoError(t, err)
	require.Equal(t, []string{"segments/1", "segments/2", "segments/3"}, keys)
	require.Equal(t, []string{"value-1", "value-2", "value-3"}, values)
	require.Equal(t, []int64{11, 12, 13}, versions)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Zero(t, server.rangeCalls.Load())
}

func TestEtcdPersistScanFallsBackWhenRangeStreamUnsupported(t *testing.T) {
	server := &scanTestKVServer{
		kvs:       scanTestKVs(),
		pageSize:  2,
		streamErr: status.Error(codes.Unimplemented, "RangeStream is unsupported"),
	}
	client := newScanTestClient(t, server)
	persist := NewOptimisticTxnEtcdPersist[string, string](client, stringMarshaler{})

	keys, values, versions, err := persist.Scan(context.Background(), "segments/")

	require.NoError(t, err)
	require.Equal(t, []string{"segments/1", "segments/2", "segments/3"}, keys)
	require.Equal(t, []string{"value-1", "value-2", "value-3"}, values)
	require.Equal(t, []int64{11, 12, 13}, versions)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Equal(t, int32(2), server.rangeCalls.Load())
}

func TestEtcdPersistScanDoesNotFallbackAfterStreamFailure(t *testing.T) {
	server := &scanTestKVServer{
		kvs:                  scanTestKVs(),
		streamChunkSize:      1,
		streamErr:            status.Error(codes.Unavailable, "stream interrupted"),
		streamErrAfterChunks: 1,
	}
	client := newScanTestClient(t, server)
	persist := NewOptimisticTxnEtcdPersist[string, string](client, stringMarshaler{})

	keys, values, versions, err := persist.Scan(context.Background(), "segments/")

	require.ErrorIs(t, err, merr.ErrIoFailed)
	require.Nil(t, keys)
	require.Nil(t, values)
	require.Nil(t, versions)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Zero(t, server.rangeCalls.Load())
}

func TestEtcdPersistScanDoesNotFallbackAfterUnmarshalFailure(t *testing.T) {
	server := &scanTestKVServer{kvs: scanTestKVs(), streamChunkSize: 1}
	client := newScanTestClient(t, server)
	decodeErr := status.Error(codes.Unimplemented, "decode failed")
	persist := NewOptimisticTxnEtcdPersist[string, string](client, unmarshalErrorMarshaler{err: decodeErr})

	keys, values, versions, err := persist.Scan(context.Background(), "segments/")

	require.ErrorIs(t, err, decodeErr)
	require.Nil(t, keys)
	require.Nil(t, values)
	require.Nil(t, versions)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Zero(t, server.rangeCalls.Load())
}
