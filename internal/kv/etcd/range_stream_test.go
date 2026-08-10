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
	"bytes"
	"context"
	"errors"
	"net"
	"sync"
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

type rangeStreamTestServer struct {
	etcdserverpb.UnimplementedKVServer

	kvs                      []*mvccpb.KeyValue
	streamChunkSize          int
	rangePageSize            int
	streamErr                error
	streamErrAfterChunks     int
	waitForCancelAfterChunks int
	streamCanceled           chan struct{}
	streamCanceledOnce       sync.Once
	streamCalls              atomic.Int32
	rangeCalls               atomic.Int32
	streamLimit              atomic.Int64
}

func (s *rangeStreamTestServer) Range(_ context.Context, req *etcdserverpb.RangeRequest) (*etcdserverpb.RangeResponse, error) {
	s.rangeCalls.Add(1)
	kvs := s.filter(req)
	pageSize := s.rangePageSize
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

func (s *rangeStreamTestServer) RangeStream(req *etcdserverpb.RangeRequest, stream etcdserverpb.KV_RangeStreamServer) error {
	s.streamCalls.Add(1)
	s.streamLimit.Store(req.Limit)
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
		completedChunks := chunks + 1
		if s.waitForCancelAfterChunks == completedChunks {
			<-stream.Context().Done()
			s.streamCanceledOnce.Do(func() { close(s.streamCanceled) })
			return stream.Context().Err()
		}
		if s.streamErr != nil && s.streamErrAfterChunks == completedChunks {
			return s.streamErr
		}
		start = end
	}
	return nil
}

func (s *rangeStreamTestServer) filter(req *etcdserverpb.RangeRequest) []*mvccpb.KeyValue {
	result := make([]*mvccpb.KeyValue, 0, len(s.kvs))
	for _, keyValue := range s.kvs {
		if bytes.Compare(keyValue.Key, req.Key) < 0 {
			continue
		}
		if len(req.RangeEnd) > 0 && bytes.Compare(keyValue.Key, req.RangeEnd) >= 0 {
			continue
		}
		result = append(result, keyValue)
	}
	return result
}

func newRangeStreamTestClient(t *testing.T, serverImpl etcdserverpb.KVServer) *clientv3.Client {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := grpc.NewServer()
	etcdserverpb.RegisterKVServer(server, serverImpl)
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

func rangeStreamTestKVs() []*mvccpb.KeyValue {
	return []*mvccpb.KeyValue{
		{Key: []byte("other/1"), Value: []byte("ignored")},
		{Key: []byte("prefix/1"), Value: []byte("value-1")},
		{Key: []byte("prefix/2"), Value: []byte("value-2")},
		{Key: []byte("prefix/3"), Value: []byte("value-3")},
	}
}

func TestWalkPrefixUsesRangeStream(t *testing.T) {
	server := &rangeStreamTestServer{kvs: rangeStreamTestKVs(), streamChunkSize: 2}
	client := newRangeStreamTestClient(t, server)

	var keys []string
	err := WalkPrefix(context.Background(), client, "prefix/", 1, time.Second, func(keyValue *mvccpb.KeyValue) error {
		keys = append(keys, string(keyValue.Key))
		return nil
	})

	require.NoError(t, err)
	require.Equal(t, []string{"prefix/1", "prefix/2", "prefix/3"}, keys)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Zero(t, server.rangeCalls.Load())
	require.Zero(t, server.streamLimit.Load(), "RangeStream must not receive a client limit")
}

func TestWalkPrefixFallsBackWhenRangeStreamIsUnsupported(t *testing.T) {
	server := &rangeStreamTestServer{
		kvs:           rangeStreamTestKVs(),
		rangePageSize: 2,
		streamErr:     status.Error(codes.Unimplemented, "RangeStream is unsupported"),
	}
	client := newRangeStreamTestClient(t, server)

	var keys []string
	err := WalkPrefix(context.Background(), client, "prefix/", 2, time.Second, func(keyValue *mvccpb.KeyValue) error {
		keys = append(keys, string(keyValue.Key))
		return nil
	})

	require.NoError(t, err)
	require.Equal(t, []string{"prefix/1", "prefix/2", "prefix/3"}, keys)
	require.Equal(t, int32(1), server.streamCalls.Load())
	require.Equal(t, int32(2), server.rangeCalls.Load())
}

func TestWalkPrefixDoesNotFallbackAfterStreamProgress(t *testing.T) {
	server := &rangeStreamTestServer{
		kvs:                  rangeStreamTestKVs(),
		streamChunkSize:      1,
		streamErr:            status.Error(codes.Unavailable, "stream interrupted"),
		streamErrAfterChunks: 1,
	}
	client := newRangeStreamTestClient(t, server)

	callbacks := 0
	err := WalkPrefix(context.Background(), client, "prefix/", 2, time.Second, func(keyValue *mvccpb.KeyValue) error {
		callbacks++
		return nil
	})

	require.ErrorIs(t, err, merr.ErrIoFailed)
	require.Equal(t, 1, callbacks)
	require.Zero(t, server.rangeCalls.Load())
}

func TestWalkPrefixCancelsStreamOnCallbackFailure(t *testing.T) {
	canceled := make(chan struct{})
	server := &rangeStreamTestServer{
		kvs:                      rangeStreamTestKVs(),
		streamChunkSize:          1,
		waitForCancelAfterChunks: 1,
		streamCanceled:           canceled,
	}
	client := newRangeStreamTestClient(t, server)
	callbackErr := errors.New("callback failed")

	err := WalkPrefix(context.Background(), client, "prefix/", 2, time.Second, func(keyValue *mvccpb.KeyValue) error {
		return callbackErr
	})

	require.ErrorIs(t, err, callbackErr)
	select {
	case <-canceled:
	case <-time.After(5 * time.Second):
		require.FailNow(t, "RangeStream server did not observe callback cancellation")
	}
	require.Zero(t, server.rangeCalls.Load())
}
