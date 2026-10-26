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
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

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

	// pinnedRangeErr fails every Range read at a pinned revision.
	pinnedRangeErr error

	// blockStreamAt blocks the stream of the range starting at it until the
	// client cancels it, then closes blockedStreamCanceled.
	blockStreamAt         string
	blockedStreamCanceled chan struct{}

	mu             sync.Mutex
	streamRequests []*etcdserverpb.RangeRequest
	rangeRequests  []*etcdserverpb.RangeRequest
}

func (s *rangeStreamTestServer) record(requests *[]*etcdserverpb.RangeRequest, req *etcdserverpb.RangeRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	*requests = append(*requests, proto.Clone(req).(*etcdserverpb.RangeRequest))
}

func (s *rangeStreamTestServer) recorded(requests *[]*etcdserverpb.RangeRequest) []*etcdserverpb.RangeRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(*requests)
}

func (s *rangeStreamTestServer) Range(_ context.Context, req *etcdserverpb.RangeRequest) (*etcdserverpb.RangeResponse, error) {
	s.rangeCalls.Add(1)
	s.record(&s.rangeRequests, req)
	if s.pinnedRangeErr != nil && req.Revision > 0 {
		return nil, s.pinnedRangeErr
	}
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
	s.record(&s.streamRequests, req)
	if s.streamErr != nil && s.streamErrAfterChunks == 0 {
		return s.streamErr
	}
	if s.blockStreamAt != "" && string(req.Key) == s.blockStreamAt {
		<-stream.Context().Done()
		close(s.blockedStreamCanceled)
		return stream.Context().Err()
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
		if len(req.RangeEnd) == 0 && !bytes.Equal(keyValue.Key, req.Key) {
			continue
		}
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

func rangeTestKeys(keyValues []*mvccpb.KeyValue) []string {
	keys := make([]string, len(keyValues))
	for i, keyValue := range keyValues {
		keys[i] = string(keyValue.Key)
	}
	return keys
}

// collectRanges walks prefix in the ranges of bounds and returns the keys of
// every range.
func collectRanges(t *testing.T, client clientv3.KV, prefix string, bounds []string, paginationSize int) ([][]string, error) {
	t.Helper()
	ranges := make([][]string, len(bounds)+1)
	err := WalkPrefixInRanges(context.Background(), client, prefix, bounds, paginationSize, time.Second,
		func(rangeIndex int, keyValue *mvccpb.KeyValue) error {
			ranges[rangeIndex] = append(ranges[rangeIndex], string(keyValue.Key))
			return nil
		})
	return ranges, err
}

func TestWalkPrefixInRangesReadsEveryRangeAtOneRevision(t *testing.T) {
	server := &rangeStreamTestServer{kvs: rangeStreamTestKVs(), streamChunkSize: 1}
	client := newRangeStreamTestClient(t, server)

	ranges, err := collectRanges(t, client, "prefix/", []string{"prefix/2", "prefix/3"}, 1)
	require.NoError(t, err)
	require.Equal(t, [][]string{{"prefix/1"}, {"prefix/2"}, {"prefix/3"}}, ranges)

	// One point read takes the revision every range is read at.
	rangeRequests := server.recorded(&server.rangeRequests)
	require.Len(t, rangeRequests, 1)
	require.Equal(t, "prefix/", string(rangeRequests[0].Key))
	require.Empty(t, rangeRequests[0].RangeEnd)
	streams := server.recorded(&server.streamRequests)
	require.Len(t, streams, 3)
	slices.SortFunc(streams, func(a, b *etcdserverpb.RangeRequest) int { return bytes.Compare(a.Key, b.Key) })
	for i, want := range [][2]string{{"prefix/", "prefix/2"}, {"prefix/2", "prefix/3"}, {"prefix/3", "prefix0"}} {
		require.Equal(t, want[0], string(streams[i].Key))
		require.Equal(t, want[1], string(streams[i].RangeEnd))
		require.EqualValues(t, 100, streams[i].Revision, "every range is pinned to the revision of the point read")
		require.Zero(t, streams[i].Limit)
	}
}

func TestWalkPrefixInRangesWithoutBoundsIsOneWalk(t *testing.T) {
	server := &rangeStreamTestServer{kvs: rangeStreamTestKVs(), streamChunkSize: 2}
	client := newRangeStreamTestClient(t, server)

	ranges, err := collectRanges(t, client, "prefix/", nil, 1)
	require.NoError(t, err)
	require.Equal(t, [][]string{{"prefix/1", "prefix/2", "prefix/3"}}, ranges)
	require.Zero(t, server.rangeCalls.Load(), "one walk needs no revision read")
	streams := server.recorded(&server.streamRequests)
	require.Len(t, streams, 1)
	require.Zero(t, streams[0].Revision)
}

func TestWalkPrefixInRangesFallsBackToPaginationAtTheRevision(t *testing.T) {
	server := &rangeStreamTestServer{
		kvs:           rangeStreamTestKVs(),
		rangePageSize: 1,
		streamErr:     status.Error(codes.Unimplemented, "RangeStream is unsupported"),
	}
	client := newRangeStreamTestClient(t, server)

	ranges, err := collectRanges(t, client, "prefix/", []string{"prefix/2"}, 3)
	require.NoError(t, err)
	require.Equal(t, [][]string{{"prefix/1"}, {"prefix/2", "prefix/3"}}, ranges)

	pages := server.recorded(&server.rangeRequests)[1:]
	require.NotEmpty(t, pages)
	for _, page := range pages {
		require.EqualValues(t, 100, page.Revision, "every page is pinned to the revision of the point read")
		require.True(t, string(page.Key) < string(page.RangeEnd))
		require.Contains(t, []string{"prefix/2", "prefix0"}, string(page.RangeEnd), "a page stays inside its range")
		require.EqualValues(t, 2, page.Limit, "the ranges share the page size of one walk")
	}
}

func TestWalkPrefixInRangesReturnsTheFirstErrorInKeyOrder(t *testing.T) {
	canceled := make(chan struct{})
	server := &rangeStreamTestServer{
		kvs:                   rangeStreamTestKVs(),
		blockStreamAt:         "prefix/3",
		blockedStreamCanceled: canceled,
	}
	client := newRangeStreamTestClient(t, server)
	firstErr := errors.New("first range failed")
	laterErr := errors.New("second range failed")
	laterFailed := make(chan struct{})

	err := WalkPrefixInRanges(context.Background(), client, "prefix/", []string{"prefix/2", "prefix/3"}, 1, time.Second,
		func(rangeIndex int, keyValue *mvccpb.KeyValue) error {
			switch rangeIndex {
			case 0:
				// Fail only after a later range did: the first range's error
				// still wins.
				<-laterFailed
				return firstErr
			case 1:
				close(laterFailed)
				return laterErr
			}
			return nil
		})
	require.ErrorIs(t, err, firstErr)
	select {
	case <-canceled:
	case <-time.After(5 * time.Second):
		require.FailNow(t, "a failed range did not cancel the ranges after it")
	}
}

func TestWalkPrefixInRangesRejectsUnorderedBounds(t *testing.T) {
	server := &rangeStreamTestServer{kvs: rangeStreamTestKVs()}
	client := newRangeStreamTestClient(t, server)

	for _, bounds := range [][]string{
		{"prefix/3", "prefix/2"},
		{"prefix/2", "prefix/2"},
		{"prefix/"},
		{"other/1"},
	} {
		_, err := collectRanges(t, client, "prefix/", bounds, 1)
		require.ErrorIs(t, err, merr.ErrServiceInternal, bounds)
	}
	require.Zero(t, server.streamCalls.Load())
	require.Zero(t, server.rangeCalls.Load())
}

func TestWalkPrefixInRangesFailsLikeOneWalkOnReadErrors(t *testing.T) {
	server := &rangeStreamTestServer{
		kvs:                  rangeStreamTestKVs(),
		streamChunkSize:      1,
		streamErr:            status.Error(codes.Unavailable, "stream interrupted"),
		streamErrAfterChunks: 1,
	}
	client := newRangeStreamTestClient(t, server)

	_, err := collectRanges(t, client, "prefix/", []string{"prefix/2"}, 1)
	require.ErrorIs(t, err, merr.ErrIoFailed)
	require.Equal(t, merr.SystemError, merr.GetErrorType(err), "the recovery retry must stay armed")
}

func TestWalkPrefixInRangesFailsOnACompactedRevisionWhilePaginating(t *testing.T) {
	server := &rangeStreamTestServer{
		kvs:            rangeStreamTestKVs(),
		streamErr:      status.Error(codes.Unimplemented, "RangeStream is unsupported"),
		pinnedRangeErr: rpctypes.ErrGRPCCompacted,
	}
	client := newRangeStreamTestClient(t, server)

	_, err := collectRanges(t, client, "prefix/", []string{"prefix/2"}, 1)
	require.ErrorIs(t, err, merr.ErrIoFailed)
	require.Contains(t, err.Error(), "compacted")
	require.Equal(t, merr.SystemError, merr.GetErrorType(err), "the recovery retry must stay armed")
}
