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
	"strings"
	"sync"
	"time"

	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"golang.org/x/time/rate"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// WalkPrefix reads an etcd prefix with RangeStream. The server chooses stream
// chunk sizes, so paginationSize only applies to the compatibility fallback.
func WalkPrefix(
	ctx context.Context,
	client clientv3.KV,
	prefix string,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(*mvccpb.KeyValue) error,
) error {
	return walkRange(ctx, client, prefix, clientv3.GetPrefixRangeEnd(prefix), 0, paginationSize, requestTimeout, fn)
}

// WalkPrefixInRanges reads an etcd prefix like WalkPrefix, as the contiguous
// key ranges bounds split it into: [prefix, bounds[0]), [bounds[0], bounds[1]),
// ..., [bounds[len(bounds)-1], end of prefix). bounds must lie under prefix and
// increase strictly. Every range is read by its own stream, concurrently, and
// all of them at the store revision of one read made before them, so together
// they return every key once, as one walk at that revision would.
//
// fn receives the index of the range of each key. It is called for the keys of
// one range in key order from one goroutine, and for different ranges
// concurrently. A failed range cancels the ranges after it while the ranges
// before it run on, so the error returned is the one a single walk would meet
// first: that of the lowest failed range. A revision compacted during the walk
// fails it like a compaction in the middle of one stream does.
//
// Without bounds the prefix is read by WalkPrefix as range 0.
func WalkPrefixInRanges(
	ctx context.Context,
	client clientv3.KV,
	prefix string,
	bounds []string,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(rangeIndex int, keyValue *mvccpb.KeyValue) error,
) error {
	if len(bounds) == 0 {
		return WalkPrefix(ctx, client, prefix, paginationSize, requestTimeout, func(keyValue *mvccpb.KeyValue) error {
			return fn(0, keyValue)
		})
	}
	starts := make([]string, 0, len(bounds)+1)
	starts = append(starts, prefix)
	for _, bound := range bounds {
		if !strings.HasPrefix(bound, prefix) || bound <= starts[len(starts)-1] {
			return merr.WrapErrServiceInternalMsg("range bound %q of prefix %q is not under the prefix or not increasing", bound, prefix)
		}
		starts = append(starts, bound)
	}
	revision, err := currentRevision(ctx, client, prefix, requestTimeout)
	if err != nil {
		return err
	}
	return walkRangesAt(ctx, client, starts, clientv3.GetPrefixRangeEnd(prefix), revision, paginationSize, requestTimeout, fn)
}

// currentRevision reads the store revision with a linearizable point read of
// key, so the ranges pinned to it see every write committed before the walk.
func currentRevision(ctx context.Context, client clientv3.KV, key string, requestTimeout time.Duration) (int64, error) {
	requestCtx, cancel := ctx, context.CancelFunc(func() {})
	if requestTimeout > 0 {
		requestCtx, cancel = context.WithTimeout(ctx, requestTimeout)
	}
	defer cancel()
	resp, err := client.Get(requestCtx, key, clientv3.WithCountOnly())
	if err != nil {
		return 0, merr.WrapErrIoFailed(key, err)
	}
	return resp.Header.Revision, nil
}

// walkRangesAt reads the ranges starting at starts, the last one ending at
// end, concurrently at revision (see WalkPrefixInRanges). The paginated
// fallback splits paginationSize among the ranges, so the pages in flight
// together hold as many keys as one page of one walk.
func walkRangesAt(
	ctx context.Context,
	client clientv3.KV,
	starts []string,
	end string,
	revision int64,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(rangeIndex int, keyValue *mvccpb.KeyValue) error,
) error {
	if paginationSize > 0 {
		paginationSize = (paginationSize + len(starts) - 1) / len(starts)
	}
	rangeCtxs := make([]context.Context, len(starts))
	cancels := make([]context.CancelFunc, len(starts))
	for i := range starts {
		rangeCtxs[i], cancels[i] = context.WithCancel(ctx)
	}
	errs := make([]error, len(starts))
	var wg sync.WaitGroup
	for i, start := range starts {
		rangeEnd := end
		if i+1 < len(starts) {
			rangeEnd = starts[i+1]
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer cancels[i]()
			errs[i] = walkRange(rangeCtxs[i], client, start, rangeEnd, revision, paginationSize, requestTimeout,
				func(keyValue *mvccpb.KeyValue) error {
					return fn(i, keyValue)
				})
			if errs[i] != nil {
				// No key after this range can hold an error that comes first.
				for _, cancel := range cancels[i+1:] {
					cancel()
				}
			}
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// walkRange reads the keys in [start, end) with RangeStream, at revision when
// it is positive, falling back to paginated reads when the server does not
// support RangeStream.
func walkRange(
	ctx context.Context,
	client clientv3.KV,
	start string,
	end string,
	revision int64,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(*mvccpb.KeyValue) error,
) error {
	fallback, err := walkRangeStream(ctx, client, start, end, revision, fn)
	if err == nil {
		return nil
	}
	if !fallback {
		return err
	}

	mlog.RatedWarn(ctx, rate.Limit(1.0/60.0), "etcd RangeStream is unsupported, falling back to paginated range",
		mlog.String("prefix", start))
	return walkRangePaginated(ctx, client, start, end, revision, paginationSize, requestTimeout, fn)
}

// rangeOptions returns the options of a read of [key, end) at revision (the
// latest one when it is not positive).
func rangeOptions(end string, revision int64, opts ...clientv3.OpOption) []clientv3.OpOption {
	opts = append(opts, clientv3.WithRange(end))
	if revision > 0 {
		opts = append(opts, clientv3.WithRev(revision))
	}
	return opts
}

func walkRangeStream(
	ctx context.Context,
	client clientv3.KV,
	start string,
	end string,
	revision int64,
	fn func(*mvccpb.KeyValue) error,
) (bool, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := client.GetStream(streamCtx, start, rangeOptions(end, revision)...)
	if err != nil {
		if status.Code(err) == codes.Unimplemented {
			return true, err
		}
		return false, merr.WrapErrIoFailed(start, err)
	}

	callbackInvoked := false
	for chunk := range stream {
		if err := chunk.Err(); err != nil {
			if !callbackInvoked && status.Code(err) == codes.Unimplemented {
				return true, err
			}
			return false, merr.WrapErrIoFailed(start, err)
		}
		for _, keyValue := range chunk.Kvs {
			callbackInvoked = true
			if err := fn(keyValue); err != nil {
				cancel()
				drainRangeStream(stream)
				return false, err
			}
		}
	}
	return false, nil
}

func drainRangeStream(stream clientv3.GetStreamChan) {
	for range stream {
	}
}

func walkRangePaginated(
	ctx context.Context,
	client clientv3.KV,
	start string,
	end string,
	revision int64,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(*mvccpb.KeyValue) error,
) error {
	key := start
	opts := rangeOptions(end, revision,
		clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend),
		clientv3.WithLimit(int64(paginationSize)),
	)

	for {
		requestCtx := ctx
		cancel := func() {}
		if requestTimeout > 0 {
			requestCtx, cancel = context.WithTimeout(ctx, requestTimeout)
		}
		resp, err := client.Get(requestCtx, key, opts...)
		cancel()
		if err != nil {
			return merr.WrapErrIoFailed(key, err)
		}

		for _, keyValue := range resp.Kvs {
			if err := fn(keyValue); err != nil {
				return err
			}
		}
		if !resp.More {
			return nil
		}
		if len(resp.Kvs) == 0 {
			return merr.WrapErrServiceInternalMsg(
				"etcd paginated range returned more=true without key values for prefix %q",
				start,
			)
		}
		key = string(resp.Kvs[len(resp.Kvs)-1].Key) + "\x00"
	}
}
