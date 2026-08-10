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
	fallback, err := walkPrefixRangeStream(ctx, client, prefix, fn)
	if err == nil {
		return nil
	}
	if !fallback {
		return err
	}

	mlog.RatedWarn(ctx, rate.Limit(1.0/60.0), "etcd RangeStream is unsupported, falling back to paginated range",
		mlog.String("prefix", prefix))
	return walkPrefixPaginated(ctx, client, prefix, paginationSize, requestTimeout, fn)
}

func walkPrefixRangeStream(
	ctx context.Context,
	client clientv3.KV,
	prefix string,
	fn func(*mvccpb.KeyValue) error,
) (bool, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := client.GetStream(streamCtx, prefix, clientv3.WithRange(clientv3.GetPrefixRangeEnd(prefix)))
	if err != nil {
		if status.Code(err) == codes.Unimplemented {
			return true, err
		}
		return false, merr.WrapErrIoFailed(prefix, err)
	}

	callbackInvoked := false
	for chunk := range stream {
		if err := chunk.Err(); err != nil {
			if !callbackInvoked && status.Code(err) == codes.Unimplemented {
				return true, err
			}
			return false, merr.WrapErrIoFailed(prefix, err)
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

func walkPrefixPaginated(
	ctx context.Context,
	client clientv3.KV,
	prefix string,
	paginationSize int,
	requestTimeout time.Duration,
	fn func(*mvccpb.KeyValue) error,
) error {
	key := prefix
	end := clientv3.GetPrefixRangeEnd(prefix)
	opts := []clientv3.OpOption{
		clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend),
		clientv3.WithLimit(int64(paginationSize)),
		clientv3.WithRange(end),
	}

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
				prefix,
			)
		}
		key = string(resp.Kvs[len(resp.Kvs)-1].Key) + "\x00"
	}
}
