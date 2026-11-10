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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/tikv/client-go/v2/oracle"
	"github.com/tikv/client-go/v2/txnkv"
	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	clientv3 "go.etcd.io/etcd/client/v3"
	"golang.org/x/sync/errgroup"
	"golang.org/x/time/rate"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	tikvutil "github.com/milvus-io/milvus/pkg/v3/util/tikv"
)

const (
	// sampleStride is how many keys the split pass skips between two kept
	// keys: 63M keys keep about 15K, a few hundred KB.
	sampleStride = 4096
	// maxBatchBytes bounds the values of one TiKV write transaction, far below
	// client-go's 100MB transaction limit.
	maxBatchBytes = 1 << 20
	// maxExamples bounds the differing keys a verify pass reports.
	maxExamples = 20
	// streamAttempts bounds the resumptions of one etcd stream in a row.
	streamAttempts = 10
)

var errSourceKeyFound = errors.New("source range contains a key")

// config is what a copy or verify run needs besides its clients.
type config struct {
	revision int64         // source revision, zero for latest per stream
	etcdRoot string        // etcd.rootPath of the cluster
	tikvRoot string        // tikv.rootPath the cluster will use
	ranges   int           // key ranges read concurrently
	writers  int           // concurrent TiKV write transactions
	batch    int           // keys per TiKV write transaction
	limiter  *rate.Limiter // shared across all streams, nil for no limit
}

// keyRange is the etcd keys in [Start, End).
type keyRange struct {
	Start []byte `json:"start"`
	End   []byte `json:"end"`
}

// sourceRanges returns the etcd keys that move to TiKV, in key order: the kv
// root, which holds the TSO and the ID allocators, and the meta root without
// the sessions, which stay in etcd.
func (c *config) sourceRanges() []keyRange {
	kv := path.Join(c.etcdRoot, "kv") + "/"
	meta := path.Join(c.etcdRoot, "meta") + "/"
	session := meta + "session/"
	return []keyRange{
		{Start: []byte(kv), End: []byte(clientv3.GetPrefixRangeEnd(kv))},
		{Start: []byte(meta), End: []byte(session)},
		{Start: []byte(clientv3.GetPrefixRangeEnd(session)), End: []byte(clientv3.GetPrefixRangeEnd(meta))},
	}
}

// targetKey maps an etcd key under etcdRoot to its TiKV key under tikvRoot.
func (c *config) targetKey(key []byte) []byte {
	if c.etcdRoot == c.tikvRoot {
		return key
	}
	target := make([]byte, 0, len(c.tikvRoot)+len(key)-len(c.etcdRoot))
	target = append(target, c.tikvRoot...)
	return append(target, key[len(c.etcdRoot):]...)
}

// family names the group of a key in reports: the two path elements after the
// root, such as meta/root-coord.
func (c *config) family(key []byte) []byte {
	rest := key[len(c.etcdRoot)+1:]
	_, tail, ok := bytes.Cut(rest, []byte("/"))
	if ok {
		if i := bytes.IndexByte(tail, '/'); i >= 0 {
			return rest[:len(rest)-len(tail)+i]
		}
	}
	return rest
}

// walk streams a range in key order. A retry resumes after the last delivered
// key at the configured revision. Zero lets each stream use its latest view.
func (c *config) walk(ctx context.Context, cli *clientv3.Client, r keyRange, keysOnly bool, limiter *rate.Limiter, fn func([]*mvccpb.KeyValue) error) error {
	start := r.Start
	failures := 0
	opts := []clientv3.OpOption{clientv3.WithRange(string(r.End))}
	if c.revision > 0 {
		opts = append(opts, clientv3.WithRev(c.revision))
	} else {
		opts = append(opts, clientv3.WithSerializable())
	}
	if keysOnly {
		opts = append(opts, clientv3.WithKeysOnly())
	}
	for {
		streamCtx, cancel := context.WithCancel(ctx)
		stream, streamErr := cli.GetStream(streamCtx, string(start), opts...)
		var fnErr error
		complete := false
		if streamErr == nil {
			for chunk := range stream {
				if streamErr = chunk.Err(); streamErr != nil {
					break
				}
				// etcd 3.7 sends a revision header on the final chunk, including for an empty range.
				complete = chunk.Header != nil && chunk.Header.Revision > 0
				if len(chunk.Kvs) == 0 {
					continue
				}
				if limiter != nil {
					for left := len(chunk.Kvs); left > 0; {
						n := min(left, limiter.Burst())
						if fnErr = limiter.WaitN(ctx, n); fnErr != nil {
							break
						}
						left -= n
					}
					if fnErr != nil {
						break
					}
				}
				if fnErr = fn(chunk.Kvs); fnErr != nil {
					break
				}
				start = append(bytes.Clone(chunk.Kvs[len(chunk.Kvs)-1].Key), 0)
				failures = 0
			}
		}
		cancel()
		if stream != nil {
			for range stream {
			}
		}
		if fnErr != nil {
			return fnErr
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if streamErr == nil {
			if complete {
				return nil
			}
			streamErr = io.ErrUnexpectedEOF
		}
		if c.revision > 0 && (errors.Is(streamErr, rpctypes.ErrCompacted) || errors.Is(streamErr, rpctypes.ErrFutureRev)) {
			return merr.Wrapf(streamErr, "source revision %d unavailable", c.revision)
		}
		if status.Code(streamErr) == codes.Unimplemented {
			return merr.Wrap(merr.WrapErrServiceUnimplemented(streamErr), "etcd RangeStream needs etcd 3.7")
		}
		switch status.Code(streamErr) {
		case codes.PermissionDenied, codes.Unauthenticated, codes.InvalidArgument:
			return merr.Wrapf(streamErr, "read etcd range starting at %q", start)
		}
		// etcd may return a normalized EtcdError without gRPC status.
		if errors.Is(streamErr, rpctypes.ErrPermissionDenied) || errors.Is(streamErr, rpctypes.ErrAuthFailed) {
			return merr.Wrapf(streamErr, "read etcd range starting at %q", start)
		}
		failures++
		if failures >= streamAttempts {
			return merr.Wrapf(streamErr, "read etcd range starting at %q", start)
		}
		mlog.Warn(ctx, "etcd stream failed, resuming after the last key read",
			mlog.String("resumeKey", string(start)), mlog.Int("failures", failures), mlog.Err(streamErr))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(failures) * time.Second):
		}
	}
}

// split divides the source ranges into about n ranges of similar key counts,
// from a keys-only stream over each of them. etcd answers a keys-only read
// from its in-memory index, without reading values from disk.
func split(ctx context.Context, cli *clientv3.Client, cfg *config, n int) ([]keyRange, error) {
	type sampled struct {
		r       keyRange
		count   int64
		samples [][]byte
	}
	sources := cfg.sourceRanges()
	results := make([]sampled, len(sources))
	group, groupCtx := errgroup.WithContext(ctx)
	for i, r := range sources {
		group.Go(func() error {
			result := sampled{r: r}
			err := cfg.walk(groupCtx, cli, r, true, cfg.limiter, func(kvs []*mvccpb.KeyValue) error {
				for _, kv := range kvs {
					if result.count%sampleStride == 0 && result.count > 0 {
						result.samples = append(result.samples, bytes.Clone(kv.Key))
					}
					result.count++
				}
				return nil
			})
			results[i] = result
			return err
		})
	}
	if err := group.Wait(); err != nil {
		return nil, err
	}
	var total int64
	for _, result := range results {
		total += result.count
	}
	var ranges []keyRange
	for _, result := range results {
		parts := 1
		if total > 0 {
			parts = max(1, int((int64(n)*result.count+total-1)/total))
		}
		parts = min(parts, len(result.samples)+1)
		start := result.r.Start
		for part := 1; part < parts; part++ {
			bound := result.samples[part*len(result.samples)/parts]
			if bytes.Compare(bound, start) <= 0 {
				continue
			}
			ranges = append(ranges, keyRange{Start: start, End: bound})
			start = bound
		}
		ranges = append(ranges, keyRange{Start: start, End: result.r.End})
	}
	mlog.Info(ctx, "split the source keys", mlog.Int64("keys", total), mlog.Int("ranges", len(ranges)))
	return ranges, nil
}

// loadOrSplit reads the ranges from file when it exists, and otherwise splits
// the source keys and saves the ranges to file when file is set. Ranges from
// an earlier split still cover every key, as only their balance depends on the
// keys.
func loadOrSplit(ctx context.Context, cli *clientv3.Client, cfg *config, file string) ([]keyRange, error) {
	if file != "" {
		if data, err := os.ReadFile(file); err == nil {
			var ranges []keyRange
			if err := json.Unmarshal(data, &ranges); err != nil {
				return nil, merr.Wrapf(err, "decode ranges file %q", file)
			}
			if err := checkCover(cfg, ranges); err != nil {
				return nil, err
			}
			return ranges, nil
		} else if !os.IsNotExist(err) {
			return nil, merr.WrapErrIoFailed(file, err)
		}
	}
	return splitAndSave(ctx, cli, cfg, file)
}

func splitAndSave(ctx context.Context, cli *clientv3.Client, cfg *config, file string) ([]keyRange, error) {
	ranges, err := split(ctx, cli, cfg, 4*cfg.ranges)
	if err != nil {
		return nil, err
	}
	if file != "" {
		data, err := json.Marshal(ranges)
		if err != nil {
			return nil, merr.WrapErrSerializationFailed(err, "ranges")
		}
		f, err := os.CreateTemp(filepath.Dir(file), ".etcd2tikv-ranges-*")
		if err != nil {
			return nil, merr.Wrap(err, "create ranges file")
		}
		defer os.Remove(f.Name())
		_, writeErr := f.Write(data)
		closeErr := f.Close()
		if err := errors.Join(writeErr, closeErr); err != nil {
			return nil, merr.Wrap(err, "write ranges file")
		}
		if err := os.Rename(f.Name(), file); err != nil {
			return nil, merr.Wrap(err, "publish ranges file")
		}
	}
	return ranges, nil
}

// checkCover reports whether ranges cover exactly the source ranges of cfg, so
// a ranges file of another root or an edited one is refused.
func checkCover(cfg *config, ranges []keyRange) error {
	sorted := append([]keyRange(nil), ranges...)
	sort.Slice(sorted, func(i, j int) bool { return bytes.Compare(sorted[i].Start, sorted[j].Start) < 0 })
	i := 0
	for _, source := range cfg.sourceRanges() {
		next := source.Start
		for i < len(sorted) && bytes.Compare(sorted[i].Start, source.End) < 0 {
			if !bytes.Equal(sorted[i].Start, next) || bytes.Compare(sorted[i].End, next) <= 0 || bytes.Compare(sorted[i].End, source.End) > 0 {
				return merr.WrapErrParameterInvalidMsg("the ranges file does not cover %q exactly at %q", source.Start, next)
			}
			next = sorted[i].End
			i++
		}
		if !bytes.Equal(next, source.End) {
			return merr.WrapErrParameterInvalidMsg("the ranges file does not cover %q up to %q", source.Start, source.End)
		}
	}
	if i != len(sorted) {
		return merr.WrapErrParameterInvalidMsg("the ranges file has ranges outside the source keys")
	}
	return nil
}

// targetRanges maps ranges to the TiKV keys they are copied to.
func targetRanges(cfg *config, ranges []keyRange) []keyRange {
	targets := make([]keyRange, 0, len(ranges))
	for _, r := range ranges {
		targets = append(targets, keyRange{Start: cfg.targetKey(r.Start), End: cfg.targetKey(r.End)})
	}
	return targets
}

// Check the selected source even if the split plan came from disk. In
// particular -overwrite must never clear TiKV on the strength of a cached plan.
func checkSource(ctx context.Context, cli *clientv3.Client, cfg *config, checkAllocators bool) error {
	ranges := cfg.sourceRanges()
	for _, root := range [][]keyRange{ranges[:1], ranges[1:]} {
		var err error
		for _, r := range root {
			err = cfg.walk(ctx, cli, r, true, nil, func([]*mvccpb.KeyValue) error { return errSourceKeyFound })
			if errors.Is(err, errSourceKeyFound) {
				break
			}
			if err != nil {
				return err
			}
		}
		if !errors.Is(err, errSourceKeyFound) {
			return merr.WrapErrParameterInvalidMsg("no source keys under %q (sessions excluded)", root[0].Start)
		}
	}
	if checkAllocators {
		for _, name := range []string{"gid/idTimestamp", "gid/timestamp"} {
			key := path.Join(cfg.etcdRoot, "kv", name)
			resp, err := cli.Get(ctx, key, clientv3.WithRev(cfg.revision))
			if err != nil {
				return merr.Wrapf(err, "read allocator %q", key)
			}
			if len(resp.Kvs) != 1 || len(resp.Kvs[0].Value) != 8 {
				return merr.WrapErrParameterInvalidMsg("source allocator %q must hold an 8-byte timestamp", key)
			}
		}
	}
	return nil
}

// Split only on interior sample keys. The small allocator and session-adjacent
// ranges need no regions of their own.
func splitRegions(ctx context.Context, store *txnkv.Client, cfg *config, ranges []keyRange) error {
	roots := cfg.sourceRanges()
	var keys [][]byte
	for _, r := range ranges {
		if !bytes.Equal(r.Start, roots[0].Start) && !bytes.Equal(r.Start, roots[1].Start) && !bytes.Equal(r.Start, roots[2].Start) {
			keys = append(keys, cfg.targetKey(r.Start))
		}
	}
	if len(keys) == 0 {
		return nil
	}
	ids, err := store.SplitRegions(ctx, keys, true, nil)
	if err != nil {
		return merr.Wrap(err, "split and scatter target TiKV regions")
	}
	for _, id := range ids {
		if err := store.WaitScatterRegionFinish(ctx, id, 0); err != nil {
			return merr.Wrapf(err, "wait for target TiKV region %d", id)
		}
	}
	return nil
}

// prepareTarget fails when TiKV already holds a key that the copy would write,
// unless overwrite is set, in which case it deletes those keys first.
func prepareTarget(ctx context.Context, store *txnkv.Client, cfg *config, overwrite bool) error {
	for _, r := range targetRanges(cfg, cfg.sourceRanges()) {
		if overwrite {
			if _, err := store.DeleteRange(ctx, r.Start, r.End, 16); err != nil {
				return merr.Wrapf(err, "clear TiKV range %q", r.Start)
			}
			continue
		}
		ts, err := store.CurrentTimestamp(oracle.GlobalTxnScope)
		if err != nil {
			return merr.Wrap(err, "get TiKV timestamp")
		}
		iter, err := store.GetSnapshot(ts).Iter(r.Start, r.End)
		if err != nil {
			return merr.Wrapf(err, "scan TiKV range %q", r.Start)
		}
		exists, key := iter.Valid(), string(iter.Key())
		iter.Close()
		if exists {
			return merr.WrapErrParameterInvalidMsg("TiKV already holds %q; pass -overwrite to replace the target keys", key)
		}
	}
	return nil
}

// copyResult counts committed batches only.
type copyResult struct {
	keys  atomic.Int64
	bytes atomic.Int64
}

// Each range has a single reader and writes its own sorted batches. The writer
// semaphore preserves the independent cap on in-flight TiKV transactions.
func copyKeys(ctx context.Context, cli *clientv3.Client, store *txnkv.Client, cfg *config, ranges []keyRange, result *copyResult) error {
	group, ctx := errgroup.WithContext(ctx)
	group.SetLimit(min(cfg.ranges, len(ranges)))
	writers := make(chan struct{}, min(cfg.writers, cfg.ranges, len(ranges)))
	for _, r := range ranges {
		group.Go(func() error {
			var batch []*mvccpb.KeyValue
			var batchBytes, valueBytes int
			flush := func() error {
				if len(batch) == 0 {
					return nil
				}
				select {
				case writers <- struct{}{}:
				case <-ctx.Done():
					return ctx.Err()
				}
				err := writeBatch(ctx, store, cfg, batch)
				<-writers
				if err != nil {
					return err
				}
				result.keys.Add(int64(len(batch)))
				result.bytes.Add(int64(valueBytes))
				clear(batch)
				batch, batchBytes, valueBytes = batch[:0], 0, 0
				return nil
			}
			err := cfg.walk(ctx, cli, r, false, cfg.limiter, func(kvs []*mvccpb.KeyValue) error {
				for _, kv := range kvs {
					size := len(kv.Key) + len(kv.Value)
					if batchBytes+size > maxBatchBytes {
						if err := flush(); err != nil {
							return err
						}
					}
					batch = append(batch, kv)
					batchBytes += size
					valueBytes += len(kv.Value)
					if len(batch) == cfg.batch {
						if err := flush(); err != nil {
							return err
						}
					}
				}
				return nil
			})
			if err != nil {
				return err
			}
			return flush()
		})
	}
	return group.Wait()
}

// Retrying an ambiguous commit is safe while the target is exclusive: every
// attempt writes exactly the same key/value pairs.
func writeBatch(ctx context.Context, store *txnkv.Client, cfg *config, kvs []*mvccpb.KeyValue) error {
	return retry.Do(ctx, func() error {
		txn, err := store.Begin()
		if err != nil {
			return merr.Wrap(err, "begin TiKV transaction")
		}
		txn.SetEnable1PC(true)
		for _, kv := range kvs {
			if err := tikvutil.SetValue(txn, cfg.targetKey(kv.Key), kv.Value); err != nil {
				_ = txn.Rollback()
				return retry.Unrecoverable(merr.Wrapf(err, "encode/write etcd key %q", kv.Key))
			}
		}
		return merr.Wrap(txn.Commit(ctx), "commit TiKV batch")
	}, retry.Attempts(10), retry.Sleep(200*time.Millisecond), retry.MaxSleepTime(5*time.Second))
}

// verifyResult is what a verify pass found.
type verifyResult struct {
	compared  atomic.Int64
	missing   atomic.Int64
	extra     atomic.Int64
	different atomic.Int64
	families  map[string]int64

	mu       sync.Mutex
	examples []string
}

func (r *verifyResult) problem(kind string, key []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.examples) < maxExamples {
		r.examples = append(r.examples, kind+" "+string(key))
	}
}

// failed reports whether TiKV differs from etcd.
func (r *verifyResult) failed() bool {
	return r.missing.Load()+r.extra.Load()+r.different.Load() > 0
}

// verifyKeys compares the keys of ranges in etcd with their copies in TiKV: a
// key only in etcd is missing, a key only in TiKV is extra, and a key whose
// values differ is different. Each range is read from TiKV at a timestamp of
// its own, taken when the range starts.
func verifyKeys(ctx context.Context, cli *clientv3.Client, store *txnkv.Client, cfg *config, ranges []keyRange, result *verifyResult) error {
	group, groupCtx := errgroup.WithContext(ctx)
	group.SetLimit(min(cfg.ranges, len(ranges)))
	for _, r := range ranges {
		group.Go(func() error {
			return verifyRange(groupCtx, cli, store, cfg, r, result)
		})
	}
	return group.Wait()
}

func verifyRange(ctx context.Context, cli *clientv3.Client, store *txnkv.Client, cfg *config, r keyRange, result *verifyResult) error {
	ts, err := store.CurrentTimestamp(oracle.GlobalTxnScope)
	if err != nil {
		return merr.Wrap(err, "get TiKV timestamp")
	}
	snapshot := store.GetSnapshot(ts)
	snapshot.SetScanBatchSize(1024)
	target := keyRange{Start: cfg.targetKey(r.Start), End: cfg.targetKey(r.End)}
	iter, err := snapshot.Iter(target.Start, target.End)
	if err != nil {
		return merr.Wrapf(err, "scan TiKV range %q", target.Start)
	}
	defer iter.Close()
	next := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := iter.Next(); err != nil {
			return merr.Wrapf(err, "advance TiKV scan in range %q", target.Start)
		}
		return nil
	}
	familyCounts := make(map[string]int64)
	err = cfg.walk(ctx, cli, r, false, cfg.limiter, func(kvs []*mvccpb.KeyValue) error {
		result.compared.Add(int64(len(kvs)))
		for i := 0; i < len(kvs); {
			family := cfg.family(kvs[i].Key)
			j := i + 1
			for j < len(kvs) && bytes.Equal(cfg.family(kvs[j].Key), family) {
				j++
			}
			familyCounts[string(family)] += int64(j - i)
			i = j
		}
		for _, kv := range kvs {
			want := cfg.targetKey(kv.Key)
			for iter.Valid() && bytes.Compare(iter.Key(), want) < 0 {
				result.extra.Add(1)
				result.problem("extra", iter.Key())
				if err := next(); err != nil {
					return err
				}
			}
			if !iter.Valid() || !bytes.Equal(iter.Key(), want) {
				result.missing.Add(1)
				result.problem("missing", want)
				continue
			}
			value := tikvutil.DecodeValue(iter.Value())
			if !bytes.Equal(value, kv.Value) {
				result.different.Add(1)
				result.problem("different", want)
			}
			if err := next(); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	for iter.Valid() {
		result.extra.Add(1)
		result.problem("extra", iter.Key())
		if err := next(); err != nil {
			return err
		}
	}
	result.mu.Lock()
	if result.families == nil {
		result.families = make(map[string]int64)
	}
	for k, v := range familyCounts {
		result.families[k] += v
	}
	result.mu.Unlock()
	return nil
}
