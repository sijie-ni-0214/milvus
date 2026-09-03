package datacoord

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	tikverr "github.com/tikv/client-go/v2/error"
	tikvkv "github.com/tikv/client-go/v2/kv"
	"github.com/tikv/client-go/v2/txnkv"
	"github.com/tikv/client-go/v2/txnkv/transaction"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
)

var (
	ErrKeyAlreadyExists = fmt.Errorf("key already exists")
	ErrKeyNotFound      = fmt.Errorf("key not found")
	// ErrVersionConflict reports that a versioned write (PutIfVersion /
	// DeleteIfVersion) found a different version in the store: the caller's
	// cached copy is stale and must be refreshed before the write is rebuilt.
	ErrVersionConflict = errors.New("optimistic txn: version conflict")
)

// OptimisticTxnPersist is a persist layer that uses optimistic transactions.
type OptimisticTxnPersist[K comparable, V any] interface {
	// Txn creates a new transaction. Add operations via Insert/Update/Upsert/Delete,
	// then call Commit to execute them atomically.
	Txn(ctx context.Context) Txn[K, V]
	// Scan reads all key-value pairs with the given prefix.
	Scan(ctx context.Context, prefix K) (keys []K, values []V, versions []int64, err error)
	// Get reads the current value and version of each key. Absent keys are
	// omitted from the result.
	Get(ctx context.Context, keys []K) (map[K]TxnResult[V], error)
}

// Txn collects operations and commits them atomically.
type Txn[K comparable, V any] interface {
	// Insert adds a write for a key that must not exist.
	Insert(key K, value V)
	// Update adds a read-modify-write for a key that must exist.
	Update(key K, f UpdateFunc[V])
	// Upsert adds an insert-or-update. If key doesn't exist, inserts value. If exists, applies f.
	Upsert(key K, value V, f UpdateFunc[V])
	// Delete adds a delete for a key that must exist.
	Delete(key K)
	// PutIfVersion writes value when the key's current version equals version;
	// version 0 means the key must not exist. The key is not read: the caller
	// supplies the version of the value it built the write from. A commit
	// marked op lands in the final batch when the transaction has to be split.
	PutIfVersion(key K, value V, version int64, commit bool)
	// DeleteIfVersion deletes the key when its current version equals version.
	DeleteIfVersion(key K, version int64)
	// Raw composes an unversioned write of another record into the same
	// transaction (e.g. a catalog-encoded DataView or channel tombstone).
	Raw(op RawOp)
	// AtomicOnly makes Commit fail instead of splitting an over-limit
	// transaction into ordered batches.
	AtomicOnly()
	// Commit executes all operations atomically. Returns results in the order ops were added.
	// On CAS failure of a read-modify-write op, retries automatically (re-reads and
	// re-applies UpdateFuncs). A stale versioned op fails with ErrVersionConflict.
	Commit() ([]TxnResult[V], error)
}

// RawOp is an unversioned put or delete composed into a transaction. Key is
// the full store key.
type RawOp struct {
	Key    string
	Value  string
	Delete bool
	Commit bool
}

// TxnResult is the result of a single operation after commit.
type TxnResult[V any] struct {
	Value   V
	Version int64
}

// UpdateFunc transforms an existing value. Returns (newValue, shouldWrite).
// If shouldWrite is false, the write is skipped and the existing value/version are returned.
type UpdateFunc[T any] func(existing T) (T, bool)

type Marshaler[T any] interface {
	Marshal(v T) ([]byte, error)
	Unmarshal(v []byte) (T, error)
}

// valueEqualer is implemented by marshalers whose values can be compared. A
// versioned write whose result was lost (e.g. a timeout after the store applied
// it) is recognized on retry by finding its own value in the store.
type valueEqualer[T any] interface {
	Equal(a, b T) bool
}

func storedValueEquals[V any](marshaler Marshaler[V], stored []byte, value V) bool {
	eq, ok := marshaler.(valueEqualer[V])
	if !ok {
		return false
	}
	current, err := marshaler.Unmarshal(stored)
	if err != nil {
		return false
	}
	return eq.Equal(current, value)
}

// --- Segment key helpers ---

const segmentMetaPrefix = "datacoord-meta/s/"

func segmentKey(collectionID, partitionID, segmentID int64) string {
	return fmt.Sprintf("%s%d/%d/%d", segmentMetaPrefix, collectionID, partitionID, segmentID)
}

func segmentIDFromKey(key string) (int64, error) {
	parts := strings.Split(key, "/")
	if len(parts) == 0 {
		return 0, fmt.Errorf("invalid segment key: %s", key)
	}
	return strconv.ParseInt(parts[len(parts)-1], 10, 64)
}

// --- SegmentInfo Marshaler (protobuf) ---

type SegmentInfoMarshaler struct{}

func (m *SegmentInfoMarshaler) Marshal(v *datapb.SegmentInfo) ([]byte, error) {
	return proto.Marshal(v)
}

func (m *SegmentInfoMarshaler) Unmarshal(data []byte) (*datapb.SegmentInfo, error) {
	v := &datapb.SegmentInfo{}
	if err := proto.Unmarshal(data, v); err != nil {
		return nil, err
	}
	return v, nil
}

func (m *SegmentInfoMarshaler) Equal(a, b *datapb.SegmentInfo) bool {
	return proto.Equal(a, b)
}

// ============================================================
// op types (internal)
// ============================================================

type opKind int

const (
	opInsert opKind = iota
	opUpdate
	opUpsert
	opDelete
	opPutIfVersion
	opDeleteIfVersion
	opRaw
)

type txnOp[K comparable, V any] struct {
	kind       opKind
	key        K
	value      V             // used by Insert, Upsert (insert case), PutIfVersion
	updateFunc UpdateFunc[V] // used by Update, Upsert (update case)
	version    int64         // used by PutIfVersion, DeleteIfVersion
	raw        RawOp         // used by Raw
	commit     bool
}

func (op txnOp[K, V]) versioned() bool {
	return op.kind == opPutIfVersion || op.kind == opDeleteIfVersion
}

// txnOps is the op recorder shared by every backend.
type txnOps[K comparable, V any] struct {
	ops        []txnOp[K, V]
	atomicOnly bool
}

func (t *txnOps[K, V]) Insert(key K, value V) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opInsert, key: key, value: value})
}

func (t *txnOps[K, V]) Update(key K, f UpdateFunc[V]) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opUpdate, key: key, updateFunc: f})
}

func (t *txnOps[K, V]) Upsert(key K, value V, f UpdateFunc[V]) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opUpsert, key: key, value: value, updateFunc: f})
}

func (t *txnOps[K, V]) Delete(key K) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opDelete, key: key})
}

func (t *txnOps[K, V]) PutIfVersion(key K, value V, version int64, commit bool) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opPutIfVersion, key: key, value: value, version: version, commit: commit})
}

func (t *txnOps[K, V]) DeleteIfVersion(key K, version int64) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opDeleteIfVersion, key: key, version: version})
}

func (t *txnOps[K, V]) Raw(op RawOp) {
	t.ops = append(t.ops, txnOp[K, V]{kind: opRaw, raw: op, commit: op.Commit})
}

func (t *txnOps[K, V]) AtomicOnly() {
	t.atomicOnly = true
}

// planBatches splits op indexes into commit batches of at most limit ops,
// mirroring the metastore composite-write contract (kv/txn.Commit): a set that
// fits is one atomic batch in recorded order; otherwise the non-commit ops are
// flushed first in recorded order and every commit-marked op lands together in
// the final batch, which is the visibility marker of the whole write.
func (t *txnOps[K, V]) planBatches(limit int) ([][]int, error) {
	if len(t.ops) == 0 {
		return nil, nil
	}
	all := make([]int, len(t.ops))
	for i := range t.ops {
		all[i] = i
	}
	if limit <= 0 || len(t.ops) <= limit {
		return [][]int{all}, nil
	}
	if t.atomicOnly {
		return nil, fmt.Errorf("atomic transaction needs %d ops but the limit is %d", len(t.ops), limit)
	}
	var nonCommit, commit []int
	for i, op := range t.ops {
		if op.commit {
			commit = append(commit, i)
		} else {
			nonCommit = append(nonCommit, i)
		}
	}
	if len(commit) > limit {
		return nil, fmt.Errorf("transaction commit set of %d ops exceeds the limit %d", len(commit), limit)
	}
	batches := make([][]int, 0, len(nonCommit)/limit+2)
	for start := 0; start < len(nonCommit); start += limit {
		end := start + limit
		if end > len(nonCommit) {
			end = len(nonCommit)
		}
		batches = append(batches, nonCommit[start:end])
	}
	if len(commit) > 0 {
		batches = append(batches, commit)
	}
	mlog.Warn(context.TODO(), "optimistic txn exceeds atomic limit, falling back to ordered batches",
		zap.Int("total", len(t.ops)), zap.Int("limit", limit), zap.Int("batches", len(batches)))
	return batches, nil
}

// ============================================================
// Etcd implementation
// ============================================================

type etcdPersist[K string, V any] struct {
	cli       *clientv3.Client
	marshaler Marshaler[V]
}

func NewOptimisticTxnEtcdPersist[K string, V any](cli *clientv3.Client, marshaler Marshaler[V]) OptimisticTxnPersist[K, V] {
	return &etcdPersist[K, V]{cli: cli, marshaler: marshaler}
}

func (p *etcdPersist[K, V]) Txn(ctx context.Context) Txn[K, V] {
	return &etcdTxn[K, V]{ctx: ctx, persist: p}
}

func (p *etcdPersist[K, V]) Scan(ctx context.Context, prefix K) ([]K, []V, []int64, error) {
	var ks []K
	var vals []V
	var vers []int64
	// Linearizable: recovery must see every segment the previous leader
	// committed, which a lagging follower may not have applied yet.
	err := etcdkv.WalkPrefix(
		ctx,
		p.cli,
		string(prefix),
		paramtable.Get().MetaStoreCfg.PaginationSize.GetAsInt(),
		0,
		func(keyValue *mvccpb.KeyValue) error {
			value, err := p.marshaler.Unmarshal(keyValue.Value)
			if err != nil {
				return err
			}
			ks = append(ks, K(keyValue.Key))
			vals = append(vals, value)
			vers = append(vers, keyValue.ModRevision)
			return nil
		},
	)
	if err != nil {
		return nil, nil, nil, err
	}
	return ks, vals, vers, nil
}

func (p *etcdPersist[K, V]) Get(ctx context.Context, keys []K) (map[K]TxnResult[V], error) {
	result := make(map[K]TxnResult[V], len(keys))
	for _, key := range keys {
		resp, err := p.cli.Get(ctx, string(key))
		if err != nil {
			return nil, err
		}
		if len(resp.Kvs) == 0 {
			continue
		}
		v, err := p.marshaler.Unmarshal(resp.Kvs[0].Value)
		if err != nil {
			return nil, err
		}
		result[key] = TxnResult[V]{Value: v, Version: resp.Kvs[0].ModRevision}
	}
	return result, nil
}

type etcdTxn[K string, V any] struct {
	txnOps[K, V]
	ctx     context.Context
	persist *etcdPersist[K, V]
}

func (t *etcdTxn[K, V]) Commit() ([]TxnResult[V], error) {
	results := make([]TxnResult[V], len(t.ops))

	// Split ops into batches to avoid exceeding etcd txn op limits.
	batches, err := t.planBatches(paramtable.Get().MetaStoreCfg.MaxEtcdTxnNum.GetAsInt())
	if err != nil {
		return nil, err
	}
	for _, batch := range batches {
		if err := t.commitBatch(batch, results); err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (t *etcdTxn[K, V]) commitBatch(batch []int, results []TxnResult[V]) error {
	commitStart := time.Now()
	retryCount := 0

	return retry.Do(t.ctx, func() error {
		retryCount++
		cmps := make([]clientv3.Cmp, 0, len(batch))
		putOps := make([]clientv3.Op, 0, len(batch))
		written := make(map[int]bool, len(batch))
		hasVersioned := false

		// Phase 1: read existing values of read-modify-write ops from etcd
		getStart := time.Now()
		for _, i := range batch {
			op := t.ops[i]
			keyStr := string(op.key)
			switch op.kind {
			case opInsert:
				resp, err := t.persist.cli.Get(t.ctx, keyStr, clientv3.WithSerializable(), clientv3.WithKeysOnly())
				if err != nil {
					return err
				}
				exists := len(resp.Kvs) > 0
				if exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyAlreadyExists, keyStr))
				}
				cmps = append(cmps, clientv3.Compare(clientv3.CreateRevision(keyStr), "=", 0))
				valBytes, err := t.persist.marshaler.Marshal(op.value)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				putOps = append(putOps, clientv3.OpPut(keyStr, string(valBytes)))
				results[i].Value = op.value
				written[i] = true

			case opUpdate:
				resp, err := t.persist.cli.Get(t.ctx, keyStr, clientv3.WithSerializable())
				if err != nil {
					return err
				}
				exists := len(resp.Kvs) > 0
				if !exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyNotFound, keyStr))
				}
				existing, err := t.persist.marshaler.Unmarshal(resp.Kvs[0].Value)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				newV, shouldWrite := op.updateFunc(existing)
				if !shouldWrite {
					results[i] = TxnResult[V]{Value: existing, Version: resp.Kvs[0].ModRevision}
					written[i] = false
					continue
				}
				cmps = append(cmps, clientv3.Compare(clientv3.ModRevision(keyStr), "=", resp.Kvs[0].ModRevision))
				valBytes, err := t.persist.marshaler.Marshal(newV)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				putOps = append(putOps, clientv3.OpPut(keyStr, string(valBytes)))
				results[i].Value = newV
				written[i] = true

			case opUpsert:
				resp, err := t.persist.cli.Get(t.ctx, keyStr, clientv3.WithSerializable())
				if err != nil {
					return err
				}
				exists := len(resp.Kvs) > 0
				if exists {
					existing, err := t.persist.marshaler.Unmarshal(resp.Kvs[0].Value)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					newV, shouldWrite := op.updateFunc(existing)
					if !shouldWrite {
						results[i] = TxnResult[V]{Value: existing, Version: resp.Kvs[0].ModRevision}
						written[i] = false
						continue
					}
					cmps = append(cmps, clientv3.Compare(clientv3.ModRevision(keyStr), "=", resp.Kvs[0].ModRevision))
					valBytes, err := t.persist.marshaler.Marshal(newV)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					putOps = append(putOps, clientv3.OpPut(keyStr, string(valBytes)))
					results[i].Value = newV
				} else {
					valBytes, err := t.persist.marshaler.Marshal(op.value)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					putOps = append(putOps, clientv3.OpPut(keyStr, string(valBytes)))
					results[i].Value = op.value
				}
				written[i] = true

			case opDelete:
				resp, err := t.persist.cli.Get(t.ctx, keyStr, clientv3.WithSerializable(), clientv3.WithKeysOnly())
				if err != nil {
					return err
				}
				exists := len(resp.Kvs) > 0

				if !exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyNotFound, keyStr))
				}
				cmps = append(cmps, clientv3.Compare(clientv3.ModRevision(keyStr), "=", resp.Kvs[0].ModRevision))
				putOps = append(putOps, clientv3.OpDelete(keyStr))
				written[i] = true

			case opPutIfVersion:
				hasVersioned = true
				cmps = append(cmps, etcdVersionCmp(keyStr, op.version))
				valBytes, err := t.persist.marshaler.Marshal(op.value)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				putOps = append(putOps, clientv3.OpPut(keyStr, string(valBytes)))
				results[i].Value = op.value
				written[i] = true

			case opDeleteIfVersion:
				hasVersioned = true
				cmps = append(cmps, etcdVersionCmp(keyStr, op.version))
				putOps = append(putOps, clientv3.OpDelete(keyStr))
				written[i] = true

			case opRaw:
				if op.raw.Delete {
					putOps = append(putOps, clientv3.OpDelete(op.raw.Key))
				} else {
					putOps = append(putOps, clientv3.OpPut(op.raw.Key, op.raw.Value))
				}
			}
		}
		getDur := time.Since(getStart)

		if len(putOps) == 0 {
			return nil // all skipped
		}

		// Phase 2: commit the transaction
		txnStart := time.Now()
		var txnResp *clientv3.TxnResponse
		var err error
		if len(cmps) == 0 {
			txnResp, err = t.persist.cli.Txn(t.ctx).Then(putOps...).Commit()
		} else {
			txnResp, err = t.persist.cli.Txn(t.ctx).If(cmps...).Then(putOps...).Commit()
		}
		txnDur := time.Since(txnStart)

		totalDur := time.Since(commitStart)
		if totalDur > 40*time.Millisecond {
			mlog.Info(t.ctx, "etcdTxn.Commit slow",
				zap.Duration("total", totalDur),
				zap.Duration("etcdGet", getDur),
				zap.Duration("etcdTxn", txnDur),
				zap.Int("numOps", len(batch)),
				zap.Int("numPuts", len(putOps)),
				zap.Int("retryCount", retryCount))
		}

		if err != nil {
			return err
		}
		if !txnResp.Succeeded {
			if hasVersioned {
				applied, err := t.resolveVersionedFailure(batch, results)
				if err != nil {
					if errors.Is(err, ErrVersionConflict) {
						return retry.Unrecoverable(err)
					}
					return err
				}
				if applied {
					return nil
				}
			}
			return fmt.Errorf("CAS failed, concurrent modification")
		}
		for i, w := range written {
			if w {
				results[i].Version = txnResp.Header.Revision
			}
		}
		return nil
	}, retry.AttemptAlways())
}

// resolveVersionedFailure rereads the keys of the batch's versioned ops after
// a failed compare. It reports ErrVersionConflict when one of them moved to a
// value this batch did not write, (false, nil) when all of them are still at
// their expected versions (the read-modify-write compares failed: retry), and
// (true, nil) when every versioned op is found already applied - an earlier
// attempt of this batch landed but its response was lost. In that case the
// unversioned raw ops are idempotently re-applied and the results filled from
// the store.
func (t *etcdTxn[K, V]) resolveVersionedFailure(batch []int, results []TxnResult[V]) (bool, error) {
	unchanged, applied := 0, 0
	revisions := make(map[int]int64, len(batch))
	var rawOps []clientv3.Op
	for _, i := range batch {
		op := t.ops[i]
		switch {
		case op.kind == opRaw:
			if op.raw.Delete {
				rawOps = append(rawOps, clientv3.OpDelete(op.raw.Key))
			} else {
				rawOps = append(rawOps, clientv3.OpPut(op.raw.Key, op.raw.Value))
			}
			continue
		case !op.versioned():
			// Mixed batches cannot tell a lost success from a conflict.
			continue
		}
		resp, err := t.persist.cli.Get(t.ctx, string(op.key))
		if err != nil {
			return false, err
		}
		exists := len(resp.Kvs) > 0
		current := int64(0)
		if exists {
			current = resp.Kvs[0].ModRevision
		}
		switch {
		case current == op.version:
			unchanged++
		case op.kind == opDeleteIfVersion && !exists:
			applied++
		case op.kind == opPutIfVersion && exists && storedValueEquals(t.persist.marshaler, resp.Kvs[0].Value, op.value):
			applied++
			revisions[i] = current
		default:
			return false, errors.Wrapf(ErrVersionConflict, "key %s: expected version %d, current %d", string(op.key), op.version, current)
		}
	}
	if applied == 0 {
		return false, nil
	}
	if unchanged > 0 || applied+len(rawOps) != len(batch) {
		return false, errors.Wrapf(ErrVersionConflict, "batch partially applied by another writer")
	}
	if len(rawOps) > 0 {
		if _, err := t.persist.cli.Txn(t.ctx).Then(rawOps...).Commit(); err != nil {
			return false, err
		}
	}
	for i, revision := range revisions {
		results[i].Value = t.ops[i].value
		results[i].Version = revision
	}
	mlog.Info(t.ctx, "etcdTxn.Commit found an earlier attempt applied", zap.Int("numOps", len(batch)))
	return true, nil
}

func etcdVersionCmp(key string, version int64) clientv3.Cmp {
	if version == 0 {
		return clientv3.Compare(clientv3.CreateRevision(key), "=", 0)
	}
	return clientv3.Compare(clientv3.ModRevision(key), "=", version)
}

// ============================================================
// TiKV implementation
// ============================================================

type tikvPersist[K string, V any] struct {
	cli       *txnkv.Client
	marshaler Marshaler[V]
}

func NewOptimisticTxnTiKVPersist[K string, V any](cli *txnkv.Client, marshaler Marshaler[V]) OptimisticTxnPersist[K, V] {
	return &tikvPersist[K, V]{cli: cli, marshaler: marshaler}
}

func (p *tikvPersist[K, V]) Txn(ctx context.Context) Txn[K, V] {
	return &tikvTxn[K, V]{ctx: ctx, persist: p}
}

func (p *tikvPersist[K, V]) captureCommitTS(txn interface{ SetCommitCallback(func(string, error)) }) *uint64 {
	var commitTS uint64
	txn.SetCommitCallback(func(info string, err error) {
		if err == nil {
			var txnInfo transaction.TxnInfo
			json.Unmarshal([]byte(info), &txnInfo)
			commitTS = txnInfo.CommitTS
		}
	})
	return &commitTS
}

func (p *tikvPersist[K, V]) tikvKeyExists(ctx context.Context, txn interface {
	Get(context.Context, []byte, ...tikvkv.GetOption) (tikvkv.ValueEntry, error)
}, key []byte,
) ([]byte, bool, error) {
	entry, err := txn.Get(ctx, key)
	if err != nil {
		if tikverr.IsErrNotFound(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	return entry.Value, true, nil
}

// tikvKeyWithCommitTS reads the key's latest value and its commit timestamp
// inside txn.
func (p *tikvPersist[K, V]) tikvKeyWithCommitTS(ctx context.Context, txn *transaction.KVTxn, key []byte) (tikvkv.ValueEntry, bool, error) {
	entry, err := txn.Get(ctx, key, tikvkv.WithReturnCommitTS())
	if err != nil {
		if tikverr.IsErrNotFound(err) {
			return tikvkv.ValueEntry{}, false, nil
		}
		return tikvkv.ValueEntry{}, false, err
	}
	return entry, true, nil
}

func (p *tikvPersist[K, V]) Scan(ctx context.Context, prefix K) ([]K, []V, []int64, error) {
	txn, err := p.cli.Begin()
	if err != nil {
		return nil, nil, nil, err
	}
	defer txn.Rollback()
	prefixBytes := []byte(prefix)
	endKey := make([]byte, len(prefixBytes))
	copy(endKey, prefixBytes)
	endKey[len(endKey)-1]++

	iter, err := txn.Iter(prefixBytes, endKey)
	if err != nil {
		return nil, nil, nil, err
	}
	defer iter.Close()

	var ks []K
	var vals []V
	var vers []int64
	for iter.Valid() {
		v, err := p.marshaler.Unmarshal(iter.Value())
		if err != nil {
			return nil, nil, nil, err
		}
		ks = append(ks, K(iter.Key()))
		vals = append(vals, v)
		vers = append(vers, int64(txn.StartTS()))
		if err := iter.Next(); err != nil {
			return nil, nil, nil, err
		}
	}
	return ks, vals, vers, nil
}

func (p *tikvPersist[K, V]) Get(ctx context.Context, keys []K) (map[K]TxnResult[V], error) {
	txn, err := p.cli.Begin()
	if err != nil {
		return nil, err
	}
	defer txn.Rollback()
	result := make(map[K]TxnResult[V], len(keys))
	for _, key := range keys {
		val, exists, err := p.tikvKeyExists(ctx, txn, []byte(key))
		if err != nil {
			return nil, err
		}
		if !exists {
			continue
		}
		v, err := p.marshaler.Unmarshal(val)
		if err != nil {
			return nil, err
		}
		// Like Scan, the snapshot timestamp bounds the commit ts of every value read.
		result[key] = TxnResult[V]{Value: v, Version: int64(txn.StartTS())}
	}
	return result, nil
}

type tikvTxn[K string, V any] struct {
	txnOps[K, V]
	ctx     context.Context
	persist *tikvPersist[K, V]
}

func (t *tikvTxn[K, V]) Commit() ([]TxnResult[V], error) {
	results := make([]TxnResult[V], len(t.ops))

	err := retry.Do(t.ctx, func() error {
		txn, err := t.persist.cli.Begin()
		if err != nil {
			return err
		}
		defer txn.Rollback()

		anyWrite := false
		versionedWritten, versionedApplied := 0, 0
		for i, op := range t.ops {
			keyBytes := []byte(op.key)
			switch op.kind {
			case opRaw:
				keyBytes = []byte(op.raw.Key)
				if op.raw.Delete {
					err = txn.Delete(keyBytes)
				} else {
					err = txn.Set(keyBytes, []byte(op.raw.Value))
				}
				if err != nil {
					return err
				}
				anyWrite = true
				continue
			case opPutIfVersion, opDeleteIfVersion:
				// The versions handed out by this persist are snapshot or
				// commit timestamps, so a value committed after the caller's
				// version is newer than the copy the write was built from.
				entry, exists, err := t.persist.tikvKeyWithCommitTS(t.ctx, txn, keyBytes)
				if err != nil {
					return err
				}
				if (op.version == 0 && exists) ||
					(op.version != 0 && (!exists || (entry.CommitTS != 0 && int64(entry.CommitTS) > op.version))) {
					// An earlier attempt whose response was lost may have
					// written exactly this op already.
					if op.kind == opDeleteIfVersion && !exists {
						versionedApplied++
						continue
					}
					if op.kind == opPutIfVersion && exists && storedValueEquals(t.persist.marshaler, entry.Value, op.value) {
						versionedApplied++
						results[i].Value = op.value
						results[i].Version = int64(entry.CommitTS)
						continue
					}
					return retry.Unrecoverable(errors.Wrapf(ErrVersionConflict,
						"key %s: expected version %d, current commit ts %d", string(op.key), op.version, entry.CommitTS))
				}
				versionedWritten++
				if op.kind == opDeleteIfVersion {
					if err := txn.Delete(keyBytes); err != nil {
						return err
					}
				} else {
					valBytes, err := t.persist.marshaler.Marshal(op.value)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					if err := txn.Set(keyBytes, valBytes); err != nil {
						return err
					}
					results[i].Value = op.value
				}
				anyWrite = true
				continue
			}

			val, exists, err := t.persist.tikvKeyExists(t.ctx, txn, keyBytes)
			if err != nil {
				return err
			}

			switch op.kind {
			case opInsert:
				if exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyAlreadyExists, string(op.key)))
				}
				valBytes, err := t.persist.marshaler.Marshal(op.value)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				if err := txn.Set(keyBytes, valBytes); err != nil {
					return err
				}
				results[i].Value = op.value
				anyWrite = true

			case opUpdate:
				if !exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyNotFound, string(op.key)))
				}
				existing, err := t.persist.marshaler.Unmarshal(val)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				newV, shouldWrite := op.updateFunc(existing)
				if !shouldWrite {
					results[i].Value = existing
					continue
				}
				valBytes, err := t.persist.marshaler.Marshal(newV)
				if err != nil {
					return retry.Unrecoverable(err)
				}
				if err := txn.Set(keyBytes, valBytes); err != nil {
					return err
				}
				results[i].Value = newV
				anyWrite = true

			case opUpsert:
				if exists {
					existing, err := t.persist.marshaler.Unmarshal(val)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					newV, shouldWrite := op.updateFunc(existing)
					if !shouldWrite {
						results[i].Value = existing
						continue
					}
					valBytes, err := t.persist.marshaler.Marshal(newV)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					if err := txn.Set(keyBytes, valBytes); err != nil {
						return err
					}
					results[i].Value = newV
				} else {
					valBytes, err := t.persist.marshaler.Marshal(op.value)
					if err != nil {
						return retry.Unrecoverable(err)
					}
					if err := txn.Set(keyBytes, valBytes); err != nil {
						return err
					}
					results[i].Value = op.value
				}
				anyWrite = true

			case opDelete:
				if !exists {
					return retry.Unrecoverable(fmt.Errorf("%w: %s", ErrKeyNotFound, string(op.key)))
				}
				if err := txn.Delete(keyBytes); err != nil {
					return err
				}
				anyWrite = true
			}
		}

		if versionedApplied > 0 && versionedWritten > 0 {
			return retry.Unrecoverable(errors.Wrapf(ErrVersionConflict, "transaction partially applied by another writer"))
		}

		if !anyWrite {
			return nil
		}

		cts := t.persist.captureCommitTS(txn)
		err = txn.Commit(t.ctx)
		if err == nil {
			for i := range t.ops {
				if results[i].Version == 0 {
					results[i].Version = int64(*cts)
				}
			}
		}
		return err
	}, retry.AttemptAlways())
	if err != nil {
		return nil, err
	}
	return results, nil
}

// ============================================================
// In-memory implementation (for testing)
// ============================================================

type memEntry[V any] struct {
	value   V
	version int64
}

type memPersist[K comparable, V any] struct {
	// mu serializes transactions like the store would: meta commits writes of
	// different segments concurrently.
	mu        sync.RWMutex
	data      map[K]*memEntry[V]
	raw       map[string]string
	nextVer   int64
	marshaler Marshaler[V]
}

func NewOptimisticTxnMemoryPersist[K comparable, V any](marshaler Marshaler[V]) OptimisticTxnPersist[K, V] {
	return &memPersist[K, V]{
		data:      make(map[K]*memEntry[V]),
		raw:       make(map[string]string),
		nextVer:   1,
		marshaler: marshaler,
	}
}

func (p *memPersist[K, V]) clone(value V) (V, error) {
	data, err := p.marshaler.Marshal(value)
	if err != nil {
		var zero V
		return zero, err
	}
	return p.marshaler.Unmarshal(data)
}

func (p *memPersist[K, V]) Txn(ctx context.Context) Txn[K, V] {
	return &memTxn[K, V]{persist: p}
}

func (p *memPersist[K, V]) Scan(ctx context.Context, prefix K) ([]K, []V, []int64, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	prefixStr := fmt.Sprintf("%v", prefix)
	var ks []K
	var vals []V
	var vers []int64
	for k, entry := range p.data {
		keyStr := fmt.Sprintf("%v", k)
		if strings.HasPrefix(keyStr, prefixStr) {
			value, err := p.clone(entry.value)
			if err != nil {
				return nil, nil, nil, err
			}
			ks = append(ks, k)
			vals = append(vals, value)
			vers = append(vers, entry.version)
		}
	}
	return ks, vals, vers, nil
}

func (p *memPersist[K, V]) Get(ctx context.Context, keys []K) (map[K]TxnResult[V], error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	result := make(map[K]TxnResult[V], len(keys))
	for _, key := range keys {
		if entry, ok := p.data[key]; ok {
			value, err := p.clone(entry.value)
			if err != nil {
				return nil, err
			}
			result[key] = TxnResult[V]{Value: value, Version: entry.version}
		}
	}
	return result, nil
}

type memTxn[K comparable, V any] struct {
	txnOps[K, V]
	persist *memPersist[K, V]
}

func (t *memTxn[K, V]) Commit() ([]TxnResult[V], error) {
	p := t.persist
	p.mu.Lock()
	defer p.mu.Unlock()
	// Validate all ops first (don't partially apply).
	for _, op := range t.ops {
		switch op.kind {
		case opInsert:
			if _, ok := p.data[op.key]; ok {
				return nil, fmt.Errorf("%w: %v", ErrKeyAlreadyExists, op.key)
			}
		case opUpdate:
			if _, ok := p.data[op.key]; !ok {
				return nil, fmt.Errorf("%w: %v", ErrKeyNotFound, op.key)
			}
		case opDelete:
			if _, ok := p.data[op.key]; !ok {
				return nil, fmt.Errorf("%w: %v", ErrKeyNotFound, op.key)
			}
		case opPutIfVersion, opDeleteIfVersion:
			current := int64(0)
			if entry, ok := p.data[op.key]; ok {
				current = entry.version
			}
			if current != op.version {
				return nil, errors.Wrapf(ErrVersionConflict, "key %v: expected version %d, current %d", op.key, op.version, current)
			}
		case opUpsert, opRaw:
			// always valid
		}
	}

	ver := p.nextVer
	p.nextVer++
	results := make([]TxnResult[V], len(t.ops))

	for i, op := range t.ops {
		switch op.kind {
		case opInsert, opPutIfVersion:
			stored, err := p.clone(op.value)
			if err != nil {
				return nil, err
			}
			p.data[op.key] = &memEntry[V]{value: stored, version: ver}
			results[i] = TxnResult[V]{Value: op.value, Version: ver}

		case opUpdate:
			entry := p.data[op.key]
			existing, err := p.clone(entry.value)
			if err != nil {
				return nil, err
			}
			newV, shouldWrite := op.updateFunc(existing)
			if !shouldWrite {
				results[i] = TxnResult[V]{Value: existing, Version: entry.version}
				continue
			}
			p.data[op.key] = &memEntry[V]{value: newV, version: ver}
			results[i] = TxnResult[V]{Value: newV, Version: ver}

		case opUpsert:
			if entry, ok := p.data[op.key]; ok {
				existing, err := p.clone(entry.value)
				if err != nil {
					return nil, err
				}
				newV, shouldWrite := op.updateFunc(existing)
				if !shouldWrite {
					results[i] = TxnResult[V]{Value: existing, Version: entry.version}
					continue
				}
				p.data[op.key] = &memEntry[V]{value: newV, version: ver}
				results[i] = TxnResult[V]{Value: newV, Version: ver}
			} else {
				stored, err := p.clone(op.value)
				if err != nil {
					return nil, err
				}
				p.data[op.key] = &memEntry[V]{value: stored, version: ver}
				results[i] = TxnResult[V]{Value: op.value, Version: ver}
			}

		case opDelete, opDeleteIfVersion:
			delete(p.data, op.key)
			results[i] = TxnResult[V]{Version: ver}

		case opRaw:
			if op.raw.Delete {
				delete(p.raw, op.raw.Key)
			} else {
				p.raw[op.raw.Key] = op.raw.Value
			}
		}
	}

	return results, nil
}
