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
	"runtime"
	"sync"
	"sync/atomic"
)

// maxRecoveryWorkers bounds the goroutines of a recovery rebuild; shards are
// numbered in a byte.
const maxRecoveryWorkers = 64

// recoveryWorkers returns the number of goroutines a CPU-bound recovery
// rebuild runs on.
func recoveryWorkers() int {
	return max(1, min(runtime.GOMAXPROCS(0), maxRecoveryWorkers))
}

// shardOf maps key to one of shards shards, spreading sequential IDs evenly.
func shardOf(key int64, shards int) int {
	return int((uint64(key) * 0x9E3779B97F4A7C15 >> 32) % uint64(shards))
}

// forEachInShards calls fn for every index below n on up to workers
// goroutines, spread by the shard of key(i): the indexes of one key reach one
// goroutine in increasing order, so state kept per key changes as in one pass
// over the indexes in order. fn gets the number of its goroutine, below
// workers, for state of its own. No index after a failed one is started, and
// the error returned is that of the lowest failed index: the one a pass in
// order would stop at.
func forEachInShards(n, workers int, key func(i int) int64, fn func(worker, i int) error) error {
	workers = max(1, min(workers, n, maxRecoveryWorkers))
	if workers == 1 {
		for i := range n {
			if err := fn(0, i); err != nil {
				return err
			}
		}
		return nil
	}

	shards := make([]uint8, n)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := start; i < min(start+chunk, n); i++ {
				shards[i] = uint8(shardOf(key(i), workers))
			}
		}()
	}
	wg.Wait()

	// failedAt is the lowest failed index so far; indexes after it need not
	// run.
	var failedAt atomic.Int64
	failedAt.Store(int64(n))
	errs := make([]error, workers)
	errIndexes := make([]int, workers)
	for worker := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, shard := range shards {
				if int(shard) != worker {
					continue
				}
				if int64(i) > failedAt.Load() {
					return
				}
				if err := fn(worker, i); err != nil {
					errs[worker], errIndexes[worker] = err, i
					for current := failedAt.Load(); int64(i) < current && !failedAt.CompareAndSwap(current, int64(i)); current = failedAt.Load() {
					}
					return
				}
			}
		}()
	}
	wg.Wait()

	var err error
	lowest := n
	for worker, workerErr := range errs {
		if workerErr != nil && errIndexes[worker] < lowest {
			err, lowest = workerErr, errIndexes[worker]
		}
	}
	return err
}

// recoveryProgressStep is how many finished items a worker counts on its own
// before it adds them to the shared total of a recoveryProgress.
const recoveryProgressStep = 1024

// recoveryProgress counts the items the workers of forEachInShards finish, for
// progress logs. A worker counts in a slot of its own and adds to the shared
// total once per recoveryProgressStep items: one counter that every worker
// writes for every item serializes them.
type recoveryProgress struct {
	every int64
	total atomic.Int64
	slots []recoveryProgressSlot
}

// recoveryProgressSlot fills a cache line so the slots of different workers do
// not share one.
type recoveryProgressSlot struct {
	pending int64
	_       [56]byte
}

func newRecoveryProgress(workers int, every int) *recoveryProgress {
	return &recoveryProgress{every: int64(every), slots: make([]recoveryProgressSlot, max(1, min(workers, maxRecoveryWorkers)))}
}

// add counts one item finished by worker. It returns the shared total and
// whether that total has just passed a multiple of every, to report then.
func (p *recoveryProgress) add(worker int) (int, bool) {
	slot := &p.slots[worker]
	slot.pending++
	if slot.pending < recoveryProgressStep {
		return 0, false
	}
	slot.pending = 0
	total := p.total.Add(recoveryProgressStep)
	return int(total), total/p.every != (total-recoveryProgressStep)/p.every
}
