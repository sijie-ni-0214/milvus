package datacoord

import (
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestForEachInShards(t *testing.T) {
	t.Run("visits every index once, the indexes of a key in order on one goroutine", func(t *testing.T) {
		for _, n := range []int{0, 1, 7, 10000} {
			for _, workers := range []int{1, 3, 16, 100} {
				keys := make([]int64, n)
				for i := range keys {
					keys[i] = int64(i % 97)
				}
				var mu sync.Mutex
				var violations []string
				visits := make([]int, n)
				workerOfKey := make(map[int64]int)
				lastOfKey := make(map[int64]int)
				err := forEachInShards(n, workers, func(i int) int64 { return keys[i] }, func(worker, i int) error {
					mu.Lock()
					defer mu.Unlock()
					if worker >= max(1, workers) {
						violations = append(violations, fmt.Sprintf("worker %d out of range", worker))
					}
					visits[i]++
					key := keys[i]
					if w, ok := workerOfKey[key]; ok && (w != worker || lastOfKey[key] >= i) {
						violations = append(violations, fmt.Sprintf("index %d of key %d on worker %d after index %d on worker %d", i, key, worker, lastOfKey[key], w))
					}
					workerOfKey[key], lastOfKey[key] = worker, i
					return nil
				})
				require.NoError(t, err)
				require.Empty(t, violations, "n %d workers %d", n, workers)
				for i, count := range visits {
					require.Equal(t, 1, count, "n %d workers %d index %d", n, workers, i)
				}
			}
		}
	})

	t.Run("returns the error of the lowest failed index", func(t *testing.T) {
		const n = 5000
		for _, failing := range [][]int{{4000}, {4000, 17}, {17, 18, 4999}} {
			fails := make(map[int]bool)
			for _, i := range failing {
				fails[i] = true
			}
			var mu sync.Mutex
			visited := make(map[int]bool)
			err := forEachInShards(n, 8, func(i int) int64 { return int64(i) }, func(_, i int) error {
				mu.Lock()
				visited[i] = true
				mu.Unlock()
				if fails[i] {
					return fmt.Errorf("index %d failed", i)
				}
				return nil
			})
			first := slices.Min(failing)
			require.EqualError(t, err, fmt.Sprintf("index %d failed", first))
			for i := range first {
				require.True(t, visited[i], "every index before the first failure runs")
			}
		}
	})
}
