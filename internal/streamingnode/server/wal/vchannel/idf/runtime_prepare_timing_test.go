//go:build test && dynamic

package idf

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBM25RuntimePrepareTimingStatsAggregatesAndResets(t *testing.T) {
	stats := bm25RuntimePrepareTimingStats{interval: time.Hour, batchSize: 2}
	startedAt := time.Unix(100, 0)
	_, ok := stats.add(startedAt.Add(10*time.Millisecond), bm25RuntimePrepareTimingSample{
		startedAt: startedAt,
		total:     10 * time.Millisecond,
	})
	require.False(t, ok)

	snapshot, ok := stats.add(startedAt.Add(time.Second), bm25RuntimePrepareTimingSample{
		startedAt: startedAt.Add(900 * time.Millisecond),
		total:     100 * time.Millisecond,
		failed:    true,
	})
	require.True(t, ok)
	assert.Equal(t, int64(2), snapshot.count)
	assert.Equal(t, int64(1), snapshot.failed)
	assert.Equal(t, time.Second, snapshot.windowDuration)
	assert.Equal(t, 110*time.Millisecond, snapshot.total.total)
	assert.Equal(t, 55*time.Millisecond, snapshot.total.average(snapshot.count))
	assert.Equal(t, 100*time.Millisecond, snapshot.total.max)

	_, ok = stats.add(startedAt.Add(2*time.Second), bm25RuntimePrepareTimingSample{
		startedAt: startedAt.Add(2 * time.Second),
	})
	assert.False(t, ok)
}

func TestBM25RuntimePrepareTimingStatsLogsOnInterval(t *testing.T) {
	stats := bm25RuntimePrepareTimingStats{interval: 5 * time.Second, batchSize: 100}
	startedAt := time.Unix(100, 0)
	snapshot, ok := stats.add(startedAt.Add(5*time.Second), bm25RuntimePrepareTimingSample{
		startedAt: startedAt,
		total:     5 * time.Second,
	})
	require.True(t, ok)
	assert.Equal(t, int64(1), snapshot.count)
	assert.Equal(t, 5*time.Second, snapshot.windowDuration)
}
