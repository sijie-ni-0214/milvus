//go:build test && dynamic

package qnview

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSegmentLoadTimingStatsAggregatesAndResets(t *testing.T) {
	stats := segmentLoadTimingStats{interval: time.Hour, batchSize: 2}
	startedAt := time.Unix(100, 0)
	_, ok := stats.add(startedAt, segmentLoadTimingSample{
		total:           10 * time.Millisecond,
		updateIndexMeta: 2 * time.Millisecond,
		reserveResource: 3 * time.Millisecond,
		physicalLoad:    4 * time.Millisecond,
		releaseResource: time.Millisecond,
		onLoaded:        2 * time.Millisecond,
	})
	require.False(t, ok)

	snapshot, ok := stats.add(startedAt.Add(time.Second), segmentLoadTimingSample{
		total:           20 * time.Millisecond,
		updateIndexMeta: 4 * time.Millisecond,
		reserveResource: 6 * time.Millisecond,
		physicalLoad:    8 * time.Millisecond,
		releaseResource: 2 * time.Millisecond,
		failed:          true,
	})
	require.True(t, ok)
	assert.Equal(t, int64(2), snapshot.count)
	assert.Equal(t, int64(1), snapshot.failed)
	assert.Equal(t, time.Second, snapshot.windowDuration)
	assert.Equal(t, 15*time.Millisecond, snapshot.total.average(snapshot.count))
	assert.Equal(t, 20*time.Millisecond, snapshot.total.max)
	assert.Equal(t, 3*time.Millisecond, snapshot.updateIndexMeta.average(snapshot.count))
	assert.Equal(t, 6*time.Millisecond, snapshot.physicalLoad.average(snapshot.count))
	assert.Equal(t, time.Millisecond, snapshot.onLoaded.average(snapshot.count))

	_, ok = stats.add(startedAt.Add(2*time.Second), segmentLoadTimingSample{})
	assert.False(t, ok)
}

func TestSegmentLoadTimingStatsLogsOnInterval(t *testing.T) {
	stats := segmentLoadTimingStats{interval: 5 * time.Second, batchSize: 100}
	startedAt := time.Unix(100, 0)
	_, ok := stats.add(startedAt, segmentLoadTimingSample{})
	require.False(t, ok)
	snapshot, ok := stats.add(startedAt.Add(5*time.Second), segmentLoadTimingSample{})
	require.True(t, ok)
	assert.Equal(t, int64(2), snapshot.count)
	assert.Equal(t, 5*time.Second, snapshot.windowDuration)
}
