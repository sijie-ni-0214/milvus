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

package metrics

import (
	"testing"

	"github.com/stretchr/testify/require"

	pkgmetrics "github.com/milvus-io/milvus/pkg/v3/metrics"
)

func TestParseTextMetrics(t *testing.T) {
	const metricsText = `# HELP milvus_c_registry_test_total Test metric.
# TYPE milvus_c_registry_test_total counter
milvus_c_registry_test_total{source="core"} 1
`

	metricFamilies, err := parseTextMetrics(metricsText)
	require.NoError(t, err)
	require.Contains(t, metricFamilies, "milvus_c_registry_test_total")
}

func TestParseTextMetricsUsesLegacyValidation(t *testing.T) {
	_, err := parseTextMetrics(`{"metric.name",source="core"} 1`)
	require.Error(t, err)
}

func TestApplyCollectionLevelMetricsModeAggregatesCacheShardMetric(t *testing.T) {
	previous := pkgmetrics.CollectionLevelMetricsMode()
	pkgmetrics.SetCollectionLevelMetricsMode(pkgmetrics.CollectionLevelMetricsModeAggregate)
	t.Cleanup(func() {
		pkgmetrics.SetCollectionLevelMetricsMode(previous)
	})

	metricFamilies, err := parseTextMetrics(`# HELP internal_cache_shard_disk_usage_bytes Cache disk usage.
# TYPE internal_cache_shard_disk_usage_bytes gauge
internal_cache_shard_disk_usage_bytes{data_type="vector_index",shard="pchannel_1_1v0"} 10
internal_cache_shard_disk_usage_bytes{data_type="vector_index",shard="pchannel_1_2v0"} 20
internal_cache_shard_disk_usage_bytes{data_type="scalar_index",shard="pchannel_1_3v0"} 5
`)
	require.NoError(t, err)

	applyCollectionLevelMetricsMode(metricFamilies)

	metricFamily := metricFamilies[cacheShardDiskUsageMetricName]
	require.Len(t, metricFamily.Metric, 2)
	values := make(map[string]float64, 2)
	for _, metric := range metricFamily.Metric {
		labels := make(map[string]string, len(metric.Label))
		for _, label := range metric.Label {
			labels[label.GetName()] = label.GetValue()
		}
		require.Equal(t, pkgmetrics.AllLabel, labels["shard"])
		values[labels["data_type"]] = metric.GetGauge().GetValue()
	}
	require.Equal(t, float64(30), values["vector_index"])
	require.Equal(t, float64(5), values["scalar_index"])
}

func TestApplyCollectionLevelMetricsModePreservesFullCacheShardMetric(t *testing.T) {
	previous := pkgmetrics.CollectionLevelMetricsMode()
	pkgmetrics.SetCollectionLevelMetricsMode(pkgmetrics.CollectionLevelMetricsModeFull)
	t.Cleanup(func() {
		pkgmetrics.SetCollectionLevelMetricsMode(previous)
	})

	metricFamilies, err := parseTextMetrics(`# TYPE internal_cache_shard_disk_usage_bytes gauge
internal_cache_shard_disk_usage_bytes{data_type="vector_index",shard="pchannel_1_1v0"} 10
internal_cache_shard_disk_usage_bytes{data_type="vector_index",shard="pchannel_1_2v0"} 20
`)
	require.NoError(t, err)

	applyCollectionLevelMetricsMode(metricFamilies)

	require.Len(t, metricFamilies[cacheShardDiskUsageMetricName].Metric, 2)
}
