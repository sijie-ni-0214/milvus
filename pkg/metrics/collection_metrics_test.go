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

package metrics

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setCollectionMetricsModeForTest(t *testing.T, mode string) {
	t.Helper()
	previous := CollectionLevelMetricsMode()
	SetCollectionLevelMetricsMode(mode)
	t.Cleanup(func() {
		SetCollectionLevelMetricsMode(previous)
	})
}

func TestCollectionCounterVecModes(t *testing.T) {
	t.Run("full preserves collection series", func(t *testing.T) {
		setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeFull)
		counter := newCollectionCounterVec(
			prometheus.CounterOpts{Name: "test_collection_counter_full_total", Help: "test"},
			[]string{databaseLabelName, collectionName},
		)

		counter.WithLabelValues("db", "collection-a").Add(1)
		counter.WithLabelValues("db", "collection-b").Add(2)

		assert.Equal(t, float64(1), testutil.ToFloat64(counter.WithLabelValues("db", "collection-a")))
		assert.Equal(t, float64(2), testutil.ToFloat64(counter.WithLabelValues("db", "collection-b")))
	})

	t.Run("aggregate collapses and preserves shared series on cleanup", func(t *testing.T) {
		setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
		counter := newCollectionCounterVec(
			prometheus.CounterOpts{Name: "test_collection_counter_aggregate_total", Help: "test"},
			[]string{databaseLabelName, collectionName},
		)

		counter.WithLabelValues("db", "collection-a").Add(1)
		counter.With(prometheus.Labels{
			databaseLabelName: "db",
			collectionName:    "collection-b",
		}).Add(2)

		assert.Equal(t, float64(3), testutil.ToFloat64(counter.WithLabelValues("db", "collection-c")))
		assert.False(t, counter.DeleteLabelValues("db", "collection-a"))
		assert.False(t, counter.Delete(prometheus.Labels{
			databaseLabelName: "db",
			collectionName:    "collection-a",
		}))
		assert.Zero(t, counter.DeletePartialMatch(prometheus.Labels{collectionName: "collection-a"}))
		assert.Equal(t, float64(3), testutil.ToFloat64(counter.WithLabelValues("db", "collection-a")))
	})
}

func TestCollectionHistogramVecAggregate(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
	histogram := newCollectionHistogramVec(
		prometheus.HistogramOpts{
			Name:    "test_collection_histogram",
			Help:    "test",
			Buckets: []float64{1, 2},
		},
		[]string{collectionIDLabelName},
	)
	registry := prometheus.NewRegistry()
	registry.MustRegister(histogram)

	histogram.WithLabelValues("1").Observe(1)
	histogram.WithLabelValues("2").Observe(2)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].Metric, 1)
	require.Equal(t, uint64(2), families[0].Metric[0].Histogram.GetSampleCount())
	require.Len(t, families[0].Metric[0].Label, 1)
	assert.Equal(t, collectionIDLabelName, families[0].Metric[0].Label[0].GetName())
	assert.Equal(t, AllLabel, families[0].Metric[0].Label[0].GetValue())
}

func TestVChannelHistogramVecAggregate(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
	histogram := newVChannelHistogramVec(
		prometheus.HistogramOpts{
			Name:    "test_vchannel_histogram",
			Help:    "test",
			Buckets: []float64{1, 2},
		},
		[]string{nodeIDLabelName, channelNameLabelName},
	)
	registry := prometheus.NewRegistry()
	registry.MustRegister(histogram)

	histogram.WithLabelValues("node", "pchannel_1_1v0").Observe(1)
	histogram.WithLabelValues("node", "pchannel_1_2v0").Observe(2)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].Metric, 1)
	require.Equal(t, uint64(2), families[0].Metric[0].Histogram.GetSampleCount())
	labels := families[0].Metric[0].Label
	require.Len(t, labels, 2)
	assert.Equal(t, AllLabel, labels[0].GetValue())
	assert.Equal(t, "node", labels[1].GetValue())
}

func TestCollectionVChannelHistogramVecAggregate(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
	histogram := newCollectionVChannelHistogramVec(
		prometheus.HistogramOpts{Name: "test_collection_vchannel_histogram", Help: "test"},
		[]string{collectionIDLabelName, channelNameLabelName},
	)
	registry := prometheus.NewRegistry()
	registry.MustRegister(histogram)

	histogram.WithLabelValues("1", "pchannel_1_1v0").Observe(1)
	histogram.WithLabelValues("2", "pchannel_1_2v0").Observe(2)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].Metric, 1)
	for _, label := range families[0].Metric[0].Label {
		assert.Equal(t, AllLabel, label.GetValue())
	}
}

func TestPChannelMetricIsUnaffectedByAggregateMode(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
	RootCoordInsertChannelTimeTick.Reset()
	t.Cleanup(RootCoordInsertChannelTimeTick.Reset)
	registry := prometheus.NewRegistry()
	registry.MustRegister(RootCoordInsertChannelTimeTick)

	const pchannel = "pchannel_1"
	RootCoordInsertChannelTimeTick.WithLabelValues(pchannel).Set(10)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].Metric, 1)
	require.Len(t, families[0].Metric[0].Label, 1)
	assert.Equal(t, channelNameLabelName, families[0].Metric[0].Label[0].GetName())
	assert.Equal(t, pchannel, families[0].Metric[0].Label[0].GetValue())
}

func TestCollectionGaugeVecAggregatePolicies(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)

	t.Run("sum policy shares additive gauge", func(t *testing.T) {
		gauge := newCollectionGaugeVec(
			prometheus.GaugeOpts{Name: "test_collection_gauge_sum", Help: "test"},
			[]string{nodeIDLabelName, collectionIDLabelName},
			collectionGaugeAggregateSum,
		)

		gauge.WithLabelValues("node", "1").Add(1)
		gauge.WithLabelValues("node", "2").Add(2)

		assert.Equal(t, float64(3), testutil.ToFloat64(gauge.WithLabelValues("node", "3")))
	})

	t.Run("disabled policy creates no series", func(t *testing.T) {
		gauge := newCollectionGaugeVec(
			prometheus.GaugeOpts{Name: "test_collection_gauge_disabled", Help: "test"},
			[]string{nodeIDLabelName, collectionIDLabelName},
			collectionGaugeAggregateDisabled,
		)
		registry := prometheus.NewRegistry()
		registry.MustRegister(gauge)

		gauge.WithLabelValues("node", "1").Set(10)
		gauge.WithLabelValues("node", "2").Set(20)

		families, err := registry.Gather()
		require.NoError(t, err)
		assert.Empty(t, families)
	})

	t.Run("vchannel disabled policy creates no series", func(t *testing.T) {
		gauge := newVChannelGaugeVec(
			prometheus.GaugeOpts{Name: "test_vchannel_gauge_disabled", Help: "test"},
			[]string{nodeIDLabelName, channelNameLabelName},
			collectionGaugeAggregateDisabled,
		)
		registry := prometheus.NewRegistry()
		registry.MustRegister(gauge)

		gauge.WithLabelValues("node", "pchannel_1_1v0").Set(10)
		gauge.WithLabelValues("node", "pchannel_1_2v0").Set(20)

		families, err := registry.Gather()
		require.NoError(t, err)
		assert.Empty(t, families)
	})
}

func TestQVCollectionCollectorCollapsesCollectionAndVChannelInAggregateMode(t *testing.T) {
	setCollectionMetricsModeForTest(t, CollectionLevelMetricsModeAggregate)
	collector := newQVViewStateMaxAgeCollector()
	providerCalled := false
	collector.SetProvider(func() []QVViewStateMaxAgeMetric {
		providerCalled = true
		return []QVViewStateMaxAgeMetric{{
			Component:    "querynode",
			State:        "ready",
			Rank:         "1",
			CollectionID: "1",
			VChannel:     "pchannel_1_1v0",
			AgeSeconds:   10,
		}}
	})
	registry := prometheus.NewRegistry()
	registry.MustRegister(collector)

	families, err := registry.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)
	require.Len(t, families[0].Metric, 1)
	assert.True(t, providerCalled)
	labels := families[0].Metric[0].Label
	collectionLabel := ""
	vchannelLabel := ""
	for _, label := range labels {
		switch label.GetName() {
		case QVCollectionIDLabel:
			collectionLabel = label.GetValue()
		case QVVChannelLabel:
			vchannelLabel = label.GetValue()
		}
	}
	assert.Equal(t, AllLabel, collectionLabel)
	assert.Equal(t, AllLabel, vchannelLabel)
}

func TestSetCollectionLevelMetricsModeRejectsInvalidValue(t *testing.T) {
	SetCollectionLevelMetricsMode(" AGGREGATE ")
	assert.Equal(t, CollectionLevelMetricsModeAggregate, CollectionLevelMetricsMode())
	assert.Panics(t, func() {
		SetCollectionLevelMetricsMode("unknown")
	})
	SetCollectionLevelMetricsMode(CollectionLevelMetricsModeFull)
}

// TestCollectionLabeledMetricsHonorAggregateMode fails when a metric labeled
// with a collection is declared with a raw prometheus constructor: such a
// metric keeps exporting real collection values in aggregate mode.
func TestCollectionLabeledMetricsHonorAggregateMode(t *testing.T) {
	labelIdentByValue := labelIdentifiers(t)
	files, err := filepath.Glob("*.go")
	require.NoError(t, err)
	checked := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		require.NoError(t, err)
		ast.Inspect(parsed, func(n ast.Node) bool {
			spec, ok := n.(*ast.ValueSpec)
			if !ok || len(spec.Names) != 1 || len(spec.Values) != 1 {
				return true
			}
			labels, ok := vecVariableLabels(spec.Values[0], labelIdentByValue)
			if !ok {
				return true
			}
			_, hasID := labels["collectionIDLabelName"]
			_, hasName := labels["collectionName"]
			if !hasID && !hasName {
				return true
			}
			checked++
			_, raw := spec.Values[0].(*ast.CallExpr).Fun.(*ast.SelectorExpr)
			assert.False(t, raw, "%s in %s carries a collection label but bypasses the collection-level metrics mode",
				spec.Names[0].Name, file)
			return true
		})
	}
	require.NotZero(t, checked)
}
