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
	"strconv"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

type quotaDisplayKey struct {
	family                      int
	db, collection, name, state string
}
type quotaDisplayValue struct {
	value        int64
	contributors int
}
type quotaDisplayCollection struct {
	db, name string
	stats    *metricsinfo.CollectionQuotaSnapshot
}
type quotaDisplayCache struct {
	revision    uint64
	aggregate   bool
	collections map[int64]quotaDisplayCollection
	series      map[quotaDisplayKey]quotaDisplayValue
}

// recordQuotaChanges updates the display metrics from snapshot. A full
// snapshot rebuilds them; partial changes update only their contributions.
func (m *meta) recordQuotaChanges(snapshot *metricsinfo.QuotaSnapshot) {
	full := snapshot.Full
	m.quotaMetricsMu.Lock()
	defer m.quotaMetricsMu.Unlock()
	aggregate := metrics.IsCollectionLevelMetricsAggregateMode()
	if m.quotaDisplay != nil && snapshot.Revision < m.quotaDisplay.revision {
		return
	}
	if m.quotaDisplay == nil || m.quotaDisplay.aggregate != aggregate || full {
		if !full {
			snapshot = m.segments.ReadQuotaSnapshot(true)
		}
		metrics.DataCoordStoredBinlogSize.Reset()
		metrics.DataCoordNumStoredRows.Reset()
		metrics.DataCoordSegmentBinLogFileCount.Reset()
		metrics.DataCoordL0DeleteEntriesNum.Reset()
		m.quotaDisplay = &quotaDisplayCache{aggregate: aggregate, collections: make(map[int64]quotaDisplayCollection), series: make(map[quotaDisplayKey]quotaDisplayValue)}
		if aggregate {
			metrics.DataCoordSegmentBinLogFileCount.WithLabelValues(metrics.AllLabel).Set(0)
		}
	}
	cache := m.quotaDisplay
	cache.revision = snapshot.Revision
	changed := make(map[quotaDisplayKey]bool)
	apply := func(id int64, c quotaDisplayCollection, sign int) {
		collection, name := strconv.FormatInt(id, 10), c.name
		if aggregate {
			collection, name = metrics.AllLabel, metrics.AllLabel
		}
		add := func(k quotaDisplayKey, value int64) {
			v := cache.series[k]
			v.value += int64(sign) * value
			v.contributors += sign
			cache.series[k] = v
			changed[k] = true
		}
		for _, state := range c.stats.Display {
			add(quotaDisplayKey{family: 0, db: c.db, collection: collection, state: state.State.String()}, state.Bytes)
			add(quotaDisplayKey{family: 1, db: c.db, collection: collection, name: name, state: state.State.String()}, state.Rows)
			add(quotaDisplayKey{family: 2, collection: collection}, state.Files)
		}
		if c.stats.HasL0 {
			add(quotaDisplayKey{family: 3, db: c.db, collection: collection}, c.stats.L0Rows)
		}
	}
	for id, value := range snapshot.Collections {
		if old, ok := cache.collections[id]; ok {
			apply(id, old, -1)
			delete(cache.collections, id)
		}
		if coll, ok := m.metaStore.GetCollection(id); ok && (value.HasQuota || value.HasL0) {
			current := quotaDisplayCollection{db: coll.DatabaseName, name: coll.Schema.GetName(), stats: value}
			cache.collections[id] = current
			apply(id, current, 1)
		}
	}
	for key := range changed {
		value := cache.series[key]
		if value.contributors == 0 {
			delete(cache.series, key)
			// Cardinality wrappers deliberately ignore collection-scoped deletes
			// in aggregate mode. Each remaining series is owned here, so delete
			// by its non-collection labels after its last contributor leaves.
			if aggregate {
				labels := prometheus.Labels{"db_name": key.db}
				switch key.family {
				case 0:
					labels["segment_state"] = key.state
					metrics.DataCoordStoredBinlogSize.DeletePartialMatch(labels)
				case 1:
					labels["segment_state"] = key.state
					metrics.DataCoordNumStoredRows.DeletePartialMatch(labels)
				case 2:
					metrics.DataCoordSegmentBinLogFileCount.WithLabelValues(metrics.AllLabel).Set(0)
				case 3:
					metrics.DataCoordL0DeleteEntriesNum.DeletePartialMatch(labels)
				}
				continue
			}
			switch key.family {
			case 0:
				metrics.DataCoordStoredBinlogSize.DeleteLabelValues(key.db, key.collection, key.state)
			case 1:
				metrics.DataCoordNumStoredRows.DeleteLabelValues(key.db, key.collection, key.name, key.state)
			case 2:
				metrics.DataCoordSegmentBinLogFileCount.DeleteLabelValues(key.collection)
			case 3:
				metrics.DataCoordL0DeleteEntriesNum.DeleteLabelValues(key.db, key.collection)
			}
			continue
		}
		switch key.family {
		case 0:
			metrics.DataCoordStoredBinlogSize.WithLabelValues(key.db, key.collection, key.state).Set(float64(value.value))
		case 1:
			metrics.DataCoordNumStoredRows.WithLabelValues(key.db, key.collection, key.name, key.state).Set(float64(value.value))
		case 2:
			metrics.DataCoordSegmentBinLogFileCount.WithLabelValues(key.collection).Set(float64(value.value))
		case 3:
			metrics.DataCoordL0DeleteEntriesNum.WithLabelValues(key.db, key.collection).Set(float64(value.value))
		}
	}
}
