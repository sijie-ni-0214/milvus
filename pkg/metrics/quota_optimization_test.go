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
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestQuotaCollectionMetricCleanup(t *testing.T) {
	previous := CollectionLevelMetricsMode()
	t.Cleanup(func() {
		SetCollectionLevelMetricsMode(previous)
		RootCoordNumEntities.Reset()
		RootCoordIndexedNumEntities.Reset()
	})
	for _, mode := range []string{CollectionLevelMetricsModeFull, CollectionLevelMetricsModeAggregate} {
		t.Run(mode, func(t *testing.T) {
			SetCollectionLevelMetricsMode(mode)
			RootCoordNumEntities.Reset()
			RootCoordIndexedNumEntities.Reset()
			RootCoordNumEntities.WithLabelValues("db", "old", TotalLabel).Set(10)
			RootCoordIndexedNumEntities.WithLabelValues("db", "old", "index", "true").Set(10)
			RootCoordNumEntities.WithLabelValues("db", "keep", TotalLabel).Set(20)
			RootCoordIndexedNumEntities.WithLabelValues("db", "keep", "index", "true").Set(20)
			CleanupRootCoordCollectionMetrics("db", "old")
			require.Equal(t, 1, testutil.CollectAndCount(RootCoordNumEntities))
			require.Equal(t, 1, testutil.CollectAndCount(RootCoordIndexedNumEntities))
			require.EqualValues(t, 20, testutil.ToFloat64(RootCoordNumEntities.WithLabelValues("db", "keep", TotalLabel)))
			require.EqualValues(t, 20, testutil.ToFloat64(RootCoordIndexedNumEntities.WithLabelValues("db", "keep", "index", "true")))
		})
	}
}
