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

package metacache

import (
	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

type segmentRecoveryMetricAggregates struct {
	segmentCounts            map[[5]string]int64
	insertFileCountFrequency map[int64]int64
	statFileCountFrequency   map[int64]int64
	deleteFileCountFrequency map[int64]int64
	numStoredRows            int64
}

func newSegmentRecoveryMetricAggregates() *segmentRecoveryMetricAggregates {
	return &segmentRecoveryMetricAggregates{
		segmentCounts:            make(map[[5]string]int64),
		insertFileCountFrequency: make(map[int64]int64),
		statFileCountFrequency:   make(map[int64]int64),
		deleteFileCountFrequency: make(map[int64]int64),
	}
}

func (a *segmentRecoveryMetricAggregates) Add(info *datapb.SegmentInfo) {
	a.segmentCounts[segMetricLabels(info)]++
	if info.GetState() != commonpb.SegmentState_Flushed {
		return
	}

	a.numStoredRows += info.GetNumOfRows()
	stats := info.GetStats()
	a.insertFileCountFrequency[stats.GetInsertBinlogCount()]++
	statFileCount := int64(0)
	for _, fieldBinlog := range info.GetStatslogs() {
		statFileCount += int64(len(fieldBinlog.GetBinlogs()))
	}
	a.statFileCountFrequency[statFileCount]++
	a.deleteFileCountFrequency[stats.GetDeltaBinlogCount()]++
}

func (a *segmentRecoveryMetricAggregates) PublishSegmentCounts() {
	metrics.DataCoordNumSegments.Reset()
	for labels, count := range a.segmentCounts {
		metrics.DataCoordNumSegments.WithLabelValues(labels[:]...).Add(float64(count))
	}
}

// ReplayFileHistogramsAsync observes the per-segment file count histograms off
// the recovery path; at millions of segments the individual observations
// would otherwise dominate the cache rebuild.
func (a *segmentRecoveryMetricAggregates) ReplayFileHistogramsAsync() {
	go func() {
		for fileCount, frequency := range a.insertFileCountFrequency {
			for range frequency {
				metrics.FlushedSegmentFileNum.WithLabelValues(metrics.InsertFileLabel).Observe(float64(fileCount))
			}
		}
		for fileCount, frequency := range a.statFileCountFrequency {
			for range frequency {
				metrics.FlushedSegmentFileNum.WithLabelValues(metrics.StatFileLabel).Observe(float64(fileCount))
			}
		}
		for fileCount, frequency := range a.deleteFileCountFrequency {
			for range frequency {
				metrics.FlushedSegmentFileNum.WithLabelValues(metrics.DeleteFileLabel).Observe(float64(fileCount))
			}
		}
	}()
}

// merge adds the aggregates of other.
func (a *segmentRecoveryMetricAggregates) merge(other *segmentRecoveryMetricAggregates) {
	for labels, count := range other.segmentCounts {
		a.segmentCounts[labels] += count
	}
	for fileCount, frequency := range other.insertFileCountFrequency {
		a.insertFileCountFrequency[fileCount] += frequency
	}
	for fileCount, frequency := range other.statFileCountFrequency {
		a.statFileCountFrequency[fileCount] += frequency
	}
	for fileCount, frequency := range other.deleteFileCountFrequency {
		a.deleteFileCountFrequency[fileCount] += frequency
	}
	a.numStoredRows += other.numStoredRows
}
