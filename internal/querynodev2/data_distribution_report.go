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

package querynodev2

import (
	"strconv"
	"sync"

	"github.com/samber/lo"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/querynodev2/delegator"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

const (
	dataDistributionReportFull     = "full"
	dataDistributionReportDelta    = "delta"
	dataDistributionReportNoChange = "no_change"
)

type dataDistributionDeltaTracker struct {
	mu sync.Mutex

	lastReportedTs int64

	dirtySegments     map[int64]struct{}
	removedSegments   map[int64]struct{}
	totalSegmentCount int64

	dirtyChannels     map[string]struct{}
	removedChannels   map[string]struct{}
	totalChannelCount int64
	dirtyLeaderViews  map[string]struct{}
}

func newDataDistributionDeltaTracker() *dataDistributionDeltaTracker {
	return &dataDistributionDeltaTracker{
		dirtySegments:    make(map[int64]struct{}),
		removedSegments:  make(map[int64]struct{}),
		dirtyChannels:    make(map[string]struct{}),
		removedChannels:  make(map[string]struct{}),
		dirtyLeaderViews: make(map[string]struct{}),
	}
}

func (c *dataDistributionDeltaTracker) reset() {
	c.lastReportedTs = 0
	c.totalSegmentCount = 0
	c.totalChannelCount = 0
	c.clearDelta()
}

func (c *dataDistributionDeltaTracker) clearDelta() {
	c.dirtySegments = make(map[int64]struct{})
	c.removedSegments = make(map[int64]struct{})
	c.dirtyChannels = make(map[string]struct{})
	c.removedChannels = make(map[string]struct{})
	c.dirtyLeaderViews = make(map[string]struct{})
}

func (c *dataDistributionDeltaTracker) canBuildDelta(req *querypb.GetDataDistributionRequest) bool {
	return req.GetSupportDelta() && req.GetLastUpdateTs() != 0 && req.GetLastUpdateTs() == c.lastReportedTs
}

func (c *dataDistributionDeltaTracker) markSegmentUpsert(segmentIDs ...int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, segmentID := range segmentIDs {
		delete(c.removedSegments, segmentID)
		c.dirtySegments[segmentID] = struct{}{}
	}
}

func (c *dataDistributionDeltaTracker) markSegmentRemove(segmentIDs ...int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, segmentID := range segmentIDs {
		delete(c.dirtySegments, segmentID)
		c.removedSegments[segmentID] = struct{}{}
	}
}

func (c *dataDistributionDeltaTracker) markChannelUpsert(channel string) {
	if channel == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.removedChannels, channel)
	c.dirtyChannels[channel] = struct{}{}
}

func (c *dataDistributionDeltaTracker) markChannelRemove(channel string) {
	if channel == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.dirtyChannels, channel)
	c.removedChannels[channel] = struct{}{}
}

func (c *dataDistributionDeltaTracker) markLeaderViewDirty(channel string) {
	if channel == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	c.dirtyLeaderViews[channel] = struct{}{}
}

func (c *dataDistributionDeltaTracker) markLeaderSegmentUpsert(channel string, segmentID, nodeID, version int64) {
	if channel == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	c.dirtyLeaderViews[channel] = struct{}{}
}

func (c *dataDistributionDeltaTracker) markLeaderSegmentRemove(channel string, segmentIDs ...int64) {
	if channel == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	c.dirtyLeaderViews[channel] = struct{}{}
}

func buildSegmentVersionInfo(s segments.Segment) *querypb.SegmentVersionInfo {
	return &querypb.SegmentVersionInfo{
		ID:                 s.ID(),
		Collection:         s.Collection(),
		Partition:          s.Partition(),
		Channel:            s.Shard().VirtualName(),
		Version:            s.Version(),
		Level:              s.Level(),
		IsSorted:           s.IsSorted(),
		LastDeltaTimestamp: s.LastDeltaTimestamp(),
		IndexInfo: lo.SliceToMap(s.Indexes(), func(info *segments.IndexedFieldInfo) (int64, *querypb.FieldIndexInfo) {
			return info.IndexInfo.IndexID, info.IndexInfo
		}),
		JsonStatsInfo: s.GetFieldJSONIndexStats(),
		ManifestPath:  s.LoadInfo().GetManifestPath(),
		DataVersion:   protoInt32(s.LoadInfo().GetDataVersion()),
	}
}

func protoInt32(v int32) *int32 {
	return &v
}

func recordDataDistributionReportMetrics(nodeID int64, reportType string, resp *querypb.GetDataDistributionResponse) {
	nodeIDStr := strconv.FormatInt(nodeID, 10)
	leaderSegmentCount := 0
	removedSegmentCount := len(resp.GetRemovedSegmentIds())
	for _, view := range resp.GetLeaderViews() {
		leaderSegmentCount += len(view.GetSegmentDist())
	}
	metrics.QueryNodeDataDistributionReportTotal.WithLabelValues(nodeIDStr, reportType).Inc()
	metrics.QueryNodeDataDistributionReportBytes.WithLabelValues(nodeIDStr, reportType).Add(float64(proto.Size(resp)))
	metrics.QueryNodeDataDistributionReportSegmentTotal.WithLabelValues(nodeIDStr, reportType).Add(float64(len(resp.GetSegments())))
	metrics.QueryNodeDataDistributionReportLeaderSegmentTotal.WithLabelValues(nodeIDStr, reportType).Add(float64(leaderSegmentCount))
	metrics.QueryNodeDataDistributionReportRemovedSegmentTotal.WithLabelValues(nodeIDStr, reportType).Add(float64(removedSegmentCount))
}

func buildFullSegmentDist(sealed []delegator.SnapshotItem) map[int64]*querypb.SegmentDist {
	segmentDist := make(map[int64]*querypb.SegmentDist)
	for _, item := range sealed {
		for _, segment := range item.Segments {
			segmentDist[segment.SegmentID] = &querypb.SegmentDist{
				NodeID:  item.NodeID,
				Version: segment.Version,
			}
		}
	}
	return segmentDist
}
