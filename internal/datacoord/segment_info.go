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
	"fmt"
	"slices"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/segmentutil"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// SegmentInfo wraps datapb.SegmentInfo and patches some extra info on it
type SegmentInfo struct {
	*datapb.SegmentInfo
	allocations     []*Allocation
	lastFlushTime   time.Time
	isCompacting    bool
	lastWrittenTime time.Time
}

// EnsureStats returns a non-nil Statistics view for read-only aggregate
// queries. It does NOT mutate s — concurrent lock-free readers would race
// otherwise. The persisted s.Stats is populated eagerly by
// NewSegmentInfo on construction and by the array-mutating operators
// (AddBinlogsOperator, UpdateBinlogsFromSaveBinlogPathsOperator,
// UpdateSegmentStats), both of which run on the update pack's private clone. When a
// caller hands us a SegmentInfo built via the struct literal
// `&SegmentInfo{SegmentInfo: ...}` with a nil Stats (the only remaining
// path is now legacy tests), we fall back to a transient recompute so
// readers see the right number; we just don't write it back.
func (s *SegmentInfo) EnsureStats() *datapb.Statistics {
	if s.SegmentInfo == nil {
		return nil
	}
	if stats := s.GetStats(); stats != nil {
		return stats
	}
	return storage.BuildStatsFromFieldBinlogs(s.GetBinlogs(), s.GetStatslogs(), s.GetBm25Statslogs(), s.GetDeltalogs())
}

func (s *SegmentInfo) GetResidualSegmentSize() int64 {
	if s.GetNumOfRows() == 0 {
		return 0
	}
	deltaRatio := float64(s.EnsureStats().GetDeleteNumRows()) / float64(s.GetNumOfRows())
	if deltaRatio >= 1.0 {
		// segments with too many deleted rows should be considered as prioritized segments and be compacted definitely
		return s.getSegmentSize()
	}
	residualRatio := 1.0 - deltaRatio
	return int64(residualRatio * float64(s.getSegmentSize()))
}

func (s *SegmentInfo) GetEarliestTs() uint64 {
	// For import segments, row timestamps predate the actual commit time.
	// Use commit_timestamp as the effective data age so compaction priority
	// and TTL decisions are not distorted by stale row timestamps.
	if commitTs := s.GetCommitTimestamp(); commitTs != 0 {
		return commitTs
	}
	// Stats.TimestampFrom is the exact min(TimestampFrom) across all insert
	// binlogs (populated by StatisticsCollector on the writer side, or by
	// BuildStatsFromFieldBinlogs on V2 fallback / migration).
	return s.EnsureStats().GetTimestampFrom()
}

// NewSegmentInfo create `SegmentInfo` wrapper from `datapb.SegmentInfo`
// assign current rows to last checkpoint and pre-allocate `allocations` slice
// Note that the allocation information is not preserved,
// the worst case scenario is to have a segment with twice size we expects
//
// Stats is populated from the FieldBinlog arrays when nil so legacy
// segments (persisted before Statistics existed) and live aggregate
// reads agree without callers needing a fallback. EnsureStats covers the
// struct-literal construction path that bypasses this constructor.
func NewSegmentInfo(info *datapb.SegmentInfo) *SegmentInfo {
	if info.Stats == nil {
		info.Stats = storage.BuildStatsFromFieldBinlogs(info.GetBinlogs(), info.GetStatslogs(), info.GetBm25Statslogs(), info.GetDeltalogs())
	}
	s := &SegmentInfo{
		SegmentInfo: info,
	}
	// setup growing fields
	if s.GetState() == commonpb.SegmentState_Growing {
		s.allocations = make([]*Allocation, 0, 16)
		s.lastFlushTime = time.Now().Add(-1 * paramtable.Get().DataCoordCfg.SegmentFlushInterval.GetAsDuration(time.Second))
		// A growing segment from recovery can be also considered idle.
		s.lastWrittenTime = getZeroTime()
	}
	return s
}

// normalizeSegmentRowCount reconciles a V1/V2 segment's NumOfRows with its
// insert binlogs before the segment enters memory. The catalog applies the same
// correction, but only to the copy it persists (kv_catalog.go), which would
// leave the in-memory segment — the one QueryCoord reads through the shared
// store — carrying the stale count until the next reload.
//
// V3 segments are exempt: their binlog arrays are a pass-through cache that can
// be empty after recovery or delta-only during a growing-source flush, so an
// array-derived count would be wrong.
func normalizeSegmentRowCount(segProto *datapb.SegmentInfo) {
	if segProto.GetStorageVersion() == storage.StorageV3 || segProto.GetManifestPath() != "" {
		return
	}
	segmentutil.ReCalcRowCount(segProto, segProto)
}
func (s *SegmentInfo) IsDeltaLogExists(logID int64) bool {
	for _, deltaLogs := range s.GetDeltalogs() {
		for _, l := range deltaLogs.GetBinlogs() {
			if l.GetLogID() == logID {
				return true
			}
		}
	}
	return false
}

func (s *SegmentInfo) IsStatsLogExists(logID int64) bool {
	for _, statsLogs := range s.GetStatslogs() {
		for _, l := range statsLogs.GetBinlogs() {
			if l.GetLogID() == logID {
				return true
			}
		}
	}
	return false
}

// Clone deep clone the segment info and return a new instance. Stats lives
// on the proto and is copied by proto.Clone, so the cloned segment's
// aggregate reads stay consistent with its (cloned) binlog arrays. Opts
// that replace binlogs should also refresh Stats eagerly (recompute via
// storage.BuildStatsFromFieldBinlogs); EnsureStats no longer writes back lazily —
// concurrent RLock readers would race.
func (s *SegmentInfo) Clone(opts ...SegmentInfoOption) *SegmentInfo {
	info := proto.Clone(s.SegmentInfo).(*datapb.SegmentInfo)
	cloned := &SegmentInfo{
		SegmentInfo:     info,
		allocations:     s.allocations,
		lastFlushTime:   s.lastFlushTime,
		isCompacting:    s.isCompacting,
		lastWrittenTime: s.lastWrittenTime,
	}
	for _, opt := range opts {
		opt(cloned)
	}
	return cloned
}

// ShadowClone shadow clone the segment and return a new instance
func (s *SegmentInfo) ShadowClone(opts ...SegmentInfoOption) *SegmentInfo {
	cloned := &SegmentInfo{
		SegmentInfo:     s.SegmentInfo,
		allocations:     s.allocations,
		lastFlushTime:   s.lastFlushTime,
		isCompacting:    s.isCompacting,
		lastWrittenTime: s.lastWrittenTime,
	}
	for _, opt := range opts {
		opt(cloned)
	}
	return cloned
}

// SegmentInfoOption is the option to set fields in segment info
type SegmentInfoOption func(segment *SegmentInfo)

// SetRowCount is the option to set row count for segment info
func SetRowCount(rowCount int64) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.NumOfRows = rowCount
	}
}

// SetExpireTime is the option to set expire time for segment info
func SetExpireTime(expireTs Timestamp) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.LastExpireTime = expireTs
	}
}

// SetState is the option to set state for segment info
func SetState(state commonpb.SegmentState) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.State = state
	}
}

// SetDmlPosition is the option to set dml position for segment info
func SetDmlPosition(pos *msgpb.MsgPosition) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.DmlPosition = pos
	}
}

// SetStartPosition is the option to set start position for segment info
func SetStartPosition(pos *msgpb.MsgPosition) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.StartPosition = pos
	}
}

// SetAllocations is the option to set allocations for segment info
func SetAllocations(allocations []*Allocation) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.allocations = allocations
	}
}

// AddAllocation is the option to add allocation info for segment info
func AddAllocation(allocation *Allocation) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.allocations = append(segment.allocations, allocation)
		segment.LastExpireTime = allocation.ExpireTime
	}
}

// SetLastWrittenTime is the option to set last writtent time for segment info
func SetLastWrittenTime() SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.lastWrittenTime = time.Now()
	}
}

// SetFlushTime is the option to set flush time for segment info
func SetFlushTime(t time.Time) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.lastFlushTime = t
	}
}

// SetIsCompacting is the option to set compaction state for segment info
func SetIsCompacting(isCompacting bool) SegmentInfoOption {
	return func(segment *SegmentInfo) {
		segment.isCompacting = isCompacting
	}
}

func (s *SegmentInfo) getSegmentSize() int64 {
	stats := s.EnsureStats()
	return stats.GetInsertBinlogSize() + stats.GetStatsBinlogSize() + stats.GetDeltaBinlogSize()
}

func (s *SegmentInfo) getFieldBinlogSize(fieldID int64) int64 {
	var size int64
	for _, binlogs := range s.GetBinlogs() {
		if binlogs.GetFieldID() == fieldID {
			for _, l := range binlogs.GetBinlogs() {
				size += l.GetMemorySize()
			}
		} else {
			for _, childFieldID := range binlogs.GetChildFields() {
				if childFieldID == fieldID {
					for _, l := range binlogs.GetBinlogs() {
						size += l.GetMemorySize()
					}
				}
			}
		}
	}
	// StorageV3 manifest segments persist no FieldBinlog arrays (the manifest
	// is the source of truth), so the loop above sums nothing for them and
	// the whole-segment fallback below would overcount the field by every
	// non-indexed field's bytes (e.g. index task slots sized by segment size
	// instead of vector size). Their per-column-group sizes ship inside
	// Stats.LoadResource instead; mirror the array matching rules (GroupId or
	// member FieldIds) so both storage versions size the same field the same.
	if size <= 0 {
		for _, group := range s.EnsureStats().GetLoadResource().GetColumnGroups() {
			if group.GetGroupId() == fieldID || slices.Contains(group.GetFieldIds(), fieldID) {
				size += group.GetMemorySize()
			}
		}
	}
	if size <= 0 {
		return s.getSegmentSize()
	}
	return size
}

func (s *SegmentInfo) getDeltaCount() int64 {
	return s.EnsureStats().GetDeleteNumRows()
}

// SegmentInfoSelector is the function type to select SegmentInfo from meta
type SegmentInfoSelector func(*SegmentInfo) bool

// ValidateManifestSegment checks that segments with manifest_path have empty
// legacy stats fields. Returns a descriptive message if validation fails,
// or empty string if the segment is valid.
func ValidateManifestSegment(info *SegmentInfo) string {
	if info.GetManifestPath() == "" {
		return ""
	}

	var nonEmpty []string
	if len(info.GetStatslogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("statslogs(%d)", len(info.GetStatslogs())))
	}
	if len(info.GetBm25Statslogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("bm25statslogs(%d)", len(info.GetBm25Statslogs())))
	}
	if len(info.GetTextStatsLogs()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("textStatsLogs(%d)", len(info.GetTextStatsLogs())))
	}
	if len(info.GetJsonKeyStats()) > 0 {
		nonEmpty = append(nonEmpty, fmt.Sprintf("jsonKeyStats(%d)", len(info.GetJsonKeyStats())))
	}

	if len(nonEmpty) > 0 {
		return fmt.Sprintf("segment %d has manifest_path but non-empty legacy stats fields: %v",
			info.GetID(), nonEmpty)
	}
	return ""
}

// segmentEffectiveTs returns the start-position timestamp that governs temporal
// decisions for a segment. For import segments with a non-zero commit_timestamp,
// commit_timestamp overrides start_position.Timestamp because the data was not
// "officially present" until the import was committed.
func segmentEffectiveTs(seg *datapb.SegmentInfo) uint64 {
	if ts := seg.GetCommitTimestamp(); ts != 0 {
		return ts
	}
	return seg.GetStartPosition().GetTimestamp()
}

// segmentEffectiveDmlTs returns the DML-position timestamp for temporal decisions.
// Same override logic as segmentEffectiveTs but for dml_position consumers
// (GC eligibility, TruncateChannelByTime).
func segmentEffectiveDmlTs(seg *datapb.SegmentInfo) uint64 {
	if ts := seg.GetCommitTimestamp(); ts != 0 {
		return ts
	}
	return seg.GetDmlPosition().GetTimestamp()
}
