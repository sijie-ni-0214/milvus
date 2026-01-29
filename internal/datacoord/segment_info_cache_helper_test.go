package datacoord

// newTestSegmentsInfo creates a SegmentsInfo pre-populated with segs; the map
// key is the cache key.
func newTestSegmentsInfo(segs map[int64]*SegmentInfo) *SegmentsInfo {
	s := NewSegmentsInfo()
	for id, seg := range segs {
		s.SetSegment(id, seg)
	}
	return s
}

// newTestCachedSegmentsInfo is newTestSegmentsInfo under the stack's name.
func newTestCachedSegmentsInfo(segs map[int64]*SegmentInfo) *CachedSegmentsInfo {
	return newTestSegmentsInfo(segs)
}

// snapshotMap returns the live segments keyed by ID.
func (s *CachedSegmentsInfo) snapshotMap() map[int64]*SegmentInfo {
	result := make(map[int64]*SegmentInfo, s.Len())
	s.Range(func(id int64, segment *SegmentInfo) bool {
		result[id] = segment
		return true
	})
	return result
}

// newTestSegmentsInfoWithoutIndexes reproduces a master-era literal that set
// only the segment map: segments are reachable by ID and by GetSegments, but
// collection/channel selections do not see them.
func newTestSegmentsInfoWithoutIndexes(segs map[int64]*SegmentInfo) *SegmentsInfo {
	s := NewSegmentsInfo()
	for id, seg := range segs {
		seg := seg
		s.segments.Swap(id, 0, func(*SegmentInfo, bool) *SegmentInfo { return seg })
	}
	return s
}
