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

package kv

import (
	"context"
	"slices"
	"strconv"
)

// RangeWalker is a MetaKv that can walk a prefix as contiguous key ranges read
// in parallel at one store revision.
type RangeWalker interface {
	// WalkWithPrefixInRanges calls fn with every key and value under prefix,
	// like WalkWithPrefix, split at bounds into the key ranges [prefix,
	// bounds[0]), [bounds[0], bounds[1]), ..., [bounds[len(bounds)-1], end of
	// prefix). bounds are keys under prefix (relative like it) in strictly
	// increasing order. The ranges are read in parallel at one store
	// revision, so together they return every key once, as one walk at that
	// revision would. fn receives the index of the range of each key: it is
	// called for the keys of one range in key order from one goroutine, and
	// for different ranges concurrently. The error returned is the one a
	// single walk would meet first.
	WalkWithPrefixInRanges(ctx context.Context, prefix string, bounds []string, paginationSize int, fn func(rangeIndex int, key, value []byte) error) error
}

// WalkWithPrefixInRanges walks prefix of metaKv as the ranges bounds split it
// into when metaKv is a RangeWalker. Any other MetaKv walks the prefix once
// with WalkWithPrefix and passes every key as range 0, so a caller that keeps
// a result per range and joins them in range order gets the key order of one
// walk either way.
func WalkWithPrefixInRanges(ctx context.Context, metaKv MetaKv, prefix string, bounds []string, paginationSize int, fn func(rangeIndex int, key, value []byte) error) error {
	if walker, ok := metaKv.(RangeWalker); ok && len(bounds) > 0 {
		return walker.WalkWithPrefixInRanges(ctx, prefix, bounds, paginationSize, fn)
	}
	return metaKv.WalkWithPrefix(ctx, prefix, paginationSize, func(key, value []byte) error {
		return fn(0, key, value)
	})
}

// maxRangeBoundSamples bounds the number of ids RangeBounds sorts.
const maxRangeBoundSamples = 1 << 16

// RangeBounds returns the bounds that split a prefix, whose keys start with
// <prefix><id>/, into at most ranges key ranges with about as many of ids in
// each: every bound is the first key <prefix><id>/ of one of ids. ids are
// ordered the way their keys are, as decimal strings rather than numbers; of
// more than maxRangeBoundSamples ids an evenly strided sample is used. Bounds
// only decide which range reads a key. The first range starts at the prefix
// and the last one ends with it, so keys of other ids, and keys of any other
// shape, are read all the same.
func RangeBounds(prefix string, ids []int64, ranges int) []string {
	if ranges <= 1 || len(ids) == 0 {
		return nil
	}
	stride := (len(ids) + maxRangeBoundSamples - 1) / maxRangeBoundSamples
	children := make([]string, 0, (len(ids)+stride-1)/stride)
	for i := 0; i < len(ids); i += stride {
		children = append(children, strconv.FormatInt(ids[i], 10)+"/")
	}
	slices.Sort(children)
	children = slices.Compact(children)

	bounds := make([]string, 0, ranges-1)
	last := 0
	for i := 1; i < ranges; i++ {
		// A bound at the first id would leave the first range only the keys
		// before every id.
		if j := i * len(children) / ranges; j > last {
			bounds = append(bounds, prefix+children[j])
			last = j
		}
	}
	return bounds
}
