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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/mocks/mock_kv"
)

func TestRangeBounds(t *testing.T) {
	t.Run("no split", func(t *testing.T) {
		assert.Empty(t, RangeBounds("p/", []int64{1, 2, 3}, 1))
		assert.Empty(t, RangeBounds("p/", []int64{1, 2, 3}, 0))
		assert.Empty(t, RangeBounds("p/", nil, 8))
		assert.Empty(t, RangeBounds("p/", []int64{7, 7, 7}, 8), "one distinct id cannot split")
	})

	t.Run("ids are ordered as their keys", func(t *testing.T) {
		// Key order: 10/ 100/ 11/ 2/ 9/ (decimal strings, not numbers).
		assert.Equal(t, []string{"p/100/", "p/2/"}, RangeBounds("p/", []int64{9, 10, 100, 11, 2}, 3))
		assert.Equal(t, []string{"p/100/", "p/11/", "p/2/", "p/9/"}, RangeBounds("p/", []int64{9, 10, 100, 11, 2, 2, 9}, 8),
			"duplicates collapse and fewer ids than ranges give fewer ranges")
	})

	t.Run("bounds increase under the prefix", func(t *testing.T) {
		ids := make([]int64, 0, 3*maxRangeBoundSamples)
		for i := range 3 * maxRangeBoundSamples {
			ids = append(ids, int64(i)*7919+1)
		}
		bounds := RangeBounds("p/", ids, 16)
		require.Len(t, bounds, 15)
		assert.True(t, slices.IsSorted(bounds))
		assert.Len(t, slices.Compact(slices.Clone(bounds)), 15)
		for _, bound := range bounds {
			id, err := strconv.ParseInt(bound[len("p/"):len(bound)-1], 10, 64)
			require.NoError(t, err)
			assert.Contains(t, ids, id)
		}
	})
}

type fakeRangeWalker struct {
	MetaKv
	bounds []string
}

func (w *fakeRangeWalker) WalkWithPrefixInRanges(_ context.Context, _ string, bounds []string, _ int, fn func(int, []byte, []byte) error) error {
	w.bounds = bounds
	for i := range bounds {
		if err := fn(i+1, []byte(bounds[i]), nil); err != nil {
			return err
		}
	}
	return nil
}

func TestWalkWithPrefixInRanges(t *testing.T) {
	ctx := context.Background()

	t.Run("a kv without ranges walks once as range 0", func(t *testing.T) {
		metaKv := mock_kv.NewMockMetaKv(t)
		metaKv.EXPECT().WalkWithPrefix(mock.Anything, "p/", 5, mock.Anything).RunAndReturn(
			func(_ context.Context, _ string, _ int, fn func([]byte, []byte) error) error {
				for _, key := range []string{"p/1", "p/2"} {
					if err := fn([]byte(key), nil); err != nil {
						return err
					}
				}
				return nil
			}).Once()
		var got []string
		require.NoError(t, WalkWithPrefixInRanges(ctx, metaKv, "p/", []string{"p/2"}, 5, func(rangeIndex int, key, _ []byte) error {
			got = append(got, strconv.Itoa(rangeIndex)+":"+string(key))
			return nil
		}))
		assert.Equal(t, []string{"0:p/1", "0:p/2"}, got)
	})

	t.Run("a range walker gets the bounds", func(t *testing.T) {
		walker := &fakeRangeWalker{}
		var got []int
		require.NoError(t, WalkWithPrefixInRanges(ctx, walker, "p/", []string{"p/2", "p/3"}, 5, func(rangeIndex int, _, _ []byte) error {
			got = append(got, rangeIndex)
			return nil
		}))
		assert.Equal(t, []string{"p/2", "p/3"}, walker.bounds)
		assert.Equal(t, []int{1, 2}, got)
	})

	t.Run("a range walker without bounds walks once", func(t *testing.T) {
		metaKv := mock_kv.NewMockMetaKv(t)
		metaKv.EXPECT().WalkWithPrefix(mock.Anything, "p/", 5, mock.Anything).Return(nil).Once()
		walker := &fakeRangeWalker{MetaKv: metaKv}
		require.NoError(t, WalkWithPrefixInRanges(ctx, walker, "p/", nil, 5, func(int, []byte, []byte) error { return nil }))
		assert.Nil(t, walker.bounds)
	})
}
